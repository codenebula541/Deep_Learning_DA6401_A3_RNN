############################################################
#  attn_seq2seq_hi_final.py
#  --------------------------------------------------------
#  • Luong‑attention, single‑layer encoder/decoder
#  • Stage "search": sweep → logs attn_train_loss, attn_val_acc, …
#  • Stage "best"  : fixed cfg → 15 ep + test + predictions + 3×3 heat‑map
############################################################

from __future__ import annotations
import os, random, math, shutil
from dataclasses import dataclass
from typing import Dict, Literal
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import wandb
os.environ["WANDB_START_METHOD"] = "thread"
wandb.login(key="")
# ---------------- Dataset ---------------- #
class DakLex(Dataset):
    def __init__(self, path: str, src_map=None, trg_map=None):
        pairs = [(p[1], p[0]) for p in (l.split('\t')[:2] for l in open(path, encoding='utf-8') if l.strip())]
        self.SOS, self.EOS, self.PAD = 1, 2, 0
        self.pairs = pairs
        if src_map is None:
            src_map = {c: i+3 for i, c in enumerate(sorted({ch for s, _ in pairs for ch in s}))}
        if trg_map is None:
            trg_map = {c: i+3 for i, c in enumerate(sorted({ch for _, t in pairs for ch in t}))}
        self.src_map, self.trg_map = src_map, trg_map

    def __len__(self):
        return len(self.pairs)

    def encode(self, s, tbl):
        return [self.SOS] + [tbl[c] for c in s] + [self.EOS]

    def __getitem__(self, idx):
        s, t = self.pairs[idx]
        return torch.tensor(self.encode(s, self.src_map)), torch.tensor(self.encode(t, self.trg_map))

def pad_batch(batch):
    s, t = zip(*batch)
    return (nn.utils.rnn.pad_sequence(s, batch_first=True),
            nn.utils.rnn.pad_sequence(t, batch_first=True))

# ---------------- Model ---------------- #
CellType = Literal["gru", "lstm"]
def rnn(cell): return nn.GRU if cell == "gru" else nn.LSTM

class Encoder(nn.Module):
    def __init__(self, v, e, h, c):
        super().__init__()
        self.emb = nn.Embedding(v, e)
        self.rnn = rnn(c)(e, h, batch_first=True)

    def forward(self, x):
        emb = self.emb(x)
        out, h = self.rnn(emb)
        return out, h

class LuongAttn(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.scale = 1 / math.sqrt(h)

    def forward(self, d, enc):
        w = torch.bmm(d, enc.transpose(1, 2)) * self.scale
        a = w.softmax(-1)
        ctx = torch.bmm(a, enc)
        return ctx.squeeze(1), a

class Decoder(nn.Module):
    def __init__(self, v, e, h, c):
        super().__init__()
        self.emb = nn.Embedding(v, e)
        self.rnn = rnn(c)(e, h, batch_first=True)
        self.attn = LuongAttn(h)
        self.fc = nn.Linear(h * 2, v)
        self.sm = nn.LogSoftmax(-1)

    def forward(self, trg, h, enc, tf=0.5, collect=False):
        B, T = trg.size()
        V = self.fc.out_features
        out = torch.zeros(B, T, V, device=trg.device)
        maps = []
        tok = trg[:, 0]
        for t in range(1, T):
            o, h = self.rnn(self.emb(tok).unsqueeze(1), h)
            ctx, att = self.attn(o, enc)
            logp = self.sm(self.fc(torch.cat((o.squeeze(1), ctx), 1)))
            out[:, t] = logp
            tok = trg[:, t] if random.random() < tf else logp.argmax(1)
            if collect:
                maps.append(att.squeeze(1).cpu())
        return out, (torch.stack(maps, 1) if collect else None)

class Seq2Seq(nn.Module):
    def __init__(self, sv, tv, e, h, c):
        super().__init__()
        self.enc = Encoder(sv, e, h, c)
        self.dec = Decoder(tv, e, h, c)

    def forward(self, s, t, tf=0.5, collect=False):
        enc, h = self.enc(s)
        return self.dec(t, h, enc, tf, collect)

# ---------------- Trainer ---------------- #
class Trainer:
    def __init__(self, m, pad, lr):
        self.m = m.to(device)
        self.pad = pad
        self.crit = nn.NLLLoss(ignore_index=pad)
        self.opt = torch.optim.Adam(m.parameters(), lr=lr)

    def _loss(self, l, t):
        return self.crit(l[:, 1:].reshape(-1, l.size(-1)), t[:, 1:].reshape(-1))

    def _acc(self, l, t):
        p = l.argmax(-1)[:, 1:]
        g = t[:, 1:]
        m = g != self.pad
        return ((p == g) | ~m).all(1).float().mean().item()

    def _pass(self, ldr, train):
        self.m.train() if train else self.m.eval()
        totL = totA = tot = 0
        for s, t in ldr:
            s, t = s.to(device), t.to(device)
            if train:
                self.opt.zero_grad()
            l, _ = self.m(s, t, 0.5 if train else 0.)
            loss = self._loss(l, t)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(self.m.parameters(), 1.)
                self.opt.step()
            bs = s.size(0)
            tot += bs
            totL += loss.item() * bs
            totA += self._acc(l, t) * bs
        return totL / tot, totA / tot

    def train_epoch(self, ldr): return self._pass(ldr, True)
    def eval_epoch (self, ldr): return self._pass(ldr, False)

# ---------------- Config ---------------- #
@dataclass
class BaseCfg:
    data_root: str = "/kaggle/input/mydataset"
    embed_dim: int = 128
    hidden_size: int = 512
    cell_type: CellType = "gru"
    dropout: float = 0.1
    lr: float = 5e-4
    batch_size: int = 64
    epochs: int = 15

def sweep_cfg():
    return {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "embed_dim": {"values": [64, 128, 256]},
            "hidden_size": {"values": [128, 256, 512]},
            "cell_type": {"values": ["gru", "lstm"]},
            "dropout": {"values": [0, 0.1, 0.3]},
            "lr": {"values": [1e-3, 5e-4]}
        }
    }

def prefixed(d: dict, p: str):
    return {f"{p}{k}": v for k, v in d.items()} if p else d

# The rest remains the same. Add your stage_search, stage_best, and CLI below as needed.


# -------------- Stage functions -------------------- #
def stage_search(cfg):
    tr=DakLex(f"{cfg.data_root}/hi.translit.sampled.train.tsv")
    dv=DakLex(f"{cfg.data_root}/hi.translit.sampled.dev.tsv",tr.src_map,tr.trg_map)
    tl=DataLoader(tr,cfg.batch_size,shuffle=True ,collate_fn=pad_batch)
    dl=DataLoader(dv,cfg.batch_size,shuffle=False,collate_fn=pad_batch)

    model=Seq2Seq(len(tr.src_map)+3,len(tr.trg_map)+3,cfg.embed_dim,cfg.hidden_size,cfg.cell_type)
    T=Trainer(model,tr.PAD,cfg.lr)
    pre="attn_"                                  # always attention here
    for e in range(1,cfg.epochs+1):
        trL,trA=T.train_epoch(tl); dvL,dvA=T.eval_epoch(dl)
        wandb.log(prefixed({"epoch":e,"train_loss":trL,"val_loss":dvL,
                            "train_acc":trA,"val_acc":dvA},pre))
        print(f"[search] Ep{e:02d} Acc {trA:.3f}/{dvA:.3f}")

def stage_best(cfg):
    tr = DakLex(f"{cfg.data_root}/hi.translit.sampled.train.tsv")
    dv = DakLex(f"{cfg.data_root}/hi.translit.sampled.dev.tsv", tr.src_map, tr.trg_map)
    ts = DakLex(f"{cfg.data_root}/hi.translit.sampled.test.tsv", tr.src_map, tr.trg_map)
    tl = DataLoader(tr, cfg.batch_size, shuffle=True, collate_fn=pad_batch)
    dl = DataLoader(dv, cfg.batch_size, shuffle=False, collate_fn=pad_batch)
    sl = DataLoader(ts, cfg.batch_size, shuffle=False, collate_fn=pad_batch)

    model = Seq2Seq(len(tr.src_map) + 3, len(tr.trg_map) + 3,
                    cfg.embed_dim, cfg.hidden_size, cfg.cell_type).to(device)
    T = Trainer(model, tr.PAD, cfg.lr)
    pre = "attn_"

    best_val_acc = -1
    best_model_path = "best_attention_model.pt"

    for e in range(1, cfg.epochs + 1):
        tL, tA = T.train_epoch(tl)
        vL, vA = T.eval_epoch(dl)

        wandb.log(prefixed({
            "epoch": e, "train_loss": tL, "val_loss": vL,
            "train_acc": tA, "val_acc": vA
        }, pre))
        print(f"[best] Ep{e:02d} {tA:.3f}/{vA:.3f}")

        # Save model if this is the best val accuracy so far
        if vA > best_val_acc:
            best_val_acc = vA
            torch.save(model.state_dict(), best_model_path)
            print(f" Saved new best model at epoch {e} with val acc: {vA:.4f}")

    # Load best model before running on test data
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    sL, sA = T.eval_epoch(sl)
    wandb.log(prefixed({"test_loss": sL, "test_acc": sA}, pre))
    print(f" Final TEST acc = {sA:.4f}")


    # predictions + heatmap grid

    # Save predictions + randomly sampled attention heatmaps
    out_dir = "predictions_attention"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)
    pf = f"{out_dir}/hi.test.pred.tsv"
    
    rev_src = {v: k for k, v in tr.src_map.items()}
    rev_trg = {v: k for k, v in tr.trg_map.items()}
    ids = lambda x, rev: "".join(rev[i] for i in x if i > 2)
    
    # Randomly sample 9 unique test examples for heatmaps
    sample_ids = set(random.sample(range(len(ts)), 9))
    
    imgs = []
    with open(pf, "w", encoding="utf-8") as fh:
        for i, (s_ids, t_ids) in enumerate(ts):
            s_ids, t_ids = s_ids.to(device), t_ids.to(device)
            logp, att = model(s_ids.unsqueeze(0), t_ids.unsqueeze(0), 0.0, True)
            pred = logp.argmax(-1).squeeze(0).tolist()[1:]
    
            # Write full prediction to output file
            fh.write(f"{ids(s_ids.tolist()[1:], rev_src)}\t{ids(t_ids.tolist()[1:], rev_trg)}\t{ids(pred, rev_trg)}\n")
    
            # Log heatmap only if this is one of the randomly chosen 9
            if i in sample_ids:
                att = att.squeeze(0).detach().cpu().numpy()[:len(t_ids) - 2, :len(s_ids) - 2]
                fig = plt.figure(figsize=(3, 3))
                sns.heatmap(att,
                            xticklabels=[rev_src[x] for x in s_ids.tolist()[1:-1]],
                            yticklabels=[rev_trg[x] for x in t_ids.tolist()[1:-1]],
                            cmap="Blues", cbar=False)
                plt.xlabel("Latin"); plt.ylabel("Dev"); plt.close(fig)
                imgs.append(wandb.Image(fig))
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        if imgs:                              # imgs now contains your 9 figures
            fig_grid, axes = plt.subplots(3, 3, figsize=(9, 9))
            for ax, wimg in zip(axes.flatten(), imgs):
                im_arr = plt.imread(wimg._image_path)     # PNG saved by wandb.Image
                ax.imshow(im_arr)
                ax.axis("off")
            plt.tight_layout()
            plt.show()                        # renders the 3×3 grid in Kaggle output
            
    wandb.log(prefixed({"attention_grid": imgs}, pre))
    print("Predictions saved →", pf)


# --------------- CLI --------------- #
def cli(stage="search"):
    project = "transliteration_model-3"      # keep same project as baseline
    cfg = BaseCfg()                          # your hand‑picked best config

    if stage == "search":                    # Stage‑1 sweep (train/val only)
        sweep_id = wandb.sweep(sweep_cfg(), project=project)
        print("Sweep ID:", sweep_id)

        def job():
            wandb.init(
                project=project,
                config={**BaseCfg().__dict__, "run_type": "attention"}  # tag runs
            )
            stage_search(wandb.config)       # logs attn_train_loss, ...

        wandb.agent(sweep_id, function=job, count=20)

    elif stage == "best":                    # Stage‑2 single fixed run
        wandb.init(
            project=project,
            config={**cfg.__dict__, "run_type": "attention"}            # tag run
        )
        stage_best(cfg)                      # trains 15 ep, tests, saves preds

cli("best")

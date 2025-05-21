"""seq2seq_hi_sweep.py (Kaggle‑compatible, GPU‑enabled, YAML‑free)
===============================================================
• Runs W&B sweep or single experiment inside Kaggle Notebook with GPU support.
• Uses **beam‑search decoding** (beam_size is a sweep hyper‑parameter).
• Logs train/val loss **and full‑word accuracy** per epoch.
• Fixed 15‑epoch training (no early stopping).
"""

from __future__ import annotations
import os, random
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb

# --- WandB login (use Kaggle secret or paste key) --- #
# wandb.login()
os.environ["WANDB_START_METHOD"] = "thread"
wandb.login(key="")
# --- Device --- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model components --- #
CellType = Literal['rnn', 'gru', 'lstm']

def _get_rnn(cell: CellType):
    return {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}[cell]

class Encoder(nn.Module):
    def __init__(self, vocab:int, embed:int, hidden:int, layers:int, cell:CellType, drop:float):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.rnn   = _get_rnn(cell)(embed, hidden, num_layers=layers, batch_first=True,
                                    dropout=drop if layers>1 else 0.)
    def forward(self, x):
        return self.rnn(self.embed(x))[1]          # hidden only

class Decoder(nn.Module):
    def __init__(self, vocab:int, embed:int, hidden:int, layers:int, cell:CellType, drop:float, sos:int, eos:int):
        super().__init__()
        self.sos, self.eos = sos, eos
        self.embed  = nn.Embedding(vocab, embed)
        self.rnn    = _get_rnn(cell)(embed, hidden, num_layers=layers, batch_first=True,
                                     dropout=drop if layers>1 else 0.)
        self.proj   = nn.Linear(hidden, vocab)
        self.logsm  = nn.LogSoftmax(-1)

    def forward(self, trg, hidden, tf_ratio=0.5):
        B,T = trg.size(); V = self.proj.out_features
        out  = trg.new_zeros(B, T, V, dtype=torch.float, device=trg.device)
        token = trg[:,0]
        for t in range(1,T):
            logp,hidden = self.step(token, hidden)
            out[:,t] = logp
            token = trg[:,t] if random.random() < tf_ratio else logp.argmax(1)
        return out

    def step(self, token, hidden):
        emb = self.embed(token.unsqueeze(1))
        o, h = self.rnn(emb, hidden)
        return self.logsm(self.proj(o.squeeze(1))), h

    @torch.no_grad()
    def beam_search(self, hidden, beam_size:int=3, max_len:int=40):
        # returns top sequence (list[int])
        device = next(self.parameters()).device
        sequences = [([self.sos], 0.0, hidden)]
        for _ in range(max_len):
            all_cand = []
            for seq, score, h in sequences:
                if seq[-1] == self.eos:
                    all_cand.append((seq, score, h)); continue
                token = torch.tensor([seq[-1]], device=device)
                logp, h_new = self.step(token, h)
                topv, topi = torch.topk(logp, beam_size)
                for k in range(beam_size):
                    idx = topi[0, k].item()
                    score_k = topv[0, k].item()
                    all_cand.append((seq + [idx], score + score_k, h_new))
            sequences = sorted(all_cand, key=lambda x: x[1], reverse=True)[:beam_size]
        return sequences[0][0]

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab:int, trg_vocab:int, embed:int, hidden:int, layers:int, cell:CellType, drop:float, sos:int, eos:int):
        super().__init__()
        self.enc = Encoder(src_vocab, embed, hidden, layers, cell, drop)
        self.dec = Decoder(trg_vocab, embed, hidden, layers, cell, drop, sos, eos)
    def forward(self, src, trg, tf=0.5):
        return self.dec(trg, self.enc(src), tf)
    @torch.no_grad()
    def translate(self, src, beam_size:int=3, max_len:int=40):
        hidden = self.enc(src)
        return self.dec.beam_search(hidden, beam_size=beam_size, max_len=max_len)

# --- Dataset --- #
class DakLex(Dataset):
    def __init__(self, path:str, src_map:Dict[str,int]|None=None, trg_map:Dict[str,int]|None=None):
        pairs = [(cols[1], cols[0]) for cols in (l.split('\t')[:2] for l in open(path, encoding='utf-8') if l.strip())]
        self.SOS, self.EOS, self.PAD = 1,2,0
        self.pairs = pairs
        if src_map is None:
            src_map = {c:i+3 for i,c in enumerate(sorted({ch for s,_ in pairs for ch in s}))}
        if trg_map is None:
            trg_map = {c:i+3 for i,c in enumerate(sorted({ch for _,t in pairs for ch in t}))}
        self.src_map, self.trg_map = src_map, trg_map
    def __len__(self): return len(self.pairs)
    def encode(self, s:str, tbl): return [self.SOS]+[tbl[c] for c in s]+[self.EOS]
    def __getitem__(self, idx):
        src, trg = self.pairs[idx]
        return (torch.tensor(self.encode(src, self.src_map)),
                torch.tensor(self.encode(trg, self.trg_map)))

def pad_batch(batch):
    s,t = zip(*batch)
    return nn.utils.rnn.pad_sequence(s, batch_first=True), nn.utils.rnn.pad_sequence(t, batch_first=True)

# --- Trainer --- #

class Trainer:
    def __init__(self, model: Seq2Seq, pad: int, lr: float):
        self.model = model.to(device)
        self.pad   = pad
        self.crit  = nn.NLLLoss(ignore_index=pad)
        self.opt   = torch.optim.Adam(model.parameters(), lr=lr)

    def _loss(self, logp, trg):
        logp = logp[:, 1:].reshape(-1, logp.size(-1))
        trg  = trg[:, 1:].reshape(-1)
        return self.crit(logp, trg)

    def _full_word_acc(self, pred_ids, gold_ids):
        # pred_ids, gold_ids = (B, T) tensors already without <sos>
        mask = gold_ids != self.pad
        batch_acc = (pred_ids.eq(gold_ids) | ~mask).all(dim=1).float()  # 1 if every non‑pad matches
        return batch_acc.mean().item()

    def train_epoch(self, loader, tf_ratio: float):
        """Greedy decoding (arg‑max) for full‑word accuracy."""
        self.model.train()
        tot_loss, tot_acc, tot = 0., 0., 0
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            self.opt.zero_grad()
            logp = self.model(src, trg, tf_ratio)          # teacher forcing
            loss = self._loss(logp, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.opt.step()

            # greedy prediction
            pred_ids = logp.argmax(-1)[:, 1:]   # drop <sos>
            gold_ids = trg[:, 1:]
            acc = self._full_word_acc(pred_ids, gold_ids)

            bs = src.size(0)
            tot      += bs
            tot_loss += loss.item() * bs
            tot_acc  += acc * bs
        return tot_loss / tot, tot_acc / tot

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        tot_loss, tot_acc, tot = 0., 0., 0
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            logp = self.model(src, trg, tf=0.0)
            loss = self._loss(logp, trg)

            pred_ids = logp.argmax(-1)[:, 1:]
            gold_ids = trg[:, 1:]
            acc = self._full_word_acc(pred_ids, gold_ids)

            bs = src.size(0)
            tot      += bs
            tot_loss += loss.item() * bs
            tot_acc  += acc * bs
        return tot_loss / tot, tot_acc / tot

# --- Config & sweep --- #
@dataclass
class BaseCfg:
    data_root:str="/kaggle/input/mydataset"
    embed_dim:int=128; hidden_size:int=256; enc_layers:int=3; cell_type:CellType='lstm'
    dropout:float=0.3; lr:float=1e-3; batch_size:int=64; epochs:int=15

def create_sweep():
    return {
        'method':'bayes',
        'metric':{'name':'val_loss','goal':'minimize'},
        'parameters':{
            'embed_dim':   {'values':[32,64,128]},
            'hidden_size': {'values':[64,128,256]},
            'enc_layers':  {'values':[1,2,3]},
            'cell_type':   {'values':['gru','lstm']},
            'dropout':     {'values':[0, 0.1,0.3]},
            'lr':          {'values':[1e-3,5e-4]}
        }
    }
    

def run(cfg):
    # -------- build dataset -------- #
    train_ds = DakLex(os.path.join(cfg.data_root, "hi.translit.sampled.train.tsv"))
    dev_ds   = DakLex(os.path.join(cfg.data_root, "hi.translit.sampled.dev.tsv"),
                      train_ds.src_map, train_ds.trg_map)
    test_ds  = DakLex(os.path.join(cfg.data_root, "hi.translit.sampled.test.tsv"),
                      train_ds.src_map, train_ds.trg_map)

    B = cfg.batch_size
    train_loader = DataLoader(train_ds, B, shuffle=True,  collate_fn=pad_batch, num_workers=0)
    dev_loader   = DataLoader(dev_ds,  B, shuffle=False, collate_fn=pad_batch, num_workers=0)
    test_loader  = DataLoader(test_ds, B, shuffle=False, collate_fn=pad_batch, num_workers=0)

    # -------- build model -------- #
    model = Seq2Seq(
        len(train_ds.src_map) + 3,
        len(train_ds.trg_map) + 3,
        cfg.embed_dim,
        cfg.hidden_size,
        cfg.enc_layers,
        cfg.cell_type,
        cfg.dropout,
        train_ds.SOS,
        train_ds.EOS,
    )

    trainer = Trainer(model, train_ds.PAD, cfg.lr)

    # -------- training loop -------- #
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, tf_ratio=0.5)
        val_loss,   val_acc   = trainer.eval_epoch(dev_loader)

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "train_acc":  train_acc,
            "val_acc":    val_acc,
        })
        print(f"Epoch {epoch:2d}: TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f}")

    # -------- test evaluation -------- #
    test_loss, test_acc = trainer.eval_epoch(test_loader)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    print(f"TEST  : acc={test_acc:.4f}")

    # -------- save predictions & 20‑sample table -------- #
    rev_src = {v: k for k, v in train_ds.src_map.items()}
    rev_trg = {v: k for k, v in train_ds.trg_map.items()}

    os.makedirs("predictions_vanilla", exist_ok=True)
    pred_file = "predictions_vanilla/hi.test.pred.tsv"
    table     = wandb.Table(columns=["latin_input", "gold", "prediction", "correct"])

    def ids_to_str(ids, rev):
        return "".join(rev[i] for i in ids if i > 2)

    with open(pred_file, "w", encoding="utf-8") as fh:
        for idx in range(len(test_ds)):
            src_ids, trg_ids = test_ds[idx]
            src_ids, trg_ids = src_ids.to(device), trg_ids.to(device)

            logp = model(src_ids.unsqueeze(0), trg_ids.unsqueeze(0), tf=0.0)
            pred = logp.argmax(-1).squeeze(0).tolist()[1:]   # drop <sos>
            gold = trg_ids.tolist()[1:]

            src_str  = ids_to_str(src_ids.tolist()[1:], rev_src)
            gold_str = ids_to_str(gold, rev_trg)
            pred_str = ids_to_str(pred, rev_trg)
            ok       = pred_str == gold_str

            fh.write(f"{src_str}\t{gold_str}\t{pred_str}\n")
            if idx < 20:                                      # first 20 rows
                table.add_data(src_str, gold_str, pred_str, ok)
                
    # ---- console grid ---- #
    import pandas as pd, textwrap
    pd.set_option("display.max_colwidth", None)
    print("\nFirst 20 test examples")
    print(pd.DataFrame(table.data, columns=table.columns).to_markdown(index=False))
    wandb.log({"test_samples": table})
    print("Predictions saved to", pred_file)

    wandb.summary["val_loss"] = val_loss
    wandb.summary["test_acc"] = test_acc


# ---------------- Kaggle-compatible CLI ---------------- #
def cli():
    mode = 'sweep'  # or 'single'
    cfg = BaseCfg()
    if mode=='single':
        wandb.init(project='transliteration_model-2', config=cfg.__dict__)
        run(wandb.config)
    else:
        sweep_id = wandb.sweep(create_sweep(), project='transliteration_model-2')
        print(f"Sweep ID: {sweep_id}")
        def sweep_run():
            wandb.init(config=BaseCfg().__dict__)
            run(wandb.config)
        wandb.agent(sweep_id, function=sweep_run, count=20)

# Run from notebook cell: cli()

cli()
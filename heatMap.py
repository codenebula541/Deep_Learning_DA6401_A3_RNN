# ---------------------------------------------------------
# 3×3 Attention‑heat‑map grid with Hindi glyph support
# ---------------------------------------------------------
import os, random, torch, matplotlib.pyplot as plt, seaborn as sns
from matplotlib import font_manager as fm

# ---------- 1.  Register Devanagari font & make it default ----------
font_path = "/kaggle/input/dev-script/NotoSansDevanagari_Condensed-Regular.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    devanagari_name = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams["font.family"] = devanagari_name         # ← global default
    print("✓ Devanagari font set to:", devanagari_name)
else:
    print("⚠️  Font not found, you’ll still see □□□ boxes")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2.  Reload dataset & best attention model ---------------
cfg = BaseCfg()                               
train_ds = DakLex(os.path.join(cfg.data_root, "hi.translit.sampled.train.tsv"))
test_ds  = DakLex(os.path.join(cfg.data_root, "hi.translit.sampled.test.tsv"),
                  train_ds.src_map, train_ds.trg_map)

model = Seq2Seq(len(train_ds.src_map)+3, len(train_ds.trg_map)+3,
                cfg.embed_dim, cfg.hidden_size, cfg.cell_type).to(device)
model.load_state_dict(torch.load("/kaggle/working/best_attention_model.pt",
                                 map_location=device))
model.eval()

rev_src = {v:k for k,v in train_ds.src_map.items()}
rev_trg = {v:k for k,v in train_ds.trg_map.items()}

# ---------- 3.  Randomly select 9 examples --------------------------
sample_ids = random.sample(range(len(test_ds)), 9)

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for ax, idx in zip(axes.flat, sample_ids):
    # ---- get one (src, trg) pair
    s_ids, t_ids = test_ds[idx]
    src = [rev_src[x] for x in s_ids.tolist()[1:-1]]
    trg = [rev_trg[x] for x in t_ids.tolist()[1:-1]]

    # ---- run model and grab attention
    with torch.no_grad():
        _, att = model(s_ids.unsqueeze(0).to(device),
                       t_ids.unsqueeze(0).to(device),
                       tf=0.0, collect=True)
    A = att.squeeze(0).detach().cpu().numpy()[:len(trg), :len(src)]   # (T_out × T_in)

    # ---- draw heat‑map
    sns.heatmap(A, ax=ax, cmap="Blues", cbar=False,
                xticklabels=src, yticklabels=trg)

    # ensure tick labels use Devanagari font
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(devanagari_name)

    ax.set_xlabel("Latin"); ax.set_ylabel("Hindi")
    ax.set_title("".join(src), fontsize=9)

plt.tight_layout()

# ---------- 4.  Save figure so you can download ---------------------
save_path = "/kaggle/working/attention_grid_3x3.png"
fig.savefig(save_path, dpi=200)
plt.show()
print(f"✅ Saved grid to {save_path}")

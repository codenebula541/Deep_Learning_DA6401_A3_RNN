import matplotlib.animation as animation

def vis_attention_grid(model, ts_ds, rev_src, rev_trg, out_dir, run_prefix="attn_"):
    os.makedirs(out_dir, exist_ok=True)
    sample_ids = random.sample(range(len(ts_ds)), 3)
    gif_paths, wandb_imgs = [], []

    for idx in sample_ids:
        s_ids, t_ids = ts_ds[idx]
        src = [rev_src[x] for x in s_ids.tolist()[1:-1]]
        trg = [rev_trg[x] for x in t_ids.tolist()[1:-1]]

        with torch.no_grad():
            _, att = model(s_ids.unsqueeze(0).to(device),
                           t_ids.unsqueeze(0).to(device),
                           tf=0.0, collect=True)
        att = att.squeeze(0).detach().cpu().numpy()[:len(trg), :len(src)]

        # Shift characters to the right (start at pos 1 instead of 0)
        xs = list(range(1, len(src) + 1))

        fig, ax = plt.subplots(figsize=(len(src) * 0.6 + 1.5, 2.5))
        ax.set_xlim(0, len(src) + 1.5)
        ax.set_ylim(-1, 2)
        ax.axis('off')
        ax.set_title(f"Transliterating: {''.join(src)}", fontsize=10)

        # Shift Latin and Hindi labels
        ax.text(0, 1, "Latin:", fontsize=10, fontweight="bold", ha="right")
        ax.text(0, 0, "Hindi:", fontsize=10, fontweight="bold", ha="right")

        # Draw input characters
        for i, ch in enumerate(src):
            ax.text(xs[i], 1, ch, ha="center", va="center", fontsize=12)

        # Pre-create target text placeholders
        out_texts = [ax.text(xs[i], 0, "", ha="center", va="center", fontsize=12) for i in range(len(trg))]

        # Attention lines (source to current target)
        lines = [ax.plot([], [], lw=1, color="tab:red", alpha=0)[0] for _ in src]

        # Golden box
        goldbox = plt.Rectangle((xs[0]-0.4, 0.85), 0.8, 0.3, fill=False, edgecolor="gold", lw=2)
        ax.add_patch(goldbox)

        def update(t):
            out_texts[t].set_text(trg[t])
            max_attn_idx = att[t].argmax()
            goldbox.set_x(xs[max_attn_idx] - 0.4)

            for j, line in enumerate(lines):
                w = att[t, j]
                if w < 0.05:
                    line.set_alpha(0.0)
                else:
                    line.set_data([xs[j], xs[t]], [1, 0])
                    line.set_linewidth(3 * w)
                    line.set_alpha(w)

            return [*out_texts, goldbox, *lines]

        ani = animation.FuncAnimation(fig, update, frames=len(trg), blit=True, interval=700)

        gif_path = os.path.join(out_dir, f"{run_prefix}{idx}.gif")
        ani.save(gif_path, writer="pillow", fps=2)
        plt.close(fig)

        gif_paths.append(gif_path)
        wandb_imgs.append(wandb.Video(gif_path, caption=f"Sample {idx}", fps=2))

    wandb.log({f"{run_prefix}attention_gifs": wandb_imgs})
    print("✅ Attention GIFs saved:", gif_paths)

# --- Step 1: Load training dataset to get correct src/trg mappings ---
train_ds = DakLex("/kaggle/input/mydataset/hi.translit.sampled.train.tsv")

# --- Step 2: Load test dataset using training vocab mappings ---
test_ds = DakLex("/kaggle/input/mydataset/hi.translit.sampled.test.tsv",
                 src_map=train_ds.src_map,
                 trg_map=train_ds.trg_map)

# --- Step 3: Create reverse maps (index → char) ---
rev_src = {v: k for k, v in train_ds.src_map.items()}
rev_trg = {v: k for k, v in train_ds.trg_map.items()}

# --- Step 4: Load best trained model using correct vocab sizes ---
model = Seq2Seq(len(train_ds.src_map)+3, len(train_ds.trg_map)+3, 
                128, 512, "gru").to(device)


model.load_state_dict(torch.load("/kaggle/working/best_attention_model.pt", map_location=device))
model.eval()

# --- Step 5: Import animation module if not already imported ---
import matplotlib.animation as animation

wandb.init(project="transliteration_model-2", name="attention_visualization")

# --- Step 6: Call your visualization function ---
vis_attention_grid(
    model=model,
    ts_ds=test_ds,
    rev_src=rev_src,
    rev_trg=rev_trg,
    out_dir="/kaggle/working/attention_gifs"
)


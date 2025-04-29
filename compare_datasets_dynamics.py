import sys, math, pathlib, argparse, torch, matplotlib.pyplot as plt

# ── helper: build a figure with an n×n grid ──────────────────────────────
def make_grid(flat, n_cols=8):
    pix = flat.size(1)
    if pix % (32 * 32) == 0:      # typical CIFAR/MNIST
        H = W = 32
        C = pix // (H * W)
    else:
        H = W = int(math.sqrt(pix))
        C = 1
    imgs = flat.view(-1, C, H, W)[: n_cols * n_cols]

    fig, axes = plt.subplots(n_cols, n_cols, figsize=(n_cols, n_cols))
    for idx, ax in enumerate(axes.flat):
        img = imgs[idx]
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1) if C == 1 \
             else ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    return fig

# ── CLI ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("old_pth"), parser.add_argument("new_pth")
parser.add_argument("--outdir", default="cache_vis", help="directory for PNGs")
args = parser.parse_args()

old_file, new_file = pathlib.Path(args.old_pth), pathlib.Path(args.new_pth)
outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

# ── numeric diff section (unchanged) -------------------------------------
rtol, atol = 1e-5, 1e-8
old_ds = torch.load(old_file, map_location="cpu")
new_ds = torch.load(new_file, map_location="cpu")

print("\n=== numeric comparison =========================================")
all_ok, names = True, ("x0", "dx", "y", "dt")
for name, old_t, new_t in zip(names, old_ds.tensors, new_ds.tensors):
    if old_t.shape != new_t.shape:
        print(f"[{name}] shape mismatch: {old_t.shape} vs {new_t.shape}")
        all_ok = False; continue
    close = torch.allclose(old_t, new_t, rtol=rtol, atol=atol)
    max_err = (old_t - new_t).abs().max().item()
    print(f"[{name}] {'OK' if close else 'DIFF'}   max |Δ| = {max_err:.2e}")
    all_ok &= close

print("\n✅ identical within tolerances." if all_ok
      else "\n⚠️  at least one tensor differs beyond tolerances.")

# ── visual save of x1 grids ---------------------------------------------
for tag, ds in (("old", old_ds), ("new", new_ds)):
    flat = ds.tensors[2][:, 1:]          # strip leading 1 column
    fig = make_grid(flat)
    fig.suptitle(f"{tag.upper()} cache – x₁", fontsize=14)
    outfile = outdir / f"{tag}_x1_grid.png"
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    print(f"[saved] {outfile.resolve()}")

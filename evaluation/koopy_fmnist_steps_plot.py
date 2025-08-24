import glob
import os, argparse, subprocess, tempfile, torch
from pathlib import Path
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image, make_grid
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from jump_wtf.utils.sampling import sample_efficient
import cleanfid.fid as fid

import matplotlib.pyplot as plt
from .cfm_fmnist import load_net as load_cfm_net
from jump_wtf.models.model import Model
from jump_wtf.models.autoencoder import Autoencoder_unet
from jump_wtf.operators.generic import GenericOperator_state
from jump_wtf.models.unet_wrapper import UNetWrapperKoopman
from torchdyn.core import NeuralODE

def load_net(ckpt_glob, device="cuda"):
    """
    Finds latest .ckpt, rebuilds AE & Koopman operator, loads weights.
    Supports 'mnist' (1×28×28) or 'train' (3×32×32).
    """
    paths = glob.glob(ckpt_glob)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {ckpt_glob}")
    latest = max(paths, key=os.path.getctime)
    print(f"> loading checkpoint: {latest}")
    ckpt_path = Path(latest)
    assert ckpt_path.is_file(), "Checkpoint not found"

    C, H, W = 1, 28, 28
    wrapper_net = UNetWrapperKoopman(
        dim=(1, 28, 28), 
        num_channels=32, 
        num_res_blocks=1,
        # attention_resolutions="14,7"
    ).to("cpu")

    # wrapper_net = UNetWrapperKoopman(
    #     dim=(1, 28, 28), 
    #     num_channels=192, 
    #     num_res_blocks=4,
    #     num_heads=4,
    # ).to("cpu")

    ckpt = torch.load("assets/unet_dynamics/fashion_mnist_full_otcfm_step-25000.pt", map_location=device, weights_only=True)
    wrapper_net.load_state_dict(ckpt)

    state_dim = C * H * W
    # Shallow
    ae = Autoencoder_unet(
        dim=(C, H, W),
        num_channels=32,
        num_res_blocks=1,
        attention_resolutions="14,7"
    )

    # Deep
    # ae = Autoencoder_unet(
    #     dim=(C, H, W),
    #     num_channels=128,
    #     channel_mult=[1, 2, 2],
    #     num_res_blocks=3,
    #     num_heads=4,
    #     attention_resolutions="14,7",
    #     bottleneck=False,
    #     resblock_updown=True,
    # )

    koop_op = GenericOperator_state(1 + 2 * state_dim)

    model = Model.load_from_checkpoint(
        ckpt_path,
        dynamics=wrapper_net,
        autoencoder=ae,
        koopman=koop_op,
        loss_function=torch.nn.MSELoss(),
        lr_scheduler="ReduceLROnPlateau",
        gamma=0.99,
        autoencoder_lr=1e-3,
        lie_lr=1e-4,
        strict=True,
    )
    model.ckpt_path = latest
    print(f"> checkpoint loaded!")
    return model.to(device).eval()

# ──────────────────────────────────────────────────────────────────────
def export_real(root, n, out):
    ds = FashionMNIST(root=root, train=True, download=True, transform=ToTensor())
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_image(ds[i][0].expand(3, -1, -1), out/f"{i:05d}.png")  # fake-RGB

# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_double_row(net_koop, net_cfm, steps_list, device, out_dir, idx, x0=None):
    """
    Build ONE figure with two rows:
      • Row-0 : Koopman pipeline
      • Row-1 : Conditional-Flow-Matching pipeline

    Columns = values in steps_list (e.g. 1⟶100).
    The same x₀ is fed to both models so rows are comparable.
    The file is written as  pair_{idx:03d}.png  inside out_dir.
    """
    C, H, W = 1, 28, 28
    out_dir.mkdir(parents=True, exist_ok=True)

    if x0 is None:
        x0 = torch.randn(1, C, H, W, device=device)

    node = NeuralODE(
                net_cfm,
                solver      ="euler",
                sensitivity ="adjoint",
                atol        =1e-4,
                rtol        =1e-4,
            )

    # first column = the raw noise that seeds both pipelines
    x0_norm  = (x0 - x0.min()) / (x0.max() - x0.min() + 1e-8)   # normalise on *device*
    noise_vis = (
        x0_norm.expand(-1, 3, -1, -1)[0]                        # fake-RGB, drop batch dim
        .cpu()                                                  # move to CPU only once
    )
    koop_imgs, cfm_imgs = [noise_vis], [noise_vis]              # start each row with noise

    for s in steps_list:                                # iterate over step counts
        k_img = (
            sample_efficient(net_koop, t_max=1.5, n_iter=s, n_samples=1,
                             device=device, x_spatial=x0)
            .add(1).div(2)                              # [-1,1] → [0,1]
            .expand(-1, 3, -1, -1).cpu()[0]             # fake-RGB
        )
        koop_imgs.append(k_img)

        # integrate for `s` Euler steps over t ∈ [0,1]
        t_span = torch.linspace(0., 1., s+1, device=device)
        c_img = (
            node.trajectory(x0, t_span)[-1]   # final state after `s` steps
            .clamp(-1, 1)
            .add(1).div(2)                   # [-1,1] → [0,1]
            .expand(-1, 3, -1, -1)           # fake-RGB
            .cpu()[0]
        )
        cfm_imgs.append(c_img)

    cols = len(steps_list) + 1               # +1 for the noise column
    grid = make_grid(torch.stack(koop_imgs + cfm_imgs),
                     nrow=cols, padding=1, pad_value=1)

    plt.figure(figsize=(1.2 * cols, 3))      
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / f"pair_{idx:03d}.png",
                dpi=400, bbox_inches="tight", pad_inches=0)           # compact output
    plt.close()

def plot_fid_curve(fid_results, out_dir):
    """Save a PNG showing FID as a function of n_iter."""
    steps, scores = zip(*sorted(fid_results.items()))
    plt.figure(figsize=(5, 3))
    plt.plot(steps, scores, marker="o")
    plt.title("FID vs. sampling steps")
    plt.xlabel("steps")
    plt.ylabel("FID score")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_dir / "fid_vs_steps.png")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koop-checkpoint", default="logs/fashion_mnist/full/20250515-205324/checkpoints/best-fid-train-step=12600-fid_train=97.618.ckpt",
                    help="Glob for Koopman .ckpt files")
    ap.add_argument("--cfm-checkpoint",  default="assets/unet_dynamics/fashion_mnist_full_otcfm_step-25000.pt",
                    help="Path to CFM checkpoint")
    ap.add_argument("--num-pairs",      type=int, default=100,
                    help="How many two-row figures to save")
    ap.add_argument("--steps",
                    default="1,2,3,5,10,20,60,100",
                    help="Comma-separated step counts")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir",     default="qualitative_pairs_fmnist")
    args = ap.parse_args()

    steps_list = [int(s) for s in args.steps.split(",")]
    out_dir = Path(args.out_dir)

    print("→ loading Koopman checkpoint")
    net_koop = load_net(args.koop_checkpoint, args.device)

    print("→ loading CFM checkpoint")
    net_cfm  = load_cfm_net(args.cfm_checkpoint, args.device)

    for i in range(args.num_pairs):
        print(f"→ figure {i+1}/{args.num_pairs}")
        sample_double_row(net_koop, net_cfm, steps_list,
                          args.device, out_dir, idx=i)

    print("Finished. Figures stored in:", out_dir)

if __name__ == "__main__":
    main()
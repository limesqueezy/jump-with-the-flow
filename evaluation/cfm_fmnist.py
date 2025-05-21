#!/usr/bin/env python3
import os, argparse, subprocess, tempfile, torch
from pathlib import Path
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel
import cleanfid.fid as fid

# ──────────────────────────────────────────────────────────────────────
def load_net(ckpt, device):
    net = UNetModel(
        dim              =(1, 28, 28),
        num_channels     =32,
        num_res_blocks   =1,
        channel_mult     =[1, 2, 2],
        attention_resolutions="16",
        dropout          =0.0,
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    for key_candidate in ("ema_model", "model"):
        if key_candidate in state:
            state = state[key_candidate]
            break
    net.load_state_dict(
        {k.replace("module.", ""): v for k, v in state.items()},
        strict=False
    )
    net.eval()
    return net

def export_real(root, n, out):
    ds = FashionMNIST(root=root, train=True, download=True, transform=ToTensor())
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        save_image(ds[i][0].expand(3, -1, -1), out/f"{i:05d}.png")  # fake-RGB

@torch.no_grad()
def sample(net, n, bs, steps, device, out):
    C, H, W = 1, 28, 28
    t_span  = torch.linspace(0., 1., steps, device=device)         # 0 → 1

    node = NeuralODE(
        net,
        solver      ="euler",
        sensitivity ="adjoint",
        atol        =1e-4,
        rtol        =1e-4,
    )

    done = 0
    out.mkdir(parents=True, exist_ok=True)
    with Progress("[progress.description]{task.description}",
                  BarColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as p:
        task = p.add_task("Generating", total=n)
        while done < n:
            cur = min(bs, n - done)
            x0  = torch.randn(cur, C, H, W, device=device)
            imgs = node.trajectory(x0, t_span)[-1].clamp(-1, 1).add_(1).div_(2)
            imgs = imgs.expand(-1, 3, -1, -1)
            for j, img in enumerate(imgs):
                save_image(img, out/f"{done+j:05d}.png")
            done += cur
            p.update(task, advance=cur)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root",   default="assets/raw_datasets")
    ap.add_argument("--num-samples", type=int, default=10_000)
    ap.add_argument("--batch-size",  type=int, default=4096)
    ap.add_argument("--ode-steps",   type=int, default=100)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work = Path("/mnt/disk1/ari/koopy_eval/cfm/f-mnist")
    real, gen = work/"real", work/"generated"

    print("→ exporting real F-MNIST images")
    export_real(args.data_root, args.num_samples, real)

    print("→ loading checkpoint")
    net = load_net(args.checkpoint, args.device)

    print("→ sampling")
    sample(net, args.num_samples, args.batch_size, args.ode_steps, args.device, gen)

    print("→ FID (pytorch-fid)")
    fid_pt = subprocess.check_output([
        "python", "-m", "pytorch_fid", str(real), str(gen), "--device", args.device
    ]).decode().strip()
    print("pytorch-fid :", fid_pt)

    print("→ FID (clean-fid)")
    fid_cl = fid.compute_fid(
        str(real), str(gen),
        mode            ="clean",
        batch_size      =32,
        device          =torch.device(args.device),
        use_dataparallel=False
    )
    print("clean-fid   :", fid_cl)
    print("Outputs in  :", work)

if __name__ == "__main__":
    main()

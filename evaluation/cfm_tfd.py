#!/usr/bin/env python3
"""
Evaluate a CFM checkpoint on the Toronto-Face-Dataset and compute FID.

• (C,H,W) = (1,48,48)   •   integrate t ∈ [0,1] with dopri

Usage
-----
python eval_tfd.py \
       --checkpoint path/to/ckpt.pt \
       --tfd-root assets/raw_datasets/TFD \
       --num-samples 50_000 \
       --batch-size 2048 \
       --ode-steps 100 \
       --device cuda
"""
import os
import argparse
import subprocess
from pathlib import Path

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from torchdyn.core import NeuralODE
from jump_wtf.data.toronto_face import TorontoFaceDataset
from torchcfm.models.unet.unet import UNetModelWrapper
import cleanfid.fid as fid


def load_net(ckpt, device):
    """Build the TFD UNetModelWrapper and load weights (ema_model)."""
    net = UNetModelWrapper(
        dim              =(1, 48, 48),
        num_res_blocks   =2,
        num_channels     =64,
        channel_mult     =[1, 1, 2, 3, 4],
        num_heads        =4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout          =0.1,
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    if "ema_model" in state:
        state = state["ema_model"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def export_real(root, n, out):
    """Export the first n real TFD images as fake-RGB PNGs."""
    ds = TorontoFaceDataset(root=root, train=None, transform=ToTensor())
    total = len(ds)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, total)):
        img, _ = ds[i]
        save_image(img.expand(3, -1, -1), out / f"{i:05d}.png")


@torch.no_grad()
def sample(net, n, bs, steps, device, out):
    """Generate n samples via NeuralODE and save as fake-RGB PNGs."""
    C, H, W = 1, 48, 48
    t_span = torch.linspace(0., 1., steps, device=device)

    node = NeuralODE(
        net,
        solver      ="dopri5",
        sensitivity ="adjoint",
    )

    done = 0
    out.mkdir(parents=True, exist_ok=True)
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn()
    ) as p:
        task = p.add_task("Generating", total=n)
        while done < n:
            cur = min(bs, n - done)
            x0  = torch.randn(cur, C, H, W, device=device)
            imgs = node.trajectory(x0, t_span)[-1].clamp(-1, 1).add_(1).div_(2)
            imgs = imgs.expand(-1, 3, -1, -1)
            for j, img in enumerate(imgs):
                save_image(img, out / f"{done + j:05d}.png")
            done += cur
            p.update(task, advance=cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to .pt file with CFM weights")
    ap.add_argument("--tfd-root",   default="assets/raw_datasets",
                    help="Folder that contains TFD_48x48.mat")
    ap.add_argument("--num-samples", type=int, default=10_000,
                    help="Number of images to export/generate for FID")
    ap.add_argument("--batch-size",  type=int, default=2048)
    ap.add_argument("--ode-steps",   type=int, default=100)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work = Path("/mnt/disk1/ari/koopy_eval/tfd")
    real = work / "real"
    gen  = work / "generated"

    print("→ exporting real TFD images")
    export_real(args.tfd_root, args.num_samples, real)

    print("→ loading checkpoint")
    net = load_net(args.checkpoint, args.device)

    print("→ sampling")
    sample(net, args.num_samples, args.batch_size, args.ode_steps, args.device, gen)

    print("→ FID (pytorch-fid)")
    fid_pt = subprocess.check_output([
        "python", "-m", "pytorch_fid", str(real), str(gen),
        "--device", args.device
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

#!/usr/bin/env python
"""train_tfd_ddp.py
===================
Train a Conditional Flow Matching (CFM) continuous normalizing flow on the
Toronto Face Dataset (TFD) using DistributedDataParallel (DDP).
This script is derived from the original `train_cifar10_ddp.py` example shipped
with TorchCFM, but adapted for:
  * grayscale 48×48 face images from TFD
  * two‑GPU data‑parallel training with `torchrun`
  * Hydra‑free CLI so it can be run as a standalone Python script

Run with (example):
    torchrun --nproc_per_node=2 train_tfd_ddp.py \
        --data_dir /datasets/TFD \
        --epochs 200 --batch_size 256 --lr 3e-4
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

# ---- TorchCFM imports ---- #
from torchcfm.models.unet import UNetModel  # velocity‑field network
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
# ---- Local dataset (provided by user) ---- #
try:
    from jump_wtf.data.toronto_face import TorontoFaceDataset
except ModuleNotFoundError as e:
    raise SystemExit("Could not import TorontoFaceDataset – make sure the package `jump_wtf` is on PYTHONPATH") from e


# -----------------------------
#  Distributed boiler‑plate
# -----------------------------

def ddp_setup(rank: int, world_size: int):
    """Initialise the default process‑group and set the current CUDA device."""
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(rank)


def ddp_cleanup():
    """Tear down the process group (avoids hangs at script exit)."""
    dist.destroy_process_group()


# -----------------------------
#  Data loading helpers
# -----------------------------

def make_dataloader(data_root: Path, batch_size: int, rank: int, world_size: int, train: bool):
    tfm = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # grayscale normalisation
        ]
    )

    dataset = TorontoFaceDataset(root=str(data_root), train=train, download=True, transform=tfm)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler


# -----------------------------
#  Main training routine
# -----------------------------

def train_one_epoch(model, loader, loss_fn, optimiser, device):
    model.train()
    epoch_loss = 0.0
    for x1, _ in loader:
        x1 = x1.to(device)
        # Sample source noise and conditional data
        x0 = torch.randn_like(x1)
        # Draw t, xt, ut from the matcher exactly like in CIFAR example
        t, xt, ut = loss_fn.sample_location_and_conditional_flow(x0, x1)
        # Move the samples to the GPU if needed
        xt = xt.to(device)
        ut = ut.to(device)
        t  = t.to(device)

        # Compute network output on the interpolated data
        vt = model(t, xt)
        # MSE loss between predicted velocity field vt and true ut
        loss = torch.mean((vt - ut) ** 2)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def main(args):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    ddp_setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 1) Data
    train_loader, train_sampler = make_dataloader(Path(args.data_dir), args.batch_size, rank, world_size, train=True)

    # 2) Velocity-field network (UNet)
    model = UNetModel(
        dim=(1, 48, 48),
        num_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 1, 2, 3, 4),
        attention_resolutions="16",
        dropout=0.1,
        num_heads=4,
        num_head_channels=64,
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        use_fp16=False,
        use_new_attention_order=False,
    ).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)


    # 3) Loss + optimiser
    loss_fn = ConditionalFlowMatcher(sigma=0.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # 4) Training loop
    if rank == 0:
        print("Starting training with", world_size, "GPUs …")
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  # shuffles each epoch when using DistributedSampler
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimiser, device)
        scheduler.step()

        if rank == 0 and (epoch + 1) % args.monitor == 0:
            print(f"[Epoch {epoch+1:04d}] loss = {avg_loss:.4f}, lr = {scheduler.get_last_lr()[0]:.2e}")

        if rank == 0 and args.ckpt_dir and (epoch + 1) % args.save_every == 0:
            args.ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model": model.module.state_dict(),
                "optim": optimiser.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, args.ckpt_dir / f"tfd_cfm_epoch{epoch+1:04d}.pt")

    ddp_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CFM‑TFD‑DDP trainer")
    parser.add_argument("--data_dir", type=str, default=Path("assets/raw_datasets/"), help="root directory where TFD_48x48.mat will be stored")
    parser.add_argument("--ckpt_dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--monitor", type=int, default=1, help="print stats every N epochs (rank‑0 only)")
    parser.add_argument("--save_every", type=int, default=10, help="checkpoint frequency (epochs)")
    args = parser.parse_args()

    main(args)

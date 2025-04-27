# Encode Decode vis

# Grayscale training

# 1-step prediction

# ‖zₖ‖ during k-step rollout starting from a real sample 

import os
import glob
import argparse
from pathlib import Path
from re import I

import torch
import torchvision.utils as vutils
from torchvision import datasets, transforms
from jump_wtf.models.model import Model
from jump_wtf.models.autoencoder import Autoencoder_unet
from jump_wtf.operators.generic import GenericOperator_state
from jump_wtf.models.unet_wrapper import UNetWrapperKoopman
from jump_wtf.utils.plot import sample_efficient_plot
from jump_wtf.utils.sampling import sample_efficient

def load_model(ckpt_glob, dataset="cifar", device="cuda"):
    """
    Finds latest .ckpt, rebuilds AE & Koopman operator, loads weights.
    Supports 'mnist' (1×28×28) or 'cifar' (3×32×32).
    """
    paths = glob.glob(ckpt_glob)
    if not paths:
        raise FileNotFoundError(f"No checkpoints found for pattern: {ckpt_glob}")
    latest = max(paths, key=os.path.getctime)
    print(f"> loading checkpoint: {latest}")
    ckpt_path = Path(latest)
    assert ckpt_path.is_file(), "Checkpoint not found"

    if dataset.lower() == "mnist":
        C, H, W = 1, 28, 28
        wrapper_net = UNetWrapperKoopman(
            dim=(1, 28, 28), 
            num_channels=32, 
            num_res_blocks=1
        ).to("cpu")

        ckpt = torch.load("assets/unet_dynamics/mnist_otcfm_epoch_20.pth", map_location=device, weights_only=True)
        wrapper_net.load_state_dict(ckpt)

    elif dataset.lower() == "cifar":
        C, H, W = 3, 32, 32
        wrapper_net = UNetWrapperKoopman(
            dim             =       [C, H, W],
            num_channels    =       128,
            num_res_blocks  =       2,
            channel_mult    =       [1, 2, 2, 2],
            num_heads       =       4,
            num_head_channels=      64,
            attention_resolutions=  "16",
            dropout         =       0.1,
        ).to(device)

        ckpt = torch.load("assets/unet_dynamics/cifar10_otcfm_step-400000.pt", map_location=device, weights_only=True)
        state = ckpt["ema_model"]
        wrapper_net.load_state_dict(state)

    else:
        raise ValueError("dataset must be 'mnist' or 'cifar'")

    state_dim = C * H * W
    ae = Autoencoder_unet(
        dim=(C, H, W),
        num_channels=32,
        num_res_blocks=1,
    )
    koop_op = GenericOperator_state(1 + 2 * state_dim)

    model = Model.load_from_checkpoint(
        ckpt_path,
        dynamics=wrapper_net,
        autoencoder=ae,
        koopman=koop_op,
        loss_function=torch.nn.MSELoss(),  # match your training loss
        lr_scheduler="ReduceLROnPlateau",
        gamma=0.99,
        autoencoder_lr=1e-3,
        lie_lr=1e-4,
        strict=True,
    )
    model.ckpt_path = latest
    print(f"> checkpoint loaded!")
    return model.to(device).eval()

def sample_and_save_grid(
    model,
    k: int,
    t_max: int = 1,
    n_iter: int = 1,
    device: str = "cuda",
    save_dir: str = "debug_samples",
):
    """
    Draw k*k samples via sample_efficient(), arrange in a k×k grid,
    and save under save_dir/grid_{k}x{k}.png.
    Assumes sample_efficient(model, t_max, n_iter, n_samples, device) is in scope.
    """
    model = model.to(device).eval()
    n_samples = k * k

    # get your samples (shape [n_samples, C, H, W], in [-1,1])
    imgs = sample_efficient(model, t_max=t_max, n_iter=n_iter, n_samples=n_samples, device=device)
    # rescale to [0,1]
    imgs = (imgs + 1.0) / 2.0

    # make output dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # build grid and write PNG
    grid = vutils.make_grid(imgs, nrow=k, padding=2)
    out_path = Path(save_dir) / f"grid_{k}x{k}.png"
    vutils.save_image(grid, out_path)

    print(f"Saved sample grid to {out_path}")

# def one_step(
#     model,
#     k: int,
#     dataset: str = "cifar",
#     device: str = "cuda",
#     save_dir: str = "debug_samples",
# ):
#     """
#     • Draw k*k real images from MNIST or CIFAR
#     • Flatten+append time‐zero → encoder → decoder → reshape
#     • Save two PNGs: original grid & reconstructed grid
#     """

#     model = model.to(device).eval()
#     n = k * k

#     # 1) Load k*k real samples
#     if dataset.lower() == "cifar":
#         tf = transforms.Compose([
#             transforms.ToTensor(),                       # [0,1]
#             transforms.Normalize((0.5,)*3, (0.5,)*3),    # [−1,1]
#         ])
#         ds = datasets.CIFAR10("/home/lemon/koopman/jump-with-the-flow/assets/raw_datasets/", train=False, download=True, transform=tf)
#     elif dataset.lower() == "mnist":
#         tf = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,)),
#         ])
#         ds = datasets.MNIST("/home/lemon/koopman/jump-with-the-flow/assets/raw_datasets/", train=False, download=True, transform=tf)
#     else:
#         raise ValueError("dataset must be 'mnist' or 'cifar'")

#     # # randomly pick n examples
#     idx = torch.randperm(len(ds))[:n]
#     imgs = torch.stack([ds[i][0] for i in idx], dim=0).to(device)  # [n,C,H,W]

#     C, H, W = imgs.shape[1:]
#     spatial_dim = C * H * W
#     x_flat = imgs.view(n, spatial_dim)           # [n, spatial_dim]
#     t0     = torch.zeros(n, 1, device=device)    # [n,1]
#     x      = torch.cat([x_flat, t0], dim=1)      # [n, spatial_dim+1]

#     with torch.no_grad():
#         z       = model.autoencoder.encoder(x)
#         decoded = model.autoencoder.decoder(z)

#     recon = decoded[:, :spatial_dim].view(2*n, C, H, W).clamp(-1, 1) # SWITCH TO n
#     breakpoint()
#     os.makedirs(save_dir, exist_ok=True)
#     orig_grid = vutils.make_grid((imgs + 1) / 2, nrow=k, padding=2)
#     recon_grid = vutils.make_grid((recon + 1) / 2, nrow=k, padding=2)

#     name = f"{dataset}_{k}x{k}"
#     vutils.save_image(orig_grid, Path(save_dir) / f"orig_grid_{name}.png")
#     vutils.save_image(recon_grid, Path(save_dir) / f"recon_grid_{name}.png")

#     print(f"Saved:\n - {save_dir}/orig_grid_{name}.png\n - {save_dir}/recon_grid_{name}.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Debug: sample from Koopman model")
    p.add_argument("--ckpt",     type=str, default="checkpoints/cifar10-koopman-*/*.ckpt",
                   help="glob for Koopman checkpoints")
    p.add_argument("--dataset",  type=str, default="cifar",
                   choices=["mnist","cifar"])
    p.add_argument("--n-rows",type=int, default=8,
                   help="number of rows to draw")
    p.add_argument("--n-iter",   type=int, default=1,
                   help="number of Koopman steps")
    p.add_argument("--t-max",    type=int, default=1,
                   help="total time horizon")
    p.add_argument("--device",   type=str, default="cuda")
    args = p.parse_args()

    model = load_model(args.ckpt, dataset=args.dataset, device=args.device)
    # sample_and_save_grid(model, k=args.n_rows, t_max=args.t_max, n_iter=args.n_iter, device="cuda")
    
    # sample_efficient_plot(
    #     model,
    #     t_max=args.t_max,
    #     n_iter=args.n_iter,
    #     n_samples=args.n_samples,
    #     device=args.device,
    # )
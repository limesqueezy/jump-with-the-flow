# =============================================================================
# Tinkered and vendored from:
#   https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/utils_cifar.py
# Upstream commit: 3fd278f9ef2f02e17e107e5769130b6cb44803e2
#
# Licensed under the MIT License (c) 2023 Alexander Tong
# See upstream LICENSE for full terms:
#   https://github.com/atong01/conditional-flow-matching/blob/main/LICENSE
#
# =============================================================================

import copy
import os

import torch
from torch import distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class LoggingSummaryWriter(SummaryWriter):
    def add_text(self, tag, text, *args, **kwargs):
        super().add_text(tag, text, *args, **kwargs)
        print(f"[TBoard:{tag}] {text}")

def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )

def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )

def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x

def log_generated_samples(writer, model, step, tag="cfm/generated"):
    """
    Run the learned flow on 64 random noise seeds, take
    the last timepoint [B,3,32,32] and log an 8×8 grid.
    """
    model.eval()
    node = NeuralODE(model, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        # 64 random CIFAR-shaped seeds
        z0     = torch.randn(64, 3, 32, 32, device=next(model.parameters()).device)
        t_span = torch.linspace(0,1,100, device=z0.device)
        traj   = node.trajectory(z0, t_span)
        imgs   = traj[-1]
        imgs   = imgs.clamp(-1, 1)
        grid   = make_grid(imgs, nrow=8, normalize=True, value_range=(-1,1))
    writer.add_image(tag, grid, global_step=step)
    model.train()

def log_final_trajectories(
    writer,
    traj: torch.Tensor,          # [T, N, C, H, W], on ANY device
    step: int = 0, # tensorboard will use step as an index so we might as well surface it
    tag: str = "cfm/trajectory_x1",
    num_samples: int = 20,
    nrow: int = 5
):
    """
    Logs a grid of `num_samples` random final-frame images (X₁) from `traj`.
    """
    # traj[-1] is [N, C, H, W]
    X1 = traj[-1]               # CPU or GPU tensor
    N, C, H, W = X1.shape

    # Pick a random permutation (wout replacement)
    idx = torch.randperm(N)[:num_samples]
    imgs = X1[idx]              # [num_samples, C, H, W]
    imgs = imgs.clamp(-1, 1)
    grid = make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1, 1))
    writer.add_image(tag, grid, global_step=step)

def generate_trajectories(net, node, cfg, device, out_path="trajectories.pth"):
    net.eval()
    n         = cfg.cfm.traj.n_traj
    t_steps   = cfg.cfm.traj.traj_steps
    chunk     = getattr(cfg.cfm.traj, "chunk_size", 500)
    t_span    = torch.linspace(0, 1, t_steps, device=device)
    all_chunks = []
    start = 0

    pbar = tqdm(total=n, desc="Gen trajectories", unit="traj")
    while start < n:
        end = min(start + chunk, n)
        try:
            z0 = torch.randn(end - start, *cfg.model.dim, device=device)
            with torch.no_grad():
                c = node.trajectory(z0, t_span).cpu()
            all_chunks.append(c)
            pbar.update(end - start)
            start = end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # back off chunk size
            chunk = max(1, chunk // 2)
            pbar.write(f"OOM, reducing chunk to {chunk}")
            # do NOT advance start; retry this slice
        # any other exception will bubble out

    pbar.close()
    traj = torch.cat(all_chunks, dim=0)
    torch.save(traj, out_path)
    return out_path

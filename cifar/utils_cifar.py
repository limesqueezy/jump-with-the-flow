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
import math
import os

import torch
from torch import distributed as dist
from torch.utils.tensorboard.writer import SummaryWriter
from torchdyn.core import NeuralODE

# from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid, save_image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

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

# def generate_samples(model, parallel, savedir, step, net_="normal", solver="dopri5"):
#     """Save 64 generated images (8 x 8) for sanity check along training.

#     Parameters
#     ----------
#     model:
#         represents the neural network that we want to generate samples from
#     parallel: bool
#         represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
#     savedir: str
#         represents the path where we want to save the generated images
#     step: int
#         represents the current step of training
#     """
#     model.eval()

#     model_ = copy.deepcopy(model)
#     if parallel:
#         # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
#         model_ = model_.module.to(device)

#     node_ = NeuralODE(model_, solver=solver, sensitivity="adjoint")
#     with torch.no_grad():
#         traj = node_.trajectory(
#             torch.randn(64, 3, 32, 32, device=device),
#             t_span=torch.linspace(0, 1, 100, device=device),
#         )

#         traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
#         traj = traj / 2 + 0.5
#     save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

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

def log_generated_samples(writer, model, cfg, step, tag="cfm/loaded_samples", solver="euler"):
    """
    Run the learned flow on 64 random noise seeds, take
    the last timepoint [B,3,32,32] and log an 8×8 grid.
    """
    model.eval()
    node = NeuralODE(model, solver=solver, sensitivity="adjoint")
    C, H, W  = cfg.dim
    B = 64
    with torch.no_grad():
        # 64 random CIFAR-shaped seeds
        z0     = torch.randn(B, C, H, W, device=next(model.parameters()).device)
        t_span = torch.linspace(0,1,100, device=z0.device)
        traj   = node.trajectory(z0, t_span)
        imgs   = traj[-1]
        imgs   = imgs.clamp(-1, 1)
        grid   = make_grid(imgs, nrow=int(math.sqrt(B)), normalize=True, value_range=(-1,1))
        grid   = grid.detach().cpu()

    writer.add_image(tag, grid, global_step=step)

def log_final_trajectories(
    writer,
    traj: torch.Tensor,          # [T, N, C, H, W], on ANY device
    step: int = 0, # tensorboard will use step as an index so we might as well surface it
    tag: str = "cfm/x1_loaded",
    num_samples: int = 64,
    nrow: int = 8
):
    writer.add_text("cfm/log_final_traj",f"logging final trajectories for debugging...")
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
    grid  = grid.detach().cpu()

    writer.add_image(tag, grid, global_step=step)

    writer.add_text("cfm/log_final_traj",f"logged final trajectories")

def generate_trajectories(net, node, cfg, device, out_path="trajectories.pth"):
    """
    Stream-generates trajectories with a single Rich progress bar and
    automatic OOM backoff (halving chunk_size on OOM).
    """
    net.eval()
    torch.cuda.empty_cache()

    n_traj     = cfg.cfm.traj.n_traj
    chunk_size = getattr(cfg.cfm.traj, "chunk_size", n_traj) or n_traj
    t_span     = torch.linspace(0, 1, cfg.cfm.traj.traj_steps, device=device)

    chunks = []
    start = 0

    # Single, clean Rich bar
    progress = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=None, complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("Generating trajectories", total=n_traj)
        while start < n_traj:
            end = min(start + chunk_size, n_traj)
            try:
                progress.update(task, description=f"chunk_size={chunk_size}")
                z0 = torch.randn(end - start, *cfg.model.dim, device=device)
                with torch.no_grad():
                    traj_chunk = node.trajectory(z0, t_span)
                chunks.append(traj_chunk.cpu())
                progress.update(task, advance=(end - start))
                start = end
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and chunk_size > 1:
                    torch.cuda.empty_cache()
                    chunk_size = max(1, chunk_size // 2)
                    progress.log(f"[red]OOM: reducing chunk_size → {chunk_size}")
                else:
                    raise

    traj = torch.cat(chunks, dim=1)
    torch.save(traj, out_path)
    progress.log(f"[bold green]Saved trajectories to {out_path}")
    return out_path
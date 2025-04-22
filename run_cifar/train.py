import os, copy
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from tqdm import trange
from torch.utils.tensorboard.writer import SummaryWriter
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper
from utils_cifar import ema, generate_samples, infiniteloop

@hydra.main(version_base=None, config_path="../jump_wtf/conf", config_name="composer")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))          # full config dumped to output dir
    writer = SummaryWriter()               # TB log → same dir

    if cfg.stage.name == "cfm":
        run_cfm(cfg, writer)
    # elif cfg.stage.name == "koopman":
    #     run_koopman(cfg, writer)
    # else:
    #     raise ValueError(f"Unknown stage {cfg.stage}")

def run_cfm(cfg, writer):

    # 1) prepare CIFAR‑10 DataLoader
    ds = hydra.utils.instantiate(cfg.dataset)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    data_iter = infiniteloop(loader)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    # 3) UNet + EMA copy
    net     = hydra.utils.instantiate(cfg.model).to(device)
    ema_net = copy.deepcopy(net).to(device)

    # 3) OT‑CFM matcher
    matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.stage.sigma)

    # 4) optimizer & LR schedule
    opt  = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(step, cfg.training.warmup) / cfg.training.warmup)

    # 5) wrap as continuous ODE (for rollout)
    node = NeuralODE(net, solver="dopri5", sensitivity="adjoint")

    # 6) training loop
    for step in trange(cfg.training.total_steps, desc="CFM training"):
        x1 = next(data_iter).to(device)   # never re‑instantiate loader
        x0 = torch.randn_like(x1, device=device)

        t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
        vt = net(t, xt)

        loss = torch.mean((vt - ut) ** 2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.training.grad_clip)
        opt.step()
        sched.step()

        # EMA update
        ema(net, ema_net, cfg.training.ema_decay)

        # log + checkpoint
        if step % cfg.training.log_every_n_steps == 0:
            writer.add_scalar("cfm/loss", loss.item(), step)

        if step and step % cfg.training.save_every_n_steps == 0:
            generate_samples(net, cfg.training.parallel,
                             cfg.paths.output_dir, step, net_="normal")
            generate_samples(ema_net, cfg.training.parallel,
                             cfg.paths.output_dir, step, net_="ema")
            torch.save({
                "net":   net.state_dict(),
                "ema":   ema_net.state_dict(),
                "opt":   opt.state_dict(),
                "sched": sched.state_dict(),
                "step":  step,
            }, os.path.join(cfg.paths.output_dir, f"cfm_step{step}.pth"))

    # final checkpoint
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(cfg.paths.output_dir, "cfm_final.pth"))

    # 7) generate & save trajectories
    net.eval()
    with torch.no_grad():
        z0     = torch.randn(cfg.stage.n_traj, *cfg.model.dim, device=device)
        t_span = torch.linspace(0, 1, cfg.stage.traj_steps, device=device)
        traj   = node.trajectory(z0, t_span)

    torch.save(traj.cpu(), "trajectories.pth")
    writer.add_text("cfm/trajectories", "trajectories.pth")

# def run_koopman(cfg, writer):
#     # 1) load frozen UNet or trajectories ---------------------------
#     if cfg.stage.traj_file:
#         traj = torch.load(cfg.stage.traj_file)
#     else:
#         unet = hydra.utils.instantiate(cfg.model).cpu()
#         unet.load_state_dict(torch.load(cfg.stage.frozen_unet_ckpt, map_location="cpu"))
#         node = NeuralODE(unet, solver="dopri5", sensitivity="adjoint")
#         with torch.no_grad():
#             traj = node.trajectory(torch.randn(2000, *cfg.model.dim, device="cpu"),
#                                    t_span=torch.linspace(0, 1, 100))
#         torch.save(traj, "trajectories.pth")

#     # 2) initialise encoder + Koopman operator ---------------------
#     # ... your existing Lightning training loop goes here ...
#     #   use cfg.training.koopman_* hyper‑params

if __name__ == "__main__":
    main()
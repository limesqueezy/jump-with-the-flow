import datetime
import os, copy, sys
from pathlib import Path
from re import I
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchdyn.core import NeuralODE
from tqdm import trange
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
from torchcfm.models.unet import UNetModel
from cifar.utils_cifar import ema, infiniteloop, log_generated_samples, log_final_trajectories, LoggingSummaryWriter, generate_trajectories
from tqdm.auto import trange
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer

from jump_wtf.utils.fid import FIDTrainCallback, FIDValCallback, compute_real_stats
from jump_wtf.models.model import Model
from jump_wtf.models.unet_wrapper import UNetWrapperKoopman
from jump_wtf.data.datamodule import DynamicsDataModule
from jump_wtf.models.autoencoder import Autoencoder_unet
from jump_wtf.operators.generic import GenericOperator_state
import torch.nn as nn

# force only GPU 0 to be visible to CUDA
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@hydra.main(version_base=None, config_path="../conf", config_name="defaults")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    writer = LoggingSummaryWriter()
    run_koop(cfg, writer)

def run_cfm(cfg, writer):

    device = torch.device(cfg.cfm.train.device if torch.cuda.is_available() else "cpu")

    raw_dir          = Path(cfg.paths.raw_dir)
    weights_dir      = Path(cfg.paths.unet_weights_dir)
    traj_dir         = Path(cfg.paths.traj_dir)
    dyn_dataset_dir  = Path(cfg.paths.dyn_dataset_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    dyn_dataset_dir.mkdir(parents=True, exist_ok=True)

    # 2) compute common stem from YAML parameters
    #    - dataset  (e.g. "cifar10")
    #    - stage.matcher (e.g. "otcfm")
    #    - training.total_steps (e.g. 400000)
    #    - stage.traj_steps     (e.g. 100)
    stem = f"{cfg.dataset_name}_single_gray_{cfg.cfm.matcher.name}_step-{cfg.cfm.train.total_steps}"
    # 3) derive all file paths
    weights_path     = weights_dir     / f"{stem}.pt"
    traj_path        = traj_dir        / f"{stem}_traj_ema_{cfg.cfm.traj.traj_steps}.pth"
    # dyn_dataset_path = dyn_dataset_dir / f"{stem}_ema_pairs.pth"
    dyn_dataset_path = dyn_dataset_dir / f"{stem}.pth"
    net = UNetModel(
            dim             =       cfg.model.dim,
            num_channels    =       cfg.model.num_channels,
            num_res_blocks  =       cfg.model.num_res_blocks,
            channel_mult    =       cfg.model.channel_mult,
            num_heads       =       cfg.model.num_heads,
            num_head_channels=      cfg.model.num_head_channels,
            attention_resolutions=  cfg.model.attention_resolutions,
            dropout         =       cfg.model.dropout,
        ).to(device)
    wrapper_net = hydra.utils.instantiate(cfg.wrapper).to(device)
    # Check if we got a learnt velocity field
    if weights_path.exists():
        writer.add_text("cfm/weights",f"Loaded existing weights from `{weights_path}`")

        ckpt = torch.load(weights_path, map_location=device, weights_only=True)
        state = ckpt["ema_model"]
        net.load_state_dict(state)

        wrapper_net.load_state_dict(state)
        log_generated_samples(writer, net, step=0, tag="cfm/loaded_samples")

    # Check if we have the final (t, x, v, x₁, Δt) pth
    if dyn_dataset_path.exists():
        writer.add_text("cfm/dyn_dataset",f"\n~FAST~\nLoading cached DynamicsDataModule from `{dyn_dataset_path}`")
        traj = torch.load(traj_path, map_location="cpu", weights_only=True)
        log_final_trajectories(writer, traj, tag="cfm/x1_loaded")
        dm = DynamicsDataModule(
            traj=traj,
            dynamics=wrapper_net,
            dynamics_path=stem,
            cache_dir=str(dyn_dataset_dir),
            batch_size=cfg.cfm.train.batch_size,
            t_grid=cfg.cfm.traj.traj_steps,
            val_frac=cfg.cfm.train.val_frac,
            chunk_steps=cfg.cfm.traj.chunk_size,
            device=cfg.cfm.train.device,
        )
        dm.setup()
        writer.add_text("cfm/dyn_dataset",f"Returning from `run_cfm`")
        return dm

    # Train the velocity field on CIFAR10
    device = torch.device(cfg.cfm.train.device if torch.cuda.is_available() else "cpu")
    if not weights_path.exists():
        writer.add_text("cfm/weights",f"Training UNet on CIFAR10")
        net     = hydra.utils.instantiate(cfg.model).to(device)
        ema_net = copy.deepcopy(net).to(device)
        matcher_obj = ExactOptimalTransportConditionalFlowMatcher(sigma=cfg.cfm.matcher.sigma)
        opt   = torch.optim.Adam(net.parameters(), lr=cfg.cfm.train.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt,
                    lambda step: min(step, cfg.cfm.train.warmup) / cfg.cfm.train.warmup)

        data_iter = infiniteloop(DataLoader(
            hydra.utils.instantiate(cfg.dataset),
            batch_size=cfg.cfm.train.batch_size,
            shuffle=True,
            num_workers=cfg.cfm.train.num_workers,
            drop_last=True,
            pin_memory=True,
        ))

        for step in trange(cfg.cfm.train.total_steps, desc="CFM training"):
            x1 = next(data_iter).to(device)
            x0 = torch.randn_like(x1, device=device)
            t, xt, ut = matcher_obj.sample_location_and_conditional_flow(x0, x1)
            vt = net(t, xt)
            loss = torch.mean((vt - ut) ** 2)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.cfm.train.grad_clip)
            opt.step(); sched.step()
            ema(net, ema_net, cfg.cfm.train.ema_decay)

            if step % cfg.cfm.train.log_every_n_steps == 0:
                writer.add_scalar("cfm/loss", loss.item(), step)
            if step and step % cfg.cfm.train.save_every_n_steps == 0:
                ckpt_path = weights_dir / f"{stem}_step{step}.pt"
                torch.save(net.state_dict(),ckpt_path)
                writer.add_text("cfm/weights",f"Saving intermediate weights to `{ckpt_path}`")
                log_generated_samples(writer, net, cfg, step=step,tag=f"cfm/intermediate_samples_step{step}")

        torch.save(net.state_dict(), weights_path)
        writer.add_text("cfm/weights",f"Saving final weights to `{weights_path}`")
        log_generated_samples(writer, net, cfg, step=cfg.cfm.train.total_steps, tag="cfm/final_samples")
        wrapper_net.load_state_dict(net.state_dict())
    # else:
    # # That's a duplicate TODO: delete it
    #     net = hydra.utils.instantiate(cfg.model).to(device)
    #     ckpt = torch.load(weights_path, map_location=device)
    #     ema_state = ckpt.get("ema_model", ckpt)
    #     net.load_state_dict(ema_state)

    # Generate trajectories TODO: can we include this step inside the pth creation
    if not traj_path.exists():
        writer.add_text("cfm/trajectories",f"Generating trajectories which will be saved to `{traj_path}`")
        node = NeuralODE(net, solver="euler", sensitivity="adjoint") # Following what they use in their code i.e. not dopri
        traj_path = generate_trajectories(net, node, cfg, device, out_path=str(traj_path))
    writer.add_text("cfm/trajectories",f"Loading existing trajectories from `{traj_path}`")
    traj = torch.load(traj_path, map_location="cpu", weights_only=True)
    log_final_trajectories(writer, traj, tag="cfm/x1_generated")

    writer.add_text("cfm/dyn_dataset",f"Caching DynamicsDataModule to `{dyn_dataset_path}`")
    dm = DynamicsDataModule(
        traj=traj,
        dynamics=wrapper_net,
        dynamics_path=stem,
        cache_dir=str(dyn_dataset_dir),
        batch_size=cfg.cfm.train.batch_size,
        t_grid=cfg.cfm.traj.traj_steps,
        chunk_steps=cfg.cfm.traj.chunk_size,
        val_frac=cfg.cfm.train.val_frac,
    )
    dm.setup()
    return dm

def run_koop(cfg: DictConfig, writer: LoggingSummaryWriter):
    """
    Train (or resume) the Koopman-encoder stage *on top of* the frozen
    dynamics learned by run_cfm.  We simply reuse run_cfm() to get the
    cached DynamicsDataModule, freeze its dynamics network, and launch
    Lightning training for the Model(autoencoder, operator).
    """

    # fetch (or build) DynamicsDataModule from the previous stage
    dm = run_cfm(cfg, writer)
    dm.dynamics.requires_grad_(False)
    dm.dynamics.eval()
    # Precompute FID stats for chosen dataset
    stats_path = Path("assets/fid_stats") / cfg.dataset_name
    stats_path.mkdir(parents=True, exist_ok=True)
    stats_file = stats_path / f"fid_stats_{cfg.dataset_name}.pt"
    stats_file = compute_real_stats(hydra.utils.instantiate(cfg.dataset), stats_file)
    writer.add_text("koopman/fid_stats_path", str(stats_file))

    # AE and Operator
    C, H, W = dm.traj.shape[2:]
    state_dim = C * H * W
    ae_cfg = cfg.koopman.autoencoder
    autoencoder = Autoencoder_unet(
        dim           = (C, H, W),          # raw image shape
        num_channels  = ae_cfg.num_channels,     # 32
        num_res_blocks= ae_cfg.num_res_blocks,   # 1
    ).to(cfg.koopman.train.device)
    # Operator size = 1 + 2 * state_dim   (state_dim = C * H * W)
    operator_dim = 1 + 2 * state_dim
    koopman_op   = GenericOperator_state(operator_dim).to(cfg.koopman.train.device)

    loss_fn = nn.MSELoss()

    # LightningModule wrapper
    model = Model(
        dynamics         = dm.dynamics,
        autoencoder      = autoencoder,
        koopman          = koopman_op,
        loss_function    = loss_fn,
        autoencoder_lr   = cfg.koopman.train.autoencoder_lr,
        lie_lr           = cfg.koopman.train.lie_lr,
        lr_scheduler     = cfg.koopman.train.lr_scheduler,
        decode_predict_bool = cfg.koopman.train.decode_predict,
        vae_loss_bool    = cfg.koopman.train.vae_loss,
        koop_reg_bool    = cfg.koopman.train.koop_reg,
        energy_bool      = cfg.koopman.train.energy_penalty,
        gamma            = cfg.koopman.train.gamma,
        delta_t          = cfg.koopman.train.delta_t,
        multistep        = cfg.koopman.train.multistep,
        period           = cfg.koopman.train.period,
        time_bool        = cfg.koopman.train.time_dep,
        num_iter         = cfg.koopman.train.num_iter,
        warmup_step      = cfg.koopman.train.warmup_step,
        fid_interval     = cfg.koopman.train.fid_interval,
        fid_real_stats_path = stats_file,
    )

    run_id     = datetime.datetime.now().strftime(f"{cfg.dataset_name}-koopman-%Y%m%d-%H%M%S")
    ckpt_root  = Path("checkpoints") / run_id
    ckpt_root.mkdir(parents=True, exist_ok=True)

    checkpoint_cb  = ModelCheckpoint(
        dirpath=ckpt_root, filename="epoch-{epoch}",
        save_top_k=-1, every_n_epochs=1)

    ckpt_cb        = ModelCheckpoint(
        dirpath=ckpt_root, filename="best-step{step:06d}-{train_loss_step:.4f}",
        monitor="total_loss", mode="min", save_top_k=1,
        every_n_train_steps=100, save_on_train_epoch_end=False, save_last=True)

    ckpt_fid_train = ModelCheckpoint(
        dirpath=ckpt_root, monitor="fid_train", mode="min",
        filename="best-fid-train-{step:.0f}-{fid_train:.3f}",
        save_top_k=1, every_n_train_steps=50, save_on_train_epoch_end=False)

    ckpt_fid_val   = ModelCheckpoint(
        dirpath=ckpt_root, monitor="fid_val", mode="min",
        filename="best-fid-val-{epoch:03d}-{fid_val:.3f}", save_top_k=1)

    trainer = Trainer(
        callbacks=[
            checkpoint_cb,
            ckpt_cb,
            FIDTrainCallback(every_n_steps=25),
            ckpt_fid_train,
            FIDValCallback(),
            ckpt_fid_val,
        ],
        default_root_dir=".",
        accelerator="gpu", devices=[0],
        max_epochs=cfg.koopman.train.max_epochs,
        log_every_n_steps=cfg.koopman.train.log_every_n_steps,
    )
    trainer.fit(model, dm)
    return model

if __name__ == "__main__":
    main()
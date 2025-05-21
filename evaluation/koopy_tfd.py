import glob
import os, argparse, subprocess, tempfile, torch
from pathlib import Path
from torchvision.transforms import ToTensor, Resize, Compose
from jump_wtf.data.toronto_face import TorontoFaceDataset
from torchvision.utils import save_image
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from jump_wtf.utils.sampling import sample_efficient
import cleanfid.fid as fid

from jump_wtf.models.model import Model
from jump_wtf.models.autoencoder import Autoencoder_unet
from jump_wtf.operators.generic import GenericOperator_state
from jump_wtf.models.unet_wrapper import UNetWrapperKoopman

def load_net(ckpt_glob, device="cuda"):
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

    C, H, W = 1, 28, 28
    wrapper_net = UNetWrapperKoopman(
        dim=(1, 28, 28), 
        num_channels=64, 
        num_res_blocks=2,
        num_heads=4,
    ).to("cpu")

    ckpt = torch.load("assets/unet_dynamics/toronto_face_toronto_face_otcfm_step-30000.pt", map_location=device, weights_only=True)
    wrapper_net.load_state_dict(ckpt, strict=False)

    state_dim = C * H * W
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

    # ae = Autoencoder_unet(
    #     dim=(1, 28, 28),
    #     num_channels=64,
    #     channel_mult=[1, 2, 2],
    #     num_res_blocks=2,
    #     num_heads=4,
    #     num_head_channels=64,
    #     attention_resolutions="14,7",
    #     dropout=0.1,
    #     learn_sigma=False,
    #     class_cond=False,
    #     use_checkpoint=False,
    #     use_fp16=False,
    #     use_new_attention_order=False,
    #     bottleneck=False,
    #     resblock_updown=True,
    # )

    ae = Autoencoder_unet(
        dim=(1, 28, 28),            # wrapper.dim
        num_channels=64,            # wrapper.num_channels
        num_res_blocks=2,           # wrapper.num_res_blocks
        channel_mult=[1, 2, 2],     # wrapper.channel_mult
        num_heads=4,                # wrapper.num_heads
        num_head_channels=64,       # wrapper.num_head_channels
        attention_resolutions="16", # wrapper.attention_resolutions
        dropout=0.1,                # wrapper.dropout
        learn_sigma=False,          # wrapper.learn_sigma
        class_cond=False,           # wrapper.class_cond
        use_checkpoint=False,       # wrapper.use_checkpoint
        use_fp16=False,             # wrapper.use_fp16
        use_new_attention_order=False,
    )

    # ae = Autoencoder_unet(
    #     dim=(1, 28, 28),            # wrapper.dim
    #     num_channels=128,            # wrapper.num_channels
    #     num_res_blocks=3,           # wrapper.num_res_blocks
    #     channel_mult=[1, 2, 2],     # wrapper.channel_mult
    #     num_heads=4,                # wrapper.num_heads
    #     num_head_channels=64,       # wrapper.num_head_channels
    #     attention_resolutions="14,7", # wrapper.attention_resolutions
    #     dropout=0.1,                # wrapper.dropout
    #     bottleneck          = False,
    #     resblock_updown     = True,
    #     learn_sigma         = False,
    #     class_cond          = False,
    #     use_checkpoint      = False,
    #     use_fp16            = False,
    #     use_new_attention_order = False,
    # )




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

# ──────────────────────────────────────────────────────────────────────
def export_real(root, n, out):
    ds = TorontoFaceDataset(
        root=root,
        train=None,
        transform=Compose([
            Resize((28, 28)),
            ToTensor()
        ])
    )
    out.mkdir(parents=True, exist_ok=True)
    total = len(ds)
    for i in range(min(n, total)):
        img, _ = ds[i]
        save_image(img.expand(3, -1, -1), out / f"{i:05d}.png")


# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample(net, n, bs, steps, device, out):
    """
    Generate ``n`` MNIST-sized images with the Koopman pipeline and
    write them to ``out`` as RGB PNG files, ready for FID.
    `steps` is forwarded to ``sample_efficient`` as ``n_iter``.
    """
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    with Progress("[progress.description]{task.description}",
                  BarColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as p:
        task = p.add_task("Generating", total=n)
        while done < n:
            cur = min(bs, n - done)
            imgs = (
                sample_efficient(
                    net,
                    n_iter   = steps,
                    n_samples= cur,
                    device   = device,
                )                                    # (cur,1,28,28) in [-1,1]
                .add_(1).div_(2)                     # → [0,1]
                .expand(-1, 3, -1, -1)               # fake-RGB for FID
            )
            for j, img in enumerate(imgs):
                save_image(img, out / f"{done+j:05d}.png")
            done += cur
            p.update(task, advance=cur)

# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root",   default="assets/raw_datasets")
    ap.add_argument("--num-samples", type=int, default=10_000)
    ap.add_argument("--batch-size",  type=int, default=2058)
    ap.add_argument("--ode-steps",   type=int, default=100)
    ap.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    work = Path("/mnt/disk1/ari/koopy_eval/koop/tfd")

    real, gen = work/"real", work/"generated"

    print("→ exporting real TFD images")
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

import copy
import torch
from pathlib import Path
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

# ——— COPY from train_cifar10.py ———
def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training."""
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5

    save_image(
        traj,
        savedir + f"{net_}_generated_FM_images_step_{step}.png",
        nrow=8
    )
# ——— end copy ———

# ——— COPY model & load logic from compute_fid.py ———
from torchcfm.models.unet.unet import UNetModelWrapper

def load_model(checkpoint_path, device):
    # instantiate exactly as in compute_fid.py
    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=128,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["ema_model"]  # same key they save in train_cifar10.py
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # strip off any "module." if it’s there
        from collections import OrderedDict
        new_state = OrderedDict((k.replace("module.", ""), v)
                                for k, v in state.items())
        model.load_state_dict(new_state)

    model.eval()
    return model
# ——— end copy ———

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python load_and_generate.py <checkpoint.pt> <output_dir>")
        sys.exit(1)

    ckpt_path, out_dir = sys.argv[1], sys.argv[2]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt_path, device)

    # This is exactly what train_cifar10.py does at step 0
    generate_samples(
        model=model,
        parallel=False,            # torchdyn only runs on 1 GPU anyway
        savedir=out_dir.rstrip("/") + "/",
        step=0,
        net_="ema"                 # use "normal" if you want the raw net instead
    )

    print(f"✅ Wrote samples to {out_dir}")

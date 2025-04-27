from jump_wtf.operators.utils import get_koop_continuous
from pathlib import Path
import torch
import math
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from .sampling import sample_efficient

def sample_efficient_plot(model_generic, t_max=1, n_iter=1, n_samples=64, device="cuda"):
    """
    Generates samples via sample_efficient(), saves them under generated_samples/<run_id>/,
    and plots them in a grid.
    """
    # determine run_id
    ckpt = getattr(model_generic, "ckpt_path", None)
    run_id = Path(ckpt).parent.name if ckpt else datetime.now().strftime("run-%Y%m%d-%H%M%S")
    save_dir = Path("generated_samples") / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time()
    imgs = sample_efficient(model_generic, t_max, n_iter, n_samples, device)
    print(f"It took: {time() - start:.2f} seconds")

    if isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().numpy()

    # plot grid
    cols = math.ceil(math.sqrt(n_samples))
    rows = math.ceil(n_samples / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()

    for i, img in enumerate(imgs):
        # normalize to [0,1] for plotting
        normed = (img + 1) / 2
        gray = normed.squeeze(0)
        axs[i].imshow(gray, cmap="gray", vmin=0, vmax=1)
        axs[i].axis("off")
        axs[i].set_title(f"Sample {i+1:02d}")
        # save for FID
        # plt.imsave(save_dir / f"sample_{i+1:02d}.png", gray, cmap="gray", vmin=0, vmax=1)

    # hide any extra axes
    for ax in axs[n_samples:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

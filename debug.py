# Encode Decode vis

# Grayscale training

# 1-step prediction

# ‖zₖ‖ during k-step rollout starting from a real sample 

import math
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
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from sklearn.cluster import SpectralClustering

import torch, matplotlib.pyplot as plt, matplotlib.animation as animation, numpy as np, math, time
from matplotlib.animation import PillowWriter
from pathlib import Path


def save_eigenvalue_spectrum(K: np.ndarray, output_dir: Path) -> None:
    """
    Compute eigenvalues of K, save raw data and a scatter plot of the spectrum.
    """
    eigs = np.linalg.eigvals(K)
    # Save raw eigenvalues
    np.save(output_dir / "eigenvalues.npy", eigs)
    # Plot spectrum
    plt.figure(figsize=(6, 6))
    plt.scatter(eigs.real, eigs.imag, s=2)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('Eigenvalue Spectrum')
    plt.grid(True)
    plt.savefig(output_dir / "eigenvalue_spectrum.png")
    plt.close()


def save_singular_value_decay(K: np.ndarray, output_dir: Path) -> float:
    """
    Compute singular values of K, save raw data, plot decay, and return effective rank.
    """
    s = np.linalg.svd(K, compute_uv=False)
    np.save(output_dir / "singular_values.npy", s)

    # Plot singular value decay
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(s)+1), np.log(s))
    plt.xlabel('Singular value index')
    plt.ylabel('Log of singular value')
    plt.title('Singular Value Decay')
    plt.grid(True)
    plt.savefig(output_dir / "singular_value_decay.png")
    plt.close()

    # Effective rank
    p = s / s.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-16))))
    with open(output_dir / "effective_rank.txt", 'w') as f:
        f.write(f"{eff_rank:.6e}\n")
    return eff_rank


def save_offdiagonal_energy_ratio(K: np.ndarray, output_dir: Path) -> float:
    """
    Compute the ratio of off-diagonal energy to total energy in K and save to file.
    """
    total_energy = np.sum(K**2)
    diag_energy = np.sum(np.diag(K)**2)
    off_energy = total_energy - diag_energy
    ratio = float(off_energy / total_energy)
    with open(output_dir / "offdiagonal_energy_ratio.txt", 'w') as f:
        f.write(f"{ratio:.6e}\n")
    return ratio


def save_ks_test_pvalue(K: np.ndarray, output_dir: Path) -> float:
    """
    Perform KS test on off-diagonal entries versus standard normal and save p-value.
    """
    mask = ~np.eye(K.shape[0], dtype=bool)
    off_vals = K[mask].ravel()
    off_standard = (off_vals - off_vals.mean()) / off_vals.std()
    stat, pval = kstest(off_standard, 'norm')
    with open(output_dir / "ks_test_pvalue.txt", 'w') as f:
        f.write(f"{pval:.6e}\n")
    return pval


def save_heatmap(K: np.ndarray, output_dir: Path) -> None:
    """
    Save a heatmap of log1p(abs(K)).
    """
    plt.figure(figsize=(6, 6))
    im = plt.imshow(np.log1p(np.abs(K)), aspect='auto')
    plt.colorbar(im)
    plt.title('Heatmap of log1p(abs(K))')
    plt.savefig(output_dir / "heatmap_log_abs_K.png")
    plt.close()


def save_spectral_clustering(K: np.ndarray, output_dir: Path, n_clusters: int = 8) -> np.ndarray:
    """
    Perform spectral clustering on |K|+|K^T| and save cluster labels.
    """
    adjacency = np.abs(K) + np.abs(K.T)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(adjacency)
    np.savetxt(output_dir / "cluster_labels.txt", labels, fmt='%d')
    return labels


def get_koop_info(K: np.ndarray, output_dir: str = 'metrics', n_clusters: int = 8) -> None:
    """
    Compute and save a suite of Koopman operator diagnostics:
      - Eigenvalue spectrum
      - Singular value decay and effective rank
      - Off-diagonal energy ratio
      - KS-test p-value of off-diagonals
      - Heatmap of abs(K)
      - Spectral clustering labels

    All outputs are saved under `output_dir`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Spectrum
    print("Computing eigenvalue spectrum...")
    save_eigenvalue_spectrum(K, out)

    # 2) Singular values
    print("Computing singular value decay and effective rank...")
    eff_rank = save_singular_value_decay(K, out)
    print(f"Effective rank: {eff_rank:.4f}")

    # 3) Off-diagonal ratio
    print("Computing off-diagonal energy ratio...")
    off_ratio = save_offdiagonal_energy_ratio(K, out)
    print(f"Off-diagonal ratio: {off_ratio:.4f}")

    # 4) KS test
    print("Performing KS test on off-diagonal entries...")
    pval = save_ks_test_pvalue(K, out)
    print(f"KS-test p-value: {pval:.4e}")

    # 5) Heatmap
    print("Saving heatmap of log1p(abs(K))...")
    save_heatmap(K, out)

    # 6) Spectral clustering
    print(f"Performing spectral clustering with {n_clusters} clusters...")
    labels = save_spectral_clustering(K, out, n_clusters)
    print("Saved cluster labels.")

    print(f"All Koopman diagnostics saved in '{output_dir}/'.")

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

        ckpt = torch.load("assets/unet_dynamics/mnist_full_otcfm_step-20.pt", map_location=device, weights_only=True)
        wrapper_net.load_state_dict(ckpt)

        state_dim = C * H * W
        ae = Autoencoder_unet(
            dim=(C, H, W),
            num_channels=32,
            num_res_blocks=1,
        )

    elif dataset.lower() == "cifar":
        C, H, W = 3, 32, 32 # CHANGE FOR COLOR CIFAR
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
        pretrained_full_cifar = "assets/unet_dynamics/cifar10_rgb_full_otcfm_step-400000.pt"
        ckpt = torch.load(pretrained_full_cifar, map_location=device, weights_only=True)
        state = ckpt["ema_model"]

        # duo_gray_cifar = "assets/unet_dynamics/cifar10_double_gray_otcfm_step-20000.pt"
        # state = torch.load(duo_gray_cifar, map_location=device, weights_only=True)


        wrapper_net.load_state_dict(state)

        state_dim = C * H * W
        ae = Autoencoder_unet(
            dim             =       [C, H, W],
            # num_channels    =       128,
            num_channels    =       256,
            # num_res_blocks  =       2,
            num_res_blocks  =       3,
            # channel_mult    =       [1, 2, 2, 2],
            channel_mult = [1,2,3,4],
            # num_heads       =       4,
            num_heads       =       8,
            num_head_channels=      64,
            # attention_resolutions=  "16",
            attention_resolutions=  "16,8",
            dropout         =       0.1,
        )

    elif dataset.lower() == "tfd":
        # Toronto Face Dataset (1×48×48)
        # C, H, W = 1, 48, 48
        C, H, W = 1, 28, 28

        # wrapper_net = UNetWrapperKoopman(
        #     dim                 = (C, H, W),
        #     num_channels        = 64,
        #     num_res_blocks      = 2,
        #     channel_mult        = [1, 1, 2, 3, 4],
        #     num_heads           = 4,
        #     num_head_channels   = 64,
        #     attention_resolutions= "16",
        #     dropout             = 0.1,
        # ).to(device)

        wrapper_net = UNetWrapperKoopman(
            dim                    = (C, H, W),
            num_channels           = 64,
            num_res_blocks         = 2,
            channel_mult           = [1, 2, 2],
            num_heads              = 4,
            num_head_channels      = 64,
            attention_resolutions  = "16",
            dropout                = 0.1,
            learn_sigma            = False,
            class_cond             = False,
            use_checkpoint         = False,
            use_fp16               = False,
            use_new_attention_order= False,
        ).to(device)


        # replace with the actual path to your TFD checkpoint
        tfd_ckpt = "assets/unet_dynamics/toronto_face_toronto_face_otcfm_step-5000.pt"
        state = torch.load(tfd_ckpt,
                           map_location=device,
                           weights_only=True)
        print("Dynamics were successfully loaded!")
        wrapper_net.load_state_dict(state)

        state_dim = C * H * W
        # ae = Autoencoder_unet(
        #     dim=(1, 48, 48),
        #     num_channels=64,
        #     num_res_blocks=2,
        #     channel_mult=[1, 1, 2, 3, 4],
        #     num_heads=4,
        #     num_head_channels=64,
        #     attention_resolutions="16",
        #     dropout=0.1,
        #     learn_sigma=False,
        #     class_cond=False,
        #     use_checkpoint=False,
        #     use_fp16=False,
        #     use_new_attention_order=False,
        # )

        ae = Autoencoder_unet(
            dim                    = (C, H, W),
            num_channels           = 64,
            num_res_blocks         = 2,
            channel_mult           = [1, 2, 2],
            num_heads              = 4,
            num_head_channels      = 64,
            attention_resolutions  = "16",
            dropout                = 0.1,
            learn_sigma            = False,
            class_cond             = False,
            use_checkpoint         = False,
            use_fp16               = False,
            use_new_attention_order= False,
        )

    else:
        raise ValueError("dataset must be 'mnist', 'cifar' or 'tfd'")

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

def strip_fid_from_checkpoint(in_ckpt: str, out_ckpt: str) -> str:
    """
    Remove all fid_train.* and fid_val.* keys from a Lightning checkpoint.
    If in_ckpt == out_ckpt, writes to a new file with '_stripped_fid' before the extension.
    Returns the path of the stripped checkpoint.
    """
    # if they passed the same path, create a new one
    if os.path.abspath(in_ckpt) == os.path.abspath(out_ckpt):
        base, ext = os.path.splitext(in_ckpt)
        out_ckpt = f"{base}_stripped_fid{ext}"

    # 1) Load the checkpoint (handles both Lightning‐style and raw state_dict)
    ckpt = torch.load(in_ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # 2) Remove all fid_train.* and fid_val.* entries
    to_remove = [k for k in sd if k.startswith("fid_train.") or k.startswith("fid_val.")]
    for k in to_remove:
        sd.pop(k)

    # 3) Save back
    if "state_dict" in ckpt:
        ckpt["state_dict"] = sd
        torch.save(ckpt, out_ckpt)
    else:
        torch.save(sd, out_ckpt)

    print(f"Removed {len(to_remove)} FID keys and wrote stripped checkpoint to\n    {out_ckpt}")
    return out_ckpt

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

def sample_analytical_plot(
    model_generic,
    t_max=1,
    n_samples=1,
    n_mode=50,
    device=None,
    output_dir="metrics"
):
    import torch, math, numpy as np, matplotlib.pyplot as plt, time
    from pathlib import Path

    # determine device if not provided
    if device is None:
        try:
            device = next(model_generic.parameters()).device
        except StopIteration:
            device = torch.device("cpu")  # fallback to CPU

    # prepare output folder
    out = Path(output_dir) / "sample_plots"
    out.mkdir(parents=True, exist_ok=True)

    # sample random initial conditions for an H×W image (here 32×32)
    x_0 = torch.randn((n_samples, 32, 32), device=device)

    # infer total pixels and Koopman-input dim = C*H*W*2 + 1
    # (pixels = H*W, C assumed 1)
    pixels = x_0.shape[1] * x_0.shape[2]   # = 32*32
    Kdim   = pixels * 2 + 1                # = 32*32*2 + 1
    identity = torch.eye(Kdim, device=device)

    # compute Koopman operator
    Kmat = model_generic.koopman(identity)
    vals, P = torch.linalg.eig(Kmat)
    idx = torch.argsort(vals.real)[-n_mode:]
    P = P[:, idx]
    P_1 = torch.linalg.pinv(P)
    vals = vals[idx]

    # preprocess initial conditions
    x_flat = x_0.reshape(n_samples, -1)
    t0 = torch.zeros((n_samples, 1), device=device)
    x_in = torch.cat((t0, x_flat), dim=1)

    # actual sampling time
    start = time.time()
    encoded = model_generic.autoencoder.encoder(x_in).to(torch.complex64)
    projected = encoded @ P
    evolved = projected @ torch.diag(torch.exp(vals * t_max))
    decoded = model_generic.autoencoder.decoder((evolved @ P_1).real)
    decoded = decoded.detach().cpu().numpy()
    torch.cuda.empty_cache()
    print(f"It took: {time.time() - start:.2f} seconds")

    # plot grid and save instead of plt.show()
    n_cols = math.ceil(math.sqrt(n_samples))
    n_rows = math.ceil(n_samples / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axs = np.atleast_1d(axs).flatten()
    for i in range(n_samples):
        img = decoded[i, 1:1 + 28*28].reshape(28, 28).clip(-1,1)
        axs[i].imshow(img, cmap="gray")
        axs[i].axis("off")
        axs[i].set_title(f"Sample {i+1}")
    for j in range(n_samples, len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    fname = f"analytical_samples_t{t_max}_m{n_mode}.png"
    plt.savefig(out / fname)          # changed: save to file
    plt.close()

def animate_sample_evolution(
    model_generic,
    t_max: float = 1.0,
    n_samples: int = 10,
    max_modes: int = 100,
    step: int = 5,
    device = None,
    output_dir: str = "metrics"           # modified: parameterize save location
):

    if device is None:
        try:
            device = next(model_generic.parameters()).device
        except StopIteration:
            device = torch.device("cpu")  # fallback to CPU

    # prepare output folder
    out = Path(output_dir) / "sample_plots"
    out.mkdir(parents=True, exist_ok=True)

    # sample random initial conditions for an H×W image (here 32×32)
    x_0 = torch.randn((n_samples, 32, 32), device=device)

    # flatten and prepend time‐channel
    x_flat = x_0.reshape(n_samples, -1)
    t0     = torch.zeros((n_samples, 1), device=device)
    x_in   = torch.cat((t0, x_flat), dim=1)

    # infer total pixels and Koopman-input dim = C*H*W*2 + 1
    # (pixels = H*W, C assumed 1)
    pixels = x_0.shape[1] * x_0.shape[2]   # = 32*32
    Kdim   = pixels * 2 + 1                # = 32*32*2 + 1
    identity = torch.eye(Kdim, device=device)

    # Koopman operator & eigendecomposition
    Kmat = model_generic.to(device).koopman(identity)
    vals, P = torch.linalg.eig(Kmat)
    vals, P = vals.to(device), P.to(device)

    # Encode once
    encoded = model_generic.autoencoder.encoder(x_in).to(torch.complex64)

    # build frames
    fig, axs = plt.subplots(1, n_samples, figsize=(n_samples*2, 2))
    axs = np.atleast_1d(axs).flatten()
    ims = []
    for n_mode in range(1, max_modes+1, step):
        idx = torch.argsort(vals.real)[-n_mode:]
        Pk = P[:, idx]; vk = vals[idx]
        Pk_inv = torch.linalg.pinv(Pk)

        proj = encoded @ Pk
        evo  = proj @ torch.diag(torch.exp(vk * t_max))
        latent = evo @ Pk_inv

        decoded = model_generic.autoencoder.decoder(latent.real)\
                  .detach().cpu().numpy()

        frames = []
        for i in range(n_samples):
            img = decoded[i, 1:1+28*28].reshape(28,28).clip(-1,1)
            im = axs[i].imshow(img, cmap="gray", animated=True)
            axs[i].axis("off")
            axs[i].set_title(f"Mode={n_mode}")
            frames.append(im)
        ims.append(frames)

    # create and save gif
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=500)
    gif_path = out / f"evolution_t{t_max}_step{step}.gif"
    ani.save(str(gif_path), writer=PillowWriter(fps=10))   # modified: save as GIF
    plt.close(fig)
    return gif_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Debug: sample from Koopman model")
    p.add_argument("--ckpt",     type=str, default="checkpoints/cifar10-koopman-20250427-183648/last.ckpt",
                   help="glob for Koopman checkpoints")
    p.add_argument("--dataset",  type=str, default="cifar",
                   choices=["mnist","cifar","tfd"])
    p.add_argument("--n-rows",type=int, default=8,
                   help="number of rows to draw")
    p.add_argument("--n-iter",   type=int, default=1,
                   help="number of Koopman steps")
    p.add_argument("--t-max",    type=int, default=1,
                   help="total time horizon")
    p.add_argument("--device",   type=str, default="cuda")
    args = p.parse_args()

    # stripped_ckpt = strip_fid_from_checkpoint(args.ckpt, args.ckpt)

    model = load_model(args.ckpt, dataset=args.dataset, device=args.device)

    sample_and_save_grid(model, k=args.n_rows, t_max=args.t_max, n_iter=args.n_iter, device="cuda")

    # sample_analytical_plot(model,
    #                    output_dir="metrics",
    #                    t_max=2.0,
    #                    n_samples=10,
    #                    n_mode=200)
    
    # sample_efficient_plot(
    #     model,
    #     t_max=args.t_max,
    #     n_iter=args.n_iter,
    #     device=args.device,
    # )

    get_koop_info(K = model.koopman.operator.cpu().detach().numpy())

    gif_file = animate_sample_evolution(
        model,
        t_max=2.0,
        n_samples=10,
        max_modes=1569,
        step=5,
        # device and output_dir are optional
    )
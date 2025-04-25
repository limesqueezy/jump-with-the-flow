import torch
from jump_wtf.operators.utils import get_koop_continuous
from time import time

def sample_efficient(model_generic, t_max=1, n_iter=100, n_samples=1, device="cuda"):
    """
    Runs the Koopman evolution and returns a NumPy array of shape
    (n_samples, 28, 28), with values in [–1, 1].
    """
    model_generic.to(device).eval()
    dt = t_max / n_iter

    enc = model_generic.autoencoder.encoder
    C   = enc.in_channels
    H   = enc.image_size
    W   = enc.image_size
    spatial_dim = C * H * W
    #    noise: [n_samples, C, H, W] flattened to [n_samples, spatial_dim]
    x_spatial = torch.randn((n_samples, C, H, W), device=device)
    x_flat    = x_spatial.view(n_samples, spatial_dim)
    t0        = torch.zeros((n_samples, 1), device=device)
    x         = torch.cat((x_flat, t0), dim=1)  # shape [n_samples, spatial_dim+1]

    koop_op = get_koop_continuous(model_generic, model_generic.koopman.operator_dim, dt)
    with torch.no_grad():
        z = model_generic.autoencoder.encoder(x)
        for _ in range(n_iter):
            z = z @ koop_op
            x_hat = model_generic.autoencoder.decoder(z)
            z = model_generic.autoencoder.encoder(x_hat) @ koop_op

        decoded = model_generic.autoencoder.decoder(z)

    # reshape to (n_samples, C, H, W) and clamp
    imgs = (
        decoded[:, :spatial_dim]
        .view(n_samples, C, H, W)
        .clamp(-1, 1)
    )
    return imgs
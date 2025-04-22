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

    # initial noise + time channel
    x = torch.randn((n_samples, 28, 28), device=device).reshape(n_samples, -1)
    t0 = torch.zeros((n_samples, 1), device=device)
    x = torch.cat((x, t0), dim=1)

    koop_op = get_koop_continuous(model_generic, model_generic.koopman.operator_dim, dt)
    torch.cuda.empty_cache()

    with torch.no_grad():
        z = model_generic.autoencoder.encoder(x)
        for _ in range(n_iter):
            z = z @ koop_op
            x_hat = model_generic.autoencoder.decoder(z)
            z = model_generic.autoencoder.encoder(x_hat) @ koop_op

        decoded = model_generic.autoencoder.decoder(z)

    # reshape and clamp to [–1,1]
    imgs = decoded[:, :28*28] \
            .reshape(n_samples, 1, 28, 28) \
            .clamp(-1, 1)
    return imgs
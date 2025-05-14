import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def get_koop_continuous_batch(model, output_dim, t):
    """
    Compute a batch of evolution operators given a batch of time values `t`.

    Args:
        model: A model with a `koopman()` method.
        output_dim: Dimension of the identity matrix.
        t: Tensor of shape (N_batch, 1) representing time values.

    Returns:
        evolution_op: Tensor of shape (N_batch, output_dim, output_dim)
    """
    N_batch = t.shape[0]
    identity = torch.eye(output_dim).to(t.device)  # use same device as t
    identity = identity.unsqueeze(0).expand(N_batch, -1, -1)  # (N_batch, output_dim, output_dim)

    scaled_identity = identity * t.view(N_batch, 1, 1)  # (N_batch, output_dim, output_dim)
    lie_module = model.koopman(scaled_identity)  # Expected output: (N_batch, output_dim, output_dim)
    evolution_op = torch.matrix_exp(lie_module)  # (N_batch, output_dim, output_dim)

    return evolution_op

def get_koop_continuous(model, output_dim, t):
    device = next(model.parameters()).device
    identity = torch.eye(output_dim).to(device)
    lie_module = model.koopman(identity)
    evolution_op = torch.matrix_exp(t * lie_module)
    return evolution_op

def plot_operator(K: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8,6), dpi=150)
    sns.heatmap(K, ax=ax, cmap="viridis", cbar_kws={"label":"Weight"})
    ax.set_title("Koopman Operator Heatmap")
    ax.set_xlabel("Input Mode")
    ax.set_ylabel("Output Mode")
    plt.tight_layout()
    return fig
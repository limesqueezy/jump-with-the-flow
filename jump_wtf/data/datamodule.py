from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning as L
from .samplers import TimeGroupedSampler, ListSampler
from pathlib import Path


class DynamicsDataModule(L.LightningDataModule):
    """Builds the (t, x, v, x₁, Δt) tuples once and serves them epoch‑after‑epoch."""
    def __init__(
        self,
        traj: torch.Tensor,
        dynamics,
        batch_size: int = 64,
        t_grid: int = 100,
        val_frac: float = 0.2,
        device = "cuda"
    ):
        super().__init__()
        self.traj       = traj          # (T, B, C, H, W)
        self.dynamics   = dynamics.to(device)
        self.batch_size = batch_size
        self.t_grid     = torch.linspace(0, 1, t_grid)
        self.val_frac   = val_frac
        self.device     = device

    def setup(self, stage=None):
        cache_path = Path("assets/full_mnist_cfm_pairs/full_ds.pth")
        if cache_path.exists():
            self.full_ds = torch.load(cache_path)
        else:
            # temp move model to CPU
            orig_dev = next(self.dynamics.parameters()).device
            self.dynamics.to("cpu").eval()

            matrix_x0 = []
            matrix_system_derivative_data = []
            matrix_targets = []
            matrix_delta_t = []

            for i, t_val in enumerate(self.t_grid):
                # images at time-step i, *CPU*
                x = self.traj[i].cpu()
                x1 = self.traj[-1].cpu()
                B = x.shape[0]

                # build the time channel and Δt, on CPU
                t = torch.full((B, 1), float(t_val), device="cpu")
                delta_t = 1.0 - t

                # dx on CPU
                with torch.no_grad():
                    dx = self.dynamics(t, x)

                # assembling inputs & targets
                matrix_x0.append(torch.hstack((t, x.reshape(B, -1))))
                matrix_system_derivative_data.append(dx)
                matrix_targets.append(torch.hstack((torch.ones(B,1), x1.reshape(B,-1))))
                matrix_delta_t.append(delta_t)

            # stack into four big CPU tensors
            matrix_x0   = torch.vstack(matrix_x0)
            matrix_dx   = torch.vstack(matrix_system_derivative_data)
            matrix_y    = torch.vstack(matrix_targets)
            matrix_dt   = torch.vstack(matrix_delta_t)

            # cache it
            self.full_ds = TensorDataset(
                matrix_x0, matrix_dx, matrix_y, matrix_dt
            )
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.full_ds, cache_path)

            # move dynamics back to original device
            self.dynamics.to(orig_dev)

        N = len(self.full_ds)
        n_val   = int(self.val_frac * N)
        n_train = N - n_val
        train_set = set(range(n_train))
        val_set   = set(range(n_train, N))

        # reverse‑time ordering over [0..N-1]
        full_order = TimeGroupedSampler(
            time_steps=len(self.t_grid),
            group_size=self.traj.size(1),
        ).indices

        self.train_ordered_indices = [i for i in full_order if i in train_set]
        self.val_ordered_indices   = [i for i in full_order if i in val_set]

        self.train_sampler = ListSampler(self.train_ordered_indices)
        self.val_sampler   = ListSampler(self.val_ordered_indices)

    def train_dataloader(self):
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

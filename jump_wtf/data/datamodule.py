from __future__ import annotations
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning as L
from .samplers import TimeGroupedSampler, ListSampler
from pathlib import Path
from typing import Union

class DynamicsDataModule(L.LightningDataModule):
    """Builds the (t, x, v, x₁, Δt) tuples once and serves them epoch‑after‑epoch."""
    def __init__(
        self,
        traj: torch.Tensor,
        dynamics,
        dynamics_path: str = "mnist_otcfm_epoch_20",
        cache_dir: Union[str, Path] = "assets/dynamics_datasets",
        batch_size: int = 64,
        t_grid: int = 100,
        val_frac: float = 0.2,
        chunk_steps: int = 2000,
        device = "cuda"
    ):
        super().__init__()
        self.traj       = traj          # (T, B, C, H, W)
        self.dynamics   = dynamics.to(device)
        self.batch_size = batch_size
        self.t_grid     = torch.linspace(0, 1, t_grid)
        self.val_frac   = val_frac
        self.chunk_steps= chunk_steps
        self.device     = device

        self.cache_path = Path(cache_dir) / f"{dynamics_path}.pth"

    def setup(self, stage=None):
        print(f"Data module from {self.cache_path}")
        if self.cache_path.exists():
            self.full_ds = torch.load(self.cache_path)
        else:
            # # temp move model to CPU
            # orig_dev = next(self.dynamics.parameters()).device
            # self.dynamics.to("cpu").eval()

            # matrix_x0 = []
            # matrix_system_derivative_data = []
            # matrix_targets = []
            # matrix_delta_t = []

            # # TODO: Would this happen faster on GPU? Could we do this on GPU?

            # for i, t_val in enumerate(tqdm(self.t_grid, desc="building data")):
            #     # images at time-step i, *CPU*
            #     x = self.traj[i].cpu()
            #     x1 = self.traj[-1].cpu()
            #     B = x.shape[0]

            #     # build the time channel and Δt, on CPU
            #     t = torch.full((B, 1), float(t_val), device="cpu")
            #     delta_t = 1.0 - t

            #     # dx on CPU
            #     with torch.no_grad():
            #         dx = self.dynamics(t, x)

            #     # assembling inputs & targets
            #     matrix_x0.append(torch.hstack((t, x.reshape(B, -1))))
            #     matrix_system_derivative_data.append(dx)
            #     matrix_targets.append(torch.hstack((torch.ones(B,1), x1.reshape(B,-1))))
            #     matrix_delta_t.append(delta_t)

            # # stack into four big CPU tensors
            # matrix_x0   = torch.vstack(matrix_x0)
            # matrix_dx   = torch.vstack(matrix_system_derivative_data)
            # matrix_y    = torch.vstack(matrix_targets)
            # matrix_dt   = torch.vstack(matrix_delta_t)

            # # cache it
            # self.full_ds = TensorDataset(
            #     matrix_x0, matrix_dx, matrix_y, matrix_dt
            # )
            # self.cache_path.parent.mkdir(exist_ok=True, parents=True)
            # torch.save(self.full_ds, self.cache_path)

            # # move dynamics back to original device
            # self.dynamics.to(orig_dev)
            
            ###────────────────── fast + streaming build ──────────────────
            dev = torch.device(self.device)     # usually "cuda"
            self.dynamics.to(dev).eval()

            T, B, C, H, W = self.traj.shape
            row_feats     = C * H * W
            cols_x0y      = 1 + row_feats      # (t, x)   and   (dt,dx)   and    (1, x₁)

            dtype      = torch.float32                         # keep legacy precision
            chunk_T    = getattr(self, "chunk_steps", T)       # optional arg
            tmp_path   = self.cache_path.with_suffix(".tmp")   # streamed file
            if tmp_path.exists(): tmp_path.unlink()            # fresh run

            torch.manual_seed(0)                               # reproducible tmp names

            offset = 0
            for t0 in tqdm(range(0, T, chunk_T),
               desc="building data",
               unit="frame",
               total=math.ceil(T / chunk_T)):
                
                t_slice = self.t_grid[t0:t0+chunk_T].to(dev)          # (chunk_T,)
                x_slice = self.traj[t0:t0+chunk_T].to(dev)            # (chunk_T,B,C,H,W)
                x1      = self.traj[-1].to(dev).view(B, -1)           # (B,row_feats)

                # build four CPU tensors for this chunk
                chunk_rows = len(t_slice) * B
                x0_chunk   = torch.empty((chunk_rows, cols_x0y), dtype=dtype)
                dx_chunk   = torch.empty((chunk_rows, cols_x0y), dtype=dtype)
                y_chunk    = torch.empty((chunk_rows, cols_x0y), dtype=dtype)
                dt_chunk   = torch.empty((chunk_rows, 1),        dtype=dtype)


                # inner = 0
                # for t_val, x in zip(t_slice, x_slice):
                for inner, i in enumerate(
                        tqdm(range(len(t_slice)),
                            desc="   ↳ chunk",
                            leave=False,
                            unit="step")):
                    t_val = t_slice[i]
                    x     = x_slice[i]

                    t = torch.full((B,1), float(t_val), device=dev, dtype=dtype)

                    
                    with torch.no_grad():
                        dx = self.dynamics(t, x).view(B, -1).to(dtype)

                    j = inner*B ; k = j+B
                    x0_chunk[j:k] = torch.cat((t.cpu(),      x.view(B,-1).cpu()), 1)
                    dx_chunk[j:k] = dx.cpu()
                    y_chunk [j:k] = torch.cat((torch.ones(B,1), x1.cpu()), 1)
                    dt_chunk[j:k] = (1.0 - t).cpu()
                    inner += 1

                # stream-append this chunk to disk
                torch.save((x0_chunk, dx_chunk, y_chunk, dt_chunk),
                        tmp_path, _use_new_zipfile_serialization=False)

                offset += chunk_rows
                del x_slice, t_slice, x0_chunk, dx_chunk, y_chunk, dt_chunk
                torch.cuda.empty_cache()

            parts = []
            with open(tmp_path, "rb") as f:                  # sequential load
                while True:
                    try:
                        parts.append(torch.load(f, map_location="cpu"))
                    except EOFError:
                        break                                # end of stream

            matrix_x0 = torch.vstack([p[0] for p in parts])
            matrix_dx = torch.vstack([p[1] for p in parts])
            matrix_y  = torch.vstack([p[2] for p in parts])
            matrix_dt = torch.vstack([p[3] for p in parts])
            tmp_path.unlink()

            self.full_ds = TensorDataset(matrix_x0, matrix_dx, matrix_y, matrix_dt)
            self.cache_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(self.full_ds, self.cache_path)
            ###────────────────── end replacement ─────────────────────────


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

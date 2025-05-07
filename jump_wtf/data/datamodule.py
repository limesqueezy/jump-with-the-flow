from __future__ import annotations
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import lightning as L
from .samplers import TimeGroupedSampler, ListSampler
from pathlib import Path
from typing import Union

import numpy as np, json, bisect, math
from numpy.lib.format import open_memmap

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
        eval_chunk: int = 2048,
        device = "cuda"
    ):
        super().__init__()
        self.traj       = traj          # (T, B, C, H, W)
        self.dynamics   = dynamics.to(device)
        self.batch_size = batch_size
        self.t_grid     = torch.linspace(0, 1, t_grid)
        self.val_frac   = val_frac
        self.chunk_steps= chunk_steps
        self.eval_chunk = eval_chunk
        self.device     = device

        self.cache_path = Path(cache_dir) / f"{dynamics_path}.pth"

    def setup(self, stage=None):
        print(f"Data module from {self.cache_path}.mmap")
        mmap_root = self.cache_path.with_suffix(".mmap")
        index_file = mmap_root / "index.json"

        if not index_file.exists():
            mmap_root.mkdir(parents=True, exist_ok=True)
            dev = torch.device(self.device)
            self.dynamics.to(dev).eval()

            T, B, C, H, W = self.traj.shape
            row_feats, cols_x0y = C*H*W, 1 + C*H*W
            np_dtype    = np.float32
            torch_dtype = torch.float32
            chunk_T = self.chunk_steps
            torch.manual_seed(0)

            index = []
            for t0 in tqdm(range(0, T, chunk_T), desc="writing memmap chunks",
                        total=math.ceil(T/chunk_T)):
                t_slice = self.t_grid[t0:t0+chunk_T].to(dev)
                x_slice = self.traj[t0:t0+chunk_T].to(dev)
                x1      = self.traj[-1].cpu().view(B, -1)

                rows_here = len(t_slice) * B
                mm = {k: open_memmap(mmap_root/f"{k}_{t0:07d}.npy",
                                    mode='w+', dtype=np_dtype,
                                    shape=(rows_here, cols_x0y if k!='dt' else 1))
                    for k in ("x0","dx","y","dt")}

                cursor = 0
                for t_val, x_T in zip(t_slice, x_slice):
                    for b0 in range(0, B, self.eval_chunk):
                        b1  = min(b0+self.eval_chunk, B)
                        x   = x_T[b0:b1].to(dev)
                        t   = torch.full((b1-b0,1), float(t_val), device=dev, dtype=torch_dtype)
                        with torch.no_grad():
                            dx = self.dynamics(t, x).view(b1-b0, -1).to(torch_dtype)
                        sl = slice(cursor, cursor+(b1-b0))
                        mm["x0"][sl] = torch.cat((t.cpu(), x.view(b1-b0,-1).cpu()),1).numpy()
                        mm["dx"][sl] = dx.cpu().numpy()
                        mm["y"][sl]  = torch.cat((torch.ones(b1-b0,1), x1[b0:b1]),1).numpy()
                        mm["dt"][sl] = (1.0 - t).cpu().numpy()
                        cursor += (b1-b0)
                for m in mm.values(): m.flush()
                index.append({"t0": int(t0), "rows": rows_here})
                torch.cuda.empty_cache()

            json.dump(index, open(index_file, "w"))

        self.full_ds = ChunkedMemmap(mmap_root)

        N       = len(self.full_ds)
        n_val   = int(self.val_frac * N)
        train_set, val_set = set(range(N-n_val)), set(range(N-n_val, N))

        full_order = TimeGroupedSampler(time_steps=len(self.t_grid),
                                        group_size=self.traj.size(1)).indices
        self.train_ordered_indices = [i for i in full_order if i in train_set]
        self.val_ordered_indices   = [i for i in full_order if i in val_set]

        self.train_sampler = ListSampler(self.train_ordered_indices)
        self.val_sampler   = ListSampler(self.val_ordered_indices)

    def train_dataloader(self):
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.full_ds,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=16,
            pin_memory=True,
            persistent_workers=False,
            drop_last=False,
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

class ChunkedMemmap(torch.utils.data.Dataset):
    """one mem‑mapped chunk at a time and reuses it until we step into the next chunk, keeps RAM < batch*workers."""
    def __init__(self, root: Path):
        self.root  = root
        self.meta  = json.load(open(root/"index.json"))
        self.starts = np.cumsum([0] + [m["rows"] for m in self.meta[:-1]])
        self._cached = (None, None)               # (chunk_id, maps)

    def __len__(self):
        return sum(m["rows"] for m in self.meta)

    def _load_chunk(self, cid: int):
        if self._cached[0] == cid:
            return self._cached[1]
        m = self.meta[cid]
        mm = {k: np.load(self.root/f"{k}_{m['t0']:07d}.npy", mmap_mode="r")
            for k in ("x0","dx","y","dt")}
        self._cached = (cid, mm)
        return mm

    def __getitem__(self, idx: int):
        cid = bisect.bisect_right(self.starts, idx) - 1
        mm  = self._load_chunk(cid)
        offset = idx - self.starts[cid]
        return tuple(torch.as_tensor(mm[k][offset].copy())  # makes writable copy to suppress warnings
             for k in ("x0", "dx", "y", "dt"))
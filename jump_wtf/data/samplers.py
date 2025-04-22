from torch.utils.data import Sampler
import numpy as np

class ListSampler(Sampler):
    """
    Wrap a fixed list of indices as a sampler (preserves order).
    """
    def __init__(self, indices: list[int]):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

"""Reverse‑time, per‑t grouping."""
class TimeGroupedSampler(Sampler):
    def __init__(self, time_steps: int, group_size: int):
        self.time_steps = time_steps  # e.g., 100 time slices
        self.group_size = group_size  # e.g., 2000 samples per t
        self.indices = self._build_indices()

    def _build_indices(self):
        all_indices = []
        for i in reversed(range(self.time_steps)):  # reverse time
            start = i * self.group_size
            end = (i + 1) * self.group_size
            indices = list(range(start, end))
            np.random.shuffle(indices)
            all_indices.extend(indices)
        return all_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
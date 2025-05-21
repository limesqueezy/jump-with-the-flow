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

class RandomTimeSampler(Sampler):
    def __init__(self, time_steps=100, group_size=2000):
        self.time_steps = time_steps
        self.group_size = group_size
        self.total_size = time_steps * group_size

    def __iter__(self):
        # Create a list of all possible indices
        all_indices = list(range(self.total_size))
        # Randomly shuffle the indices to sample random timesteps
        indices = []
        time_indices = list(range(self.time_steps))
        np.random.shuffle(time_indices)  # Shuffle the time indices
        for t in time_indices:
            # Get the indices for this timestep
            start_idx = t * self.group_size
            end_idx = start_idx + self.group_size
            # Get all indices for this timestep
            time_slice_indices = list(range(start_idx, end_idx))
            np.random.shuffle(time_slice_indices)  # Shuffle within the time slice
            # Add to our list of indices
            indices.extend(time_slice_indices)
        return iter(indices)
        
    def __len__(self):
        return self.total_size
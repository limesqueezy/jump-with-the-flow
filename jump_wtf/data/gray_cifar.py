# mylib/datasets/cifar_subset_gray.py
from typing import Sequence, List, Union

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

class CIFARSubsetGray(CIFAR10):
    """
    CIFAR-10 where:
      • only the requested classes are kept
      • every image is converted to grayscale (1 channel)
    """

    def __init__(
        self,
        root: str,
        classes_to_keep: Sequence[Union[int, str]],
        train: bool = True,
        download: bool = False,
        transform=None,
    ):
        super().__init__(root=root, train=train, download=download)

        if isinstance(classes_to_keep[0], str):
            name2idx = {name: idx for idx, name in enumerate(self.classes)}
            classes_idx: List[int] = [name2idx[name] for name in classes_to_keep]
        else:
            classes_idx = list(classes_to_keep)

        keep_mask = [t in classes_idx for t in self.targets]
        self.data = self.data[keep_mask]
        self.targets = [t for t in self.targets if t in classes_idx]
        gray_first = T.Grayscale(num_output_channels=1)
        if transform is None:
            self.transform = T.Compose([gray_first, T.ToTensor()])
        else:
            # be careful not to double-wrap an existing Compose
            self.transform = T.Compose([gray_first] + list(transform.transforms))

        # adjust default `self.mean` / `self.std`

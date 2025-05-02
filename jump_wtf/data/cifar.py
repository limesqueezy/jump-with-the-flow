from typing import Optional, Sequence, Union
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose, RandomHorizontalFlip, ToTensor,
    Grayscale, Normalize
)

class CIFARWrapper(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = False,
        *,
        grayscale: bool = False,
        classes_to_keep: Optional[Sequence[Union[int, str]]] = None,
        flip_prob: float = 0.5,
    ):
        super().__init__(root=root, train=train, download=download, transform=None)

        if classes_to_keep:
            if isinstance(classes_to_keep[0], str):
                name2idx = {n: i for i, n in enumerate(self.classes)}
                classes_idx = [name2idx[n] for n in classes_to_keep]
            else:
                classes_idx = list(classes_to_keep)
            mask = [t in classes_idx for t in self.targets]
            self.data    = self.data[mask]
            self.targets = [t for t in self.targets if t in classes_idx]

        tfms = [
            RandomHorizontalFlip(p=flip_prob),
            ToTensor(),
        ]

        if grayscale:
            # first convert, then normalize with 1‐channel stats
            tfms.insert(0, Grayscale(num_output_channels=1))
            tfms.append(Normalize(mean=[0.5], std=[0.5]))
        else:
            # 3‐channel normalize
            tfms.append(Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std= [0.2023, 0.1994, 0.2010]
            ))

        self.transform = Compose(tfms)

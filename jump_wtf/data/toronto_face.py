from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.datasets.vision import VisionDataset
import scipy.io as sio
import numpy as np
from PIL import Image

class TorontoFaceDataset(VisionDataset):
    url      = "http://www.cs.toronto.edu/~jsusskin/TFD/TFD_48x48.mat"
    filename = "TFD_48x48.mat"
    md5      = "9d113650c197719b750ecc6864faf87c"

    def __init__(self, root, train=None,
                transform=None, target_transform=None,
                download=False):
        super().__init__(root, transform=transform,
                        target_transform=target_transform)
        if download:
            download_url(self.url, self.root, self.filename,
                        md5=self.md5)

        mat     = sio.loadmat(str(Path(self.root) / self.filename))
        images  = mat["images"]          # (N,48,48)
        raw_lbl = mat["labs_ex"].flatten() - 1
        folds   = mat["folds"][:, 0].flatten()  # use only fold #0

        # 1) If train is None → ALL samples
        # 2) If train=True → folds==1; if train=False → folds==2
        if train is None:
            mask = np.ones(len(images), dtype=bool)
        else:
            mask = folds == (1 if train else 2)

        self.data = images[mask]
        self.targets = [
            int(l) if l >= 0 else None
            for l in raw_lbl[mask]
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_arr = self.data[idx]                                 # NumPy array (48×48)
        img = Image.fromarray(img_arr.astype(np.uint8), mode="L")
        # Fetch target, but allow None without casting
        target = self.targets[idx]                               # may be None

        if self.transform:
            img = self.transform(img)
        if self.target_transform and (target is not None):
            target = self.target_transform(target)

        # Return label exactly as stored (None or int) <-- this will cause errors to our loader
        return img, target

    def extra_repr(self):
        return f"Split: {'train' if len(self.data)>0 else 'test'}"

import os
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from PIL import Image
import torch
from jump_wtf.data.toronto_face import TorontoFaceDataset

# 1) Instantiate dataset & dataloader on CPU
ds = TorontoFaceDataset(
    root="./assets/raw_datasets/",
    train=None,           # load all samples
    download=True,
    transform=ToTensor()  # tensor in [0,1]
)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=False)

# 2) Get one batch of 64 images
imgs = next(iter(loader))  # imgs shape: (64, 1, 48, 48)

# 3) Make an 8x8 grid
grid = make_grid(imgs, nrow=8, padding=2, normalize=True)

# 4) Convert grid to PIL image
# grid: tensor [C, H, W] in [0,1] -> convert to [0,255]
ndarr = (grid * 255).byte().permute(1, 2, 0).numpy()
# For grayscale, remove channel dim if exists
if ndarr.shape[2] == 1:
    ndarr = ndarr[..., 0]
mosaic = Image.fromarray(ndarr)

# 5) Save mosaic
output_path = Path("mosaics/mosaic.png")
mosaic.save(output_path)

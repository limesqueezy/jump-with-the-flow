from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image

from jump_wtf.data.cifar import CIFARWrapper

def make_and_save_cifar_mosaics(
    root: str,
    out_dir: str = "./mosaics",
    batch_size: int = 64,
):
    # 1) Define the four scenarios: (filename_suffix, grayscale, classes_to_keep)
    configs = [
        ("single_gray",  True,  ["cat"]),        # only class 0, in gray
        ("three_color", False, [0, 1, 2]),   # classes 0,1,2, in color
        ("full_gray",    True,  []),         # all classes, in gray
        ("full_color",   False, []),         # all classes, in color
    ]

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    for name, grayscale, classes in configs:
        # 2) Instantiate wrapper (no external transform needed)
        ds = CIFARWrapper(
            root=root,
            train=True,
            download=True,
            grayscale=grayscale,
            classes_to_keep=classes or None,
            flip_prob=0.0,      # disable randomness for a stable mosaic
        )

        # 3) Sample a batch and make a grid
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        imgs, _ = next(iter(loader))             # drop labels
        grid = make_grid(imgs, nrow=8, normalize=True)

        # 4) Convert to H×W×C uint8
        arr = (grid * 255).byte().permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:  # PIL wants H×W for single‐channel
            arr = arr[:, :, 0]

        # 5) Save
        out_file = out_path / f"mosaic_{name}.png"
        Image.fromarray(arr).save(out_file)
        print(f"→ {out_file}")

if __name__ == "__main__":
    make_and_save_cifar_mosaics(root="./assets/raw_datasets/")

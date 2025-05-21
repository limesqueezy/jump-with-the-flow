# ---------------------------------------------------------------
#  Build the three mosaics directly from the on-disk image folders
# ---------------------------------------------------------------
import os, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image      # pip install pillow

# ----------------------------------------------------------------
#  1)  Where the images live and how many class-rows each contains
# ----------------------------------------------------------------
SAMPLES_DIR   = "samples"
DATASET_DIRS  = {
    "MNIST"        : "mnist_koopy",   # 10 rows (0-9)
    "Fashion-MNIST": "fashion_mnist", # 10 rows (0-9)
    "TFD"          : "tfd",           #  7 rows (0-6)
}
N_ROWS = {"MNIST": 10, "Fashion-MNIST": 10, "TFD": 7}

# Each tuple is (sub-folder name, column header)
METHODS = [("real",  "Real"),
           ("cfm",   "CFM"),
           ("koopy", "Ours")]

# ----------------------------------------------------------------
#  2)  Helper: load a whole dataset into the structure your plotter
#      already expects →  list[row]     of dict[col_name] = H×W×1
# ----------------------------------------------------------------
def load_dataset(root: str, n_rows: int):
    rows = []
    for cls in range(n_rows):
        row = {}
        for folder_name, col_name in (METHODS if root != "mnist_koopy" else METHODS[1:]):
            pattern = os.path.join(SAMPLES_DIR, root, folder_name, f"{cls}.*")
            match   = glob.glob(pattern)
            if not match:
                raise FileNotFoundError(f"No file matched “{pattern}”")
            # convert to grayscale array shaped (H, W, 1)
            img = np.array(Image.open(match[0]).convert("L"))[..., None]
            row[col_name] = img
        rows.append(row)
    return rows

# ----------------------------------------------------------------
#  3)  Assemble all three datasets
# ----------------------------------------------------------------
datasets = {
    ds_name: load_dataset(ds_folder, N_ROWS[ds_name])
    for ds_name, ds_folder in DATASET_DIRS.items()
}

# ----------------------------------------------------------------
#  4)  Make the mosaics (same plotting logic you already had)
# ----------------------------------------------------------------
fixed_fig_width  = 6.0   # keeps column width identical
pad_between_imgs = 0.20  # tight-layout padding
dpi_out          = 300

for panel_idx, (ds_name, ds_imgs) in enumerate(datasets.items()):

    n_rows = len(ds_imgs)
    col_subset = METHODS[1:] if ds_name == "MNIST" else METHODS
    n_cols     = len(col_subset)

    pad = 0.50 if ds_name == "MNIST" else 0.30   # thicker gaps for MNIST



    fig_height = fixed_fig_width / n_cols * n_rows
    fig, axs   = plt.subplots(n_rows, n_cols,
                              figsize=(fixed_fig_width, fig_height),
                              squeeze=False)

    header_font = 35
    for c, (_, col_name) in enumerate(col_subset):
        axs[0, c].set_title(col_name, fontsize=header_font, pad=12)

    for r in range(n_rows):
        for c, (_, col_name) in enumerate(col_subset):
            ax = axs[r, c]
            ax.imshow(ds_imgs[r][col_name][:, :, 0], cmap="gray")
            ax.axis("off")
        axs[r, 0].set_ylabel(str(r), rotation=0, labelpad=12,
                             va='center', fontsize=12)


    fig.tight_layout(pad=pad, rect=[0, 0, 1, 0.91])   # leaves ~7 % top margin for titles

    fig.savefig(f"mosaic_{ds_name.lower().replace('-', '_')}.png",
                dpi=dpi_out, bbox_inches="tight")
    plt.close(fig)

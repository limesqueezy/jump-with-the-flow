# ─────────────────────────────────────────────────────────────────────────────
#  Overlay plot: Koopman (“Ours”) vs. Rectified Flow baseline
# ─────────────────────────────────────────────────────────────────────────────
import csv
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np


def plot_overlay_scaling(
    ours_csv       = "sampler_scaling.csv",
    rf_csv         = "rf_scaling.csv",
    save_path      = "scaling_comparison.png",
    dpi            = 400,
    ours_color     = "#3C4735",
    rf_color       = "#48232D",
):
    """
    Draws both throughput–vs–batch-size curves on a #FFFFFF background and
    writes a high-resolution PNG/PDF.

    • Solid line (ours_color)  = our Koopman sampler
    • Dotted line (rf_color)   = Rectified Flow baseline
    """
    # ---------------- 1) helper to read CSV ---------------------------
    def read_scaling_csv(path):
        b, ips = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                b.append(int(row["batch_size"]))
                ips.append(float(row["imgs_per_sec"]))
        if not b:
            raise RuntimeError(f"No rows in {path}")
        b, ips = np.asarray(b), np.asarray(ips)
        order  = np.argsort(b)
        return b[order], ips[order]

    batch_ours, ips_ours = read_scaling_csv(ours_csv)
    batch_rf,   ips_rf   = read_scaling_csv(rf_csv)

    # ---------------- 2) figure --------------------------------------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(3.5, 2.625), dpi=dpi, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # ours – solid
    ax.plot(
        batch_ours,
        ips_ours,
        color=ours_color,
        marker="o",
        linewidth=1.4,
        label="Ours",
    )

    # baseline – dotted
    ax.plot(
        batch_rf,
        ips_rf,
        color=rf_color,
        linestyle=":",
        linewidth=1.6,
        marker="o",
        label="Rectified Flow",
    )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size (log₂ scale)")
    ax.set_ylabel("Images / second")
    ax.set_title("Sampling Efficiency", pad=4)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)

    # colour legend text to match the lines
    legend = ax.legend(frameon=False)
    legend.get_texts()[0].set_color(ours_color)
    legend.get_texts()[1].set_color(rf_color)

    fig.tight_layout()

    # ---------------- 3) save ----------------------------------------
    ext = Path(save_path).suffix.lower()
    if ext == ".pdf":
        fig.savefig(save_path, bbox_inches="tight")
    else:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)
    return fig

plot_overlay_scaling(
    ours_csv="sampler_scaling.csv",
    rf_csv="rf_scaling.csv",
    save_path="scaling_comparison.png",   # or ".pdf"
    dpi=400,
)

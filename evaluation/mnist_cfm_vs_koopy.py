# evaluating_mnist.py

import argparse
import torch
from pathlib import Path

# import koopman pipeline
from .koopy_mnist import load_net as load_koopman_net, sample as sample_koopman
# import continuous‐flow (CFM) pipeline
from .cfm_mnist import load_net as load_cfm_net, sample as sample_cfm

def main():
    ap = argparse.ArgumentParser(
        description="Generate MNIST samples via Koopman and/or CFM pipelines"
    )
    ap.add_argument(
        "--mode",
        choices=["koopman", "cfm", "both"],
        default="both",
        help="Which model(s) to sample",
    )
    ap.add_argument(
        "--koop-checkpoint",
        required=False,
        help="Glob pattern for Koopman .ckpt files (e.g. 'runs/koop/*.ckpt')",
    )
    ap.add_argument(
        "--cfm-checkpoint",
        required=False,
        help="Path to CFM checkpoint (e.g. 'models/cfm_mnist.pt')",
    )
    ap.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Total number of samples to generate",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for sampling loops",
    )
    ap.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="Number of integration steps (forward passes) per sample",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs",
        help="Root folder under which subfolders 'koopman' and/or 'cfm' will be created",
    )
    args = ap.parse_args()

    device = args.device
    work_dir = Path(args.output_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    x0 = torch.randn(args.num_samples, 1, 28, 28, device=device)

    if args.mode in ("koopman", "both"):
        if not args.koop_checkpoint:
            raise ValueError("You must pass --koop-checkpoint when mode includes 'koopman'")
        print(f"→ Loading Koopman model from '{args.koop_checkpoint}'")
        koop_net = load_koopman_net(args.koop_checkpoint, device=device)

        koop_out = work_dir / "koopman"
        print(f"→ Sampling {args.num_samples} images with Koopman; writing to '{koop_out}'")
        sample_koopman(
            net     = koop_net,
            n       = args.num_samples,
            bs      = args.batch_size,
            steps   = args.ode_steps,
            # steps   = 1,
            device  = device,
            out     = koop_out,
            x0      = x0
        )

    if args.mode in ("cfm", "both"):
        if not args.cfm_checkpoint:
            raise ValueError("You must pass --cfm-checkpoint when mode includes 'cfm'")
        print(f"→ Loading CFM model from '{args.cfm_checkpoint}'")
        cfm_net = load_cfm_net(args.cfm_checkpoint, device=device)

        cfm_out = work_dir / "cfm"
        print(f"→ Sampling {args.num_samples} images with CFM; writing to '{cfm_out}'")
        sample_cfm(
            net     = cfm_net,
            n       = args.num_samples,
            bs      = args.batch_size,
            steps   = args.ode_steps,
            device  = device,
            out     = cfm_out,
            x0      = x0
        )

    print("→ Done.")

if __name__ == "__main__":
    main()

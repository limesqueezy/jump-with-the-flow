import argparse, csv, time, os
from pathlib import Path
import torch
from torchdyn.core import NeuralODE
from torchcfm.models.unet import UNetModel

def load_net(ckpt: str, device: torch.device) -> torch.nn.Module:
    """Load the UNet model from a checkpoint (EMA or raw)."""
    net = UNetModel(
        dim              =(1, 28, 28),
        num_channels     =32,
        num_res_blocks   =1,
        channel_mult     =[1, 2, 2],
        attention_resolutions="16",
        dropout          =0.0,
    ).to(device)

    state = torch.load(ckpt, map_location=device)
    for k in ("ema_model", "model"):
        if k in state:
            state = state[k]
            break
    net.load_state_dict({kk.replace("module.", ""): vv for kk, vv in state.items()},
                        strict=False)
    net.eval()
    return net

@torch.no_grad()
def bench_once(net: torch.nn.Module, bs: int, steps: int,
               device: torch.device) -> float:
    """
    Time a single NeuralODE trajectory pass (no disk I/O).

    Returns
    -------
    elapsed_sec : float
        Wall-clock time in seconds for the forward pass.
    """
    # Create a reusable ODE wrapper only once per batch size
    node = NeuralODE(net,
                     solver="euler",
                     sensitivity="adjoint",
                     atol=1e-4,
                     rtol=1e-4)

    t_span = torch.linspace(0., 1., steps, device=device)
    x0 = torch.randn(bs, 1, 28, 28, device=device)

    # Accurate timing on CUDA: sync before & after
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = node.trajectory(x0, t_span)[-1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - start

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to the trained CFM checkpoint (.pt)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ode-steps", type=int, default=1,
                        help="Number of solver steps (n_steps column)")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
                        help="List of batch sizes to benchmark")
    parser.add_argument("--out-csv", default="cfm_speed.csv",
                        help="CSV file to create or append results to")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"→ loading checkpoint from {args.checkpoint}")
    net = load_net(args.checkpoint, device)

    csv_path = Path(args.out_csv)
    # If it exists, delete so we start fresh
    if csv_path.exists():
        csv_path.unlink()
    print(f"→ creating new {csv_path}")

    # Open in write mode, write header, then all rows
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "elapsed_sec", "imgs_per_sec", "n_steps"])

        for bs in args.batch_sizes:
            elapsed = bench_once(net, bs, args.ode_steps, device)
            ips = bs / elapsed
            writer.writerow([bs, elapsed, ips, args.ode_steps])
            print(f"[bs={bs:4d}] {elapsed:.6f} sec → {ips:,.1f} imgs/s")

    print("✓ Done – results written to", csv_path.resolve())

if __name__ == "__main__":
    main()
"""
train.py — Supervised training loop for TSPGNN

Label strategies
----------------
  n ≤ 10  → brute-force optimal tour  (--label optimal, default when n ≤ 10)
  any n   → nearest-neighbour tour    (--label nn)

The NN label strategy is key for generalisation: it lets you train on larger
instances (e.g. --n 50 --label nn) so the model sees the graph structure it
will face at inference time. The model learns to imitate NN, then generalises.

Mixed-size training (--n_min / --n_max) samples a random size each step,
which further improves generalisation across instance sizes.

Device
------
Automatically uses CUDA > MPS (Apple Silicon) > CPU.
Override with --device cpu|cuda|mps.

Usage:
    # Small instances, optimal labels (original behaviour)
    python train.py --size small --n 8 --steps 1000

    # Larger instances with NN pseudo-labels
    python train.py --size medium --n 50 --label nn --steps 3000 --source tsp

    # Mixed sizes (best for generalisation)
    python train.py --size large --n_min 10 --n_max 100 --label nn --steps 5000 --source tsp

    # Fine-tune an existing model
    python train.py --resume model/gnn.pt --size medium --n_min 10 --n_max 100 --label nn --lr 1e-4

    # Force CPU
    python train.py --size small --n 8 --steps 1000 --device cpu
"""

import argparse
import os
import random as _random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_cities, random_instance, optimal_tour_labels, nn_tour_labels
from model import TSPGNN, MODEL_SIZES

POOL_SIZE = 1000


def get_device(requested: str = "auto") -> torch.device:
    """
    Resolve the best available device.

    Priority: CUDA (NVIDIA) → XPU (Intel Arc via IPEX) → MPS (Apple Silicon) → CPU
    Pass requested="cpu" / "cuda" / "xpu" / "mps" to override.

    Intel Arc setup:  pip install intel-extension-for-pytorch
    NVIDIA setup:     pip install torch --index-url https://download.pytorch.org/whl/cu124
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
        if torch.xpu.is_available():
            return torch.device("xpu")
    except ImportError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model: TSPGNN, n_nodes: int = 8, n_steps: int = 500, lr: float = 1e-3,
          city_pool: torch.Tensor = None, label: str = "auto",
          n_min: int = None, n_max: int = None,
          device: torch.device = torch.device("cpu"),
          amp: bool = False, accum_steps: int = 1):
    """
    Supervised training loop.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max are both set)
    label        : "auto"    — brute-force if n<=10, else NN pseudo-labels
                   "optimal" — brute-force only (asserts n<=10)
                   "nn"      — nearest-neighbour pseudo-labels (any n)
    n_min/max    : when both set, sample a random size in [n_min, n_max] each step
    city_pool    : (N, 2) tensor already on device — cities sampled from it each step
    device       : torch.device to run forward/backward on
    amp          : enable mixed precision (float16 on CUDA, bfloat16 elsewhere)
                   reduces VRAM ~50% and speeds up training on supported GPUs
    accum_steps  : accumulate gradients over N steps before a weight update
                   simulates a larger batch without extra memory cost
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Mixed precision setup ─────────────────────────────────────────────────
    # float16 on CUDA needs a GradScaler to recover from underflow.
    # bfloat16 (XPU/MPS/CPU) is numerically stable enough without one.
    device_type = device.type if hasattr(device, "type") else "cpu"
    amp_dtype   = torch.float16 if device_type == "cuda" else torch.bfloat16
    scaler      = torch.cuda.amp.GradScaler() if (amp and device_type == "cuda") else None
    amp_ctx     = (torch.autocast(device_type=device_type, dtype=amp_dtype)
                   if amp else nullcontext())

    losses = []
    optimizer.zero_grad()

    bar = tqdm(range(n_steps), desc=f"Training [{device}]", unit="step",
               dynamic_ncols=True)
    for step, _ in enumerate(bar):
        # ── Sample instance size ──────────────────────────────────────────────
        n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes

        # ── Sample coordinates → move to device ───────────────────────────────
        if city_pool is not None:
            idx    = torch.randperm(city_pool.shape[0])[:n]
            coords = city_pool[idx].to(device)
        else:
            coords = random_instance(n).to(device)

        # ── Compute labels on CPU, then move to device ────────────────────────
        use_label = label
        if use_label == "auto":
            use_label = "optimal" if n <= 10 else "nn"

        if use_label == "optimal":
            assert n <= 10, f"Brute-force labels require n <= 10, got n={n}. Use --label nn."
            y = optimal_tour_labels(coords.cpu()).to(device)
        else:
            y = nn_tour_labels(coords.cpu()).to(device)

        # ── Forward + loss (with optional AMP) ───────────────────────────────
        with amp_ctx:
            p_hat = model(coords)
            loss  = F.binary_cross_entropy(p_hat, y) / accum_steps

        # ── Backward ──────────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Weight update every accum_steps (gradient accumulation) ──────────
        is_update = (step + 1) % accum_steps == 0 or (step + 1) == n_steps
        if is_update:
            if scaler is not None:
                scaler.unscale_(optimizer)
            # Gradient clipping — caps gradient magnitude to prevent GPU spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        raw_loss = loss.item() * accum_steps   # undo /accum_steps for display
        losses.append(raw_loss)
        avg = sum(losses[-50:]) / len(losses[-50:])
        bar.set_postfix(n=n, loss=f"{raw_loss:.4f}", avg50=f"{avg:.4f}",
                        best=f"{min(losses):.4f}")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=8,
                        help="Cities per training instance (use with fixed size training)")
    parser.add_argument("--n_min",  type=int,   default=None,
                        help="Min cities for mixed-size training (use with --n_max)")
    parser.add_argument("--n_max",  type=int,   default=None,
                        help="Max cities for mixed-size training (use with --n_min)")
    parser.add_argument("--label",  type=str,   default="auto",
                        help="Label strategy: 'auto', 'optimal' (n≤10 only), 'nn' (any n)")
    parser.add_argument("--steps",  type=int,   default=500)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--size",   type=str,   default=None,
                        help="Model preset: 'small', 'medium', 'large'. Overrides --d/--L.")
    parser.add_argument("--d",      type=int,   default=64)
    parser.add_argument("--L",      type=int,   default=4)
    parser.add_argument("--out",    type=str,   default="model/gnn.pt")
    parser.add_argument("--source", type=str,   default="random",
                        help="'random', 'tsp', or 'solomon'")
    parser.add_argument("--resume", type=str,   default=None,
                        help="Path to existing weights to resume/fine-tune from")
    parser.add_argument("--device", type=str,   default="auto",
                        help="Device: 'auto' (cuda>xpu>mps>cpu), 'cuda', 'xpu', 'mps', 'cpu'")
    parser.add_argument("--amp",    action="store_true",
                        help="Mixed precision: float16 on CUDA, bfloat16 elsewhere. "
                             "Cuts VRAM ~50%%, speeds up training on supported GPUs.")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (default 1 = off). "
                             "Use 4-8 to simulate a larger batch without extra memory.")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile() for ~20-30%% speedup (PyTorch 2.0+, "
                             "first step will be slow due to JIT compilation).")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, 'type') and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.label == "optimal" and args.n > 10:
        raise ValueError("--label optimal requires n ≤ 10. Use --label nn for larger instances.")
    if args.label in ("optimal", "auto") and args.n_max and args.n_max > 10:
        print("Note: --n_max > 10 with label=auto will use NN labels for instances > 10.")

    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}.")
        args.d, args.L = MODEL_SIZES[args.size]

    # ── City pool ─────────────────────────────────────────────────────────────
    city_pool = None
    if args.source != "random":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' …")
        city_pool = load_cities(POOL_SIZE, source=args.source).to(device)
        print(f"Pool ready: {city_pool.shape} on {device}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TSPGNN(d=args.d, L=args.L)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: file not found: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")

    # ── torch.compile (optional) ──────────────────────────────────────────────
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled (first step will be slow — JIT compiling)")
        except Exception as e:
            print(f"torch.compile: not available ({e}), skipping")

    base    = getattr(model, "_orig_mod", model)
    n_params = sum(p.numel() for p in base.parameters())
    size_info = f"n={args.n}" if not (args.n_min and args.n_max) \
                else f"n in [{args.n_min},{args.n_max}]"
    print(f"TSPGNN(d={args.d}, L={args.L})  --  {n_params:,} parameters")
    print(f"Training: {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}")
    print(f"AMP: {args.amp}  |  accum_steps: {args.accum_steps}  |  "
          f"compile: {args.compile}\n")

    losses = train(
        model, n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        amp=args.amp, accum_steps=args.accum_steps,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))

    print(f"\nWeights saved → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

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
import math
import os
import random as _random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import (load_cities, random_instance, optimal_tour_labels,
                  nn_tour_labels, save_city_pool, load_city_pool_mmap,
                  save_label_cache, load_label_cache)
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
          amp: bool = False, accum_steps: int = 1,
          scheduler: str = "cosine", warmup_steps: int = 100,
          patience: int = 0, label_cache: tuple = None):
    """
    Supervised training loop.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max are both set)
    label        : "auto" / "optimal" (n<=10) / "nn" (any n)
    n_min/max    : sample random size in [n_min, n_max] each step
    city_pool    : (N, 2) tensor on device — cities sampled each step
    device       : torch.device
    amp          : mixed precision (float16 CUDA, bfloat16 elsewhere)
    accum_steps  : gradient accumulation steps
    scheduler    : "cosine" — cosine annealing with linear warmup | "none"
    warmup_steps : linear warmup steps before cosine decay kicks in
    patience     : early stopping — stop after N steps with no improvement (0=off)
    label_cache  : (coords_np, labels_np) from load_label_cache() — skips
                   per-step label computation for fixed-n training
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── LR scheduler: linear warmup + cosine annealing ───────────────────────
    if scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        sched = None

    # ── Mixed precision setup ─────────────────────────────────────────────────
    device_type = device.type if hasattr(device, "type") else "cpu"
    amp_dtype   = torch.float16 if device_type == "cuda" else torch.bfloat16
    scaler      = torch.cuda.amp.GradScaler() if (amp and device_type == "cuda") else None
    amp_ctx     = (torch.autocast(device_type=device_type, dtype=amp_dtype)
                   if amp else nullcontext())

    losses      = []
    best_loss   = float("inf")
    no_improve  = 0
    optimizer.zero_grad()

    bar = tqdm(range(n_steps), desc=f"Training [{device}]", unit="step",
               dynamic_ncols=True)
    for step, _ in enumerate(bar):
        # ── Sample instance ───────────────────────────────────────────────────
        if label_cache is not None:
            # Use pre-computed instance — no label computation needed this step
            coords_cache, labels_cache = label_cache
            i      = _random.randint(0, len(coords_cache) - 1)
            import numpy as _np
            coords = torch.from_numpy(_np.array(coords_cache[i])).to(device, non_blocking=True)
            y      = torch.from_numpy(_np.array(labels_cache[i])).to(device, non_blocking=True)
            n      = coords.shape[0]
        else:
            n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
            if city_pool is not None:
                idx    = torch.randperm(city_pool.shape[0])[:n]
                coords = city_pool[idx].to(device, non_blocking=True)
            else:
                coords = random_instance(n).to(device, non_blocking=True)

            use_label = "optimal" if (label == "auto" and n <= 10) else \
                        ("nn" if label == "auto" else label)
            if use_label == "optimal":
                assert n <= 10, f"Brute-force labels require n <= 10, got n={n}."
                y = optimal_tour_labels(coords.cpu()).to(device, non_blocking=True)
            else:
                y = nn_tour_labels(coords.cpu()).to(device, non_blocking=True)

        # ── Forward + loss ────────────────────────────────────────────────────
        with amp_ctx:
            p_hat = model(coords)
            loss  = F.binary_cross_entropy(p_hat, y) / accum_steps

        # ── Backward ──────────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ── Weight update ─────────────────────────────────────────────────────
        is_update = (step + 1) % accum_steps == 0 or (step + 1) == n_steps
        if is_update:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if sched is not None:
                sched.step()

        # ── Periodic GPU memory flush ─────────────────────────────────────────
        if (step + 1) % 100 == 0:
            if device_type == "cuda":
                torch.cuda.empty_cache()
            elif device_type == "xpu":
                torch.xpu.empty_cache()

        # ── Logging ───────────────────────────────────────────────────────────
        raw_loss = loss.item() * accum_steps
        losses.append(raw_loss)
        if raw_loss < best_loss:
            best_loss  = raw_loss
            no_improve = 0
        else:
            no_improve += 1

        avg     = sum(losses[-50:]) / len(losses[-50:])
        cur_lr  = optimizer.param_groups[0]["lr"]
        bar.set_postfix(n=n, loss=f"{raw_loss:.4f}", avg50=f"{avg:.4f}",
                        best=f"{best_loss:.4f}", lr=f"{cur_lr:.2e}")

        # ── Early stopping ────────────────────────────────────────────────────
        if patience > 0 and no_improve >= patience:
            bar.write(f"Early stop: no improvement for {patience} steps.")
            break

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
    parser.add_argument("--pool_cache", type=str, default=None,
                        help="Path to a .npy city pool cache (memory-mapped, saves RAM). "
                             "If the file does not exist it is created from --source. "
                             "Example: --pool_cache model/city_pool.npy")
    parser.add_argument("--compile", action="store_true",
                        help="Apply torch.compile() for ~20-30%% speedup (PyTorch 2.0+, "
                             "first step will be slow due to JIT compilation).")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"],
                        help="LR scheduler: 'cosine' (warmup + cosine decay) or 'none'.")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Linear warmup steps before cosine decay (default 100).")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping: halt after N steps with no improvement "
                             "(default 0 = disabled).")
    parser.add_argument("--label_cache", type=str, default=None,
                        help="Path to a .npz label cache (pre-computed coords+labels). "
                             "If file does not exist it is built from --n, --label, --source. "
                             "Only for fixed-n training (no --n_min/--n_max). "
                             "Example: --label_cache model/labels_n50.npz")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, 'type') and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True          # fastest kernel per input size
        torch.set_float32_matmul_precision("high")     # TF32 on Ampere+ (~10% free speedup)

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
        if args.pool_cache:
            if not os.path.exists(args.pool_cache):
                print(f"Building city pool cache from source='{args.source}' -> {args.pool_cache} ...")
                raw = load_cities(POOL_SIZE, source=args.source)
                save_city_pool(raw, args.pool_cache)
                print("Cache saved.")
            print(f"Loading city pool from cache (memory-mapped): {args.pool_cache}")
            city_pool = load_city_pool_mmap(args.pool_cache).to(device)
        else:
            print(f"Loading {POOL_SIZE} cities from source='{args.source}' ...")
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

    # ── Label cache ───────────────────────────────────────────────────────────
    lbl_cache = None
    if args.label_cache and not (args.n_min and args.n_max):
        if not os.path.exists(args.label_cache):
            print(f"Building label cache (n={args.n}, label={args.label}) "
                  f"-> {args.label_cache} ...")
            save_label_cache(
                n=args.n, pool_size=2000, label=args.label,
                path=args.label_cache, city_pool=city_pool,
            )
            print("Label cache saved.")
        print(f"Loading label cache: {args.label_cache}")
        lbl_cache = load_label_cache(args.label_cache)

    base     = getattr(model, "_orig_mod", model)
    n_params = sum(p.numel() for p in base.parameters())
    size_info = f"n={args.n}" if not (args.n_min and args.n_max) \
                else f"n in [{args.n_min},{args.n_max}]"
    print(f"TSPGNN(d={args.d}, L={args.L})  --  {n_params:,} parameters")
    print(f"Training: {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}")
    print(f"AMP: {args.amp}  |  accum_steps: {args.accum_steps}  |  compile: {args.compile}")
    print(f"Scheduler: {args.scheduler} (warmup={args.warmup})  |  "
          f"patience: {args.patience or 'off'}  |  "
          f"label_cache: {'yes' if lbl_cache else 'no'}\n")

    losses = train(
        model, n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        amp=args.amp, accum_steps=args.accum_steps,
        scheduler=args.scheduler, warmup_steps=args.warmup,
        patience=args.patience, label_cache=lbl_cache,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))

    print(f"\nWeights saved → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

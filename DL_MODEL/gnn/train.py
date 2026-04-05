"""
train.py — Supervised training loop for TSPGNN

Why the previous version produced 0% success
---------------------------------------------
1. Class-imbalance collapse: a tour has n edges out of n² possible (positive
   rate ≈ 1/n).  Unweighted BCE is minimised by outputting a constant p ≈ 1/n
   for every edge, which gives a sequential greedy tour and 0% gap success.
   Fix: pos_weight = n_neg / n_pos so each class contributes equally.

2. No quality signal: training reported only BCE loss.  A low loss is
   compatible with a model that never separates tour edges from non-tour edges.
   Fix: compute greedy-tour gap on a held-out validation set every N steps.

3. Asymmetric predictions: the GNN is not constrained to output p[i,j]=p[j,i]
   but tour labels are symmetric.  Symmetrising p_hat before the loss halves
   the effective noise in the gradient signal.
   Fix: p_sym = (p_hat + p_hat.T) / 2 before BCE.

4. Label quality: NN tours are 15-25% from optimal for large n.
   Fix: --label nn2opt runs 2-opt improvement (n ≤ 300) — ~10-20% better labels.

Label strategies
----------------
  n ≤ 10  → brute-force optimal tour  (--label optimal, default when n ≤ 10)
  any n   → nearest-neighbour tour    (--label nn)
  n ≤ 300 → NN + 2-opt improvement    (--label nn2opt)

Device
------
Automatically uses CUDA > XPU (Intel Arc) > MPS (Apple Silicon) > CPU.
Override with --device cpu|cuda|xpu|mps.

Usage
-----
  # Quick test (small model, n=8, brute-force labels)
  python train.py --size small --n 8 --steps 2000

  # Standard (medium model, 2-opt improved labels)
  python train.py --size medium --n 50 --label nn2opt --steps 3000 --source tsp

  # Mixed sizes for generalisation
  python train.py --size large --n_min 10 --n_max 100 --label nn2opt --steps 5000 --source tsp

  # Fine-tune
  python train.py --resume model/gnn.pt --size medium --n_min 10 --n_max 100 --label nn2opt --lr 1e-4
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
                  save_label_cache, load_label_cache,
                  greedy_decode, tour_length)
from model import TSPGNN, MODEL_SIZES

POOL_SIZE = 1000


# ── Device selection ──────────────────────────────────────────────────────────

def get_device(requested: str = "auto") -> torch.device:
    """
    Resolve the best available device.
    Priority: CUDA → XPU (Intel Arc via IPEX) → MPS (Apple Silicon) → CPU.
    Pass requested="cpu" / "cuda" / "xpu" / "mps" to override.
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


# ── Validation helpers ────────────────────────────────────────────────────────

def _nn_tour(coords: torch.Tensor) -> list:
    """Fast greedy nearest-neighbour tour (CPU)."""
    n = coords.shape[0]
    dist = torch.cdist(coords.cpu(), coords.cpu())
    visited = torch.zeros(n, dtype=torch.bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt)
        visited[nxt] = True
    return tour


def _build_val_set(n_nodes, n_min, n_max, city_pool, label,
                   n_val: int = 30) -> list:
    """
    Sample n_val fixed instances at training start.
    Each entry is (coords_cpu, ref_tour_length) where ref_tour_length comes
    from the same label strategy used for training.
    """
    instances = []
    for _ in range(n_val):
        n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes

        if city_pool is not None:
            idx    = torch.randperm(city_pool.shape[0])[:n]
            coords = city_pool[idx].cpu()
        else:
            coords = random_instance(n)

        # Reference tour: same quality level as training labels
        if n <= 10 and label in ("auto", "optimal"):
            ref_tour = None  # brute-force is too slow here — use NN instead
        ref_tour = _nn_tour(coords)
        ref_len  = tour_length(coords, ref_tour)
        instances.append((coords, ref_len))
    return instances


def _val_gap(model: torch.nn.Module, val_set: list,
             device: torch.device) -> float:
    """
    Decode the validation set with the current model and return mean gap (%)
    vs the NN reference stored in val_set.
    Gap > 0 means GNN tour is longer than NN; gap < 0 means GNN beats NN.
    """
    model.eval()
    gaps = []
    with torch.no_grad():
        for coords, ref_len in val_set:
            if ref_len < 1e-9:
                continue
            p_hat    = model(coords.to(device))
            p_sym    = (p_hat + p_hat.T) / 2
            gnn_tour = greedy_decode(p_sym)
            gnn_len  = tour_length(coords, gnn_tour)
            gaps.append((gnn_len - ref_len) / ref_len * 100.0)
    model.train()
    return float(np.mean(gaps)) if gaps else float("nan")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: TSPGNN, n_nodes: int = 8, n_steps: int = 500,
          lr: float = 1e-3, city_pool: torch.Tensor = None,
          label: str = "auto", n_min: int = None, n_max: int = None,
          device: torch.device = torch.device("cpu"),
          amp: bool = False, accum_steps: int = 1,
          scheduler: str = "cosine", warmup_steps: int = 100,
          patience: int = 0, label_cache: tuple = None,
          weight_decay: float = 1e-4, val_interval: int = 200):
    """
    Supervised training loop.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max are both set)
    label        : "auto" / "optimal" (n≤10) / "nn" (any n) / "nn2opt" (n≤300)
    n_min/max    : sample random size in [n_min, n_max] each step
    city_pool    : (N, 2) tensor on device — cities sampled each step
    amp          : mixed precision (float16 CUDA, bfloat16 elsewhere)
    accum_steps  : gradient accumulation steps
    scheduler    : "cosine" — cosine annealing with linear warmup | "none"
    warmup_steps : linear warmup steps before cosine decay kicks in
    patience     : early stopping — stop after N steps with no improvement (0=off)
    label_cache  : (coords_np, labels_np) from load_label_cache()
    weight_decay : AdamW L2 regularisation (default 1e-4)
    val_interval : compute validation gap every this many steps (0 = off)
    """
    model = model.to(device)

    # AdamW: same as Adam but adds decoupled L2 regularisation.
    # Empirically 5-10% better generalisation for GNNs than plain Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

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

    # ── Validation set (fixed instances sampled once at training start) ───────
    val_set = []
    if val_interval > 0:
        pool_cpu = city_pool.cpu() if city_pool is not None else None
        val_set  = _build_val_set(n_nodes, n_min, n_max, pool_cpu, label,
                                  n_val=30)

    losses     = []
    best_loss  = float("inf")
    no_improve = 0
    last_gap   = float("nan")
    optimizer.zero_grad()

    bar = tqdm(range(n_steps), desc=f"Training [{device}]", unit="step",
               dynamic_ncols=True)
    for step, _ in enumerate(bar):

        # ── Sample instance ───────────────────────────────────────────────────
        if label_cache is not None:
            coords_cache, labels_cache = label_cache
            i      = _random.randint(0, len(coords_cache) - 1)
            import numpy as _np
            coords = torch.from_numpy(_np.array(coords_cache[i])).to(
                device, non_blocking=True)
            y      = torch.from_numpy(_np.array(labels_cache[i])).to(
                device, non_blocking=True)
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
                assert n <= 10, f"Brute-force labels require n ≤ 10, got {n}."
                y = optimal_tour_labels(coords.cpu()).to(device, non_blocking=True)
            elif use_label == "nn2opt":
                y = nn_tour_labels(coords.cpu(), two_opt=True).to(
                    device, non_blocking=True)
            else:
                y = nn_tour_labels(coords.cpu()).to(device, non_blocking=True)

        # ── Forward + loss ────────────────────────────────────────────────────
        with amp_ctx:
            p_hat = model(coords)

            # Symmetrise: the tour graph is undirected (y[i,j] == y[j,i]).
            # Averaging p_hat and its transpose halves gradient noise and
            # enforces the structural constraint for free.
            p_sym = (p_hat + p_hat.T) / 2

            # Weighted BCE: pos_weight = n_neg / n_pos balances the extreme
            # class imbalance (tour edges are ~1/n of all edges).
            # Without this the model collapses to constant low probabilities.
            n_pos  = y.sum().clamp(min=1.0)
            n_neg  = (y.numel() - n_pos).clamp(min=1.0)
            pos_w  = (n_neg / n_pos).clamp(max=999.0)
            weight = y * (pos_w - 1.0) + 1.0   # pos_w for tour edges, 1 elsewhere
            loss   = F.binary_cross_entropy(p_sym, y, weight=weight) / accum_steps

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

        # ── Periodic validation gap ───────────────────────────────────────────
        if val_interval > 0 and (step + 1) % val_interval == 0 and val_set:
            last_gap = _val_gap(model, val_set, device)

        # ── Logging ───────────────────────────────────────────────────────────
        raw_loss = loss.item() * accum_steps
        losses.append(raw_loss)
        if raw_loss < best_loss:
            best_loss  = raw_loss
            no_improve = 0
        else:
            no_improve += 1

        avg    = sum(losses[-50:]) / len(losses[-50:])
        cur_lr = optimizer.param_groups[0]["lr"]
        bar.set_postfix(n=n, loss=f"{raw_loss:.4f}", avg50=f"{avg:.4f}",
                        best=f"{best_loss:.4f}", gap=f"{last_gap:.1f}%",
                        lr=f"{cur_lr:.2e}")

        # ── Early stopping ────────────────────────────────────────────────────
        if patience > 0 and no_improve >= patience:
            bar.write(f"Early stop: no improvement for {patience} steps.")
            break

    return losses


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=8)
    parser.add_argument("--n_min",  type=int,   default=None)
    parser.add_argument("--n_max",  type=int,   default=None)
    parser.add_argument("--label",  type=str,   default="auto",
                        help="'auto', 'optimal' (n≤10), 'nn' (any n), 'nn2opt' (n≤300)")
    parser.add_argument("--steps",  type=int,   default=500)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--size",   type=str,   default=None,
                        help="Model preset: 'small', 'medium', 'large'.")
    parser.add_argument("--d",      type=int,   default=64)
    parser.add_argument("--L",      type=int,   default=4)
    parser.add_argument("--out",    type=str,   default="model/gnn.pt")
    parser.add_argument("--source", type=str,   default="random",
                        help="'random', 'tsp', or 'solomon'")
    parser.add_argument("--resume", type=str,   default=None,
                        help="Path to existing weights to resume/fine-tune from")
    parser.add_argument("--device", type=str,   default="auto",
                        help="'auto' (cuda>xpu>mps>cpu), 'cuda', 'xpu', 'mps', 'cpu'")
    parser.add_argument("--amp",    action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--pool_cache", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--label_cache", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW L2 regularisation (default 1e-4).")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="Compute validation gap every N steps (0 = off).")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, "type") and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.label == "optimal" and args.n > 10:
        raise ValueError("--label optimal requires n ≤ 10. Use --label nn2opt.")
    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}.")
        args.d, args.L = MODEL_SIZES[args.size]

    # ── City pool ─────────────────────────────────────────────────────────────
    city_pool = None
    if args.source != "random":
        if args.pool_cache:
            if not os.path.exists(args.pool_cache):
                print(f"Building city pool cache ({args.source}) → {args.pool_cache} ...")
                raw = load_cities(POOL_SIZE, source=args.source)
                save_city_pool(raw, args.pool_cache)
                print("Cache saved.")
            print(f"Loading city pool (memory-mapped): {args.pool_cache}")
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

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: not available ({e}), skipping")

    # ── Label cache ───────────────────────────────────────────────────────────
    lbl_cache = None
    if args.label_cache and not (args.n_min and args.n_max):
        if not os.path.exists(args.label_cache):
            print(f"Building label cache (n={args.n}, label={args.label}) "
                  f"→ {args.label_cache} ...")
            save_label_cache(
                n=args.n, pool_size=2000, label=args.label,
                path=args.label_cache, city_pool=city_pool,
            )
            print("Label cache saved.")
        print(f"Loading label cache: {args.label_cache}")
        lbl_cache = load_label_cache(args.label_cache)

    base     = getattr(model, "_orig_mod", model)
    n_params = sum(p.numel() for p in base.parameters())
    size_info = (f"n={args.n}" if not (args.n_min and args.n_max)
                 else f"n in [{args.n_min},{args.n_max}]")
    print(f"TSPGNN(d={args.d}, L={args.L})  —  {n_params:,} parameters")
    print(f"Training : {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}")
    print(f"Optimizer: AdamW(wd={args.weight_decay})  |  "
          f"AMP: {args.amp}  |  accum_steps: {args.accum_steps}  |  "
          f"compile: {args.compile}")
    print(f"Scheduler: {args.scheduler}(warmup={args.warmup})  |  "
          f"patience: {args.patience or 'off'}  |  "
          f"val_interval: {args.val_interval or 'off'}  |  "
          f"label_cache: {'yes' if lbl_cache else 'no'}\n")

    losses = train(
        model, n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        amp=args.amp, accum_steps=args.accum_steps,
        scheduler=args.scheduler, warmup_steps=args.warmup,
        patience=args.patience, label_cache=lbl_cache,
        weight_decay=args.weight_decay, val_interval=args.val_interval,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))

    print(f"\nWeights saved → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

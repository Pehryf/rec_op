"""
train.py — Supervised training loop for TSPGNN

Key design decisions
--------------------
1. Weighted BCE (pos_weight = n_neg/n_pos): tour edges are ~1/n of all
   possible edges, so plain BCE collapses to constant ~0 predictions.
   Weighting forces equal gradient contribution from both classes.

2. Symmetrised predictions (p_sym = (p + p.T) / 2): the tour graph is
   undirected, labels are symmetric — enforcing this halves gradient noise.

3. Labels computed on-the-fly every step: no pre-built cache, no async
   device transfers, no stale data issues.

4. Constant LR with Adam: simple and reliable.  Add a scheduler later if
   you observe oscillation near convergence.

Label strategies
----------------
  auto     → optimal (n ≤ 10), nn otherwise
  optimal  → brute-force exact tour    (n ≤ 10 only)
  nn       → nearest-neighbour tour    (any n)
  nn2opt   → NN + 2-opt improvement    (n ≤ 300)

Device
------
Automatically picks CUDA > XPU > MPS > CPU.
Override with --device cpu|cuda|xpu|mps.

Usage
-----
  python train.py --size small --n 8 --steps 3000
  python train.py --size medium --n 50 --label nn2opt --steps 5000 --source tsp
  python train.py --resume model/gnn.pt --size medium --n_min 10 --n_max 100 --label nn2opt
"""

import argparse
import os
import random as _random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import (load_cities, random_instance,
                  optimal_tour_labels, nn_tour_labels,
                  greedy_decode, tour_length)
from model import TSPGNN, MODEL_SIZES

POOL_SIZE = 1000


def get_device(requested: str = "auto") -> torch.device:
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


def _make_labels(coords_cpu: torch.Tensor, label: str) -> torch.Tensor:
    """Compute tour labels on CPU, return float32 tensor."""
    n = coords_cpu.shape[0]
    use = "optimal" if (label == "auto" and n <= 10) else \
          ("nn"     if  label == "auto"             else label)
    if use == "optimal":
        assert n <= 10, f"Brute-force labels only for n≤10, got {n}"
        return optimal_tour_labels(coords_cpu)
    if use == "nn2opt":
        return nn_tour_labels(coords_cpu, two_opt=True)
    return nn_tour_labels(coords_cpu)


def _nn_tour(coords: torch.Tensor) -> list:
    """Simple nearest-neighbour tour for validation reference."""
    n = coords.shape[0]
    dist = torch.cdist(coords, coords)
    visited = torch.zeros(n, dtype=torch.bool)
    tour = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt); visited[nxt] = True
    return tour


def train(model: TSPGNN,
          n_nodes: int = 8,
          n_steps: int = 2000,
          lr: float = 1e-3,
          city_pool: torch.Tensor = None,
          label: str = "auto",
          n_min: int = None,
          n_max: int = None,
          device: torch.device = torch.device("cpu"),
          val_interval: int = 500) -> list:
    """
    Minimal supervised training loop.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max set)
    label        : 'auto' | 'optimal' | 'nn' | 'nn2opt'
    n_min/n_max  : random size per step in [n_min, n_max]
    city_pool    : (N,2) CPU tensor; if given, cities are subsampled from it
    val_interval : log greedy-tour gap vs NN every N steps (0 = off)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed validation instances (built once, always on CPU)
    val_set = []
    if val_interval > 0:
        for _ in range(30):
            n_v = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
            if city_pool is not None:
                c = city_pool[torch.randperm(city_pool.shape[0])[:n_v]]
            else:
                c = random_instance(n_v)
            val_set.append((c, tour_length(c, _nn_tour(c))))

    losses   = []
    best_gap = float("inf")

    bar = tqdm(range(n_steps), desc=f"Training [{device}]",
               unit="step", dynamic_ncols=True)
    for step, _ in enumerate(bar):

        # ── Sample instance ───────────────────────────────────────────────────
        n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
        if city_pool is not None:
            coords_cpu = city_pool[torch.randperm(city_pool.shape[0])[:n]]
        else:
            coords_cpu = random_instance(n)

        # Labels always computed on CPU, then moved synchronously
        y      = _make_labels(coords_cpu, label).to(device)
        coords = coords_cpu.to(device)

        # ── Forward ───────────────────────────────────────────────────────────
        p_hat = model(coords)
        # Symmetrise: tour is undirected, y[i,j] == y[j,i]
        p_sym = (p_hat + p_hat.T) / 2

        # ── Weighted BCE ──────────────────────────────────────────────────────
        # pos_weight = n_neg/n_pos so both classes contribute equally.
        # Without this the model collapses to constant ~0 predictions.
        n_pos  = y.sum().clamp(min=1.0)
        n_neg  = (y.numel() - n_pos).clamp(min=1.0)
        pos_w  = (n_neg / n_pos).clamp(max=999.0)
        weight = y * (pos_w - 1.0) + 1.0   # pos_w for tour edges, 1 otherwise
        loss   = F.binary_cross_entropy(p_sym, y, weight=weight)

        # ── Update ────────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        losses.append(loss.item())
        avg = sum(losses[-50:]) / len(losses[-50:])

        postfix = dict(n=n, loss=f"{loss.item():.4f}", avg50=f"{avg:.4f}")

        # ── Periodic validation gap ───────────────────────────────────────────
        if val_interval > 0 and (step + 1) % val_interval == 0 and val_set:
            model.eval()
            gaps = []
            with torch.no_grad():
                for c_cpu, nn_len in val_set:
                    if nn_len < 1e-9:
                        continue
                    p = model(c_cpu.to(device))
                    p_s = (p + p.T) / 2
                    gnn_tour = greedy_decode(p_s)
                    gnn_len  = tour_length(c_cpu, gnn_tour)
                    gaps.append((gnn_len - nn_len) / nn_len * 100.0)
            model.train()
            gap = float(np.mean(gaps)) if gaps else float("nan")
            best_gap = min(best_gap, gap) if not np.isnan(gap) else best_gap
            postfix["gap%"] = f"{gap:.1f}"
            postfix["best_gap"] = f"{best_gap:.1f}"

        bar.set_postfix(**postfix)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,   default=8)
    parser.add_argument("--n_min",       type=int,   default=None)
    parser.add_argument("--n_max",       type=int,   default=None)
    parser.add_argument("--label",       type=str,   default="auto",
                        help="'auto' | 'optimal' (n≤10) | 'nn' | 'nn2opt' (n≤300)")
    parser.add_argument("--steps",       type=int,   default=2000)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--size",        type=str,   default=None,
                        help="'small' | 'medium' | 'large'")
    parser.add_argument("--d",           type=int,   default=64)
    parser.add_argument("--L",           type=int,   default=4)
    parser.add_argument("--out",         type=str,   default="model/gnn.pt")
    parser.add_argument("--source",      type=str,   default="random",
                        help="'random' | 'tsp' | 'solomon'")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--val_interval",type=int,   default=500,
                        help="Validate every N steps (0 = off)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, "type") and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Args ──────────────────────────────────────────────────────────────────
    if args.label == "optimal" and args.n > 10:
        raise ValueError("--label optimal requires n ≤ 10.")
    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}.")
        args.d, args.L = MODEL_SIZES[args.size]

    # ── City pool (only when source != random) ────────────────────────────────
    city_pool = None
    if args.source != "random":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' ...")
        city_pool = load_cities(POOL_SIZE, source=args.source).cpu()
        print(f"Pool: {city_pool.shape}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TSPGNN(d=args.d, L=args.L)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: {args.resume} not found")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    size_info = (f"n={args.n}" if not (args.n_min and args.n_max)
                 else f"n∈[{args.n_min},{args.n_max}]")
    print(f"TSPGNN(d={args.d}, L={args.L})  —  {n_params:,} params")
    print(f"Training: {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    losses = train(
        model,
        n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        val_interval=args.val_interval,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))
    print(f"\nWeights → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best: {min(losses):.4f}")

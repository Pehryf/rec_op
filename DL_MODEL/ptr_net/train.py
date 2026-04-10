"""
train.py — Supervised training loop for PointerNetwork

Key design decisions
--------------------
1. NLL loss (cross-entropy) over each decoding step with teacher forcing:
   at step t, the model must assign highest log-probability to tour[t].
   This is the natural loss for a sequence model that predicts one city at a time.

2. Label strategies:
     auto     → optimal (n ≤ 10), nn otherwise
     optimal  → brute-force exact tour    (n ≤ 10 only)
     nn       → nearest-neighbour tour    (any n)
     nn2opt   → NN + 2-opt improvement    (n ≤ 300)

3. Validation gap: every val_interval steps, run greedy decoding on a held-out
   set and compute the gap vs the NN baseline.  No optimal labels needed at
   validation time — NN baseline is used as a proxy for quality.

4. Gradient clipping (max norm 1.0): LSTM training with BPTT can produce large
   gradients on long sequences; clipping prevents instability.

5. Constant LR with Adam.  Add a scheduler if you observe oscillation near
   convergence.

6. --mode tsp    → node_dim=2, standard TSP instances (x, y coordinates).
   --mode tsptwd → node_dim=5, TSPTW-D instances ([x, y, a/T, b/T, s/T]).
   Labels are always Euclidean-distance based (nn/nn2opt/optimal on coords).

Device
------
Automatically picks CUDA > XPU > MPS > CPU.
Override with --device cpu|cuda|xpu|mps.

Usage
-----
  python train.py --mode tsp    --size small  --n 8  --label optimal --epochs 1000
  python train.py --mode tsp    --size medium --n 10 --label optimal --epochs 3000 --source tsp
  python train.py --mode tsptwd --size medium --n 10 --label nn2opt  --epochs 3000
  python train.py --resume model/ptr_net_medium.pt --mode tsp --size medium \\
                  --n 50 --label nn2opt --source tsp --epochs 2000 --lr 5e-4
"""

import argparse
import os
import random as _random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import (load_cities, random_instance, tour_length,
                  optimal_tour, nn_tour, two_opt_improve,
                  random_tsptwd_instance)
from model import PointerNetwork, MODEL_SIZES

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


def _make_labels(coords_cpu: torch.Tensor, label: str) -> list:
    """
    Compute a tour label (list of city indices) on CPU.

    label strategies
    ----------------
    auto    → optimal if n ≤ 10, otherwise nn
    optimal → brute-force exact tour (n ≤ 10 only)
    nn      → nearest-neighbour tour (any n)
    nn2opt  → NN + 2-opt improvement (n ≤ 300, too slow otherwise)
    """
    n   = coords_cpu.shape[0]
    use = "optimal" if (label == "auto" and n <= 10) else \
          ("nn"     if  label == "auto"             else label)
    if use == "optimal":
        assert n <= 10, f"Brute-force labels only for n ≤ 10, got {n}"
        return optimal_tour(coords_cpu)
    if use == "nn2opt":
        return two_opt_improve(coords_cpu, nn_tour(coords_cpu))
    return nn_tour(coords_cpu)


def _nn_tour(coords: torch.Tensor) -> list:
    """Simple NN tour for validation reference (CPU only)."""
    return nn_tour(coords)


def train(model: PointerNetwork,
          n_nodes: int = 8,
          n_steps: int = 2000,
          lr: float = 1e-3,
          city_pool: torch.Tensor = None,
          label: str = "auto",
          n_min: int = None,
          n_max: int = None,
          device: torch.device = torch.device("cpu"),
          val_interval: int = 500,
          mode: str = "tsp") -> list:
    """
    Supervised training loop for PointerNetwork.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max set)
    label        : 'auto' | 'optimal' (n ≤ 10) | 'nn' | 'nn2opt' (n ≤ 300)
    n_min/n_max  : random size per step in [n_min, n_max] — variable-size training
    city_pool    : (N, 2) CPU tensor; cities subsampled from it when provided
    val_interval : log greedy-tour gap vs NN every N steps (0 = off)
    mode         : 'tsp' (node_dim=2) | 'tsptwd' (node_dim=5, TSPTW-D features)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed validation instances (always on CPU)
    val_set = []
    if val_interval > 0:
        for _ in range(30):
            n_v = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
            if mode == "tsptwd":
                inst = random_tsptwd_instance(n_v)
                c_coords = inst["coords"]
                c_feats  = inst["node_feats"]
            elif city_pool is not None:
                c_coords = city_pool[torch.randperm(city_pool.shape[0])[:n_v]]
                c_feats  = c_coords
            else:
                c_coords = random_instance(n_v)
                c_feats  = c_coords
            val_set.append((c_feats, c_coords, tour_length(c_coords, _nn_tour(c_coords))))

    losses   = []
    best_gap = float("inf")

    bar = tqdm(range(n_steps), desc=f"Training [{device}] mode={mode}",
               unit="step", dynamic_ncols=True)
    for step, _ in enumerate(bar):

        # ── Sample instance ───────────────────────────────────────────────────
        n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
        if mode == "tsptwd":
            inst       = random_tsptwd_instance(n)
            coords_cpu = inst["coords"]
            feats_cpu  = inst["node_feats"]   # (n, 5)
        elif city_pool is not None:
            coords_cpu = city_pool[torch.randperm(city_pool.shape[0])[:n]]
            feats_cpu  = coords_cpu
        else:
            coords_cpu = random_instance(n)
            feats_cpu  = coords_cpu

        # Labels computed on CPU coords, kept as a list of ints for teacher forcing
        tour  = _make_labels(coords_cpu, label)
        feats = feats_cpu.to(device)

        # ── Teacher-forced forward pass ───────────────────────────────────────
        # log_probs: (n, n) — log_probs[t, i] = log P(π_t = i | π_<t, X)
        log_probs, _ = model(feats, tour=tour)

        # Cross-entropy: at step t the target city is tour[t]
        targets = torch.tensor(tour, dtype=torch.long, device=device)
        loss    = F.nll_loss(log_probs, targets)

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
                for c_feats, c_coords, nn_len in val_set:
                    if nn_len < 1e-9:
                        continue
                    _, ptr_tour = model(c_feats.to(device))
                    ptr_len = tour_length(c_coords, ptr_tour)
                    gaps.append((ptr_len - nn_len) / nn_len * 100.0)
            model.train()
            gap = float(np.mean(gaps)) if gaps else float("nan")
            best_gap = min(best_gap, gap) if not np.isnan(gap) else best_gap
            postfix["gap%"]     = f"{gap:.1f}"
            postfix["best_gap"] = f"{best_gap:.1f}"

        bar.set_postfix(**postfix)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",            type=int,   default=8)
    parser.add_argument("--n_min",        type=int,   default=None)
    parser.add_argument("--n_max",        type=int,   default=None)
    parser.add_argument("--label",        type=str,   default="auto",
                        help="'auto' | 'optimal' (n≤10) | 'nn' | 'nn2opt' (n≤300)")
    parser.add_argument("--steps",        type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="Alias for --steps (same meaning).")
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--size",         type=str,   default=None,
                        help="'small' | 'medium' | 'large'. Overrides --embed/--hidden/--layers.")
    parser.add_argument("--embed",        type=int,   default=64)
    parser.add_argument("--hidden",       type=int,   default=128)
    parser.add_argument("--layers",       type=int,   default=1)
    parser.add_argument("--out",          type=str,   default=None,
                        help="Output path for weights. Auto-named from --size and --mode if omitted.")
    parser.add_argument("--source",       type=str,   default="random",
                        help="'random' | 'tsp' | 'solomon'")
    parser.add_argument("--mode",         type=str,   default="tsp",
                        help="'tsp' (node_dim=2) | 'tsptwd' (node_dim=5).")
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--device",       type=str,   default="auto")
    parser.add_argument("--val_interval", type=int,   default=500,
                        help="Validate every N steps (0 = off)")
    args = parser.parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, "type") and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Steps / epochs ────────────────────────────────────────────────────────
    if args.epochs is not None and args.steps is not None:
        raise ValueError("Specify only one of --steps or --epochs, not both.")
    if args.epochs is not None:
        args.steps = args.epochs
    if args.steps is None:
        args.steps = 2000

    # ── Mode ──────────────────────────────────────────────────────────────────
    if args.mode not in ("tsp", "tsptwd"):
        raise ValueError("--mode must be 'tsp' or 'tsptwd'.")
    node_dim = 5 if args.mode == "tsptwd" else 2

    # ── Args ──────────────────────────────────────────────────────────────────
    if args.label == "optimal" and args.n > 10:
        raise ValueError("--label optimal requires n ≤ 10.")
    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}.")
        args.embed, args.hidden, args.layers = MODEL_SIZES[args.size]

    # ── Auto output path ──────────────────────────────────────────────────────
    if args.out is None:
        if args.size is not None:
            suffix = f"_{args.mode}" if args.mode != "tsp" else ""
            args.out = f"model/ptr_net_{args.size}{suffix}.pt"
        else:
            args.out = "model/ptr_net.pt"

    # ── City pool (only when source != random and mode == tsp) ───────────────
    city_pool = None
    if args.source != "random" and args.mode == "tsp":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' ...")
        city_pool = load_cities(POOL_SIZE, source=args.source).cpu()
        print(f"Pool: {city_pool.shape}\n")
    elif args.source != "random" and args.mode == "tsptwd":
        print("Note: --source is ignored in tsptwd mode (instances are randomly generated).")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PointerNetwork(embed_dim=args.embed, hidden_dim=args.hidden,
                           n_layers=args.layers, node_dim=node_dim)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: {args.resume} not found")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")

    n_params  = sum(p.numel() for p in model.parameters())
    size_info = (f"n={args.n}" if not (args.n_min and args.n_max)
                 else f"n∈[{args.n_min},{args.n_max}]")
    print(f"PointerNetwork(embed={args.embed}, hidden={args.hidden}, "
          f"layers={args.layers}, node_dim={node_dim})  —  {n_params:,} params")
    print(f"Training: {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}, mode={args.mode}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    losses = train(
        model,
        n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        val_interval=args.val_interval,
        mode=args.mode,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))
    print(f"\nWeights → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best: {min(losses):.4f}")

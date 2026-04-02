"""
train.py — Supervised training loop for TSPGNN

Usage:
    # Fresh training
    python train.py
    python train.py --n 10 --steps 1000 --source tsp

    # Fine-tune an existing model on a different dataset
    python train.py --resume model/gnn.pt --source solomon --steps 300 --lr 1e-4
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_cities, random_instance, optimal_tour_labels
from model import TSPGNN, MODEL_SIZES

POOL_SIZE = 1000   # cities pre-loaded from dataset when source != "random"


def train(model: TSPGNN, n_nodes: int = 8, n_steps: int = 500, lr: float = 1e-3,
          city_pool: torch.Tensor = None):
    """
    Supervised training loop.

    Labels are computed by brute-force exact solver (feasible for n ≤ 10).
    Loss: binary cross-entropy over all n² edges.

    Parameters
    ----------
    city_pool : (N, 2) tensor of pre-loaded dataset cities.
                If provided, each step samples n_nodes cities at random from the pool
                instead of generating a synthetic random instance.
                If None, falls back to uniform random coordinates.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    bar = tqdm(range(n_steps), desc="Training", unit="step", dynamic_ncols=True)
    for step in bar:
        if city_pool is not None:
            idx    = torch.randperm(city_pool.shape[0])[:n_nodes]
            coords = city_pool[idx]
        else:
            coords = random_instance(n_nodes)

        y     = optimal_tour_labels(coords)   # (n, n) ground truth
        p_hat = model(coords)                 # (n, n) predictions

        loss = F.binary_cross_entropy(p_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        avg = sum(losses[-50:]) / len(losses[-50:])
        bar.set_postfix(loss=f"{loss.item():.4f}", avg50=f"{avg:.4f}", best=f"{min(losses):.4f}")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=8,              help="Cities per training instance (≤ 10)")
    parser.add_argument("--steps",  type=int,   default=500,            help="Training steps")
    parser.add_argument("--lr",     type=float, default=1e-3,           help="Learning rate")
    parser.add_argument("--size",   type=str,   default=None,
                        help="Model preset: 'small' (d=64,L=4), 'medium' (d=128,L=6), 'large' (d=256,L=8). "
                             "Overrides --d and --L when set.")
    parser.add_argument("--d",      type=int,   default=64,             help="Embedding dimension (ignored if --size is set)")
    parser.add_argument("--L",      type=int,   default=4,              help="Number of GNN layers (ignored if --size is set)")
    parser.add_argument("--out",    type=str,   default="model/gnn.pt", help="Path to save weights")
    parser.add_argument("--source", type=str,   default="random",
                        help="City source: 'random' (synthetic), 'tsp', or 'solomon'")
    parser.add_argument("--resume", type=str,   default=None,
                        help="Path to existing weights to resume/fine-tune from")
    args = parser.parse_args()

    assert args.n <= 10, "Brute-force labels require n ≤ 10. Use --n 8 or --n 10."

    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}. Got '{args.size}'.")
        args.d, args.L = MODEL_SIZES[args.size]

    # ── Load city pool from dataset (if requested) ────────────────────────────
    city_pool = None
    if args.source != "random":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' …")
        city_pool = load_cities(POOL_SIZE, source=args.source)
        print(f"Pool ready: {city_pool.shape}  "
              f"(each step samples {args.n} cities at random)\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TSPGNN(d=args.d, L=args.L)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: file not found: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")
    print(f"TSPGNN(d={args.d}, L={args.L})  —  {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training: n={args.n}, steps={args.steps}, lr={args.lr}, source={args.source}\n")

    losses = train(model, n_nodes=args.n, n_steps=args.steps, lr=args.lr, city_pool=city_pool)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)

    losses_path = os.path.join(out_dir, "losses.npy")
    import numpy as np
    np.save(losses_path, np.array(losses))

    print(f"\nWeights saved → {args.out}")
    print(f"Losses  saved → {losses_path}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

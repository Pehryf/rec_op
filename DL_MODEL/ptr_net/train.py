"""
train.py — Supervised training loop for PointerNetwork

Loss: cross-entropy over each decoding step against the optimal tour.
      At step t, the model must assign highest probability to tour[t].

Usage:
    # Fresh training
    python train.py
    python train.py --size medium --steps 2000 --source tsp

    # Fine-tune an existing model
    python train.py --resume model/ptr_net.pt --source solomon --steps 500 --lr 1e-4
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import random_instance, optimal_tour, tour_length, load_cities
from model import PointerNetwork, MODEL_SIZES

POOL_SIZE = 1000


def train(model: PointerNetwork, n_nodes: int = 8, n_steps: int = 500,
          lr: float = 1e-3, city_pool: torch.Tensor = None):
    """
    Supervised training on randomly generated (or pooled) instances.
    Labels are the optimal tours computed by brute-force (n ≤ 10 only).

    Loss: mean cross-entropy over the n pointer steps.
          At step t, target = tour[t] (the correct next city).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses    = []

    bar = tqdm(range(n_steps), desc="Training", unit="step", dynamic_ncols=True)
    for _ in bar:
        if city_pool is not None:
            idx    = torch.randperm(city_pool.shape[0])[:n_nodes]
            coords = city_pool[idx]
        else:
            coords = random_instance(n_nodes)

        tour = optimal_tour(coords)                  # ground-truth list[int]

        # Teacher-forced forward pass → log_probs: (n, n)
        log_probs, _ = model(coords, tour=tour)

        # Cross-entropy: at step t, target is tour[t]
        targets = torch.tensor(tour, dtype=torch.long)
        loss    = F.nll_loss(log_probs, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        avg = sum(losses[-50:]) / len(losses[-50:])
        bar.set_postfix(loss=f"{loss.item():.4f}", avg50=f"{avg:.4f}",
                        best=f"{min(losses):.4f}")

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,   default=8)
    parser.add_argument("--steps",  type=int,   default=500)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--size",   type=str,   default=None,
                        help="Preset: 'small', 'medium', 'large'. Overrides --embed/--hidden/--layers.")
    parser.add_argument("--embed",  type=int,   default=64)
    parser.add_argument("--hidden", type=int,   default=128)
    parser.add_argument("--layers", type=int,   default=1)
    parser.add_argument("--out",    type=str,   default="model/ptr_net.pt")
    parser.add_argument("--source", type=str,   default="random",
                        help="'random', 'tsp', or 'solomon'")
    parser.add_argument("--resume", type=str,   default=None)
    args = parser.parse_args()

    assert args.n <= 10, "Brute-force labels require n ≤ 10."

    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}")
        args.embed, args.hidden, args.layers = MODEL_SIZES[args.size]

    city_pool = None
    if args.source != "random":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' …")
        city_pool = load_cities(POOL_SIZE, source=args.source)
        print(f"Pool ready: {city_pool.shape}\n")

    model = PointerNetwork(embed_dim=args.embed, hidden_dim=args.hidden,
                           n_layers=args.layers)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: {args.resume} not found")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointerNetwork(embed={args.embed}, hidden={args.hidden}, layers={args.layers})"
          f"  —  {n_params:,} parameters")
    print(f"Training: n={args.n}, steps={args.steps}, lr={args.lr}, source={args.source}\n")

    losses = train(model, n_nodes=args.n, n_steps=args.steps,
                   lr=args.lr, city_pool=city_pool)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))

    print(f"\nWeights saved → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

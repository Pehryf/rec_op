"""
train.py — Supervised training loop for TSPGNN

Usage:
    python train.py                        # defaults: n=8, 500 steps
    python train.py --n 10 --steps 1000
    python train.py --n 8 --steps 500 --lr 2e-3 --out model/gnn.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import random_instance, optimal_tour_labels
from model import TSPGNN


def train(model: TSPGNN, n_nodes: int = 8, n_steps: int = 500, lr: float = 1e-3):
    """
    Supervised training on randomly generated instances.
    Labels are computed by brute-force exact solver (feasible for n ≤ 10).

    Loss: binary cross-entropy over all n² edges.
    Optimizer: Adam.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    bar = tqdm(range(n_steps), desc="Training", unit="step", dynamic_ncols=True)
    for step in bar:
        coords = random_instance(n_nodes)
        y      = optimal_tour_labels(coords)   # (n, n) ground truth
        p_hat  = model(coords)                 # (n, n) predictions

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
    parser.add_argument("--n",     type=int,   default=8,              help="Number of cities (≤ 10 for brute-force labels)")
    parser.add_argument("--steps", type=int,   default=500,            help="Training steps")
    parser.add_argument("--lr",    type=float, default=1e-3,           help="Learning rate")
    parser.add_argument("--d",     type=int,   default=64,             help="Embedding dimension")
    parser.add_argument("--L",     type=int,   default=4,              help="Number of GNN layers")
    parser.add_argument("--out",   type=str,   default="model/gnn.pt", help="Path to save weights")
    args = parser.parse_args()

    assert args.n <= 10, "Brute-force labels require n ≤ 10. Use --n 8 or --n 10."

    model = TSPGNN(d=args.d, L=args.L)
    print(f"TSPGNN(d={args.d}, L={args.L})  —  {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Training: n={args.n}, steps={args.steps}, lr={args.lr}\n")

    losses = train(model, n_nodes=args.n, n_steps=args.steps, lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"\nWeights saved → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best loss: {min(losses):.4f}")

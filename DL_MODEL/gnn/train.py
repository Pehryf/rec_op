"""
train.py — Supervised training loop for TSPGNN
"""

import torch
import torch.nn.functional as F

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

    for step in range(n_steps):
        coords = random_instance(n_nodes)
        y      = optimal_tour_labels(coords)   # (n, n) ground truth
        p_hat  = model(coords)                 # (n, n) predictions

        loss = F.binary_cross_entropy(p_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1:4d}/{n_steps}  BCE loss = {loss.item():.4f}")

    return losses

"""
data.py — TSP data helpers
"""

import torch
from itertools import permutations


def random_instance(n: int, seed: int = None) -> torch.Tensor:
    """n random cities uniformly distributed in [0, 1]²."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2)


def tour_length(coords: torch.Tensor, tour: list) -> float:
    """Total Euclidean length of a closed tour."""
    idx = torch.tensor(tour + [tour[0]])
    return (coords[idx[:-1]] - coords[idx[1:]]).norm(dim=1).sum().item()


def greedy_decode(p: torch.Tensor, start: int = 0) -> list:
    """
    Greedy tour construction from edge probabilities.
    At each step, move to the highest-scoring unvisited city.
    """
    n = p.shape[0]
    visited = torch.zeros(n, dtype=torch.bool, device=p.device)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        scores = p[tour[-1]].clone()
        scores[visited] = -1.0
        next_city = scores.argmax().item()
        tour.append(next_city)
        visited[next_city] = True
    return tour


def optimal_tour_labels(coords: torch.Tensor) -> torch.Tensor:
    """
    Brute-force optimal tour for small instances (n ≤ 10).
    Returns a binary (n, n) edge matrix: y[i,j] = 1 iff (i,j) is in the optimal tour.
    """
    n = coords.shape[0]
    best_len, best_tour = float("inf"), None
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        length = tour_length(coords, tour)
        if length < best_len:
            best_len, best_tour = length, tour
    y = torch.zeros(n, n)
    for k in range(n):
        a, b = best_tour[k], best_tour[(k + 1) % n]
        y[a, b] = y[b, a] = 1.0
    return y

"""
data.py — TSP data helpers
"""

import ast
import csv
import os

import torch
from itertools import permutations


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_tsp_chunks(chunks_dir: str, max_instances: int = None) -> list:
    """
    Load TSP instances from chunked CSV files (one instance per file).

    Each CSV row contains:
      - city_coordinates : Python-literal list of [x, y] pairs (scale ~[0, 100])
      - distance_matrix  : pre-computed Euclidean distances (unused here)
      - best_route       : placeholder string — NOT a valid tour
      - total_distance   : scalar float

    Returns a list of dicts with keys:
      - 'coords'  : torch.Tensor (n, 2) normalised to [0, 1]
      - 'n'       : int, number of cities
    """
    csv.field_size_limit(10_000_000)   # distance_matrix fields can be very large
    instances = []
    chunk_files = sorted(
        f for f in os.listdir(chunks_dir) if f.endswith(".csv")
    )
    for fname in chunk_files:
        if max_instances is not None and len(instances) >= max_instances:
            break
        path = os.path.join(chunks_dir, fname)
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                coords_raw = ast.literal_eval(row["city_coordinates"])   # [[x,y], ...]
                coords = torch.tensor(coords_raw, dtype=torch.float32)   # (n, 2)
                # Normalise to [0, 1] — coordinates are in ~[0, 100]
                coords = coords / 100.0
                instances.append({"coords": coords, "n": coords.shape[0]})
                if max_instances is not None and len(instances) >= max_instances:
                    break
    return instances


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

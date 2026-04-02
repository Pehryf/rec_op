"""
data.py — TSP data helpers for Tabu Search
"""

import ast
import csv
import os

import torch
from itertools import permutations


def load_cities(n: int, source: str = "tsp",
                chunks_dir: str = None,
                solomon_dir: str = None) -> torch.Tensor:
    """
    Load exactly n cities as a (n, 2) tensor normalised to [0, 1].

    source: "tsp"     — TSP chunk files
            "solomon" — Solomon TSPTW CSV files
    """
    csv.field_size_limit(10_000_000)
    raw_points = []

    if source == "tsp":
        base = chunks_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "dataset_raw", "_chunks", "tsp_dataset"
        )
        for fname in sorted(f for f in os.listdir(base) if f.endswith(".csv")):
            if len(raw_points) >= n:
                break
            with open(os.path.join(base, fname), newline="") as fh:
                for row in csv.DictReader(fh):
                    for point in ast.literal_eval(row["city_coordinates"]):
                        raw_points.append([float(point[0]), float(point[1])])
                        if len(raw_points) >= n:
                            break

    elif source == "solomon":
        base = solomon_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "dataset_raw", "solomon_dataset"
        )
        for group in sorted(os.listdir(base)):
            group_path = os.path.join(base, group)
            if not os.path.isdir(group_path):
                continue
            for fname in sorted(f for f in os.listdir(group_path) if f.endswith(".csv")):
                if len(raw_points) >= n:
                    break
                with open(os.path.join(group_path, fname), newline="") as fh:
                    reader = csv.reader(fh)
                    next(reader)
                    for row in reader:
                        if len(row) < 3:
                            continue
                        raw_points.append([float(row[1]), float(row[2])])
                        if len(raw_points) >= n:
                            break
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'tsp' or 'solomon'.")

    if len(raw_points) < n:
        raise ValueError(f"Only {len(raw_points)} cities available, requested {n}.")

    coords = torch.tensor(raw_points[:n], dtype=torch.float32)
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return (coords - mins) / (maxs - mins).clamp(min=1e-8)


def random_instance(n: int, seed: int = None) -> torch.Tensor:
    """n random cities uniformly distributed in [0, 1]²."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2)


def tour_length(coords: torch.Tensor, tour: list) -> float:
    """Total Euclidean length of a closed tour."""
    idx = torch.tensor(tour + [tour[0]])
    return (coords[idx[:-1]] - coords[idx[1:]]).norm(dim=1).sum().item()


def optimal_tour(coords: torch.Tensor) -> tuple:
    """
    Brute-force optimal tour for small instances (n ≤ 10).
    Returns (tour, length).
    """
    n = coords.shape[0]
    best_len, best_tour = float("inf"), None
    for perm in permutations(range(1, n)):
        t = [0] + list(perm)
        l = tour_length(coords, t)
        if l < best_len:
            best_len, best_tour = l, t
    return best_tour, best_len

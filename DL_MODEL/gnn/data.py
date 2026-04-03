"""
data.py — TSP data helpers
"""

import ast
import csv
import os

import numpy as np
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


def load_cities(n: int, source: str = "tsp",
                chunks_dir: str = None,
                solomon_dir: str = None) -> torch.Tensor:
    """
    Load exactly n cities from the dataset as a (n, 2) tensor normalised to [0, 1].

    Mirrors the GA helper `charger_villes_depuis_split(chunk_size, source)`.
    Cities are collected across instances/files until n are gathered, then
    min-max normalised so the model receives coordinates in [0, 1].

    Parameters
    ----------
    n           : number of cities to load
    source      : "tsp"     — TSP chunk files (dataset_raw/_chunks/tsp_dataset/)
                  "solomon" — Solomon TSPTW CSV files (dataset_raw/solomon_dataset/)
    chunks_dir  : override for TSP chunks directory
    solomon_dir : override for Solomon dataset root directory

    Returns
    -------
    torch.Tensor of shape (n, 2)
    """
    csv.field_size_limit(10_000_000)

    raw_points = []   # list of [x, y] floats, collected until len == n

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
                    next(reader)   # skip header
                    for row in reader:
                        if len(row) < 3:
                            continue
                        raw_points.append([float(row[1]), float(row[2])])
                        if len(raw_points) >= n:
                            break

    else:
        raise ValueError(f"Unknown source '{source}'. Use 'tsp' or 'solomon'.")

    if len(raw_points) < n:
        raise ValueError(
            f"Only {len(raw_points)} cities available in source '{source}', requested {n}."
        )

    coords = torch.tensor(raw_points[:n], dtype=torch.float32)   # (n, 2)

    # Min-max normalise to [0, 1] — works for any coordinate scale
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    coords = (coords - mins) / (maxs - mins).clamp(min=1e-8)

    return coords


def save_city_pool(cities: torch.Tensor, path: str):
    """
    Save a city pool tensor to a binary .npy file for memory-mapped reuse.
    Call this once after load_cities() to avoid reloading the dataset each run.

    Usage:
        pool = load_cities(10000, source="tsp")
        save_city_pool(pool, "model/city_pool.npy")
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, cities.numpy())


def load_city_pool_mmap(path: str) -> torch.Tensor:
    """
    Load a city pool from a .npy file using memory mapping.
    Only the slices actually accessed are read from disk — keeps RAM usage flat
    regardless of how large the pool is.

    Usage:
        pool = load_city_pool_mmap("model/city_pool.npy")
        # pass as city_pool to train()
    """
    arr = np.load(path, mmap_mode="r")   # mmap_mode="r" = read-only, no full load
    return torch.from_numpy(arr.copy())  # .copy() needed to make it writable for torch


def save_label_cache(n: int, pool_size: int, label: str, path: str,
                     city_pool: torch.Tensor = None):
    """
    Pre-compute pool_size instances of size n with their labels and save to disk.
    Only useful for fixed-n training — avoids recomputing NN labels every step.

    Usage:
        save_label_cache(n=50, pool_size=2000, label="nn", path="model/labels_n50.npz")
    Then pass --label_cache model/labels_n50.npz to train.py.
    """
    from tqdm import tqdm as _tqdm
    coords_all = np.zeros((pool_size, n, 2), dtype=np.float32)
    labels_all = np.zeros((pool_size, n, n), dtype=np.float32)
    pool_np    = city_pool.numpy() if city_pool is not None else None

    for i in _tqdm(range(pool_size), desc=f"Building label cache (n={n})", unit="inst"):
        if pool_np is not None:
            idx    = np.random.permutation(len(pool_np))[:n]
            coords = torch.tensor(pool_np[idx], dtype=torch.float32)
        else:
            coords = random_instance(n)
        y = optimal_tour_labels(coords) if label == "optimal" else nn_tour_labels(coords)
        coords_all[i] = coords.numpy()
        labels_all[i] = y.numpy()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, coords=coords_all, labels=labels_all)


def load_label_cache(path: str):
    """
    Load a pre-computed label cache (memory-mapped).
    Returns (coords, labels) as numpy arrays — index them per step.

    Usage:
        coords_cache, labels_cache = load_label_cache("model/labels_n50.npz")
        i = random.randint(0, len(coords_cache) - 1)
        coords = torch.from_numpy(coords_cache[i])
        y      = torch.from_numpy(labels_cache[i])
    """
    data = np.load(path, mmap_mode="r")
    return data["coords"], data["labels"]


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


def nn_tour_labels(coords: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbour tour labels for any instance size.
    Returns a binary (n, n) edge matrix: y[i,j] = 1 iff (i,j) is in the NN tour.

    Used as pseudo-labels to train the GNN on larger instances (n > 10)
    where brute-force is infeasible. The model learns to imitate NN quality,
    then generalises beyond it.
    """
    n = coords.shape[0]
    dist = torch.cdist(coords, coords)

    visited = torch.zeros(n, dtype=torch.bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt)
        visited[nxt] = True

    y = torch.zeros(n, n)
    for k in range(n):
        a, b = tour[k], tour[(k + 1) % n]
        y[a, b] = y[b, a] = 1.0
    return y

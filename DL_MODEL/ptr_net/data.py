"""
data.py — TSP data helpers for Pointer Networks

Key design decisions
--------------------
1. load_cities() normalises to [0, 1] using per-batch min-max — same as GNN data.py.
2. optimal_tour() returns a list of city indices (teacher-forcing target for Ptr-Net).
3. nn_tour() / two_opt_improve() provide pseudo-labels for larger instances.
4. save_city_pool / load_city_pool_mmap allow memory-efficient reuse of large pools.
5. save_label_cache / load_label_cache pre-compute (coords, tour) pairs to avoid
   recomputing NN labels every training step.
"""

import ast
import csv
import os

import numpy as np
import torch
from itertools import permutations


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_tsp_chunks(chunks_dir: str, max_instances: int = None) -> list:
    """
    Load TSP instances from chunked CSV files.

    Each CSV row contains:
      - city_coordinates : Python-literal list of [x, y] pairs (scale ~[0, 100])
      - distance_matrix  : pre-computed Euclidean distances (unused here)
      - best_route       : placeholder string — NOT a valid tour
      - total_distance   : scalar float

    Returns a list of dicts:
      - 'coords' : torch.Tensor (n, 2) normalised to [0, 1]
      - 'n'      : int, number of cities
    """
    csv.field_size_limit(10_000_000)
    instances = []
    for fname in sorted(f for f in os.listdir(chunks_dir) if f.endswith(".csv")):
        if max_instances is not None and len(instances) >= max_instances:
            break
        with open(os.path.join(chunks_dir, fname), newline="") as fh:
            for row in csv.DictReader(fh):
                coords_raw = ast.literal_eval(row["city_coordinates"])
                coords = torch.tensor(coords_raw, dtype=torch.float32) / 100.0
                instances.append({"coords": coords, "n": coords.shape[0]})
                if max_instances is not None and len(instances) >= max_instances:
                    break
    return instances


def load_cities(n: int, source: str = "tsp",
                chunks_dir: str = None,
                solomon_dir: str = None) -> torch.Tensor:
    """
    Load exactly n cities as a (n, 2) tensor normalised to [0, 1].

    Parameters
    ----------
    n           : number of cities to load
    source      : "tsp"     — TSP chunk files (dataset_raw/_chunks/tsp_dataset/)
                  "solomon" — Solomon TSPTW CSV files (dataset_raw/solomon_dataset/)
    chunks_dir  : override for TSP chunks directory
    solomon_dir : override for Solomon dataset root directory
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
        raise ValueError(
            f"Only {len(raw_points)} cities available in source '{source}', requested {n}."
        )

    coords = torch.tensor(raw_points[:n], dtype=torch.float32)
    mins = coords.min(dim=0).values
    maxs = coords.max(dim=0).values
    return (coords - mins) / (maxs - mins).clamp(min=1e-8)


def save_city_pool(cities: torch.Tensor, path: str):
    """
    Save a city pool tensor to a .npy file for memory-mapped reuse.
    Call once after load_cities() to avoid reloading the dataset each run.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, cities.numpy())


def load_city_pool_mmap(path: str) -> torch.Tensor:
    """
    Load a city pool from a .npy file using memory mapping.
    Only slices actually accessed are read from disk.
    """
    arr = np.load(path, mmap_mode="r")
    return torch.from_numpy(arr.copy())


def save_label_cache(n: int, pool_size: int, label: str, path: str,
                     city_pool: torch.Tensor = None):
    """
    Pre-compute pool_size instances of size n with their tour labels and save to disk.
    Labels are stored as integer arrays of shape (pool_size, n) — city index sequences.
    """
    from tqdm import tqdm as _tqdm
    coords_all = np.zeros((pool_size, n, 2), dtype=np.float32)
    tours_all  = np.zeros((pool_size, n),    dtype=np.int32)
    pool_np    = city_pool.cpu().numpy() if city_pool is not None else None

    for i in _tqdm(range(pool_size), desc=f"Building label cache (n={n})", unit="inst"):
        if pool_np is not None:
            idx    = np.random.permutation(len(pool_np))[:n]
            coords = torch.tensor(pool_np[idx], dtype=torch.float32)
        else:
            coords = random_instance(n)
        tour = _make_tour(coords, label)
        coords_all[i] = coords.numpy()
        tours_all[i]  = np.array(tour, dtype=np.int32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, coords=coords_all, tours=tours_all)


def load_label_cache(path: str):
    """
    Load a pre-computed label cache (memory-mapped).
    Returns (coords, tours) as numpy arrays.
    """
    data = np.load(path, mmap_mode="r")
    return data["coords"], data["tours"]


def _make_tour(coords: torch.Tensor, label: str) -> list:
    """Internal helper: compute a tour list from coords and a label strategy."""
    n   = coords.shape[0]
    use = "optimal" if (label == "auto" and n <= 10) else \
          ("nn"     if  label == "auto"             else label)
    if use == "optimal":
        return optimal_tour(coords)
    if use == "nn2opt":
        return two_opt_improve(coords, nn_tour(coords))
    return nn_tour(coords)


# ── Instance generation ───────────────────────────────────────────────────────

def random_instance(n: int, seed: int = None) -> torch.Tensor:
    """n random cities uniformly distributed in [0, 1]²."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2)


# ── Tour helpers ──────────────────────────────────────────────────────────────

def tour_length(coords: torch.Tensor, tour: list) -> float:
    """Total Euclidean length of a closed tour."""
    idx = torch.tensor(tour + [tour[0]])
    return (coords[idx[:-1]] - coords[idx[1:]]).norm(dim=1).sum().item()


def optimal_tour(coords: torch.Tensor) -> list:
    """
    Brute-force optimal tour for small instances (n ≤ 10).
    Returns the tour as a list of city indices starting at 0.
    """
    n = coords.shape[0]
    best_len, best_tour = float("inf"), None
    for perm in permutations(range(1, n)):
        t = [0] + list(perm)
        l = tour_length(coords, t)
        if l < best_len:
            best_len, best_tour = l, t
    return best_tour


def nn_tour(coords: torch.Tensor, start: int = 0) -> list:
    """
    Nearest-neighbour greedy tour construction.
    Returns a list of city indices.  O(n²) time, O(n) space.
    """
    n = coords.shape[0]
    dist = torch.cdist(coords, coords)
    visited = torch.zeros(n, dtype=torch.bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt)
        visited[nxt] = True
    return tour


def two_opt_improve(coords: torch.Tensor, tour: list,
                    max_iter: int = 100) -> list:
    """
    2-opt local search: repeatedly reverse sub-segments of the tour if doing
    so reduces total length.  Practical for n ≤ 300 (O(n²) per pass).
    """
    n = len(tour)
    if n <= 3:
        return tour
    tour = list(tour)
    xy = coords.numpy() if isinstance(coords, torch.Tensor) else coords

    def d(i: int, j: int) -> float:
        return float(np.linalg.norm(xy[i] - xy[j]))

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue   # trivial full-tour reversal
                ni, ni1 = tour[i], tour[i + 1]
                nj, nj1 = tour[j], tour[(j + 1) % n]
                if d(ni, nj) + d(ni1, nj1) < d(ni, ni1) + d(nj, nj1) - 1e-10:
                    tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour

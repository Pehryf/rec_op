"""
data.py — TSP and TSPTW-D data helpers for Pointer Networks

Key design decisions
--------------------
1. load_cities() normalises to [0, 1] using per-batch min-max — same as GNN data.py.
2. optimal_tour() returns a list of city indices (teacher-forcing target for Ptr-Net).
3. nn_tour() / two_opt_improve() provide pseudo-labels for larger instances.
4. save_city_pool / load_city_pool_mmap allow memory-efficient reuse of large pools.
5. save_label_cache / load_label_cache pre-compute (coords, tour) pairs to avoid
   recomputing NN labels every training step.
6. TSPTW-D helpers: generate_time_windows, generate_perturbations,
   build_tsptwd_node_features, random_tsptwd_instance.
   Node features (n, 5): [x, y, a_i/T, b_i/T, s_i/T] — consistent with GNN data.py.
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


# ── TSPTW-D helpers ───────────────────────────────────────────────────────────

def generate_time_windows(
    coords: torch.Tensor,
    speed: float = 1.0,
    tw_width_ratio: float = 0.4,
    seed: int = None,
) -> tuple:
    """
    Generate feasible time windows and service times for a TSPTW-D instance.

    Strategy: simulate a nearest-neighbour tour to get reference arrival times,
    then place windows of width tw_width_ratio * total_tour_time centred on
    each arrival time.  The depot always has window [0, T_max].

    Returns
    -------
    time_windows  : torch.Tensor (n, 2)   — [a_i, b_i] per city
    service_times : torch.Tensor (n,)     — s_i per city (depot s_0 = 0)
    """
    if seed is not None:
        torch.manual_seed(seed)
    n = coords.shape[0]

    dist_mat = torch.cdist(coords, coords)
    visited  = torch.zeros(n, dtype=torch.bool)
    tour_    = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist_mat[tour_[-1]].clone(); d[visited] = float("inf")
        nxt = d.argmin().item(); tour_.append(nxt); visited[nxt] = True

    mean_leg = dist_mat[dist_mat > 0].mean().item()
    svc = torch.full((n,), mean_leg * 0.05)
    svc[0] = 0.0

    arrivals = [0.0] * n
    t = 0.0
    for k in range(len(tour_) - 1):
        i, j = tour_[k], tour_[k + 1]
        t = max(t, arrivals[i]) + svc[i].item()
        t += dist_mat[i, j].item() / speed
        arrivals[j] = t
    total_time = t + dist_mat[tour_[-1], 0].item() / speed

    half_w = total_time * tw_width_ratio
    jitter = (torch.rand(n) * 0.5 + 0.75) * half_w
    jitter[0] = total_time

    a = torch.tensor(arrivals) - jitter
    b = torch.tensor(arrivals) + jitter
    a = a.clamp(min=0.0)
    b = b.clamp(min=a + mean_leg * 0.1)
    a[0] = 0.0; b[0] = total_time * 1.5

    return torch.stack([a, b], dim=1), svc


def generate_perturbations(
    n: int,
    total_time: float,
    n_perturb: int = None,
    alpha_range: tuple = (0.3, 1.5),
    seed: int = None,
) -> list:
    """
    Generate random perturbation events for a TSPTW-D instance.

    Each perturbation is a tuple (i, j, t_start, t_end, alpha) meaning:
      cost(i→j) = base_dist(i,j) × (1 + alpha)  for departures in [t_start, t_end].

    Returns
    -------
    list of (i, j, t_start, t_end, alpha) tuples
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    if n_perturb is None:
        n_perturb = max(1, n // 10)

    perturbs = []
    for _ in range(n_perturb):
        i = torch.randint(0, n, (1,), generator=rng).item()
        j = torch.randint(0, n, (1,), generator=rng).item()
        while j == i:
            j = torch.randint(0, n, (1,), generator=rng).item()
        t0 = (torch.rand(1, generator=rng) * total_time * 0.6).item()
        t1 = t0 + total_time * 0.3
        alpha = (
            torch.rand(1, generator=rng) * (alpha_range[1] - alpha_range[0])
            + alpha_range[0]
        ).item()
        perturbs.append((int(i), int(j), float(t0), float(t1), float(alpha)))
    return perturbs


def build_tsptwd_node_features(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    service_times: torch.Tensor,
) -> torch.Tensor:
    """
    Build node feature tensor for TSPTW-D input to the Pointer Network.

    Node features (n, 5):
      [x, y, a_i/T, b_i/T, s_i/T]  where T = max(b_i)

    Returns
    -------
    node_feats : torch.Tensor (n, 5)
    """
    T = time_windows[:, 1].max().clamp(min=1e-8).item()
    a = (time_windows[:, 0] / T).unsqueeze(1)
    b = (time_windows[:, 1] / T).unsqueeze(1)
    s = (service_times       / T).unsqueeze(1)
    return torch.cat([coords, a, b, s], dim=1)   # (n, 5)


def random_tsptwd_instance(n: int, seed: int = None) -> dict:
    """
    Generate a random TSPTW-D instance.

    Returns a dict with keys:
      'coords'        : torch.Tensor (n, 2)  — city coordinates in [0, 1]²
      'time_windows'  : torch.Tensor (n, 2)  — [a_i, b_i] per city
      'service_times' : torch.Tensor (n,)    — s_i per city
      'perturbations' : list of (i,j,t0,t1,alpha) tuples
      'node_feats'    : torch.Tensor (n, 5)  — model input features
    """
    coords = random_instance(n, seed=seed)
    tw, svc = generate_time_windows(coords, seed=seed)
    total_time = tw[:, 1].max().item()
    perturbs = generate_perturbations(n, total_time, seed=seed)
    node_feats = build_tsptwd_node_features(coords, tw, svc)
    return {
        "coords":        coords,
        "time_windows":  tw,
        "service_times": svc,
        "perturbations": perturbs,
        "node_feats":    node_feats,
    }


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

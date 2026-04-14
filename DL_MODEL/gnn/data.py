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
    pool_np    = city_pool.cpu().numpy() if city_pool is not None else None

    for i in _tqdm(range(pool_size), desc=f"Building label cache (n={n})", unit="inst"):
        if pool_np is not None:
            idx    = np.random.permutation(len(pool_np))[:n]
            coords = torch.tensor(pool_np[idx], dtype=torch.float32)
        else:
            coords = random_instance(n)
        use_2opt = label == "nn2opt"
        base_label = "nn" if use_2opt else label
        y = (optimal_tour_labels(coords) if base_label == "optimal"
             else nn_tour_labels(coords, two_opt=use_2opt))
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
    Uses masked_fill for reliable cross-device masking (CPU, CUDA, MPS).
    """
    p = p.float()
    n = p.shape[0]
    visited = torch.zeros(n, dtype=torch.bool, device=p.device)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        scores = p[tour[-1]].clone().masked_fill(visited, float("-inf"))
        next_city = scores.argmax().item()
        tour.append(next_city)
        visited[next_city] = True
    return tour


def two_opt_improve(coords: torch.Tensor, tour: list,
                    max_iter: int = 100) -> list:
    """
    2-opt local search: repeatedly reverse sub-segments of the tour if doing
    so reduces total length.  Runs until no improving swap exists or max_iter
    passes are completed.  Practical for n ≤ 300 (O(n²) per pass).
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
                # Skip the wrap-around edge (i == 0, j == n-1) to avoid reversing
                # the whole tour which is a no-op.
                if i == 0 and j == n - 1:
                    continue
                ni, ni1 = tour[i], tour[i + 1]
                nj, nj1 = tour[j], tour[(j + 1) % n]
                if d(ni, nj) + d(ni1, nj1) < d(ni, ni1) + d(nj, nj1) - 1e-10:
                    tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
                    improved = True
        if not improved:
            break
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

    Parameters
    ----------
    coords          : (n, 2) city coordinates in [0, 1]²
    speed           : travel speed (distance / time unit)
    tw_width_ratio  : window half-width as fraction of total NN tour time
    seed            : optional RNG seed

    Returns
    -------
    time_windows  : torch.Tensor (n, 2)   — [a_i, b_i] per city
    service_times : torch.Tensor (n,)     — s_i per city (depot s_0 = 0)
    """
    if seed is not None:
        torch.manual_seed(seed)
    n = coords.shape[0]

    # NN tour to get reference arrival times
    dist_mat = torch.cdist(coords, coords)
    visited  = torch.zeros(n, dtype=torch.bool)
    tour     = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist_mat[tour[-1]].clone(); d[visited] = float("inf")
        nxt = d.argmin().item(); tour.append(nxt); visited[nxt] = True

    # Service times: ~5 % of mean leg distance
    mean_leg = dist_mat[dist_mat > 0].mean().item()
    svc = torch.full((n,), mean_leg * 0.05)
    svc[0] = 0.0   # depot has no service time

    # Simulate arrivals along the NN tour
    arrivals = [0.0] * n
    t = 0.0
    for k in range(len(tour) - 1):
        i, j = tour[k], tour[k + 1]
        t = max(t, arrivals[i]) + svc[i].item()   # depart after service
        t += dist_mat[i, j].item() / speed
        arrivals[j] = t
    total_time = t + dist_mat[tour[-1], 0].item() / speed   # return

    half_w = total_time * tw_width_ratio
    # Add random jitter so windows are not all the same width
    jitter = (torch.rand(n) * 0.5 + 0.75) * half_w   # 0.75–1.25 × half_w
    jitter[0] = total_time   # depot open all day

    a = torch.tensor(arrivals) - jitter
    b = torch.tensor(arrivals) + jitter
    a = a.clamp(min=0.0)
    b = b.clamp(min=a + mean_leg * 0.1)   # window at least slightly positive width
    a[0] = 0.0; b[0] = total_time * 1.5   # depot: open from 0 to 1.5× tour

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

    Parameters
    ----------
    n           : number of cities
    total_time  : reference tour duration (used to scale time windows)
    n_perturb   : number of perturbed edges (default: max(1, n//10))
    alpha_range : (min_alpha, max_alpha) for perturbation severity
    seed        : optional RNG seed

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
        # Random active time window covering ~30% of tour
        t0 = (torch.rand(1, generator=rng) * total_time * 0.6).item()
        t1 = t0 + total_time * 0.3
        alpha = (
            torch.rand(1, generator=rng) * (alpha_range[1] - alpha_range[0])
            + alpha_range[0]
        ).item()
        perturbs.append((int(i), int(j), float(t0), float(t1), float(alpha)))
    return perturbs


def perturb_edge_matrix(
    coords: torch.Tensor,
    perturbations: list,
    t: float = 0.0,
) -> torch.Tensor:
    """
    Return an (n, n) tensor of effective travel cost multipliers at time t.

    entry[i, j] = 1 + alpha  if a perturbation on (i,j) is active at time t,
                  1.0        otherwise.
    Symmetric: perturbations apply in both directions.
    """
    n = coords.shape[0]
    mat = torch.ones(n, n)
    for pi, pj, t_start, t_end, alpha in perturbations:
        if t_start <= t <= t_end:
            mat[pi, pj] = 1.0 + alpha
            mat[pj, pi] = 1.0 + alpha
    return mat


def worst_case_perturb_matrix(coords: torch.Tensor, perturbations: list) -> torch.Tensor:
    """
    Return an (n, n) tensor of the WORST-CASE perturbation factor per edge
    (max alpha across all perturbations on that edge, regardless of time).

    Used as a static edge feature for the GNN.
    """
    n = coords.shape[0]
    mat = torch.zeros(n, n)
    for pi, pj, _t0, _t1, alpha in perturbations:
        if alpha > mat[pi, pj].item():
            mat[pi, pj] = alpha
            mat[pj, pi] = alpha
    return mat


def build_tsptwd_features(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    service_times: torch.Tensor,
    perturbations: list,
) -> tuple:
    """
    Build node and edge feature tensors for TSPTW-D input to the GNN.

    Node features (n, 5):
      [x, y, a_i/T, b_i/T, s_i/T]  where T = max(b_i)

    Edge features (n, n, 1):
      [alpha_ij]  worst-case perturbation factor per edge (0 = no perturbation)

    Returns
    -------
    node_feats : torch.Tensor (n, 5)
    edge_feats : torch.Tensor (n, n, 1)
    """
    T = time_windows[:, 1].max().clamp(min=1e-8).item()
    a = (time_windows[:, 0] / T).unsqueeze(1)
    b = (time_windows[:, 1] / T).unsqueeze(1)
    s = (service_times       / T).unsqueeze(1)
    node_feats = torch.cat([coords, a, b, s], dim=1)   # (n, 5)

    perturb_mat = worst_case_perturb_matrix(coords, perturbations)
    edge_feats  = perturb_mat.unsqueeze(-1)             # (n, n, 1)

    return node_feats, edge_feats


def evaluate_tsptwd(
    coords: torch.Tensor,
    tour: list,
    time_windows: torch.Tensor,
    service_times: torch.Tensor,
    perturbations: list,
    penalty_coeff: float = 1000.0,
    speed: float = 1.0,
) -> tuple:
    """
    Simulate a TSPTW-D tour and return the penalised objective.

    Travel cost:  c_ij(t) = dist_ij / speed × (1 + alpha) if perturbation active
    Time window:  early arrival → wait; late arrival → penalty

    Returns
    -------
    total_time  : float  — return time to depot (raw, before penalty)
    tw_penalty  : float  — total time-window violation penalty
    obj         : float  — total_time + tw_penalty
    """
    dist_mat = torch.cdist(coords, coords)
    n = coords.shape[0]
    t = 0.0
    penalty = 0.0

    for k in range(n):
        i = tour[k]
        j = tour[(k + 1) % n]

        # Departure from i: wait until window opens
        if k > 0:
            a_i = time_windows[i, 0].item()
            b_i = time_windows[i, 1].item()
            if t < a_i:
                t = a_i
            elif t > b_i:
                penalty += (t - b_i) * penalty_coeff
            t += service_times[i].item()

        # Travel cost i → j at departure time t
        base = dist_mat[i, j].item() / speed
        mult = 1.0
        for pi, pj, t_start, t_end, alpha in perturbations:
            if (pi == i and pj == j) or (pi == j and pj == i):
                if t_start <= t <= t_end:
                    mult = max(mult, 1.0 + alpha)
        t += base * mult

    total_time = t
    return total_time, penalty, total_time + penalty


def make_benchmark_instance(
    n: int,
    source: str = "tsp",
    tw_width_ratio: float = 0.4,
    n_perturb: int = None,
    seed: int = None,
    chunks_dir: str = None,
    solomon_dir: str = None,
) -> dict:
    """
    Build a complete TSPTW-D benchmark instance for consistent cross-algorithm comparison.

    Returns a dict with keys:
      coords        : (n, 2)  city coordinates normalised to [0, 1]
      time_windows  : (n, 2)  [a_i, b_i] time windows
      service_times : (n,)    service times per city
      perturbations : list of (i, j, t_start, t_end, alpha)
      node_feats    : (n, 5)  GNN node features
      edge_feats    : (n, n, 1) GNN edge features
      depot         : int     index of depot (always 0)
    """
    if source == "solomon":
        coords, time_windows, service_times = load_solomon_instance(
            n, solomon_dir=solomon_dir, seed=seed
        )
    elif source == "random":
        coords = random_instance(n, seed=seed)
        time_windows, service_times = generate_time_windows(
            coords, tw_width_ratio=tw_width_ratio, seed=seed
        )
    else:
        coords = load_cities(n, source=source, chunks_dir=chunks_dir, solomon_dir=solomon_dir)
        time_windows, service_times = generate_time_windows(
            coords, tw_width_ratio=tw_width_ratio, seed=seed
        )

    # Estimate tour time for perturbation scaling
    dist_mat = torch.cdist(coords, coords)
    total_time = time_windows[:, 1].max().item()

    perturbations = generate_perturbations(
        n, total_time=total_time, n_perturb=n_perturb, seed=seed
    )
    node_feats, edge_feats = build_tsptwd_features(
        coords, time_windows, service_times, perturbations
    )

    return {
        "coords":        coords,
        "time_windows":  time_windows,
        "service_times": service_times,
        "perturbations": perturbations,
        "node_feats":    node_feats,
        "edge_feats":    edge_feats,
        "depot":         0,
    }


def load_tsptwd_json(path: str) -> dict:
    """
    Load a TSPTW-D dataset generated by datasetsgenerator.ipynb and return
    the same dict format as make_benchmark_instance().

    The JSON stores time values in minutes (scaled by `scale`).  They are
    divided back by `scale` here so every quantity lives in the same unit as
    the [0, 1]² coordinates — consistent with evaluate_tsptwd() and the GNN.

    Parameters
    ----------
    path : path to a tsptwd_n*.json file

    Returns
    -------
    dict with keys: coords, time_windows, service_times, perturbations,
                    node_feats, edge_feats, depot
    """
    import json as _json
    with open(path, encoding="utf-8") as fh:
        data = _json.load(fh)

    scale    = float(data["meta"]["scale"])
    horizon  = float(data["meta"].get("horizon", scale))   # fallback = scale
    all_nodes = [data["depot"]] + data["clients"]   # depot at index 0

    coords = torch.tensor(
        [[node["x"], node["y"]] for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1, 2)

    def _b(node):
        # depot "b" is null in the generator — substitute the full horizon
        return float(node["b"]) / scale if node["b"] is not None else horizon / scale

    tw = torch.tensor(
        [[float(node["a"]) / scale, _b(node)] for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1, 2)

    svc = torch.tensor(
        [node["service"] / scale for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1,)

    perturbs = [
        (int(p["arc"][0]), int(p["arc"][1]),
         float(p["t_start"]) / scale, float(p["t_end"]) / scale,
         float(p["alpha"]))
        for p in data.get("perturbations", [])
    ]

    node_feats, edge_feats = build_tsptwd_features(coords, tw, svc, perturbs)

    return {
        "coords":        coords,
        "time_windows":  tw,
        "service_times": svc,
        "perturbations": perturbs,
        "node_feats":    node_feats,
        "edge_feats":    edge_feats,
        "depot":         0,
    }


def load_solomon_instance(
    n: int,
    solomon_dir: str = None,
    seed: int = None,
) -> tuple:
    """
    Load up to n cities from a random Solomon TSPTW file (rc_*.csv).

    Returns
    -------
    coords        : torch.Tensor (n, 2)   normalised to [0, 1]
    time_windows  : torch.Tensor (n, 2)   [ready_time, due_date] normalised
    service_times : torch.Tensor (n,)     normalised
    """
    base = solomon_dir or os.path.join(
        os.path.dirname(__file__), "..", "..", "dataset_raw", "SolomonTSPTW"
    )
    files = sorted(f for f in os.listdir(base) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No Solomon CSV files found in {base}")

    rng = np.random.default_rng(seed)
    fname = rng.choice(files)
    path  = os.path.join(base, fname)

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break

    if len(rows) < n:
        raise ValueError(
            f"Solomon file {fname} has only {len(rows)} nodes, requested {n}."
        )

    raw_xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    raw_a  = np.array([float(r["ready_time"])   for r in rows])
    raw_b  = np.array([float(r["due_date"])     for r in rows])
    raw_s  = np.array([float(r["service_time"]) for r in rows])

    # Normalise coordinates to [0, 1]
    mins, maxs = raw_xy.min(0), raw_xy.max(0)
    xy = (raw_xy - mins) / (maxs - mins + 1e-8)

    # Normalise times by T_max (depot due date = row 0)
    T = raw_b[0] if raw_b[0] > 0 else raw_b.max()
    a = raw_a / T; b = raw_b / T; s = raw_s / T

    coords        = torch.tensor(xy,  dtype=torch.float32)
    time_windows  = torch.tensor(np.stack([a, b], axis=1), dtype=torch.float32)
    service_times = torch.tensor(s,   dtype=torch.float32)
    return coords, time_windows, service_times


def nn_tour_labels(coords: torch.Tensor, two_opt: bool = False) -> torch.Tensor:
    """
    Nearest-neighbour tour labels for any instance size.
    Returns a binary (n, n) edge matrix: y[i,j] = 1 iff (i,j) is in the NN tour.

    Parameters
    ----------
    two_opt : if True, apply 2-opt local search after NN construction (only
              for n ≤ 300 — too slow otherwise).  Produces significantly better
              labels (~10-20% shorter tours) at the cost of label-build time.

    Used as pseudo-labels to train the GNN on larger instances (n > 10)
    where brute-force is infeasible.
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

    if two_opt and n <= 300:
        tour = two_opt_improve(coords, tour)

    y = torch.zeros(n, n)
    for k in range(n):
        a, b = tour[k], tour[(k + 1) % n]
        y[a, b] = y[b, a] = 1.0
    return y

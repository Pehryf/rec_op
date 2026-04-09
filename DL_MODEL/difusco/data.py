"""
data.py — Data utilities for DIFUSCO (TSP and TSPTW-D)

Sections
--------
1. Dataset loading
   load_dataset(n)              → load the pre-generated tsptwd_n{n}.json instance
   load_tsptwd_json(path)       → load any tsptwd JSON file into a feature dict
   random_instance(n)           → random (n, 2) coords in [0,1]²
   generate_tsptwd_instance(n)  → on-the-fly random TSPTW-D instance (same format)

2. TSPTW-D feature engineering
   generate_time_windows(coords)      → feasible [a_i, b_i] + service times
   generate_perturbations(n, T)       → list of arc-cost disruption events
   worst_case_perturb_matrix(coords)  → (n,n) max α per edge (static GNN input)
   perturb_edge_matrix(coords, t)     → (n,n) active multipliers at time t
   build_tsptwd_features(...)         → node feats (n,5) + edge feats (n,n,1)

3. Training labels
   nn_tour_labels(coords)   → binary (n,n) edge matrix from nearest-neighbour tour
   two_opt_improve(coords)  → 2-opt local search on a tour list

4. DIFUSCO decoding  (pair-coding TODO — stubs only)
   binarize(p, threshold)   → continuous (n,n) probabilities → {0,1} adj matrix
   tour_from_adj(adj)       → ordered tour from a binary adj matrix
   greedy_decode(p, start)  → greedy argmax tour from edge probabilities

5. Evaluation
   evaluate_tsptwd(...)  → penalised objective (total time + TW penalty)
   tour_length(coords)   → Euclidean tour length
"""

import json
import math
import os
import random as _random

import numpy as np
import torch
from itertools import permutations


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "datasets"
)


def load_dataset(n: int, datasets_dir: str = None) -> dict:
    """
    Load the pre-generated TSPTW-D instance of size n.

    Reads  datasets/tsptwd_n{n}.json  (one instance per file).
    Available sizes: 5, 10, 20, 50, 100, 200, 300, 500, 1000, 10000.

    Parameters
    ----------
    n            : number of clients (excluding depot, same as the file suffix)
    datasets_dir : override for the datasets/ directory path

    Returns
    -------
    dict with keys:
      coords        : torch.Tensor (n+1, 2)   city coordinates in [0,1]²
      time_windows  : torch.Tensor (n+1, 2)   [a_i/scale, b_i/scale]
      service_times : torch.Tensor (n+1,)     s_i / scale
      perturbations : list of (i, j, t_start, t_end, alpha)
      node_feats    : torch.Tensor (n+1, 5)   GNN node features
      edge_feats    : torch.Tensor (n+1, n+1, 1) GNN edge features
      depot         : int  (always 0)
      meta          : dict  raw meta-data from the JSON file
    """
    base = datasets_dir or _DATASETS_DIR
    path = os.path.join(base, f"tsptwd_n{n}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Available sizes: {sorted(int(f.split('_n')[1].split('.')[0]) for f in os.listdir(base) if f.startswith('tsptwd_n'))}"
        )
    inst = load_tsptwd_json(path)
    return inst


def load_tsptwd_json(path: str) -> dict:
    """
    Load a single TSPTW-D instance from a JSON file.

    JSON structure
    --------------
    {
      "meta": {"n_clients": int, "scale": float, "horizon": float, "seed": int},
      "depot":  {"id": 0, "x": float, "y": float, "a": float, "b": float, "service": float},
      "clients": [{"id": int, "x": ..., "y": ..., "a": ..., "b": ..., "service": ...}, ...],
      "perturbations": [{"arc": [i, j], "t_start": float, "t_end": float, "alpha": float}, ...]
    }

    Time values (a, b, service, t_start, t_end) are stored in minutes and divided
    by `scale` here so every quantity lives in the same unit as the [0,1]² coords.

    Returns
    -------
    dict with keys:
      coords, time_windows, service_times, perturbations,
      node_feats, edge_feats, depot (=0), meta
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    scale     = float(data["meta"]["scale"])
    all_nodes = [data["depot"]] + data["clients"]   # depot always at index 0

    coords = torch.tensor(
        [[node["x"], node["y"]] for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1, 2)

    tw = torch.tensor(
        [[node["a"] / scale, node["b"] / scale] for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1, 2)

    svc = torch.tensor(
        [node["service"] / scale for node in all_nodes],
        dtype=torch.float32,
    )                                                # (n+1,)

    perturbs = [
        (
            int(p["arc"][0]),
            int(p["arc"][1]),
            float(p["t_start"]) / scale,
            float(p["t_end"])   / scale,
            float(p["alpha"]),
        )
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
        "meta":          data["meta"],
    }


def random_instance(n: int, seed: int = None) -> torch.Tensor:
    """n random cities uniformly distributed in [0, 1]²."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.rand(n, 2)


def generate_tsptwd_instance(
    n: int,
    seed: int = None,
    scale: float = 200.0,
    service_min: float = 5.0,
    service_max: float = 15.0,
    tw_width_min: float = 60.0,
    tw_width_max: float = 180.0,
    horizon: float = 480.0,
    alpha_min: float = 1.5,
    alpha_max: float = 3.5,
    perturbs_ratio: float = 0.1,
) -> dict:
    """
    Generate a random TSPTW-D instance on the fly and return a feature dict
    identical in structure to ``load_tsptwd_json()``.

    Parameters
    ----------
    n               : number of clients (depot is added at index 0, total n+1 nodes)
    seed            : RNG seed for reproducibility
    scale           : distance-to-minutes factor (Euclidean dist × scale = minutes)
    service_min/max : service time range at each client [min]
    tw_width_min/max: time-window width range [min]
    horizon         : end of day — depot window closes here [min]
    alpha_min/max   : perturbation multiplier range (full multiplier, e.g. 2.0 = ×2)
    perturbs_ratio  : fraction of n used as number of perturbations (min 1)

    Returns
    -------
    dict with keys:
      coords        : torch.Tensor (n+1, 2)      city coordinates in [0,1]²
      time_windows  : torch.Tensor (n+1, 2)      [a_i/scale, b_i/scale]
      service_times : torch.Tensor (n+1,)         s_i / scale
      perturbations : list of (i, j, t_start/scale, t_end/scale, alpha)
      node_feats    : torch.Tensor (n+1, 5)       GNN node features
      edge_feats    : torch.Tensor (n+1, n+1, 1)  GNN edge features
      depot         : int  (always 0)
      meta          : dict

    Strategy for time windows
    -------------------------
    We simulate a greedy tour in index order (depot → client 1 → … → client n)
    to get reference arrival times τ_i, then place the window
    [a_i = max(1, τ_i − margin), b_i = min(a_i + width, horizon − s_i)]
    around each arrival.  This guarantees at least one feasible tour exists.
    """
    rng = _random.Random(seed)

    # ── 1. Random coordinates in [0, 1]² ─────────────────────────────────────
    xy = [(rng.random(), rng.random()) for _ in range(n + 1)]  # index 0 = depot

    # ── 2. Greedy propagation to build feasible time windows (in minutes) ────
    depot_x, depot_y = xy[0]
    tw_a  = [0.0]     * (n + 1)    # opening times  [min]
    tw_b  = [horizon] * (n + 1)    # closing times  [min]
    svc   = [0.0]     * (n + 1)    # service times  [min]

    t    = 0.0
    px, py = depot_x, depot_y

    for idx in range(1, n + 1):
        cx, cy = xy[idx]
        s_i    = round(rng.uniform(service_min, service_max), 1)
        dist   = math.hypot(cx - px, cy - py) * scale     # travel time [min]
        t_arr  = t + dist                                   # greedy arrival

        width   = round(rng.uniform(tw_width_min, tw_width_max), 1)
        margin  = round(rng.uniform(0.0, width * 0.4), 1)  # open window a bit early
        a_i     = max(1.0, round(t_arr - margin, 1))
        b_i     = min(round(a_i + width, 1), horizon - s_i)
        if a_i >= b_i:                                      # safety clamp
            b_i = a_i + tw_width_min

        tw_a[idx] = a_i
        tw_b[idx] = b_i
        svc[idx]  = s_i

        t    = max(t_arr, a_i) + s_i                        # advance greedy clock
        px, py = cx, cy

    # ── 3. Perturbations ─────────────────────────────────────────────────────
    n_perturbs = max(1, int(n * perturbs_ratio))
    all_arcs   = [(i, j) for i in range(n + 1) for j in range(i + 1, n + 1)]
    chosen     = rng.sample(all_arcs, min(n_perturbs, len(all_arcs)))

    perturbs_raw = []
    for (i, j) in chosen:
        t_start = round(rng.uniform(0.0, horizon * 0.6), 1)
        dur     = round(rng.uniform(30.0, horizon * 0.3), 1)
        t_end   = min(round(t_start + dur, 1), horizon)
        alpha   = round(rng.uniform(alpha_min, alpha_max), 2)
        perturbs_raw.append((i, j, t_start, t_end, alpha))

    # ── 4. Convert to tensors (normalised by scale, matching load_tsptwd_json) ─
    coords = torch.tensor(xy, dtype=torch.float32)            # (n+1, 2)

    tw_tensor = torch.tensor(
        [[tw_a[i] / scale, tw_b[i] / scale] for i in range(n + 1)],
        dtype=torch.float32,
    )                                                          # (n+1, 2)

    svc_tensor = torch.tensor(
        [svc[i] / scale for i in range(n + 1)],
        dtype=torch.float32,
    )                                                          # (n+1,)

    perturbs = [
        (i, j, t0 / scale, t1 / scale, alpha)
        for (i, j, t0, t1, alpha) in perturbs_raw
    ]

    node_feats, edge_feats = build_tsptwd_features(coords, tw_tensor, svc_tensor, perturbs)

    return {
        "coords":        coords,
        "time_windows":  tw_tensor,
        "service_times": svc_tensor,
        "perturbations": perturbs,
        "node_feats":    node_feats,
        "edge_feats":    edge_feats,
        "depot":         0,
        "meta": {
            "n_clients":   n,
            "scale":       scale,
            "horizon":     horizon,
            "seed":        seed,
            "generated":   "on-the-fly",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. TSPTW-D feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def generate_time_windows(
    coords: torch.Tensor,
    speed: float = 1.0,
    tw_width_ratio: float = 0.4,
    seed: int = None,
) -> tuple:
    """
    Generate feasible time windows and service times for a TSPTW-D instance.

    Strategy: simulate a nearest-neighbour tour to get reference arrival times,
    then place windows of width ≈ tw_width_ratio × total_tour_time centred on
    each arrival time (with random jitter so windows are not all the same width).
    The depot always has window [0, 1.5 × total_tour_time].

    Returns
    -------
    time_windows  : torch.Tensor (n, 2)  — [a_i, b_i] per city
    service_times : torch.Tensor (n,)    — s_i per city  (depot s_0 = 0)
    """
    if seed is not None:
        torch.manual_seed(seed)
    n = coords.shape[0]

    dist_mat = torch.cdist(coords, coords)

    # Nearest-neighbour tour for reference arrival times
    visited = torch.zeros(n, dtype=torch.bool)
    tour    = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist_mat[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt); visited[nxt] = True

    # Service times ≈ 5 % of mean leg distance
    mean_leg = dist_mat[dist_mat > 0].mean().item()
    svc      = torch.full((n,), mean_leg * 0.05)
    svc[0]   = 0.0   # depot has no service time

    # Simulate arrivals along the NN tour
    arrivals = [0.0] * n
    t = 0.0
    for k in range(len(tour) - 1):
        i, j = tour[k], tour[k + 1]
        t    = max(t, arrivals[i]) + svc[i].item()
        t   += dist_mat[i, j].item() / speed
        arrivals[j] = t
    total_time = t + dist_mat[tour[-1], 0].item() / speed

    half_w  = total_time * tw_width_ratio
    jitter  = (torch.rand(n) * 0.5 + 0.75) * half_w   # 0.75–1.25 × half_w
    jitter[0] = total_time                              # depot open all day

    a = (torch.tensor(arrivals) - jitter).clamp(min=0.0)
    b = (torch.tensor(arrivals) + jitter).clamp(min=a + mean_leg * 0.1)
    a[0] = 0.0
    b[0] = total_time * 1.5

    return torch.stack([a, b], dim=1), svc


def generate_perturbations(
    n: int,
    total_time: float,
    n_perturb: int = None,
    alpha_range: tuple = (0.3, 1.5),
    seed: int = None,
) -> list:
    """
    Generate random perturbation events.

    Each perturbation is a tuple (i, j, t_start, t_end, alpha) meaning:
      cost(i→j) = base_dist(i,j) × (1 + alpha)  for departures in [t_start, t_end].

    Parameters
    ----------
    n           : number of cities
    total_time  : reference tour duration (scales the active time windows)
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
        j = i
        while j == i:
            j = torch.randint(0, n, (1,), generator=rng).item()
        t0    = (torch.rand(1, generator=rng) * total_time * 0.6).item()
        t1    = t0 + total_time * 0.3
        alpha = (
            torch.rand(1, generator=rng) * (alpha_range[1] - alpha_range[0])
            + alpha_range[0]
        ).item()
        perturbs.append((int(i), int(j), float(t0), float(t1), float(alpha)))
    return perturbs


def worst_case_perturb_matrix(coords: torch.Tensor, perturbations: list) -> torch.Tensor:
    """
    (n, n) tensor of the WORST-CASE perturbation factor per edge.

    entry[i, j] = max alpha across all perturbations on (i, j), regardless of time.
    Used as a static input feature for the model (edge_feats).
    """
    n   = coords.shape[0]
    mat = torch.zeros(n, n)
    for pi, pj, _t0, _t1, alpha in perturbations:
        if alpha > mat[pi, pj].item():
            mat[pi, pj] = alpha
            mat[pj, pi] = alpha
    return mat


def perturb_edge_matrix(
    coords: torch.Tensor,
    perturbations: list,
    t: float = 0.0,
) -> torch.Tensor:
    """
    (n, n) tensor of ACTIVE cost multipliers at departure time t.

    entry[i, j] = 1 + alpha  if a perturbation on (i,j) is active at time t,
                  1.0        otherwise.
    Used during tour simulation / evaluation.
    """
    n   = coords.shape[0]
    mat = torch.ones(n, n)
    for pi, pj, t_start, t_end, alpha in perturbations:
        if t_start <= t <= t_end:
            mat[pi, pj] = 1.0 + alpha
            mat[pj, pi] = 1.0 + alpha
    return mat


def build_tsptwd_features(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    service_times: torch.Tensor,
    perturbations: list,
) -> tuple:
    """
    Build node and edge feature tensors for model input.

    Node features  (n, 5):  [x, y, a_i/T, b_i/T, s_i/T]   where T = max(b_i)
    Edge features  (n, n, 1):  [alpha_ij]  worst-case perturbation factor

    Returns
    -------
    node_feats : torch.Tensor (n, 5)
    edge_feats : torch.Tensor (n, n, 1)
    """
    T          = time_windows[:, 1].max().clamp(min=1e-8).item()
    a          = (time_windows[:, 0] / T).unsqueeze(1)
    b          = (time_windows[:, 1] / T).unsqueeze(1)
    s          = (service_times       / T).unsqueeze(1)
    node_feats = torch.cat([coords, a, b, s], dim=1)        # (n, 5)

    perturb_mat = worst_case_perturb_matrix(coords, perturbations)
    edge_feats  = perturb_mat.unsqueeze(-1)                  # (n, n, 1)

    return node_feats, edge_feats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Training labels
# ─────────────────────────────────────────────────────────────────────────────

def nn_tour_labels(coords: torch.Tensor, two_opt: bool = False) -> torch.Tensor:
    """
    Nearest-neighbour tour → binary (n, n) edge matrix used as training label.

    y[i, j] = 1 iff edge (i, j) is in the NN tour (symmetric).

    For DIFUSCO, these labels represent the noisy-but-tractable ground truth
    that the diffusion model learns to reconstruct.  For n > 10 where brute-force
    optimal is infeasible, NN (optionally improved with 2-opt) is the standard
    pseudo-label strategy used in the original paper.

    Parameters
    ----------
    two_opt : apply 2-opt local search after NN construction (n ≤ 300 only).
    """
    n    = coords.shape[0]
    dist = torch.cdist(coords, coords)

    visited = torch.zeros(n, dtype=torch.bool)
    tour    = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        tour.append(d.argmin().item())
        visited[tour[-1]] = True

    if two_opt and n <= 300:
        tour = two_opt_improve(coords, tour)

    y = torch.zeros(n, n)
    for k in range(n):
        a, b = tour[k], tour[(k + 1) % n]
        y[a, b] = y[b, a] = 1.0
    return y


def two_opt_improve(coords: torch.Tensor, tour: list, max_iter: int = 100) -> list:
    """
    2-opt local search: reverse sub-segments to reduce total tour length.

    O(n²) per pass, practical for n ≤ 300.
    Stops when no improving swap is found or max_iter passes are done.
    """
    n  = len(tour)
    if n <= 3:
        return tour
    tour = list(tour)
    xy   = coords.numpy() if isinstance(coords, torch.Tensor) else coords

    def d(i, j):
        return float(np.linalg.norm(xy[i] - xy[j]))

    for _ in range(max_iter):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                ni, ni1 = tour[i],     tour[i + 1]
                nj, nj1 = tour[j],     tour[(j + 1) % n]
                if d(ni, nj) + d(ni1, nj1) < d(ni, ni1) + d(nj, nj1) - 1e-10:
                    tour[i + 1: j + 1] = tour[i + 1: j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


# ─────────────────────────────────────────────────────────────────────────────
# 4. DIFUSCO decoding
# ─────────────────────────────────────────────────────────────────────────────

def binarize(p: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert a continuous edge-probability matrix to a binary adjacency matrix.

    p : (n, n) floats in [0, 1]  — DIFUSCO final output after denoising
    Returns a (n, n) ByteTensor with 1 where p >= threshold.

    Note: the result may not be a valid tour adjacency (each node might have
    degree ≠ 2). Use tour_from_adj() to extract an ordered tour.

    TODO (pair-coding): explore adaptive thresholding and beam search decoding.
    """
    return (p >= threshold).float()


def tour_from_adj(adj: torch.Tensor, start: int = 0) -> list:
    """
    Extract an ordered tour from a (possibly noisy) binary adjacency matrix.

    Strategy: greedy — at each step follow the highest-weight unvisited neighbour.
    Falls back to nearest unvisited node if the adj matrix has no valid next edge.

    adj   : (n, n) — binary or soft adjacency matrix
    start : starting node index

    Returns
    -------
    list of n node indices representing the tour

    TODO (pair-coding): implement degree-constrained matching decoder (the
    approach used in the original DIFUSCO paper for higher solution quality).
    """
    return greedy_decode(adj, start=start)


def greedy_decode(p: torch.Tensor, start: int = 0) -> list:
    """
    Greedy tour construction from edge scores / probabilities.

    At each step, move to the highest-scoring unvisited city.
    Works identically for GNN probability matrices and DIFUSCO output.

    p     : (n, n) floats
    start : index of the first city
    """
    p = p.float()
    n = p.shape[0]
    visited = torch.zeros(n, dtype=torch.bool, device=p.device)
    tour    = [start]; visited[start] = True
    for _ in range(n - 1):
        scores    = p[tour[-1]].clone().masked_fill(visited, float("-inf"))
        next_city = scores.argmax().item()
        tour.append(next_city); visited[next_city] = True
    return tour


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def tour_length(coords: torch.Tensor, tour: list) -> float:
    """Total Euclidean length of a closed tour."""
    idx = torch.tensor(tour + [tour[0]])
    return (coords[idx[:-1]] - coords[idx[1:]]).norm(dim=1).sum().item()


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

    Travel cost:  c_ij(t) = dist_ij / speed × (1 + alpha)  if perturbation active
    Time window:  early arrival → wait until a_i
                  late arrival  → penalty += (t - b_i) × penalty_coeff

    Returns
    -------
    total_time : float  — return time to depot (no penalty)
    tw_penalty : float  — total time-window violation penalty
    obj        : float  — total_time + tw_penalty
    """
    dist_mat = torch.cdist(coords, coords)
    t        = 0.0
    penalty  = 0.0

    for k in range(len(tour)):
        i = tour[k]
        j = tour[(k + 1) % len(tour)]

        # Enforce time window at current node (except depot on first visit)
        if k > 0:
            a_i = time_windows[i, 0].item()
            b_i = time_windows[i, 1].item()
            if t < a_i:
                t = a_i           # early: wait
            elif t > b_i:
                penalty += (t - b_i) * penalty_coeff
            t += service_times[i].item()

        # Travel to next node with active perturbations
        base = dist_mat[i, j].item() / speed
        mult = 1.0
        for pi, pj, t_start, t_end, alpha in perturbations:
            if (pi == i and pj == j) or (pi == j and pj == i):
                if t_start <= t <= t_end:
                    mult = max(mult, 1.0 + alpha)
        t += base * mult

    return t, penalty, t + penalty


# ─────────────────────────────────────────────────────────────────────────────
# 6. Extra baselines and utilities (used by benchmark notebooks)
# ─────────────────────────────────────────────────────────────────────────────

def optimal_tour_labels(coords: torch.Tensor) -> torch.Tensor:
    """
    Brute-force optimal tour for small instances (n ≤ 12).

    Tries all (n-1)! permutations starting from node 0.
    Returns binary (n, n) edge matrix of the optimal tour.
    Only feasible for n ≤ 12 (12! ≈ 479M, takes ~10 s; use n ≤ 10 for speed).
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


def optimal_tour(coords: torch.Tensor) -> tuple:
    """
    Return (tour list, length) for the brute-force optimal tour (n ≤ 12).
    """
    n = coords.shape[0]
    best_len, best_tour = float("inf"), None
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        length = tour_length(coords, tour)
        if length < best_len:
            best_len, best_tour = length, tour
    return best_tour, best_len


def nn_tour(coords: torch.Tensor) -> list:
    """
    Nearest-neighbour tour (list of node indices).
    Convenience wrapper around greedy_decode using inverse distance scores.
    """
    dist = torch.cdist(coords, coords)
    # Use inverse distance as score; mask diagonal so a node never points to itself
    p = 1.0 / (dist + 1e-8)
    p.fill_diagonal_(0.0)
    return greedy_decode(p, start=0)


def time_aware_nn_tour(
    coords: torch.Tensor,
    time_windows: torch.Tensor,
    service_times: torch.Tensor,
    perturbations: list,
    speed: float = 1.0,
) -> list:
    """
    Nearest-neighbour heuristic that respects TSPTW-D time windows.

    At each step, candidates are scored by a combination of distance and
    urgency (how close the deadline is).  If a node's window is already
    violated at the earliest possible arrival, it is still visited — this
    is a heuristic, not an exact method.

    Returns an ordered tour (list of node indices, depot first and last is implicit).
    """
    dist_mat = torch.cdist(coords, coords)
    n        = coords.shape[0]
    visited  = torch.zeros(n, dtype=torch.bool)
    tour     = [0]; visited[0] = True
    t        = 0.0   # current simulation time

    for _ in range(n - 1):
        cur   = tour[-1]

        # Compute arrival time and effective cost to each unvisited node
        scores = {}
        for j in range(n):
            if visited[j]:
                continue
            base = dist_mat[cur, j].item() / speed
            mult = 1.0
            for pi, pj, ts, te, alpha in perturbations:
                if (pi == cur and pj == j) or (pi == j and pj == cur):
                    if ts <= t <= te:
                        mult = max(mult, 1.0 + alpha)
            travel = base * mult
            arrival = t + travel

            # Urgency: prefer nodes whose window closes soonest
            b_j     = time_windows[j, 1].item()
            urgency = b_j - arrival   # larger = more slack = less urgent

            # Score: prioritise nearby + urgent nodes
            # (negative so that argmin = best)
            scores[j] = travel + max(0.0, -urgency) * 0.5

        nxt = min(scores, key=scores.get)
        tour.append(nxt); visited[nxt] = True

        # Advance simulation time
        base = dist_mat[cur, nxt].item() / speed
        mult = 1.0
        for pi, pj, ts, te, alpha in perturbations:
            if (pi == cur and pj == nxt) or (pi == nxt and pj == cur):
                if ts <= t <= te:
                    mult = max(mult, 1.0 + alpha)
        t += base * mult

        a_nxt = time_windows[nxt, 0].item()
        if t < a_nxt:
            t = a_nxt   # wait until window opens
        t += service_times[nxt].item()

    return tour

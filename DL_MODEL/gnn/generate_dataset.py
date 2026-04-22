"""
generate_dataset.py — TSPTW-D training dataset with OR-Tools labels.

Strategy
--------
Each instance is solved with OR-Tools routing + Guided Local Search under a
per-size time budget.  Only feasible solutions are kept.

Travel-time model for feasibility:
  - Pessimistic: edge cost = dist(i,j) × (1 + max_alpha_ij) where max_alpha_ij
    is the worst-case perturbation severity on that arc.
  - This produces routes that are robust to the worst observed disruption.

Objective minimised by OR-Tools:
  - Euclidean distance (unperturbed), so the model learns short tours that
    also satisfy time windows.

Output
------
  datasets/train/tsptwd_ortools_n{N}.json   (one file per size)

The format is identical to the existing tsptwd_train_n*.json files so the
existing _load_tsptwd_json_pool loader picks them up without changes.
scale=1.0 (all values already in [0,1] coordinate units).

Install
-------
  pip install ortools

Usage
-----
  python generate_dataset.py                     # all default sizes
  python generate_dataset.py --sizes 50 100 200  # specific sizes only
  python generate_dataset.py --n 75 --count 2000 # single size
  python generate_dataset.py --workers 4
"""

import argparse
import json
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


# (n, count, or_tools_time_limit_s)
DEFAULT_SIZES = [
    (20,  3000,  5),
    (30,  3000,  5),
    (40,  2500,  8),
    (50,  2500,  8),
    (60,  2000, 12),
    (75,  2000, 15),
    (100, 1500, 20),
    (125, 1200, 25),
    (150, 1000, 30),
    (175,  800, 40),
    (200,  700, 45),
]


# ── Pure-numpy helpers (no torch in workers) ──────────────────────────────────

def _dist_matrix(xy: np.ndarray) -> np.ndarray:
    diff = xy[:, None] - xy[None, :]
    return np.sqrt((diff ** 2).sum(-1))


def _generate_tw(xy: np.ndarray, rng: np.random.Generator,
                 tw_width_ratio: float = 0.4) -> tuple:
    """Generate feasible time windows. Returns (tw (n,2), svc (n,), total_time)."""
    n = len(xy)
    dist = _dist_matrix(xy)

    # NN tour for reference arrival times
    visited = np.zeros(n, dtype=bool)
    tour = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].copy()
        d[visited] = np.inf
        nxt = int(d.argmin()); tour.append(nxt); visited[nxt] = True

    mean_leg = dist[dist > 0].mean()
    svc = np.full(n, mean_leg * 0.05); svc[0] = 0.0

    arrivals = np.zeros(n)
    t = 0.0
    for k in range(n - 1):
        i, j = tour[k], tour[k + 1]
        t = max(t, arrivals[i]) + svc[i] + dist[i, j]
        arrivals[j] = t
    total_time = t + dist[tour[-1], 0]

    half_w = total_time * tw_width_ratio
    jitter = (rng.random(n) * 0.5 + 0.75) * half_w; jitter[0] = total_time
    a = np.maximum(arrivals - jitter, 0.0)
    b = np.maximum(arrivals + jitter, a + mean_leg * 0.1)
    a[0] = 0.0; b[0] = total_time * 1.5

    return np.stack([a, b], axis=1).astype(np.float64), svc.astype(np.float64), float(total_time)


def _generate_perturbs(n: int, total_time: float, rng: np.random.Generator,
                       alpha_range: tuple = (0.3, 1.5)) -> list:
    n_perturb = max(1, n // 10)
    out = []
    for _ in range(n_perturb):
        i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        t0 = float(rng.random() * total_time * 0.6)
        t1 = t0 + total_time * 0.3
        alpha = float(rng.random() * (alpha_range[1] - alpha_range[0]) + alpha_range[0])
        out.append((i, j, t0, t1, alpha))
    return out


# ── OR-Tools solver ───────────────────────────────────────────────────────────

def _solve_ortools(xy: np.ndarray, tw: np.ndarray, svc: np.ndarray,
                   perturbs: list, time_limit_s: int) -> list | None:
    """
    Solve TSPTW with OR-Tools routing.
    Objective  : minimise Euclidean distance.
    Feasibility: time windows with pessimistic (worst-case) perturbation times.
    Returns tour (list of node indices, starting at 0) or None if infeasible.
    """
    from ortools.constraint_solver import routing_enums_pb2, pywrapcp

    n    = len(xy)
    dist = _dist_matrix(xy)

    # Pessimistic effective distances
    alpha_mat = np.zeros((n, n))
    for pi, pj, _, _, alpha in perturbs:
        if alpha > alpha_mat[pi, pj]:
            alpha_mat[pi, pj] = alpha_mat[pj, pi] = alpha
    eff_dist = dist * (1.0 + alpha_mat)

    INT  = 1_000_000                                          # float → int scale
    d_i  = np.round(dist     * INT).astype(np.int64)
    e_i  = np.round(eff_dist * INT).astype(np.int64)
    tw_i = np.round(tw       * INT).astype(np.int64)
    sv_i = np.round(svc      * INT).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # Cost: minimise Euclidean distance
    def _dist_cb(fi, ti):
        return int(d_i[manager.IndexToNode(fi), manager.IndexToNode(ti)])
    routing.SetArcCostEvaluatorOfAllVehicles(
        routing.RegisterTransitCallback(_dist_cb)
    )

    # Time dimension: effective travel + service at origin node
    def _time_cb(fi, ti):
        i = manager.IndexToNode(fi)
        j = manager.IndexToNode(ti)
        return int(sv_i[i] + e_i[i, j])
    horizon = int(tw_i[:, 1].max()) + int(sv_i.sum()) + 1
    routing.AddDimension(
        routing.RegisterTransitCallback(_time_cb),
        horizon, horizon, False, "Time",
    )
    td = routing.GetDimensionOrDie("Time")
    for i in range(n):
        td.CumulVar(manager.NodeToIndex(i)).SetRange(
            int(tw_i[i, 0]), int(tw_i[i, 1])
        )

    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    sp.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    sp.time_limit.seconds = time_limit_s
    sp.log_search = False

    sol = routing.SolveWithParameters(sp)
    if sol is None:
        return None

    tour, idx = [], routing.Start(0)
    while not routing.IsEnd(idx):
        tour.append(manager.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    return tour


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _worker(args: tuple) -> dict | None:
    seed, n, time_limit = args
    rng = np.random.default_rng(seed)
    xy  = rng.random((n, 2)).astype(np.float32)

    try:
        tw, svc, total_time = _generate_tw(xy.astype(np.float64), rng)
        perturbs = _generate_perturbs(n, total_time, rng)
        tour = _solve_ortools(xy.astype(np.float64), tw, svc, perturbs, time_limit)
    except Exception:
        return None

    if tour is None:
        return None

    return {
        "seed":       seed,
        "n":          n,
        "xy":         xy.tolist(),
        "tw":         tw.tolist(),
        "svc":        svc.tolist(),
        "perturbs":   perturbs,
        "tour":       tour,
        "horizon":    float(tw[:, 1].max()),
    }


# ── JSON serialisation ────────────────────────────────────────────────────────

def _to_json_record(inst: dict) -> dict:
    """Convert solved instance to the existing tsptwd_train JSON format (scale=1)."""
    xy, tw, svc = inst["xy"], inst["tw"], inst["svc"]
    depot = {
        "x": xy[0][0], "y": xy[0][1],
        "a": tw[0][0], "b": None,       # depot deadline = full horizon
        "service": svc[0],
    }
    clients = [
        {"x": xy[i][0], "y": xy[i][1],
         "a": tw[i][0], "b": tw[i][1], "service": svc[i]}
        for i in range(1, inst["n"])
    ]
    perturbations = [
        {"arc": [p[0], p[1]], "t_start": p[2], "t_end": p[3], "alpha": p[4]}
        for p in inst["perturbs"]
    ]
    return {
        "seed":          inst["seed"],
        "depot":         depot,
        "clients":       clients,
        "perturbations": perturbations,
        "tour":          inst["tour"],
    }


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_size(n: int, count: int, time_limit: int,
                  out_dir: str, workers: int) -> int:
    """Generate `count` instances of size n, return number actually saved."""
    seeds = list(range(n * 100_000, n * 100_000 + count * 4))  # extra seeds for rejects

    records  = []
    rejected = 0
    t0       = time.time()

    ctx  = mp.get_context("spawn")
    args = [(s, n, time_limit) for s in seeds]

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, a): a for a in args[:count + count // 2]}
        pbar = tqdm(total=count, desc=f"n={n:>3}", unit="inst", leave=False)

        for fut in as_completed(futures):
            if len(records) >= count:
                break
            result = fut.result()
            if result is None:
                rejected += 1
            else:
                records.append(_to_json_record(result))
                pbar.update(1)
        pbar.close()

    if not records:
        print(f"  n={n}: 0 feasible instances — skipping.")
        return 0

    horizon = max(
        max(c["b"] for c in r["clients"]) for r in records
    )
    out = {
        "meta": {
            "n_clients":   n - 1,
            "n_instances": len(records),
            "scale":       1.0,
            "horizon":     float(horizon),
            "solver":      "ortools_gls",
            "time_limit_s": time_limit,
        },
        "instances": records,
    }
    path = os.path.join(out_dir, f"tsptwd_ortools_n{n}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))

    elapsed = time.time() - t0
    rate    = len(records) / elapsed
    print(f"  n={n:>3}: {len(records):>5} saved  "
          f"({rejected} rejected)  {elapsed/60:.1f} min  {rate:.1f} inst/s  → {path}")
    return len(records)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",   type=int, nargs="+", default=None,
                        help="Subset of n values to generate (default: all)")
    parser.add_argument("--n",       type=int, default=None,
                        help="Single size override")
    parser.add_argument("--count",   type=int, default=None,
                        help="Instance count override (used with --n)")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers (default: all cores - 1)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: datasets/train/)")
    args = parser.parse_args()

    try:
        from ortools.constraint_solver import pywrapcp  # noqa: F401
    except ImportError:
        raise SystemExit("OR-Tools not installed. Run: pip install ortools")

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "datasets", "train"
    )

    if args.n is not None:
        # Single-size run
        size_plan = [(args.n, args.count or 2000, max(5, args.n // 5))]
    elif args.sizes is not None:
        size_plan = [(n, c, t) for n, c, t in DEFAULT_SIZES if n in args.sizes]
    else:
        size_plan = DEFAULT_SIZES

    print(f"Workers: {workers}  |  Output: {out_dir}")
    print(f"Sizes: {[n for n, _, _ in size_plan]}\n")

    total = 0
    for n, count, time_limit in size_plan:
        total += generate_size(n, count, time_limit, out_dir, workers)

    print(f"\nDone. {total:,} instances total.")

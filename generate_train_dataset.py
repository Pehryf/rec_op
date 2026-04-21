"""
generate_train_dataset.py — Generate a multi-instance TSPTW-D training dataset.

The benchmark datasets in datasets/ contain one instance per size (seed=42) and
are used by everyone to compare model performance — they must never be modified.
This script generates a separate training pool: many diverse instances per size,
written to datasets/train/ which is excluded from git.

Output format (datasets/train/tsptwd_train_n{N}.json)
------------------------------------------------------
{
  "meta": { "n_clients": int, "n_instances": int, "scale": float, "horizon": float },
  "instances": [
    {
      "seed": int,
      "depot":  { "x", "y", "a", "b", "service" },
      "clients": [ { "x", "y", "a", "b", "service" }, ... ],
      "perturbations": [ { "arc", "t_start", "t_end", "alpha" }, ... ]
    },
    ...
  ]
}

train.py reads this format when --source tsptwd_json is used: it randomly picks
one instance per training step, so the model sees a different graph every time.

Usage
-----
  python generate_train_dataset.py
  python generate_train_dataset.py --sizes 10 20 50 100 200 --n_instances 1000
  python generate_train_dataset.py --out_dir datasets/train --seed 0 --scale 200
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

# Allow importing data.py from DL_MODEL/gnn/ when --nn2opt is requested
_ROOT    = Path(__file__).parent
_GNN_DIR = _ROOT / "DL_MODEL" / "gnn"
_DEFAULT_OUT = _ROOT / "datasets" / "train"


# ---------------------------------------------------------------------------
# Core generation (mirrors the logic in datasetsgenerator.ipynb)
# ---------------------------------------------------------------------------

def _distance_minutes(x1, y1, x2, y2, scale: float) -> float:
    return math.hypot(x1 - x2, y1 - y2) * scale


def _generate_instance(
    n_clients: int,
    *,
    scale: float = 200.0,
    horizon: float = 1440.0,
    service_min: float = 5.0,
    service_max: float = 15.0,
    tw_width_min: float = 120.0,
    tw_width_max: float = 240.0,
    n_perturbations: int = None,
    alpha_min: float = 1.5,
    alpha_max: float = 3.5,
    seed: int = 0,
) -> dict:
    """Generate one TSPTW-D instance; returns the instance dict (no file I/O)."""
    rng = random.Random(seed)

    if n_perturbations is None:
        n_perturbations = max(1, n_clients // 5)

    # Depot
    depot = {"x": rng.uniform(0, 1), "y": rng.uniform(0, 1), "a": 0.0, "b": None, "service": 0.0}

    # Clients — time windows generated independently (same logic as benchmark generator)
    clients = []
    for _ in range(n_clients):
        s_i = round(rng.uniform(service_min, service_max), 1)
        a_i = round(rng.uniform(0.0, horizon - tw_width_min), 1)
        width = round(rng.uniform(tw_width_min, tw_width_max), 1)
        b_i = min(round(a_i + width, 1), horizon)
        if b_i - a_i < tw_width_min:
            b_i = min(round(a_i + tw_width_min, 1), horizon)
        clients.append({"x": rng.uniform(0, 1), "y": rng.uniform(0, 1),
                        "a": a_i, "b": b_i, "service": s_i})

    # Perturbations — random arcs, random active intervals
    n_nodes = n_clients + 1
    all_arcs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    chosen = rng.sample(all_arcs, min(n_perturbations, len(all_arcs)))
    perturbations = []
    for arc in chosen:
        t_start = round(rng.uniform(0, horizon * 0.6), 1)
        duration = round(rng.uniform(30, horizon * 0.3), 1)
        t_end = min(round(t_start + duration, 1), horizon)
        alpha = round(rng.uniform(alpha_min, alpha_max), 2)
        perturbations.append({"arc": list(arc), "t_start": t_start, "t_end": t_end, "alpha": alpha})

    return {"seed": seed, "depot": depot, "clients": clients, "perturbations": perturbations}


def _compute_nn2opt_tour(inst: dict, n_clients: int, scale: float, horizon: float) -> list:
    """Pre-compute TW-aware nn2opt tour and return as list of node indices."""
    import torch
    sys.path.insert(0, str(_GNN_DIR))
    from data import tsptwd_nn2opt_tour, _tsptwd_greedy_tour

    depot = inst["depot"]
    clients = inst["clients"]
    all_nodes = [depot] + clients
    coords = torch.tensor([[nd["x"], nd["y"]] for nd in all_nodes], dtype=torch.float32)
    tw = torch.tensor([[nd["a"], nd["b"] if nd["b"] is not None else horizon]
                       for nd in all_nodes], dtype=torch.float32)
    svc = torch.tensor([nd["service"] for nd in all_nodes], dtype=torch.float32)
    perturbs = [(p["arc"][0], p["arc"][1], p["t_start"], p["t_end"], p["alpha"])
                for p in inst["perturbations"]]

    # Scale coords to minutes-distance space
    coords_scaled = coords * scale

    if n_clients <= 100:
        tour = tsptwd_nn2opt_tour(coords_scaled, tw, svc, perturbs, max_passes=3)
    else:
        tour = _tsptwd_greedy_tour(coords_scaled, tw, svc, perturbs)
    return tour


def generate_train_pool(
    n_clients: int,
    n_instances: int,
    *,
    out_dir: str | Path = None,
    scale: float = 200.0,
    horizon: float = 1440.0,
    n_perturbations: int = None,
    base_seed: int = 0,
    nn2opt: bool = False,
    **kwargs,
) -> Path:
    """
    Generate *n_instances* diverse TSPTW-D instances for *n_clients* and write
    them as a single JSON file to *out_dir*/tsptwd_train_n{n_clients}.json.

    Each instance uses seed = base_seed + i so the pool is fully reproducible
    and independent of the benchmark datasets (which all use seed=42).

    If nn2opt=True, pre-computes a TW-aware nn2opt tour (n≤100) or greedy tour
    (n>100) and stores it as "tour" field for use as training labels.
    """
    out_dir = Path(out_dir) if out_dir is not None else _DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_perturbations is None:
        n_perturbations = max(1, n_clients // 5)

    instances = []
    for i in range(n_instances):
        inst = _generate_instance(
            n_clients,
            scale=scale,
            horizon=horizon,
            n_perturbations=n_perturbations,
            seed=base_seed + i,
            **kwargs,
        )
        if nn2opt:
            inst["tour"] = _compute_nn2opt_tour(inst, n_clients, scale, horizon)
        instances.append(inst)

    out_path = out_dir / f"tsptwd_train_n{n_clients}.json"
    payload = {
        "meta": {
            "n_clients": n_clients,
            "n_instances": n_instances,
            "scale": scale,
            "horizon": horizon,
        },
        "instances": instances,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    print(f"  n={n_clients:>5}  {n_instances} instances  → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SIZES = [10, 20, 50, 100, 200, 300, 500]
DEFAULT_N_INSTANCES = {10: 2000, 20: 2000, 50: 1000, 100: 500, 200: 200, 300: 100, 500: 50}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSPTW-D training dataset pool")
    parser.add_argument("--sizes",       type=int, nargs="+", default=DEFAULT_SIZES,
                        help="Problem sizes (n_clients) to generate, e.g. 10 50 100")
    parser.add_argument("--n_instances", type=int, default=None,
                        help="Instances per size (default: size-dependent, see DEFAULT_N_INSTANCES)")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help=f"Output directory (default: {_DEFAULT_OUT})")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Base seed; instance i uses seed+i (0 avoids collision with benchmark seed=42)")
    parser.add_argument("--scale",       type=float, default=200.0)
    parser.add_argument("--horizon",     type=float, default=1440.0)
    parser.add_argument("--nn2opt",      action="store_true",
                        help="Pre-compute TW-aware nn2opt tour (n≤100) or greedy tour (n>100) "
                             "and embed as 'tour' field. Recommended for n≤100 only.")
    args = parser.parse_args()

    print(f"Generating training pool → {args.out_dir}/")
    print(f"Sizes: {args.sizes}  nn2opt={args.nn2opt}")
    for n in args.sizes:
        count = args.n_instances if args.n_instances is not None else DEFAULT_N_INSTANCES.get(n, 500)
        generate_train_pool(
            n,
            count,
            out_dir=args.out_dir,
            scale=args.scale,
            horizon=args.horizon,
            base_seed=args.seed,
            nn2opt=args.nn2opt,
        )
    print("Done.")

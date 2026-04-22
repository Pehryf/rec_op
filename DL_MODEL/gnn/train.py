"""
train.py — Supervised training loop for TSPGNN

Key design decisions
--------------------
1. Weighted BCE (pos_weight = n_neg/n_pos): tour edges are ~1/n of all
   possible edges, so plain BCE collapses to constant ~0 predictions.
   Weighting forces equal gradient contribution from both classes.

2. Symmetrised predictions (p_sym = (p + p.T) / 2): the tour graph is
   undirected, labels are symmetric — enforcing this halves gradient noise.

3. Labels computed on-the-fly every step: no pre-built cache, no async
   device transfers, no stale data issues.

4. Constant LR with Adam: simple and reliable.  Add a scheduler later if
   you observe oscillation near convergence.

Label strategies
----------------
  auto     → optimal (n ≤ 10), nn otherwise
  optimal  → brute-force exact tour    (n ≤ 10 only)
  nn       → nearest-neighbour tour    (any n)
  nn2opt   → NN + 2-opt improvement    (n ≤ 300)

Device
------
Automatically picks CUDA > XPU > MPS > CPU.
Override with --device cpu|cuda|xpu|mps.

Sources
-------
  random        → random instances on-the-fly (default)
  tsp           → cities subsampled from a TSP city pool
  tsptwd_json   → TSPTW-D instances loaded from datasets/tsptwd_n*.json
                  with random axis-aligned augmentation; falls back to
                  random generation for sizes without a matching JSON file.

Usage
-----
  python train.py --size small --n 8 --steps 3000
  python train.py --size medium --n 50 --label nn2opt --steps 5000 --source tsp
  python train.py --resume model/gnn.pt --size medium --n_min 10 --n_max 100 --label nn2opt
  python train.py --size small --mode tsptwd --resume model/gnn_small_tsptwd.pt \\
                  --n 50 --label nn2opt --steps 3000 --source tsptwd_json
"""

import argparse
import glob as _glob
import json as _json
import os
import random as _random
import re as _re

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import (load_cities, random_instance,
                  optimal_tour_labels, nn_tour_labels, tsptwd_nn_tour_labels,
                  tsptwd_nn2opt_tour_labels,
                  greedy_decode, tour_length,
                  generate_time_windows, generate_perturbations,
                  build_tsptwd_features, evaluate_tsptwd)
from model import TSPGNN, MODEL_SIZES

POOL_SIZE = 1000


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tour_to_label_matrix(tour: list, n: int) -> torch.Tensor:
    """Convert a tour (list of node indices) to a binary (n,n) edge matrix."""
    y = torch.zeros(n, n)
    for k in range(len(tour)):
        a, b = tour[k], tour[(k + 1) % len(tour)]
        y[a, b] = y[b, a] = 1.0
    return y


def _make_labels(
    coords_cpu: torch.Tensor,
    label: str,
    time_windows: torch.Tensor = None,
    service_times: torch.Tensor = None,
    perturbations: list = None,
    stored_tour: list = None,
) -> torch.Tensor:
    """Compute tour labels on CPU, return float32 tensor.

    When time_windows/service_times/perturbations are provided (TSPTWD mode)
    the label respects the `label` argument:
      - stored_tour present  → use pre-computed tour from JSON (fastest)
      - label == "nn2opt"    → TW-aware NN + 2-opt
      - otherwise            → TW-aware NN
    """
    tsptwd = time_windows is not None
    n = coords_cpu.shape[0]

    if tsptwd:
        if stored_tour is not None:
            return _tour_to_label_matrix(stored_tour, n)
        if label == "nn2opt":
            return tsptwd_nn2opt_tour_labels(
                coords_cpu, time_windows, service_times, perturbations or []
            )
        return tsptwd_nn_tour_labels(
            coords_cpu, time_windows, service_times, perturbations or []
        )

    use = "optimal" if (label == "auto" and n <= 10) else \
          ("nn"     if  label == "auto"             else label)
    if use == "optimal":
        assert n <= 10, f"Brute-force labels only for n≤10, got {n}"
        return optimal_tour_labels(coords_cpu)
    if use == "nn2opt":
        return nn_tour_labels(coords_cpu, two_opt=True)
    return nn_tour_labels(coords_cpu)


def _nn_tour(coords: torch.Tensor) -> list:
    """Simple nearest-neighbour tour for validation reference."""
    n = coords.shape[0]
    dist = torch.cdist(coords, coords)
    visited = torch.zeros(n, dtype=torch.bool)
    tour = [0]; visited[0] = True
    for _ in range(n - 1):
        d = dist[tour[-1]].clone()
        d[visited] = float("inf")
        nxt = d.argmin().item()
        tour.append(nxt); visited[nxt] = True
    return tour


def _load_tsptwd_json_pool(dataset_dir: str) -> dict:
    """
    Scan *dataset_dir* for tsptwd_n*.json files and load them into a pool.

    Supports two formats:
      Single-instance (benchmark datasets, datasets/tsptwd_n*.json):
        Top-level keys: meta, depot, clients, perturbations.
        Loaded as a one-element list.

      Multi-instance (training pool, datasets/train/tsptwd_train_n*.json):
        Top-level keys: meta, instances (list of {seed, depot, clients, perturbations}).
        Loaded as a list of instances, picked randomly during training.

    Returns
    -------
    dict mapping n (int) → list of {coords, tw, svc, perturbs} dicts.
    All times are divided by scale to match the [0,1]-normalised coordinate space.
    """
    pool = {}
    pattern = os.path.join(dataset_dir, 'tsptwd_*n*.json')
    for path in sorted(_glob.glob(pattern)):
        m = _re.search(r'n(\d+)\.json$', path)
        if not m:
            continue
        n = int(m.group(1))
        with open(path, encoding='utf-8') as fh:
            data = _json.load(fh)

        scale   = float(data['meta']['scale'])
        horizon = float(data['meta']['horizon'])

        def _b(v):
            return float(v['b']) / scale if v['b'] is not None else horizon / scale

        def _parse(depot, clients, perturbations, tour=None):
            nodes = [depot] + clients
            entry = {
                'coords': torch.tensor([[v['x'], v['y']] for v in nodes],
                                       dtype=torch.float32),
                'tw':     torch.tensor([[float(v['a']) / scale, _b(v)] for v in nodes],
                                       dtype=torch.float32),
                'svc':    torch.tensor([float(v['service']) / scale for v in nodes],
                                       dtype=torch.float32),
                'perturbs': [
                    (int(p['arc'][0]), int(p['arc'][1]),
                     float(p['t_start']) / scale, float(p['t_end']) / scale,
                     float(p['alpha']))
                    for p in perturbations
                ],
                'tour': tour,  # list of ints or None
            }
            return entry

        if 'instances' in data:
            # Multi-instance training format
            pool[n] = [
                _parse(inst['depot'], inst['clients'], inst.get('perturbations', []),
                       inst.get('tour'))
                for inst in data['instances']
            ]
        else:
            # Single-instance benchmark format
            pool[n] = [_parse(data['depot'], data['clients'], data.get('perturbations', []),
                               data.get('tour'))]

    return pool


def _augment_coords(coords: torch.Tensor) -> torch.Tensor:
    """Random axis-aligned flip — preserves pairwise distances up to reflection."""
    coords = coords.clone()
    if _random.random() < 0.5:
        coords[:, 0] = 1.0 - coords[:, 0]
    if _random.random() < 0.5:
        coords[:, 1] = 1.0 - coords[:, 1]
    return coords


def _sample_tsptwd_instance(n, city_pool, n_perturb, json_pool=None):
    """
    Sample one TSPTW-D instance.

    Priority (when mode=tsptwd_json):
      1. JSON pool entry matching n  (with random axis flip augmentation)
      2. City pool subsample         (with randomly generated TW)
      3. Fully random instance

    Returns (node_feats_cpu, edge_feats_cpu, coords_cpu, tw_cpu, svc_cpu, perturbs, stored_tour).
    stored_tour is a list of ints (pre-computed tour) or None.
    """
    if json_pool is not None and n in json_pool:
        inst   = _random.choice(json_pool[n])   # random pick from pool
        coords = _augment_coords(inst['coords'])
        tw, svc, perturbs = inst['tw'], inst['svc'], inst['perturbs']
        stored_tour = inst.get('tour')
        nf, ef = build_tsptwd_features(coords, tw, svc, perturbs)
        return nf, ef, coords, tw, svc, perturbs, stored_tour

    if city_pool is not None:
        coords = city_pool[torch.randperm(city_pool.shape[0])[:n]]
    else:
        coords = random_instance(n)
    tw, svc = generate_time_windows(coords)
    total_time = tw[:, 1].max().item()
    perturbs = generate_perturbations(n, total_time=total_time, n_perturb=n_perturb)
    node_feats, edge_feats = build_tsptwd_features(coords, tw, svc, perturbs)
    return node_feats, edge_feats, coords, tw, svc, perturbs, None


def train(model: TSPGNN,
          n_nodes: int = 8,
          n_steps: int = 2000,
          lr: float = 1e-3,
          city_pool: torch.Tensor = None,
          json_pool: dict = None,
          label: str = "auto",
          n_min: int = None,
          n_max: int = None,
          device: torch.device = torch.device("cpu"),
          val_interval: int = 500,
          mode: str = "tsp") -> list:
    """
    Minimal supervised training loop.

    Parameters
    ----------
    n_nodes      : fixed instance size (ignored when n_min/n_max set)
    label        : 'auto' | 'optimal' | 'nn' | 'nn2opt'
    n_min/n_max  : random size per step in [n_min, n_max]
    city_pool    : (N,2) CPU tensor; if given, cities are subsampled from it
    json_pool    : dict {n → instance} from _load_tsptwd_json_pool(); used
                   when source='tsptwd_json' to train on real dataset instances
    val_interval : log greedy-tour gap vs NN every N steps (0 = off)
    mode         : 'tsp' (default) or 'tsptwd' — adds time-window + perturbation
                   features; model must have node_dim=5, edge_dim=2.
    """
    tsptwd = mode == "tsptwd"
    n_perturb = None   # auto: n//10 per instance

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed validation instances (built once, always on CPU)
    val_set = []
    if val_interval > 0:
        for _ in range(30):
            n_v = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
            if tsptwd:
                nf, ef, c, _tw, _svc, _perturbs, _stored = _sample_tsptwd_instance(
                    n_v, city_pool, n_perturb, json_pool
                )
                val_set.append((nf, ef, c, tour_length(c, _nn_tour(c))))
            else:
                if city_pool is not None:
                    c = city_pool[torch.randperm(city_pool.shape[0])[:n_v]]
                else:
                    c = random_instance(n_v)
                val_set.append((c, None, c, tour_length(c, _nn_tour(c))))

    losses   = []
    best_gap = float("inf")

    bar = tqdm(range(n_steps), desc=f"Training [{device}] mode={mode}",
               unit="step", dynamic_ncols=True)
    for step, _ in enumerate(bar):

        # ── Sample instance ───────────────────────────────────────────────────
        n = _random.randint(n_min, n_max) if (n_min and n_max) else n_nodes
        if tsptwd:
            node_feats_cpu, edge_feats_cpu, coords_cpu, tw_cpu, svc_cpu, perturbs_cpu, stored_tour = \
                _sample_tsptwd_instance(n, city_pool, n_perturb, json_pool)
            y      = _make_labels(coords_cpu, label, tw_cpu, svc_cpu, perturbs_cpu,
                                  stored_tour).to(device)
            x_dev  = node_feats_cpu.to(device)
            e_dev  = edge_feats_cpu.to(device)
        else:
            if city_pool is not None:
                coords_cpu = city_pool[torch.randperm(city_pool.shape[0])[:n]]
            else:
                coords_cpu = random_instance(n)
            y      = _make_labels(coords_cpu, label).to(device)
            x_dev  = coords_cpu.to(device)
            e_dev  = None

        # ── Forward ───────────────────────────────────────────────────────────
        p_hat = model(x_dev, e_dev)
        # Symmetrise: tour is undirected, y[i,j] == y[j,i]
        p_sym = (p_hat + p_hat.T) / 2

        # ── Weighted BCE ──────────────────────────────────────────────────────
        # pos_weight = n_neg/n_pos so both classes contribute equally.
        # Without this the model collapses to constant ~0 predictions.
        n_pos  = y.sum().clamp(min=1.0)
        n_neg  = (y.numel() - n_pos).clamp(min=1.0)
        pos_w  = (n_neg / n_pos).clamp(max=999.0)
        weight = y * (pos_w - 1.0) + 1.0   # pos_w for tour edges, 1 otherwise
        loss   = F.binary_cross_entropy(p_sym, y, weight=weight)

        # ── Update ────────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Non-negative weight constraint: project all parameters to ≥ 0.
        # Remove this block if you do not need weights to be non-negative.
        #with torch.no_grad():
        #    for p in model.parameters():
        #        p.clamp_(min=0.0)

        # ── Logging ───────────────────────────────────────────────────────────
        losses.append(loss.item())
        avg = sum(losses[-50:]) / len(losses[-50:])

        postfix = dict(n=n, loss=f"{loss.item():.4f}", avg50=f"{avg:.4f}")

        # ── Periodic validation gap ───────────────────────────────────────────
        if val_interval > 0 and (step + 1) % val_interval == 0 and val_set:
            model.eval()
            gaps = []
            with torch.no_grad():
                for x_cpu, e_cpu, c_cpu, nn_len in val_set:
                    if nn_len < 1e-9:
                        continue
                    e_dev_v = e_cpu.to(device) if e_cpu is not None else None
                    p = model(x_cpu.to(device), e_dev_v)
                    p_s = (p + p.T) / 2
                    gnn_tour = greedy_decode(p_s)
                    gnn_len  = tour_length(c_cpu, gnn_tour)
                    gaps.append((gnn_len - nn_len) / nn_len * 100.0)
            model.train()
            gap = float(np.mean(gaps)) if gaps else float("nan")
            best_gap = min(best_gap, gap) if not np.isnan(gap) else best_gap
            postfix["gap%"] = f"{gap:.1f}"
            postfix["best_gap"] = f"{best_gap:.1f}"

        bar.set_postfix(**postfix)

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",           type=int,   default=8)
    parser.add_argument("--n_min",       type=int,   default=None)
    parser.add_argument("--n_max",       type=int,   default=None)
    parser.add_argument("--label",       type=str,   default="auto",
                        help="'auto' | 'optimal' (n≤10) | 'nn' | 'nn2opt' (n≤300)")
    parser.add_argument("--steps",        type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None,
                        help="Alias for --steps (same meaning)")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--size",        type=str,   default=None,
                        help="'small' | 'medium' | 'large'")
    parser.add_argument("--d",           type=int,   default=64)
    parser.add_argument("--L",           type=int,   default=4)
    parser.add_argument("--out",         type=str,   default=None,
                        help="Output path for weights. Defaults to model/gnn_{SIZE}.pt "
                             "or model/gnn_{SIZE}_tsptwd.pt when --size is given, "
                             "else model/gnn.pt")
    parser.add_argument("--source",      type=str,   default="random",
                        help="'random' | 'tsp' | 'tsptwd_json'")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--val_interval",type=int,   default=500,
                        help="Validate every N steps (0 = off)")
    parser.add_argument("--mode",        type=str,   default="tsp",
                        help="'tsp' (plain) | 'tsptwd' (time windows + perturbations)")
    args = parser.parse_args()

    # ── Resolve --epochs / --steps alias ─────────────────────────────────────
    if args.epochs is not None and args.steps is not None:
        raise ValueError("Specify only one of --steps or --epochs, not both.")
    if args.epochs is not None:
        args.steps = args.epochs
    if args.steps is None:
        args.steps = 2000

    # ── Device ────────────────────────────────────────────────────────────────
    device = get_device(args.device)
    print(f"Device: {device}")
    if hasattr(device, "type") and device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ── Args ──────────────────────────────────────────────────────────────────
    if args.label == "optimal" and args.n > 10:
        raise ValueError("--label optimal requires n ≤ 10.")
    if args.size is not None:
        if args.size not in MODEL_SIZES:
            raise ValueError(f"--size must be one of {list(MODEL_SIZES)}.")
        args.d, args.L = MODEL_SIZES[args.size]

    # ── Resolve default output path ───────────────────────────────────────────
    if args.out is None:
        if args.size is not None:
            suffix = f"_{args.mode}" if args.mode != "tsp" else ""
            args.out = f"model/gnn_{args.size}{suffix}.pt"
        else:
            args.out = "model/gnn.pt"

    # ── City / JSON pool ──────────────────────────────────────────────────────
    city_pool = None
    json_pool = None
    if args.source == "tsptwd_json":
        _dataset_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets'
        )
        json_pool = _load_tsptwd_json_pool(_dataset_dir)
        _loaded = sorted(json_pool.keys())
        print(f"JSON pool: {len(_loaded)} instances loaded  n={_loaded}")
        if args.mode != "tsptwd":
            raise ValueError("--source tsptwd_json requires --mode tsptwd")
    elif args.source != "random":
        print(f"Loading {POOL_SIZE} cities from source='{args.source}' ...")
        city_pool = load_cities(POOL_SIZE, source=args.source).cpu()
        print(f"Pool: {city_pool.shape}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    node_dim = 5 if args.mode == "tsptwd" else 2
    edge_dim = 4 if args.mode == "tsptwd" else 1  # dist + alpha + t_start/T + t_end/T
    model = TSPGNN(d=args.d, L=args.L, node_dim=node_dim, edge_dim=edge_dim)
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume: {args.resume} not found")
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print(f"Resumed from {args.resume}")

    n_params = sum(p.numel() for p in model.parameters())
    size_info = (f"n={args.n}" if not (args.n_min and args.n_max)
                 else f"n∈[{args.n_min},{args.n_max}]")
    print(f"TSPGNN(d={args.d}, L={args.L}, node_dim={node_dim}, edge_dim={edge_dim})"
          f"  —  {n_params:,} params")
    print(f"Training: {size_info}, steps={args.steps}, lr={args.lr}, "
          f"label={args.label}, source={args.source}, mode={args.mode}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    losses = train(
        model,
        n_nodes=args.n, n_steps=args.steps, lr=args.lr,
        city_pool=city_pool, json_pool=json_pool,
        label=args.label,
        n_min=args.n_min, n_max=args.n_max,
        device=device,
        val_interval=args.val_interval,
        mode=args.mode,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    np.save(os.path.join(out_dir or ".", "losses.npy"), np.array(losses))
    print(f"\nWeights → {args.out}")
    print(f"Final loss: {losses[-1]:.4f}  |  Best: {min(losses):.4f}")

    # ── ONNX export (Netron-compatible graph) ─────────────────────────────────
    model.eval()
    model = model.cpu()
    n_dummy = 10
    dummy_x = torch.zeros(n_dummy, model.node_dim)
    onnx_path = args.out.replace(".pt", ".onnx")
    if args.mode == "tsptwd":
        dummy_e = torch.zeros(n_dummy, n_dummy, model.edge_dim - 1)
        torch.onnx.export(
            model, (dummy_x, dummy_e), onnx_path,
            input_names=["x", "edge_extra"],
            output_names=["edge_probs"],
            opset_version=18,
            dynamic_axes={
                "x":          {0: "n"},
                "edge_extra": {0: "n", 1: "n"},
                "edge_probs": {0: "n", 1: "n"},
            },
        )
    else:
        torch.onnx.export(
            model, (dummy_x, None), onnx_path,
            input_names=["x"],
            output_names=["edge_probs"],
            opset_version=18,
            dynamic_axes={
                "x":          {0: "n"},
                "edge_probs": {0: "n", 1: "n"},
            },
        )
    print(f"ONNX graph → {onnx_path}")

"""
train.py — DIFUSCO training loop

┌─────────────────────────────────────────────────────────────────────────┐
│  How diffusion training works — step by step                            │
│                                                                         │
│  The model learns by playing a denoising game:                          │
│                                                                         │
│  1. Take a clean tour label y_0 ∈ {0,1}^(n×n)                         │
│     (binary adjacency matrix from a nearest-neighbour tour)             │
│                                                                         │
│  2. Pick a random timestep  t ~ Uniform(1, T)                          │
│                                                                         │
│  3. Corrupt y_0 with Gaussian noise at level t:                        │
│       y_t = sqrt(ᾱ_t) · y_0  +  sqrt(1 − ᾱ_t) · ε                    │
│       where ε ~ N(0, I)  and  ᾱ_t ∈ (0, 1] decreases with t          │
│                                                                         │
│       ᾱ_t close to 1 → y_t looks almost like y_0   (small t)         │
│       ᾱ_t close to 0 → y_t looks like pure noise   (large t)         │
│                                                                         │
│  4. Ask the model:  given y_t and the graph x, what was y_0?           │
│       ŷ_0 = model(x, y_t, t)                                           │
│                                                                         │
│  5. Loss = BCE(ŷ_0, y_0)  — "how wrong was your guess?"               │
│                                                                         │
│  After training, the model can denoise from pure noise to a tour:      │
│    t = T → T-1 → … → 1 → 0                                            │
│  At each step it predicts ŷ_0, then re-adds the right amount of       │
│  noise for the NEXT step (DDPM formula).                                │
└─────────────────────────────────────────────────────────────────────────┘

Usage
-----
    python train.py --n 10 --epochs 200 --size small
    python train.py --n 10 --epochs 200 --size small --use_dataset
    python train.py --n 50 --epochs 500 --size medium --two_opt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from model import DifuscoModel, MODEL_SIZES
from data  import (
    load_dataset, random_instance,
    generate_time_windows, generate_perturbations,
    build_tsptwd_features, nn_tour_labels,
    greedy_decode, evaluate_tsptwd,
)

# Usage string updated for --mode
#   python train.py --n 10 --epochs 3000 --size small --mode tsptwd
#   python train.py --n 10 --epochs 3000 --size small --mode tsp


# ─────────────────────────────────────────────────────────────────────────────
# Noise schedule
# ─────────────────────────────────────────────────────────────────────────────

class NoiseSchedule:
    """
    Pre-computes the diffusion coefficients for T timesteps.

    ┌──────────────────────────────────────────────────────────────────┐
    │  We need three sequences, all indexed t = 1 … T:                │
    │                                                                  │
    │  β_t  ("beta")  — how much NEW noise is added at step t         │
    │         linearly spaced: β_1 = β_min,  β_T = β_max             │
    │                                                                  │
    │  α_t  ("alpha")  = 1 − β_t                                      │
    │         how much of the PREVIOUS signal is kept                 │
    │                                                                  │
    │  ᾱ_t  ("alpha_bar")  = α_1 · α_2 · … · α_t   (cumulative)     │
    │         how much of the ORIGINAL y_0 survives up to step t      │
    │         ᾱ_1 ≈ 1  (almost nothing added)                        │
    │         ᾱ_T ≈ 0  (original signal almost gone)                 │
    │                                                                  │
    │  Forward process (closed form, no loop needed):                  │
    │    y_t = sqrt(ᾱ_t) · y_0  +  sqrt(1−ᾱ_t) · ε,  ε~N(0,I)     │
    │    ↑ this is what lets us jump to ANY timestep in one shot      │
    └──────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    T        : total number of diffusion steps (e.g. 1000)
    beta_min : noise at first step  (default 1e-4, from Ho et al.)
    beta_max : noise at last step   (default 0.02, from Ho et al.)
    """

    def __init__(self, T: int = 1000, beta_min: float = 1e-4, beta_max: float = 0.02):
        self.T = T

        # β_t linearly from beta_min to beta_max  (shape: T,)
        betas      = torch.linspace(beta_min, beta_max, T)           # (T,)
        alphas     = 1.0 - betas                                      # (T,)
        alpha_bars = torch.cumprod(alphas, dim=0)                     # (T,)

        # Store as buffers (not parameters — we never backprop through these)
        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bars = alpha_bars                  # ᾱ_t, indexed 0…T-1
                                                      # i.e. alpha_bars[t-1] = ᾱ_t

    def to(self, device):
        self.betas      = self.betas.to(device)
        self.alphas     = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def alpha_bar(self, t: int) -> torch.Tensor:
        """ᾱ_t as a scalar tensor (t is 1-indexed)."""
        return self.alpha_bars[t - 1]


def forward_diffuse(y0: torch.Tensor, t: int, schedule: NoiseSchedule):
    """
    Corrupt y_0 to y_t using the closed-form forward process.

    ┌──────────────────────────────────────────────────────────────────┐
    │  y_t = sqrt(ᾱ_t) · y_0  +  sqrt(1 − ᾱ_t) · ε                  │
    │                                                                  │
    │  Intuition: ᾱ_t acts as a "signal weight".                       │
    │    t small → ᾱ_t ≈ 1 → y_t ≈ y_0  (barely corrupted)          │
    │    t large → ᾱ_t ≈ 0 → y_t ≈ ε   (mostly noise)               │
    │                                                                  │
    │  This is called the "reparameterisation trick" — instead of     │
    │  simulating t sequential noise additions, we compute y_t        │
    │  DIRECTLY from y_0 in one shot using ᾱ_t.                       │
    └──────────────────────────────────────────────────────────────────┘

    y0 : (n, n) clean binary tour adjacency
    t  : int    timestep ∈ [1, T]

    Returns
    -------
    y_t   : (n, n) noisy version
    noise : (n, n) the Gaussian noise that was added (used if predicting ε)
    """
    ab   = schedule.alpha_bar(t)                       # ᾱ_t  scalar
    noise = torch.randn_like(y0)                       # ε ~ N(0, I)
    y_t   = ab.sqrt() * y0 + (1.0 - ab).sqrt() * noise
    return y_t, noise


# ─────────────────────────────────────────────────────────────────────────────
# Instance generation
# ─────────────────────────────────────────────────────────────────────────────

def make_random_instance(n: int, seed: int = None, two_opt: bool = False) -> dict:
    """
    Generate a random TSPTW-D instance and compute its NN-tour label.

    Returns a dict ready for one training step:
      node_feats : (n, 5)
      edge_feats : (n, n, 1)
      y0         : (n, n)  binary tour adjacency (training target)
      coords     : (n, 2)
      + the full instance fields for evaluation
    """
    coords = random_instance(n, seed=seed)
    tw, svc = generate_time_windows(coords, seed=seed)
    total_time = tw[:, 1].max().item()
    perturbs = generate_perturbations(n, total_time, seed=seed)
    node_feats, edge_feats = build_tsptwd_features(coords, tw, svc, perturbs)
    y0 = nn_tour_labels(coords, two_opt=two_opt)

    return {
        "coords":        coords,
        "time_windows":  tw,
        "service_times": svc,
        "perturbations": perturbs,
        "node_feats":    node_feats,
        "edge_feats":    edge_feats,
        "y0":            y0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    n: int            = 10,
    epochs: int       = 200,
    size: str         = "small",
    lr: float         = 1e-3,
    use_dataset: bool = False,   # True → use datasets/tsptwd_n{n}.json
    two_opt: bool     = False,   # True → improve NN labels with 2-opt
    resume: str       = None,    # path to .pt checkpoint to resume from
    out: str          = None,    # override output checkpoint path
    save_every: int   = 50,
    device: str       = None,
):
    """
    Train DIFUSCO on TSPTW-D instances of size n.

    Each "epoch" is one gradient step on one randomly generated instance.
    For serious training, increase epochs to 3000+ and n to 50–100.

    Parameters
    ----------
    n           : number of cities (including depot)
    epochs      : number of gradient steps
    size        : model size preset ("small" / "medium" / "large")
    lr          : Adam learning rate
    use_dataset : if True, load the pre-generated instance instead of random
    two_opt     : if True, improve NN labels with 2-opt (better but slower)
    resume      : path to an existing .pt checkpoint to continue training from
    out         : override the output checkpoint path (default: model/difusco_{size}_n{n}.pt)
    save_every  : save checkpoint every N epochs
    device      : "cpu" / "cuda" / "mps" — auto-detected if None
    """
    # ── Device ────────────────────────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    print(f"Training on {dev}  |  n={n}  |  size={size}  |  epochs={epochs}")

    # ── Model + schedule ──────────────────────────────────────────────────────
    d, L, T = MODEL_SIZES[size]

    # TSPTW-D: node_dim=5 (x, y, a/T, b/T, s/T),  edge_dim=2 (dist + alpha)
    model    = DifuscoModel(d=d, L=L, node_dim=5, edge_dim=2).to(dev)
    schedule = NoiseSchedule(T=T).to(dev)
    optim    = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    if resume:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"Checkpoint not found: {resume}")
        model.load_state_dict(torch.load(resume, map_location=dev))
        print(f"Resumed from {resume}")

    # ── Load fixed dataset instance (optional) ────────────────────────────────
    fixed_inst = None
    if use_dataset:
        fixed_inst = load_dataset(n)
        fixed_inst["y0"] = nn_tour_labels(fixed_inst["coords"], two_opt=two_opt)
        print(f"Using pre-generated dataset instance  (n_clients={n})")

    # ── Training ──────────────────────────────────────────────────────────────
    losses = []
    model.train()

    for epoch in range(1, epochs + 1):

        # ── Get instance ──────────────────────────────────────────────────────
        if fixed_inst is not None:
            inst = fixed_inst
        else:
            inst = make_random_instance(n, seed=epoch, two_opt=two_opt)

        x  = inst["node_feats"].to(dev)    # (n, 5)
        ef = inst["edge_feats"].to(dev)    # (n, n, 1)
        y0 = inst["y0"].to(dev)            # (n, n)  ← what we want to learn

        # ── Sample timestep ───────────────────────────────────────────────────
        #
        # t ~ Uniform(1, T)
        # The model must be able to denoise at ANY noise level,
        # so we train it on random levels each step.
        #
        t = torch.randint(1, T + 1, (1,)).item()
        t_tensor = torch.tensor(t, device=dev)

        # ── Forward diffusion ─────────────────────────────────────────────────
        #
        # y_t = sqrt(ᾱ_t) · y_0  +  sqrt(1−ᾱ_t) · ε
        #
        y_t, _ = forward_diffuse(y0, t, schedule)   # (n, n)

        # ── Model forward ─────────────────────────────────────────────────────
        #
        # Given the noisy y_t, the graph x, and the timestep t,
        # predict the clean tour adjacency y_0.
        #
        y0_pred = model(x, y_t, t_tensor, edge_extra=ef)   # (n, n)

        # ── Loss ──────────────────────────────────────────────────────────────
        #
        # Binary cross-entropy between predicted and true y_0.
        #
        # Why BCE and not MSE?
        #   y_0 is a binary matrix (0 or 1).  BCE is the natural loss for
        #   predicting probabilities of binary outcomes — it penalises being
        #   confident and wrong much more than MSE does.
        #
        loss = F.binary_cross_entropy(y0_pred, y0)

        # ── Gradient step ─────────────────────────────────────────────────────
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        losses.append(loss.item())

        # ── Logging ───────────────────────────────────────────────────────────
        if epoch % max(1, epochs // 20) == 0 or epoch == 1:
            avg = sum(losses[-50:]) / min(50, len(losses))
            print(f"  epoch {epoch:>5d}/{epochs}  loss={loss.item():.4f}  avg50={avg:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            _save_checkpoint(model, losses, size, n, epoch, out=out)

    print("Training complete.")
    return model, losses


def _save_checkpoint(model, losses, size, n, epoch, out=None):
    """Save model weights and loss curve to model/."""
    save_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(save_dir, exist_ok=True)
    pt_path   = out if out else os.path.join(save_dir, f"difusco_{size}_n{n}.pt")
    loss_path = pt_path.replace(".pt", "_losses.npy")
    torch.save(model.state_dict(), pt_path)
    np.save(loss_path, np.array(losses))
    print(f"  → checkpoint saved  ({pt_path},  epoch {epoch})")


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation helper (greedy decode from ŷ_0 at t=1)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, inst, schedule, device="cpu"):
    """
    Run a quick single-step evaluation: denoise from t=1 and decode.

    This is NOT full DDPM inference (which would start from t=T and run
    T denoising steps).  It's a fast sanity check — pass slightly-noisy
    labels through the model and see if it reconstructs a reasonable tour.

    Full inference will be implemented in the benchmark notebooks.
    """
    dev = torch.device(device)
    model.eval()

    x    = inst["node_feats"].to(dev)
    ef   = inst["edge_feats"].to(dev)
    y0   = inst["y0"].to(dev)
    n    = x.shape[0]

    # Corrupt at t=1 (smallest noise level) so the model has an easy task
    y_t, _ = forward_diffuse(y0, t=1, schedule=schedule)
    t_tensor = torch.tensor(1, device=dev)

    y0_pred = model(x, y_t, t_tensor, edge_extra=ef)   # (n, n)
    tour    = greedy_decode(y0_pred, start=0)

    t_total, penalty, obj = evaluate_tsptwd(
        inst["coords"], tour,
        inst["time_windows"], inst["service_times"], inst["perturbations"],
    )
    print(f"  Eval  →  tour={tour}  obj={obj:.3f}  (time={t_total:.3f}, penalty={penalty:.1f})")
    model.train()
    return tour, obj


# ─────────────────────────────────────────────────────────────────────────────
# Full DDPM / DDIM inference  (the actual denoising chain used at test time)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample(
    model,
    node_feats: torch.Tensor,
    schedule: "NoiseSchedule",
    edge_extra: torch.Tensor = None,
    n_steps: int = 50,
    ddim: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Full reverse diffusion chain: y_T ~ N(0,I)  →  ŷ_0 ∈ (0,1)^(n×n).

    ┌──────────────────────────────────────────────────────────────────┐
    │  Two modes:                                                      │
    │                                                                  │
    │  DDPM (ddim=False):                                             │
    │    The original stochastic reverse process.                     │
    │    Each step adds a small amount of noise back (σ_t · ε).       │
    │    Run all T steps for best quality; slow.                      │
    │                                                                  │
    │    y_{t-1} = coef1 · ŷ_0  +  coef2 · y_t  +  σ_t · ε         │
    │    where                                                        │
    │      coef1 = sqrt(ᾱ_{t-1}) · β_t  /  (1 − ᾱ_t)              │
    │      coef2 = sqrt(α_t) · (1 − ᾱ_{t-1})  /  (1 − ᾱ_t)        │
    │      σ_t   = sqrt( (1−ᾱ_{t-1}) / (1−ᾱ_t) · β_t )            │
    │                                                                  │
    │  DDIM (ddim=True, default):                                     │
    │    Deterministic — no noise added back each step.               │
    │    Can skip timesteps → 50 steps instead of 1000, same quality. │
    │                                                                  │
    │    noise_dir  = (y_t − sqrt(ᾱ_t) · ŷ_0)  /  sqrt(1−ᾱ_t)     │
    │    y_{t-1}    = sqrt(ᾱ_{t-1}) · ŷ_0  +  sqrt(1−ᾱ_{t-1}) · noise_dir │
    └──────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    model      : trained DifuscoModel
    node_feats : (n, node_dim)   node features (coords for TSP, full feats for TSPTW-D)
    schedule   : NoiseSchedule
    edge_extra : (n, n, k) or None   extra edge features (perturbations for TSPTW-D)
    n_steps    : number of denoising steps (≤ T; fewer = faster, slight quality loss)
    ddim       : True = deterministic DDIM (recommended), False = stochastic DDPM
    device     : "cpu" / "cuda" / "mps"

    Returns
    -------
    p : (n, n) predicted edge probability matrix in (0, 1)
        Pass to greedy_decode() or binarize() to get a tour.
    """
    dev = torch.device(device)
    model.to(dev)   # move model to the target device (no-op if already there)
    model.eval()
    schedule.to(dev)
    T  = schedule.T
    x  = node_feats.to(dev)
    ef = edge_extra.to(dev) if edge_extra is not None else None
    n  = x.shape[0]

    # Evenly-spaced timestep sequence from T down to 1
    if n_steps >= T:
        timesteps = list(range(T, 0, -1))
    else:
        timesteps = torch.linspace(T, 1, n_steps).long().tolist()

    # Start from pure Gaussian noise
    y = torch.randn(n, n, device=dev)

    for idx, t in enumerate(timesteps):
        t_tensor = torch.tensor(t, dtype=torch.long, device=dev)

        # Predict clean tour from current noisy state
        y0_pred = model(x, y, t_tensor, edge_extra=ef)   # (n, n) ∈ (0,1)

        # Final step: return the prediction directly
        if idx == len(timesteps) - 1:
            y = y0_pred
            break

        t_prev  = int(timesteps[idx + 1])
        ab_t    = schedule.alpha_bar(t).to(dev)
        ab_prev = schedule.alpha_bar(t_prev).to(dev)

        if ddim:
            # Estimated noise direction (the "direction pointing toward y_t from ŷ_0")
            noise_dir = (y - ab_t.sqrt() * y0_pred) / (1.0 - ab_t).clamp(min=1e-8).sqrt()
            y = ab_prev.sqrt() * y0_pred + (1.0 - ab_prev).sqrt() * noise_dir
        else:
            # DDPM posterior: blend ŷ_0 and y_t, then add noise
            beta_t  = schedule.betas[t - 1].to(dev)
            alpha_t = schedule.alphas[t - 1].to(dev)
            coef1   = ab_prev.sqrt() * beta_t  / (1.0 - ab_t).clamp(min=1e-8)
            coef2   = alpha_t.sqrt() * (1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8)
            mu      = coef1 * y0_pred + coef2 * y
            sigma   = ((1.0 - ab_prev) / (1.0 - ab_t).clamp(min=1e-8) * beta_t).clamp(min=0).sqrt()
            y       = mu + sigma * torch.randn_like(y)

    return y   # (n, n) soft probabilities


def sample_best_of(
    model,
    node_feats: torch.Tensor,
    schedule: "NoiseSchedule",
    edge_extra: torch.Tensor = None,
    n_samples: int = 8,
    n_steps: int = 50,
    device: str = "cpu",
) -> tuple:
    """
    Run `n_samples` independent DDIM chains and return the tour with the
    shortest Euclidean length.

    DIFUSCO is stochastic at the start (y_T is sampled fresh each time),
    so multiple runs from the same instance can give different tours.
    Best-of-N is a simple way to leverage this diversity.

    Returns
    -------
    best_tour : list of node indices
    best_len  : float, Euclidean tour length
    all_tours : list of all sampled tours
    """
    from data import greedy_decode, tour_length
    coords = node_feats[:, :2]   # first two dims are always x, y

    best_tour, best_len = None, float("inf")
    all_tours = []

    for _ in range(n_samples):
        p    = sample(model, node_feats, schedule, edge_extra=edge_extra,
                      n_steps=n_steps, ddim=True, device=device)
        tour = greedy_decode(p, start=0)
        l    = tour_length(coords, tour)
        all_tours.append(tour)
        if l < best_len:
            best_len, best_tour = l, tour

    return best_tour, best_len, all_tours


def make_random_instance_tsp(n: int, seed: int = None) -> dict:
    """
    Generate a plain TSP instance (no time windows, no perturbations).

    Returns the same dict structure as make_random_instance but for TSP:
      node_feats = coords  (n, 2)   ← passed directly to the model
      edge_extra = None
      y0         = NN tour labels  (n, n)
    """
    coords = random_instance(n, seed=seed)
    y0     = nn_tour_labels(coords)
    return {
        "coords":     coords,
        "node_feats": coords,        # TSP: raw coords are the node features
        "edge_extra": None,
        "y0":         y0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DIFUSCO on TSP or TSPTW-D")
    parser.add_argument("--n",           type=int,   default=10,       help="number of cities (incl. depot for TSPTW-D)")
    parser.add_argument("--epochs",      type=int,   default=200,      help="training steps")
    parser.add_argument("--size",        type=str,   default="small",  choices=list(MODEL_SIZES.keys()))
    parser.add_argument("--mode",        type=str,   default="tsptwd", choices=["tsp", "tsptwd"],
                        help="tsp = plain TSP (node_dim=2), tsptwd = full problem (node_dim=5)")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--use_dataset", action="store_true",          help="use pre-generated tsptwd JSON (tsptwd mode only)")
    parser.add_argument("--two_opt",     action="store_true",          help="improve NN labels with 2-opt")
    parser.add_argument("--resume",      type=str,   default=None,     help="path to .pt checkpoint to resume from")
    parser.add_argument("--out",         type=str,   default=None,     help="override output checkpoint path")
    parser.add_argument("--save_every",  type=int,   default=50)
    parser.add_argument("--device",      type=str,   default=None)
    args = parser.parse_args()

    if args.mode == "tsp":
        # ── Plain TSP mode ────────────────────────────────────────────────────
        d, L, T = MODEL_SIZES[args.size]
        if args.device is None:
            dev_str = "cuda" if torch.cuda.is_available() else \
                      "mps"  if torch.backends.mps.is_available() else "cpu"
        else:
            dev_str = args.device
        dev = torch.device(dev_str)

        model    = DifuscoModel(d=d, L=L, node_dim=2, edge_dim=1).to(dev)
        schedule = NoiseSchedule(T=T).to(dev)
        optim    = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.resume:
            model.load_state_dict(torch.load(args.resume, map_location=dev))
            print(f"Resumed from {args.resume}")

        out = args.out or os.path.join(
            os.path.dirname(__file__), "model", f"difusco_{args.size}_tsp.pt"
        )
        losses = []
        print(f"Training TSP  |  n={args.n}  |  size={args.size}  |  device={dev_str}")

        for epoch in range(1, args.epochs + 1):
            inst = make_random_instance_tsp(args.n, seed=epoch)
            x  = inst["node_feats"].to(dev)
            y0 = inst["y0"].to(dev)

            t        = torch.randint(1, T + 1, (1,)).item()
            t_tensor = torch.tensor(t, device=dev)
            y_t, _   = forward_diffuse(y0, t, schedule)

            y0_pred = model(x, y_t, t_tensor, edge_extra=None)
            loss    = F.binary_cross_entropy(y0_pred, y0)

            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

            if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
                avg = sum(losses[-50:]) / min(50, len(losses))
                print(f"  epoch {epoch:>5d}/{args.epochs}  loss={loss.item():.4f}  avg50={avg:.4f}")

            if epoch % args.save_every == 0 or epoch == args.epochs:
                _save_checkpoint(model, losses, args.size + "_tsp", args.n, epoch, out=out)

        print("TSP training complete.")
    else:
        # ── TSPTW-D mode (default) ────────────────────────────────────────────
        model, losses = train(
            n=args.n, epochs=args.epochs, size=args.size, lr=args.lr,
            use_dataset=args.use_dataset, two_opt=args.two_opt,
            resume=args.resume, out=args.out,
            save_every=args.save_every, device=args.device,
        )

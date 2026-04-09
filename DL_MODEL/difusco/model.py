"""
model.py — DIFUSCO denoising model (Sun & Yang, NeurIPS 2023)

┌─────────────────────────────────────────────────────────────────────────┐
│  Big picture                                                            │
│                                                                         │
│  The GNN outputs edge probabilities in ONE forward pass.                │
│  DIFUSCO does the same, but over T denoising steps.                     │
│                                                                         │
│  At training time:                                                      │
│    y_0  = clean binary tour adjacency  (our "ground truth")             │
│    y_t  = y_0 corrupted with Gaussian noise at level t                  │
│    goal = given y_t and the graph x, predict the clean y_0             │
│                                                                         │
│  At inference time:                                                     │
│    start from y_T ~ N(0, I)  (pure noise)                              │
│    run the model T times, each time reducing the noise level            │
│    → converges to a binary edge matrix → decode tour                   │
└─────────────────────────────────────────────────────────────────────────┘

Classes
-------
SinusoidalTimeEmbedding  : encode scalar timestep t → d-dim vector
DifuscoGNNLayer          : one residual GCN layer conditioned on timestep
DifuscoModel             : full denoising network

Presets
-------
MODEL_SIZES["small" | "medium" | "large"] → (d, L, T)
  d = embedding dim
  L = number of GNN layers
  T = total diffusion timesteps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (d, L, T)
MODEL_SIZES = {
    "small":  (64,  4, 500),
    "medium": (128, 6, 1000),
    "large":  (256, 8, 1000),
}


# ─────────────────────────────────────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Encode a scalar timestep t into a d-dimensional vector.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Why do we need this?                                            │
    │                                                                  │
    │  The model sees y_t at many different noise levels (t = 1…T).   │
    │  Without knowing t, it can't tell whether it's looking at a     │
    │  slightly-noisy tour (t small) or near-random noise (t large).  │
    │  The timestep embedding gives the model this context.           │
    │                                                                  │
    │  Math (sinusoidal, same as Transformer positional encoding):    │
    │                                                                  │
    │    emb(t)_k = sin(t / 10000^(2k/d))   for k = 0, 2, 4, …      │
    │    emb(t)_k = cos(t / 10000^(2k/d))   for k = 1, 3, 5, …      │
    │                                                                  │
    │  This creates a unique fingerprint for each timestep.           │
    │  Then a small MLP lifts it to the model's hidden dimension d.   │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d: int):
        super().__init__()
        # Small MLP to project raw sinusoidal encoding into the model dim
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.SiLU(),           # SiLU (swish) smoother than ReLU for diffusion
            nn.Linear(d * 2, d),
        )
        self.d = d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : (,) or (1,) scalar timestep (integer, 1 ≤ t ≤ T)
        returns : (d,) timestep embedding
        """
        t = t.float().reshape(1)
        half = self.d // 2
        # Frequencies: 1/10000^(k / half) for k in 0…half-1
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )                                           # (half,)
        args = t * freqs                            # (half,)
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)  # (d,)
        return self.mlp(emb)                        # (d,)


# ─────────────────────────────────────────────────────────────────────────────
# GNN layer (conditioned on timestep)
# ─────────────────────────────────────────────────────────────────────────────

class DifuscoGNNLayer(nn.Module):
    """
    One residual GCN layer with joint edge + node updates, conditioned on t.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Same equations as the GNN (Joshi et al.) with ONE addition:    │
    │  the timestep embedding is injected into the edge features      │
    │  before the update, so every layer knows "what noise level      │
    │  we are at" when refining the edge representations.             │
    │                                                                  │
    │  Edge update:                                                    │
    │    e_ij^l = ReLU(LN(W1·h_i + W2·h_j + W3·e_ij^{l-1} + t_ij)) │
    │    where t_ij = W_t · t_emb  (same for all (i,j))              │
    │                                                                  │
    │  Attention gate:                                                 │
    │    η_ij = softmax_j( gate(e_ij^l) )                            │
    │                                                                  │
    │  Node update (residual):                                        │
    │    h_i^l = h_i^{l-1} + ReLU(LN(W4·h_i^{l-1} + Σ_j η·W5·e))  │
    └──────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, d: int):
        super().__init__()
        # Edge weights (same as GNN)
        self.W1   = nn.Linear(d, d, bias=False)  # h_i  → edge
        self.W2   = nn.Linear(d, d, bias=False)  # h_j  → edge
        self.W3   = nn.Linear(d, d, bias=False)  # e_ij → edge
        # Timestep conditioning on edges
        self.W_t  = nn.Linear(d, d, bias=False)  # t_emb → edge bias
        # Node weights
        self.W4   = nn.Linear(d, d, bias=False)  # self  → node
        self.W5   = nn.Linear(d, d, bias=False)  # agg   → node
        # Scalar attention gate
        self.gate = nn.Linear(d, 1)
        # Layer norms
        self.ln_e = nn.LayerNorm(d)
        self.ln_h = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor, t_emb: torch.Tensor):
        """
        h     : (n, d)    node embeddings
        e     : (n, n, d) edge embeddings
        t_emb : (d,)      timestep embedding (same for all edges)
        """
        n, d = h.shape

        # ── Inject timestep into edge features ────────────────────────────────
        # t_emb is (d,) → broadcast to (n, n, d)
        t_bias = self.W_t(t_emb).unsqueeze(0).unsqueeze(0)  # (1, 1, d)

        # ── Edge update ───────────────────────────────────────────────────────
        h_i   = self.W1(h).unsqueeze(1).expand(-1, n, -1)   # (n, n, d)
        h_j   = self.W2(h).unsqueeze(0).expand(n, -1, -1)   # (n, n, d)
        e_new = F.relu(self.ln_e(h_i + h_j + self.W3(e) + t_bias))  # (n, n, d)

        # ── Attention gate ────────────────────────────────────────────────────
        eta   = torch.softmax(self.gate(e_new), dim=1)       # (n, n, 1)

        # ── Node update (residual) ────────────────────────────────────────────
        agg   = (eta * self.W5(e_new)).sum(dim=1)            # (n, d)
        h_new = h + F.relu(self.ln_h(self.W4(h) + agg))     # (n, d)

        return h_new, e_new


# ─────────────────────────────────────────────────────────────────────────────
# Full denoising model
# ─────────────────────────────────────────────────────────────────────────────

class DifuscoModel(nn.Module):
    """
    DIFUSCO denoising network: given (x, y_t, t) → predict y_0.

    ┌──────────────────────────────────────────────────────────────────┐
    │  Inputs                                                          │
    │    x          node features  (n, node_dim)                      │
    │               TSP     : (n, 2)  → [x_coord, y_coord]           │
    │               TSPTW-D : (n, 5)  → [x, y, a/T, b/T, s/T]       │
    │                                                                  │
    │    y_t        noisy edge matrix  (n, n)  ← the key new input   │
    │               At t=0 this is the clean binary tour adjacency.   │
    │               At t=T this is near-Gaussian noise.               │
    │                                                                  │
    │    t          scalar timestep ∈ [1, T]                          │
    │                                                                  │
    │    edge_extra  optional extra edge features (n, n, k)           │
    │               TSPTW-D: worst-case perturbation α_ij  (n,n,1)   │
    │                                                                  │
    │  Architecture                                                    │
    │    1. Embed t → t_emb  (d,)                                    │
    │    2. Project node features  (n, node_dim) → (n, d)            │
    │    3. Build edge input: [dist | y_t | edge_extra] → (n, n, d)  │
    │    4. L × DifuscoGNNLayer(h, e, t_emb)                         │
    │    5. Edge MLP: (n, n, d) → (n, n)  + sigmoid → ŷ_0 ∈ (0,1)  │
    │                                                                  │
    │  Output                                                          │
    │    y0_pred : (n, n)  predicted clean edge probabilities         │
    │             interpret as P(edge i→j is in the tour)             │
    └──────────────────────────────────────────────────────────────────┘

    Modes
    -----
    Plain TSP  : node_dim=2, edge_dim=1  (dist only)
    TSPTW-D    : node_dim=5, edge_dim=2  (dist + worst-case α_ij)
    """

    def __init__(self, d: int = 128, L: int = 6,
                 node_dim: int = 2, edge_dim: int = 1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Timestep embedding
        self.time_emb = SinusoidalTimeEmbedding(d)

        # Input projections
        # Edge input = [static edge features | y_t] so edge_dim + 1
        self.node_in = nn.Linear(node_dim, d)
        self.edge_in = nn.Linear(edge_dim + 1, d)  # +1 for y_t channel

        # GNN layers
        self.layers = nn.ModuleList([DifuscoGNNLayer(d) for _ in range(L)])

        # Output head: edge embeddings → predicted y_0 logits
        self.edge_out = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                y_t: torch.Tensor,
                t: torch.Tensor,
                edge_extra: torch.Tensor = None) -> torch.Tensor:
        """
        x          : (n, node_dim)
        y_t        : (n, n)         noisy edge labels at timestep t
        t          : scalar or (1,) current timestep
        edge_extra : (n, n, k)      optional extra edge features (TSPTW-D: k=1)

        returns y0_pred : (n, n)  predicted clean tour probabilities ∈ (0,1)
        """
        # ── 1. Timestep embedding ─────────────────────────────────────────────
        t_emb = self.time_emb(t.reshape(1))   # (d,)

        # ── 2. Node embeddings ────────────────────────────────────────────────
        h = self.node_in(x)                   # (n, d)

        # ── 3. Edge embeddings ────────────────────────────────────────────────
        # Always use the first two dims as spatial coordinates
        dist = torch.cdist(x[:, :2], x[:, :2]).unsqueeze(-1)  # (n, n, 1)

        # y_t is the key DIFUSCO-specific input: the current noisy edge labels
        y_t_feat = y_t.unsqueeze(-1)                           # (n, n, 1)

        if edge_extra is not None:
            # TSPTW-D: also include perturbation features
            e_in = torch.cat([dist, edge_extra, y_t_feat], dim=-1)  # (n,n, edge_dim+1)
        else:
            e_in = torch.cat([dist, y_t_feat], dim=-1)              # (n,n, 2)

        e = self.edge_in(e_in)                # (n, n, d)

        # ── 4. GNN layers ─────────────────────────────────────────────────────
        for layer in self.layers:
            h, e = layer(h, e, t_emb)

        # ── 5. Output: predicted y_0 ──────────────────────────────────────────
        y0_pred = torch.sigmoid(self.edge_out(e)).squeeze(-1)  # (n, n)
        return y0_pred

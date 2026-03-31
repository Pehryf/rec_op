"""
model.py — TSPGNN architecture (Joshi et al., 2019)

GNNLayer : one residual GCN layer with joint edge and node updates
TSPGNN   : full encoder + edge classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    """
    One residual GCN layer with joint edge and node updates (Joshi et al., 2019).

    Edge update:
        e_ij^(l) = ReLU(LN(W1·h_i + W2·h_j + W3·e_ij^(l-1)))

    Attention gate:
        η_ij^(l) = softmax_j( gate(e_ij^(l)) )

    Node update (residual):
        h_i^(l) = h_i^(l-1) + ReLU(LN(W4·h_i^(l-1) + Σ_j η_ij · W5·e_ij^(l)))
    """

    def __init__(self, d: int):
        super().__init__()
        # Edge weights
        self.W1 = nn.Linear(d, d, bias=False)   # h_i contribution
        self.W2 = nn.Linear(d, d, bias=False)   # h_j contribution
        self.W3 = nn.Linear(d, d, bias=False)   # e_ij contribution
        # Node weights
        self.W4 = nn.Linear(d, d, bias=False)   # self contribution
        self.W5 = nn.Linear(d, d, bias=False)   # aggregated edge contribution
        # Scalar attention gate: d → 1
        self.gate = nn.Linear(d, 1)
        # Layer norms (work on any batch/instance size, unlike BatchNorm)
        self.ln_e = nn.LayerNorm(d)
        self.ln_h = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """
        h : (n, d)    node embeddings
        e : (n, n, d) edge embeddings
        """
        n, d = h.shape

        # ── Edge update ───────────────────────────────────────────────────────
        h_i   = self.W1(h).unsqueeze(1).expand(-1, n, -1)   # (n, n, d)
        h_j   = self.W2(h).unsqueeze(0).expand(n, -1, -1)   # (n, n, d)
        e_new = F.relu(self.ln_e(h_i + h_j + self.W3(e)))   # (n, n, d)

        # ── Attention gate ────────────────────────────────────────────────────
        eta = torch.softmax(self.gate(e_new), dim=1)         # (n, n, 1)

        # ── Node update (residual) ────────────────────────────────────────────
        agg   = (eta * self.W5(e_new)).sum(dim=1)            # (n, d)
        h_new = h + F.relu(self.ln_h(self.W4(h) + agg))     # (n, d)

        return h_new, e_new


class TSPGNN(nn.Module):
    """
    GNN encoder + edge classification head for TSP.

    Architecture:
      1. Linear projection: coords (n,2) → node embeddings (n,d)
      2. Linear projection: distances (n,n,1) → edge embeddings (n,n,d)
      3. L stacked GNNLayer blocks
      4. MLP edge head: (n,n,d) → (n,n) edge probabilities
    """

    def __init__(self, d: int = 128, L: int = 6):
        super().__init__()
        self.node_in  = nn.Linear(2, d)
        self.edge_in  = nn.Linear(1, d)
        self.layers   = nn.ModuleList([GNNLayer(d) for _ in range(L)])
        self.edge_out = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (n, 2)   node coordinates in [0, 1]²
        returns p : (n, n)  edge probabilities ∈ (0, 1)
        """
        h    = self.node_in(x)                    # (n, d)
        dist = torch.cdist(x, x).unsqueeze(-1)    # (n, n, 1)
        e    = self.edge_in(dist)                  # (n, n, d)

        for layer in self.layers:
            h, e = layer(h, e)

        p = torch.sigmoid(self.edge_out(e)).squeeze(-1)   # (n, n)
        return p

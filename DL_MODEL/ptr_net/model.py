"""
model.py — Pointer Network architecture (Vinyals et al., 2015)

PointerNetwork : LSTM encoder-decoder with attention-based pointer head.

Predefined size presets
-----------------------
MODEL_SIZES["small"|"medium"|"large"] → (embed_dim, hidden_dim, n_layers)

| Preset  | embed | hidden | layers | Parameters  | Recommended for       |
|---------|-------|--------|--------|-------------|-----------------------|
| small   |  64   |  128   |   1    |   ~230 K    | quick tests, n ≤ 20   |
| medium  | 128   |  256   |   1    |   ~920 K    | standard TSP, n ≤ 100 |
| large   | 256   |  512   |   2    |  ~3.70 M    | larger instances      |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SIZES = {
    "small":  (64,  128, 1),
    "medium": (128, 256, 1),
    "large":  (256, 512, 2),
}


class Encoder(nn.Module):
    """
    LSTM encoder: embeds each city coordinate and processes the sequence.

    Input  : (n, node_dim) city features
    Output : encoder_outputs (n, hidden_dim), final hidden state (h, c)
    """

    def __init__(self, embed_dim: int, hidden_dim: int, n_layers: int = 1,
                 node_dim: int = 2):
        super().__init__()
        self.embed   = nn.Linear(node_dim, embed_dim)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True)

    def forward(self, x: torch.Tensor):
        """x : (n, 2)"""
        emb = self.embed(x).unsqueeze(0)            # (1, n, embed_dim)
        out, (h, c) = self.lstm(emb)                # out: (1, n, hidden)
        return out.squeeze(0), h, c                 # (n, hidden), h, c


class Attention(nn.Module):
    """
    Additive (Bahdanau-style) attention used as a pointer.

    Computes a score for each encoder output given the current decoder state:
        u_i = v^T tanh(W1·enc_i + W2·dec)
        a_i = softmax(u_i)   ← pointer distribution over cities

    The city with the highest a_i is selected at each decoding step.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v  = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, dec_h: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        enc_out : (n, hidden_dim)
        dec_h   : (hidden_dim,)
        mask    : (n,) bool — True for already-visited cities
        Returns : (n,) log-softmax probabilities
        """
        e = self.v(torch.tanh(
            self.W1(enc_out) + self.W2(dec_h.unsqueeze(0))
        )).squeeze(-1)                               # (n,)
        e = e.masked_fill(mask, float("-inf"))      # out-of-place — safe for autograd
        return F.log_softmax(e, dim=0)              # (n,)


class PointerNetwork(nn.Module):
    """
    Pointer Network for TSP (Vinyals et al., 2015).

    Architecture:
      1. Encoder LSTM reads all city coordinates → encoder outputs + hidden state
      2. Decoder LSTM steps one city at a time:
         - Input: embedding of the last chosen city
         - Attention over encoder outputs → pointer distribution
         - Greedy or sampled selection of next city
      3. Returns the decoded tour as a list of city indices.

    Training mode  (teacher_forcing=True):
        Ground-truth tour is fed as decoder input at each step.
        Returns log-probabilities for REINFORCE or supervised cross-entropy.

    Inference mode (teacher_forcing=False):
        Each step selects the city with the highest pointer score.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256,
                 n_layers: int = 1, node_dim: int = 2):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.node_dim   = node_dim

        self.encoder    = Encoder(embed_dim, hidden_dim, n_layers, node_dim)
        self.decoder    = nn.LSTMCell(embed_dim, hidden_dim)
        self.attention  = Attention(hidden_dim)
        self.city_embed = nn.Linear(node_dim, embed_dim)  # shared city embedding

        # Learnable first decoder input (replaces the "start" token)
        self.start_token = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x: torch.Tensor, tour: list = None):
        """
        x    : (n, node_dim)  city features in [0, 1]
        tour : list of n city indices (ground-truth, for teacher forcing)

        Returns
        -------
        log_probs : (n,) sum of log-probabilities of each selection (for loss)
        chosen    : list of n city indices (decoded tour)
        """
        n = x.shape[0]
        enc_out, h, c = self.encoder(x)              # (n, H), h, c

        # Collapse n_layers dimension to single vector for LSTMCell
        hx = h[-1]                                   # (1, H) → (H,) after squeeze below
        cx = c[-1]
        hx, cx = hx.squeeze(0), cx.squeeze(0)       # (H,)

        city_embs = self.city_embed(x)               # (n, E)

        visited   = set()                             # Python set — never touches autograd
        log_probs = []
        chosen    = []
        dec_input = self.start_token                  # (E,)

        for step in range(n):
            hx, cx    = self.decoder(dec_input.unsqueeze(0),
                                     (hx.unsqueeze(0), cx.unsqueeze(0)))
            hx, cx    = hx.squeeze(0), cx.squeeze(0)

            # Build a fresh mask tensor each step — no in-place modification
            mask = torch.tensor([i in visited for i in range(n)],
                                 dtype=torch.bool, device=x.device)
            lp   = self.attention(enc_out, hx, mask)           # (n,)
            log_probs.append(lp)

            if tour is not None:
                idx = tour[step]
            else:
                idx = lp.argmax().item()

            chosen.append(idx)
            visited.add(idx)
            dec_input = city_embs[idx]               # (E,)

        return torch.stack(log_probs), chosen        # (n, n), list[int]

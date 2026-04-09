"""
model.py — DIFUSCO architecture (Sun & Yang, 2023)

TODO (pair-coding session):
  - DifuscoGNN   : sparse GNN backbone (node + edge embeddings)
  - DifuscoModel : full diffusion wrapper
                   forward  → denoising score prediction
                   sample   → full DDPM / DDIM reverse chain

Predefined size presets will mirror the GNN convention:
  MODEL_SIZES["small" | "medium" | "large"] → (d, L, T)
  where d = embedding dim, L = GNN layers, T = diffusion steps
"""

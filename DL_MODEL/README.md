# Deep Learning Model

This module studies and implements deep learning approaches for combinatorial optimization. Unlike classical solvers, these models **learn heuristics or policies** from data, enabling fast inference on new instances without re-running a full optimization. The TSP (and its TSPTW-D variant) is used as the reference benchmark problem.

## Why Deep Learning for combinatorial optimization?

Classical exact methods (branch & bound, dynamic programming) are intractable at scale. DL models trade optimality guarantees for speed: once trained, they produce near-optimal solutions in milliseconds. They also generalize across problem instances, which makes them valuable for studying algorithm behavior at scale.

## Models

- **Pointer Networks (Ptr-Net)** — sequence-to-sequence model that outputs a permutation of input nodes using attention as a pointer
- **Transformers** — encoder-decoder architecture adapted for combinatorial optimization; captures global node relationships
- **Graph Neural Networks (GNN)** — operate directly on the graph structure; learn node/edge embeddings to guide tour construction
- **DIFUSCO** (`difusco/`) — diffusion model that generates TSP tours as a denoising process on the graph; current state of the art on supervised benchmarks (Sun & Yang, 2023)

## Structure

```
DL_MODEL/
├── difusco/
│   ├── model.py                      # DIFUSCO architecture (GNN backbone + diffusion wrapper)
│   ├── data.py                       # Data loading, feature building, evaluation
│   ├── train.py                      # Training loop (forward diffusion + score network)
│   ├── train.sh / train.ps1          # Platform launchers
│   ├── notebook_theory.ipynb         # Architecture & diffusion theory walkthrough
│   ├── modelisation_mathematique.ipynb  # Mathematical formulation
│   ├── benchmark_tsp.ipynb           # Plain TSP benchmark (documentation + quality metrics)
│   ├── benchmark_tsptwd.ipynb        # TSPTW-D benchmark (comparison against all methods)
│   ├── figures/                      # Saved plots (benchmark/, demo, etc.)
│   └── model/                        # Saved checkpoints (.pt) and loss curves
└── README.md
```

## References

- Sun, Z., & Yang, Y., *DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization* (NeurIPS 2023)
- Ho et al., *Denoising Diffusion Probabilistic Models* (NeurIPS 2020)
- Song et al., *Denoising Diffusion Implicit Models* (ICLR 2021)
- Vinyals et al., *Pointer Networks* (2015)
- Kool et al., *Attention, Learn to Solve Routing Problems!* (2019)

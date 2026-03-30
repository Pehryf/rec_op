# Deep Learning Model

This module studies and implements deep learning approaches for combinatorial optimization. Unlike classical solvers, these models **learn heuristics or policies** from data, enabling fast inference on new instances without re-running a full optimization. The TSP is used as the reference benchmark problem.

## Why Deep Learning for combinatorial optimization?

Classical exact methods (branch & bound, dynamic programming) are intractable at scale. DL models trade optimality guarantees for speed: once trained, they produce near-optimal solutions in milliseconds. They also generalize across problem instances, which makes them valuable for studying algorithm behavior at scale.

## Models

- **Pointer Networks (Ptr-Net)** — sequence-to-sequence model that outputs a permutation of input nodes using attention as a pointer
- **Transformers** — encoder-decoder architecture adapted for combinatorial optimization; captures global node relationships
- **Graph Neural Networks (GNN)** — operate directly on the graph structure; learn node/edge embeddings to guide tour construction
- **Lin-Kernighan-Helsgaun (LKH)** — classical heuristic augmented with learned components to guide the search

## References

- Vinyals et al., *Pointer Networks* (2015)
- Kool et al., *Attention, Learn to Solve Routing Problems!* (2019)
- Nazari et al., *Reinforcement Learning for Solving the Vehicle Routing Problem* (2018)
- Helsgott & Keld, *An Effective Implementation of the Lin-Kernighan TSP Heuristic* (2000)

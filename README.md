# Travelling Salesman Problem — Multi-approach Solver

## Table of Contents

- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Classical Methods](#classical-methods)
- [AI-Based Methods](#ai-based-methods)
- [Hybrid Methods](#hybrid-neural--classical)
- [Benchmarking](#benchmarking)
- [Contributing](#contributing)

---

## Problem Statement

The **Travelling Salesman Problem (TSP)** asks:

> Given a list of cities and the distances between them, what is the shortest possible route that visits each city exactly once and returns to the origin?

Formally, given a complete weighted graph $G = (V, E, w)$ with $n$ nodes, find a Hamiltonian cycle of minimum total weight.

It is **NP-hard** — no known polynomial-time exact algorithm exists for the general case.
This repo explores and benchmarks multiple resolution strategies, from exact solvers to neural approaches.

---

## Repository Structure

```
r_op/
├── dataset_raw/
│   ├── chunks/
│   │   ├── tsp_dataset/        # tsp_dataset.csv (LFS)
│   │   └── world/              # world.csv (LFS)
│   ├── solomon_dataset/        # VRPTW Solomon benchmarks (C1, C2, R1, R2, RC1, RC2)
│   └── SolomonTSPTW/           # Solomon TSPTW instances (.txt / .csv)
├── dataset_split/
│   ├── split_60_20_20/         # 60% train / 20% val / 20% test
│   └── split_70_15_15/         # 70% train / 15% val / 15% test
├── docs/
│   ├── README.md               # Contribution rules
│   └── bibliography/           # One entry per model/method
├── split_dataset.py            # Dataset splitter (stratified)
├── normalize_data.py           # Data normalization utilities
└── README.md
```

---

## Dataset

| Source | Format | Size | Description |
|--------|--------|------|-------------|
| `tsp_dataset.csv` | CSV | ~215 MB | Synthetic TSP instances (random cities, precomputed optimal routes) |
| `world.csv` | CSV | ~1.9M rows | Real-world geographic coordinates |
| Solomon benchmarks | CSV | — | Classic VRPTW instances (C1/C2/R1/R2/RC1/RC2) |
| SolomonTSPTW | CSV/TXT | — | TSP with Time Windows instances |

Large files are stored via **Git LFS**. After cloning, run:

```bash
git lfs pull
```

To regenerate the split:

```bash
python split_dataset.py --chunk_size 20000
# Options:
#   --raw_dir    Path to dataset_raw (default: ./dataset_raw)
#   --out_dir    Output directory   (default: ./dataset_split)
#   --chunk_size Max rows per chunk  (default: 20000)
```

---

## Classical Methods

These methods do not use machine learning. They are essential baselines and remain competitive on small to medium instances.

### Exact Methods

| Method | Complexity | Optimal? | Notes |
|--------|-----------|----------|-------|
| **Brute Force** | $O(n!)$ | Yes | Feasible only for $n \leq 12$ |
| **Dynamic Programming** (Held-Karp) | $O(n^2 \cdot 2^n)$ | Yes | Feasible up to $n \approx 20$ |
| **Branch & Bound** | Exponential (pruned) | Yes | Better in practice than brute force |
| **Integer Linear Programming** (ILP) | NP-hard (solver-dependent) | Yes | Exponential worst-case, but modern solvers (CPLEX, Gurobi, OR-Tools) handle $n \leq$ a few thousands via cutting planes & branch-and-cut |
| **Concorde** | Exponential (highly optimized) | Yes | Absolute gold standard solver; uses B&C with TSP-specific cuts (comb, blossom); solved instances up to ~100 000 nodes |

### Heuristics (constructive)

| Method | Complexity | Gap to optimal | Notes |
|--------|-----------|----------------|-------|
| **Nearest Neighbor** | $O(n^2)$ | ~20–25% | Greedy, fast, poor quality |
| **Greedy Edge Insertion** | $O(n^2 \log n)$ | ~15–20% | Build tour by adding shortest edges |
| **Christofides Algorithm** | $O(n^3)$ | ≤ 50% (guaranteed) | Approximation ratio 1.5× — slightly improved by Karlin et al. (2020) to $1.5 - \varepsilon$ |
| **Savings Algorithm** (Clarke-Wright) | $O(n^2 \log n)$ | ~10–15% | Originally for VRP, applies to TSP |

### Metaheuristics (improvement)

| Method | Key idea | Strengths |
|--------|----------|-----------|
| **2-opt / 3-opt** | Swap edges to eliminate crossings | Simple, effective local search |
| **Or-opt** | Relocate segments of 1–3 cities | Faster than 3-opt, often paired with LK |
| **Lin-Kernighan (LK)** | Variable-depth edge swaps | Foundation of the best classical solvers |
| **LKH-3** (Lin-Kernighan-Helsgaun) | LK + candidate lists + penalty functions | State of the art classical heuristic; wins most TSP competitions |
| **Simulated Annealing (SA)** | Accept worse solutions probabilistically | Escapes local optima |
| **Tabu Search** | Short-term memory to avoid revisiting | Good on medium instances |
| **Variable Neighborhood Search (VNS)** | Systematic neighborhood change | Robust, easy to combine with other methods |
| **Ant Colony Optimization (ACO)** | Pheromone-guided probabilistic paths | Parallelizable, robust |
| **Genetic Algorithms (GA)** | Crossover + mutation on tour population | Good diversity, slow convergence |
| **EAX (Edge Assembly Crossover)** | Genetic crossover preserving edge structure | Best known evolutionary algorithm for TSP |
| **POPMUSIC** | Decompose into overlapping sub-tours, optimize locally | State of the art for very large instances (>100k nodes) |
| **Slime Mold Algorithm** | Bio-inspired network flow minimization | Recent, competitive on large graphs |

---

## AI-Based Methods

### Machine Learning

| Method | Description |
|--------|-------------|
| **Self-Organizing Maps (SOM)** | Unsupervised neural network that deforms a ring onto city positions |

### Deep Learning

| Method | Description |
|--------|-------------|
| **Pointer Networks (Ptr-Net)** | Seq2seq with attention that outputs a permutation of input cities |
| **Graph Neural Networks (GNN)** | Learns on graph structure; edge scores guide tour construction |
| **Transformers** | Attention-based encoder-decoder adapted for combinatorial optimization |
| **DIFUSCO** (2023) | Diffusion model that generates tours as a denoising process on the graph; current state of the art on supervised benchmarks |

### Reinforcement Learning

| Method | Description |
|--------|-------------|
| **Policy Gradient (REINFORCE)** | Trains a policy to construct tours, rewarded by tour length |
| **Actor-Critic (A2C / PPO)** | Reduces variance in gradient estimation |
| **Attention Model (AM)** | Transformer-based policy trained end-to-end with RL |
| **POMO** (2020) | Policy Optimization with Multiple Optima — exploits TSP rotational symmetry to train on $n$ starting points simultaneously; dominant RL baseline post-2020 |
| **Sym-NCO** (2022) | Extends POMO by enforcing all symmetries of the TSP (rotation + reflection) during training |

### Hybrid (Neural + Classical)

| Method | Description |
|--------|-------------|
| **EAS** (2022) | Efficient Active Search — fine-tunes the neural policy at inference time on a single instance |
| **L2I** (Learning to Improve) | Neural network learns to select which local-search move to apply next; combines deep learning with 2-opt/or-opt |
| **GLOP** (2023) | Global-Local Policy — neural model decomposes large instances into sub-tours, each solved by LKH; handles >1 000 nodes |

---

## Benchmarking

Every method must be evaluated on random graphs with the following sizes and on the provided test split:

| Nodes | 1 | 10 | 100 | 1 000 | 10 000 | 100 000 |
|-------|---|----|-----|-------|--------|---------|

### Metrics

| Metric | Definition |
|--------|-----------|
| **% success** | Runs where gap to optimal is ≤ 1% |
| **% non-detection** | Algorithm believes it converged but real gap > 1% |
| **% false detection** | Algorithm stops thinking it failed but solution is within 1% |
| **Inference time** | Average time to produce a solution (ms or s, CPU and GPU if applicable) |

See [`docs/README.md`](docs/README.md) for the full contribution and documentation format.

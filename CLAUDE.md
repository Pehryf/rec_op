# Claude Code — Project Context

## Project

Multi-approach solver for the **Travelling Salesman Problem (TSP)**.
Each algorithm lives in its own branch and must satisfy every item in `TODO.md`.

---

## Branch rules

- **Every new branch must be created from `DL_gnn`**, not from `master` or any other branch.
- One branch = one algorithm.
- Branch names use lowercase with underscores.

```
DL_gnn  (base for all new branches)
├── tabu_search
├── ptr_net
└── difusco
```

### Create a new branch

```bash
git checkout DL_gnn
git checkout -b <branch_name>
```

---

## Giacomo's algorithms

| Algorithm | Category | Branch | Status |
|-----------|----------|--------|--------|
| Graph Neural Network (GNN) | Deep Learning | `DL_gnn` | In progress |
| Tabu Search | Metaheuristic | `tabu_search` | To do |
| Pointer Networks (Ptr-Net) | Deep Learning | `ptr_net` | To do |
| DIFUSCO | Deep Learning | `difusco` | To do |

---

## Required deliverables per algorithm (from TODO.md)

Each branch must contain a notebook with all of the following:

| # | Item | Notes |
|---|------|-------|
| 1 | Definition | Origin, category, key concepts |
| 2 | Formal description | Mathematical formulation, objective function, variables |
| 3 | Architecture / algorithm steps | Pseudocode or diagram |
| 4 | Complexity analysis | Time and space |
| 5 | Strengths and limitations | |
| 6 | Use case explanation | When and why to use this approach |
| 7 | Code implementation | Readable, commented, PEP-compliant — **in separate `.py` files, imported by the notebook** |
| 8 | Code demonstration | Small instance walkthrough |
| 9 | Benchmark | TSP instances at n = 1, 10, 100, 10 000, 100 000 |
| 10 | Per-size metrics | Success rate, false positive rate, undetermined rate |
| 11 | Experimental analysis | Behavior, scalability, comparison with other models |
| 12 | Model file | Saved weights or parameters (if applicable) |
| 13 | Bibliographic references | |

---

## Notebook structure convention (established in DL_gnn)

Each algorithm has **two notebooks**:

| Notebook | Contents |
|----------|----------|
| `notebook_theory.ipynb` | Items 1–6 + 11 + 13 (all markdown, no code) |
| `notebook_implementation.ipynb` | Items 7–10 + 12 (code cells, imports from `.py` files) |

The implementation notebook must have a **model/config selection cell at the top** (cell 1)
so the user can switch between presets (e.g. small / medium / large) without touching the rest.

---

## File structure convention (established in DL_gnn)

```
<MODEL_DIR>/
├── <algo>/
│   ├── model.py              # architecture / algorithm
│   ├── data.py               # data helpers and dataset loader
│   ├── train.py              # training loop (CLI + importable function)
│   ├── notebook_theory.ipynb
│   ├── notebook_implementation.ipynb
│   └── model/               # saved weights (.pt) and losses (.npy)
```

For non-DL algorithms (e.g. Tabu Search), `train.py` and `model/` are replaced by
a `solver.py` that contains the algorithm and any parameter-tuning logic.

---

## Dataset

- TSP chunks: `dataset_raw/_chunks/tsp_dataset/` — one CSV per instance, `city_coordinates` column
- Solomon TSPTW: `dataset_raw/solomon_dataset/` — subdirs C1, C2, R1, R2, RC1 (Git LFS)
- Loader: `data.py::load_cities(n, source="tsp"|"solomon")` — returns `torch.Tensor(n, 2)` in `[0, 1]`

Coordinates in the TSP chunks are in `~[0, 100]` — always normalise to `[0, 1]`.

---

## Key reminders

- Training labels for GNN require brute-force optimal tour — only feasible for **n ≤ 10**.
- `train.py` supports `--size small|medium|large`, `--source random|tsp|solomon`, `--resume <path>`.
- `MODEL_SIZES = {"small": (64,4), "medium": (128,6), "large": (256,8)}` is defined in `model.py`.
- Always keep training logic in `.py` files; notebooks only import and call.

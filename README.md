# AI Approaches for Combinatorial Optimization

## Context

This repository is the AI axis of the **CesiCDP × ADEME** project. The objective is to study, implement, and compare AI-based algorithms for combinatorial optimization problems. The **Travelling Salesman Problem (TSP)** is used as the benchmark problem to evaluate and compare the approaches.

## Approaches explored

- Machine Learning: Graph Neural Networks (GNN), Transformers, Pointer Networks (Ptr-Net)
- Deep Learning: Lin-Kernighan-Helsgaun (LKH)
- Reinforcement Learning (RL)
- Genetic Algorithms
- Simulated Annealing
- Ant Colony Optimization (ACO)
- Self-Organizing Maps (SOMs)

Each approach is developed in its own subdirectory.

## Structure

```
rec_op/
├── README.md
└── DL_MODEL/       # Deep Learning model implementations
```

## Deliverables

### Livrable 1 — Modélisation (check)

Jupyter Notebook covering:
- Formal representation of the data, problem, and objective function
- Theoretical complexity analysis
- At least one additional constraint incorporated into the model
- Bibliographic references

### Livrable final

**Part 1 — Modélisation** (updated from livrable 1):
- Formal modeling
- Description of the chosen resolution methods

**Part 2 — Implémentation & Exploitation**:
- Implementation of the algorithms (Python, PEP-compliant, readable and commented)
- Demonstration on test instances
- Full experimental study: performance benchmarks, behavioral analysis, limitations, and improvement proposals

### Soutenance

Results-oriented oral presentation with live code demo on small instances, progress summary, and challenges/next steps.

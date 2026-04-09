"""
data.py — Data utilities for DIFUSCO (TSP and TSPTW-D)

TODO (pair-coding session):
  - load_dataset(n)              : load pre-split instances of size n
  - generate_time_windows(n)     : random [a_i, b_i] TW generation
  - generate_perturbations(n)    : random arc-cost disruption events
  - build_tsp_features(coords)   : (n, 2) coords → (node_feats, edge_feats)
  - build_tsptwd_features(...)   : TSPTW-D version with TW + perturbations
  - binarize(p, threshold)       : continuous edge probabilities → {0,1} adj
  - tour_from_adj(adj)           : adjacency matrix → ordered tour (greedy)
  - evaluate(tour, inst)         : compute cost + TW violations
"""

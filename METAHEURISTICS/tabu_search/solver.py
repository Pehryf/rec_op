"""
solver.py — Tabu Search for TSP

TabuSearch : main solver class
  - Initial solution via nearest-neighbour heuristic
  - Neighbourhood: all 2-opt edge swaps
  - Tabu list: fixed-length deque of (i, j) move pairs
  - Aspiration criterion: accept a tabu move if it beats the best-known tour
  - Stopping criterion: max_iterations or patience (no improvement)

Usage
-----
    from solver import TabuSearch
    from data import load_cities

    coords = load_cities(100, source="tsp")
    ts = TabuSearch(coords, tabu_tenure=20, max_iter=5000, patience=500)
    best_tour, best_length, history = ts.run()
"""

import random
from collections import deque

import torch


class TabuSearch:
    """
    Tabu Search solver for TSP.

    Parameters
    ----------
    coords       : (n, 2) tensor of city coordinates in [0, 1]²
    tabu_tenure  : number of iterations a move stays forbidden
    max_iter     : hard iteration limit
    patience     : stop early if no improvement for this many iterations
    seed         : random seed for reproducibility
    """

    def __init__(self, coords: torch.Tensor, tabu_tenure: int = 20,
                 max_iter: int = 5000, patience: int = 500, seed: int = None):
        self.coords      = coords
        self.n           = coords.shape[0]
        self.tabu_tenure = tabu_tenure
        self.max_iter    = max_iter
        self.patience    = patience

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Pre-compute full distance matrix — O(n²) once, O(1) per lookup
        self.dist = torch.cdist(coords, coords)   # (n, n)

    # ── Tour evaluation ───────────────────────────────────────────────────────

    def tour_length(self, tour: list) -> float:
        """Total Euclidean length of a closed tour."""
        total = 0.0
        for k in range(self.n):
            total += self.dist[tour[k], tour[(k + 1) % self.n]].item()
        return total

    # ── Initial solution ──────────────────────────────────────────────────────

    def _nearest_neighbour(self, start: int = 0) -> list:
        """Greedy nearest-neighbour construction from a given start city."""
        visited = [False] * self.n
        tour    = [start]
        visited[start] = True
        for _ in range(self.n - 1):
            last  = tour[-1]
            dists = self.dist[last].clone()
            for i, v in enumerate(visited):
                if v:
                    dists[i] = float("inf")
            nxt = dists.argmin().item()
            tour.append(nxt)
            visited[nxt] = True
        return tour

    # ── Neighbourhood: 2-opt ─────────────────────────────────────────────────

    @staticmethod
    def _apply_2opt(tour: list, i: int, j: int) -> list:
        """
        Reverse the segment tour[i+1 … j] (standard 2-opt move).
        Returns a new tour list — does not modify in place.
        """
        return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]

    def _delta_2opt(self, tour: list, i: int, j: int) -> float:
        """
        Cost change of applying 2-opt(i, j) without rebuilding the full tour.

        Removes edges (tour[i], tour[i+1]) and (tour[j], tour[j+1 % n]).
        Adds    edges (tour[i], tour[j])   and (tour[i+1], tour[j+1 % n]).
        """
        n  = self.n
        a, b = tour[i],         tour[(i + 1) % n]
        c, d = tour[j],         tour[(j + 1) % n]
        d_ab = self.dist[a, b].item()
        d_cd = self.dist[c, d].item()
        d_ac = self.dist[a, c].item()
        d_bd = self.dist[b, d].item()
        return (d_ac + d_bd) - (d_ab + d_cd)   # negative = improvement

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> tuple:
        """
        Run Tabu Search.

        Returns
        -------
        best_tour   : list of city indices
        best_length : float
        history     : list of best-length values per iteration (for plotting)
        """
        current_tour   = self._nearest_neighbour()
        current_length = self.tour_length(current_tour)
        best_tour      = current_tour[:]
        best_length    = current_length

        tabu_list      = deque(maxlen=self.tabu_tenure)
        tabu_set       = set()
        history        = [best_length]
        no_improve     = 0

        for iteration in range(self.max_iter):
            best_delta = float("inf")
            best_move  = None
            best_candidate = None

            # Evaluate all 2-opt neighbours
            for i in range(self.n - 1):
                for j in range(i + 2, self.n):
                    if i == 0 and j == self.n - 1:
                        continue   # trivial reverse of full tour

                    delta = self._delta_2opt(current_tour, i, j)
                    move  = (i, j)

                    is_tabu = move in tabu_set
                    candidate_length = current_length + delta

                    # Accept if: not tabu, OR aspiration (beats global best)
                    if (not is_tabu or candidate_length < best_length):
                        if delta < best_delta:
                            best_delta     = delta
                            best_move      = move
                            best_candidate = self._apply_2opt(current_tour, i, j)

            if best_move is None:
                break   # no valid move found

            # Apply best move
            current_tour   = best_candidate
            current_length = current_length + best_delta

            # Update tabu list
            if len(tabu_list) == self.tabu_tenure:
                evicted = tabu_list[0]
                tabu_set.discard(evicted)
            tabu_list.append(best_move)
            tabu_set.add(best_move)

            # Update global best
            if current_length < best_length:
                best_tour   = current_tour[:]
                best_length = current_length
                no_improve  = 0
            else:
                no_improve += 1

            history.append(best_length)

            if no_improve >= self.patience:
                break

        return best_tour, best_length, history

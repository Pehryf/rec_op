"""
solver.py — Tabu Search for TSP / TSPTW-D

TabuSearch : main solver class
  - Initial solution via nearest-neighbour heuristic
  - Neighbourhood: all 2-opt edge swaps
  - Tabu list: fixed-length deque of (i, j) move pairs
  - Aspiration criterion: accept a tabu move if it beats the best-known tour
  - Stopping criterion: max_iterations or patience (no improvement)
  - Optional TSPTW-D extensions:
      * Time windows [a_i, b_i]: vehicle must arrive within [a_i, b_i];
        waiting is allowed (arrive early → wait), late arrivals are penalised.
      * Dynamic travel costs: c_ij(t) = dist_ij * (1 + alpha) for any
        perturbation active on edge (i, j) at departure time t.
      * Objective: minimise total tour time (return time to depot) instead
        of total Euclidean distance.

Usage (plain TSP)
-----------------
    from solver import TabuSearch
    from data import load_cities

    coords = load_cities(100, source="tsp")
    ts = TabuSearch(coords, tabu_tenure=20, max_iter=5000, patience=500)
    best_tour, best_obj, history = ts.run()

Usage (TSPTW-D)
---------------
    from solver import TabuSearch
    from data import load_solomon_instance

    coords, time_windows, service_times = load_solomon_instance(50)
    perturbations = [(2, 5, 100.0, 300.0, 0.5)]  # edge (2,5) slowed 50% in [100, 300]
    ts = TabuSearch(
        coords,
        time_windows=time_windows,
        service_times=service_times,
        perturbations=perturbations,
        penalty_coeff=1000.0,
        tabu_tenure=20, max_iter=5000, patience=500,
    )
    best_tour, best_obj, history = ts.run()
"""

import random
from collections import deque

import torch


class TabuSearch:
    """
    Tabu Search solver for TSP / TSPTW-D.

    Plain TSP parameters
    --------------------
    coords       : (n, 2) tensor of city coordinates

    TSPTW-D extensions (all optional — omit for plain TSP)
    -------------------------------------------------------
    time_windows  : (n, 2) tensor  [[a_i, b_i], ...]
                    ready time a_i and deadline b_i for each city.
    service_times : (n,)   tensor  [s_i, ...]
                    time spent servicing city i (added after waiting if needed).
    perturbations : list of (i, j, t_start, t_end, alpha) tuples
                    c_ij(t) = dist[i,j] * (1 + alpha) when t ∈ [t_start, t_end].
    penalty_coeff : weight λ for TW-violation penalty in the objective:
                    Z(tour) = return_time + λ * total_violation

    Common parameters
    -----------------
    tabu_tenure  : number of iterations a move stays forbidden
    max_iter     : hard iteration limit
    patience     : stop early if no improvement for this many iterations
    seed         : random seed for reproducibility
    """

    def __init__(self, coords: torch.Tensor,
                 time_windows: torch.Tensor = None,
                 service_times: torch.Tensor = None,
                 perturbations: list = None,
                 penalty_coeff: float = 1000.0,
                 tabu_tenure: int = 20,
                 max_iter: int = 5000,
                 patience: int = 500,
                 seed: int = None):
        self.coords        = coords
        self.n             = coords.shape[0]
        self.time_windows  = time_windows
        self.service_times = service_times
        self.perturbations = perturbations or []
        self.penalty_coeff = penalty_coeff
        self.tabu_tenure   = tabu_tenure
        self.max_iter      = max_iter
        self.patience      = patience

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Pre-compute base distance matrix — O(n²) once, O(1) per lookup
        self.dist = torch.cdist(coords, coords)   # (n, n)

    # ── Tour evaluation ───────────────────────────────────────────────────────

    def tour_length(self, tour: list) -> float:
        """Total Euclidean length of a closed tour (plain TSP objective)."""
        total = 0.0
        for k in range(self.n):
            total += self.dist[tour[k], tour[(k + 1) % self.n]].item()
        return total

    def _travel_cost(self, i: int, j: int, t: float) -> float:
        """
        Travel cost from city i to city j departing at time t.

        Returns the base Euclidean distance scaled by any active perturbation:
            c_ij(t) = dist[i,j] * (1 + alpha)   if edge (i,j) is perturbed at t
            c_ij(t) = dist[i,j]                  otherwise
        """
        base = self.dist[i, j].item()
        for (pi, pj, t_start, t_end, alpha) in self.perturbations:
            if (pi == i and pj == j) or (pi == j and pj == i):
                if t_start <= t <= t_end:
                    return base * (1.0 + alpha)
        return base

    def _compute_schedule(self, tour: list) -> tuple:
        """
        Propagate the schedule through the tour under TSPTW-D constraints.

        Starting at tour[0] (depot) at time 0, for each city in order:
          τ_i  = arrival time
          if τ_i > b_i  → violation += τ_i - b_i   (late arrival)
          d_i  = max(τ_i, a_i) + s_i                (depart after service)
          τ_next = d_i + c(city, next_city, d_i)    (time-dependent travel)

        Returns
        -------
        return_time     : float — arrival time back at depot (raw objective)
        total_violation : float — sum of deadline exceedances (penalty term)
        """
        t = 0.0
        total_violation = 0.0

        for k in range(self.n):
            city = tour[k]
            a_i = self.time_windows[city, 0].item()
            b_i = self.time_windows[city, 1].item()
            s_i = self.service_times[city].item()

            # Late arrival → accumulate violation
            if t > b_i:
                total_violation += t - b_i

            # Departure: wait if early, then serve
            depart = max(t, a_i) + s_i

            # Travel to next city (wraps to depot after last client)
            next_city = tour[(k + 1) % self.n]
            t = depart + self._travel_cost(city, next_city, depart)

        # t is the arrival time back at tour[0] (depot)
        return t, total_violation

    def _penalized_obj(self, tour: list) -> float:
        """
        Penalized objective for TSPTW-D:
            Z(tour) = return_time + penalty_coeff * total_violation
        """
        return_time, violation = self._compute_schedule(tour)
        return return_time + self.penalty_coeff * violation

    def tour_time(self, tour: list) -> tuple:
        """
        Return (return_time, total_violation) for a TSPTW-D tour.
        Alias for _compute_schedule — public API for notebooks.
        """
        return self._compute_schedule(tour)

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
        O(1) cost change of applying 2-opt(i, j) — valid for plain TSP only.

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
        Run Tabu Search (plain TSP or TSPTW-D depending on time_windows).

        In plain TSP mode  (time_windows=None):
          - objective: total Euclidean distance
          - delta evaluated in O(1) per move → O(n²) per iteration

        In TSPTW-D mode (time_windows provided):
          - objective: return_time + penalty_coeff * total_violation
          - schedule propagated in O(n) per move → O(n³) per iteration

        Returns
        -------
        best_tour : list of city indices
        best_obj  : float — objective value (distance or penalized time)
        history   : list of best objective values per iteration (for plotting)
        """
        use_tw = self.time_windows is not None

        current_tour = self._nearest_neighbour()
        current_obj  = self._penalized_obj(current_tour) if use_tw else self.tour_length(current_tour)
        best_tour    = current_tour[:]
        best_obj     = current_obj

        tabu_list  = deque(maxlen=self.tabu_tenure)
        tabu_set   = set()
        history    = [best_obj]
        no_improve = 0

        for _ in range(self.max_iter):
            best_delta    = float("inf")
            best_move     = None
            best_cand_obj = float("inf")
            best_cand_tour = None

            # Evaluate all 2-opt neighbours
            for i in range(self.n - 1):
                for j in range(i + 2, self.n):
                    if i == 0 and j == self.n - 1:
                        continue   # trivial reverse of full tour

                    move    = (i, j)
                    is_tabu = move in tabu_set

                    if use_tw:
                        # O(n): build candidate tour and propagate schedule
                        candidate = self._apply_2opt(current_tour, i, j)
                        cand_obj  = self._penalized_obj(candidate)
                        delta     = cand_obj - current_obj
                    else:
                        # O(1): closed-form distance delta
                        delta     = self._delta_2opt(current_tour, i, j)
                        cand_obj  = current_obj + delta
                        candidate = None   # materialise only if selected

                    # Accept if not tabu, or aspiration (beats global best)
                    if not is_tabu or cand_obj < best_obj:
                        if delta < best_delta:
                            best_delta     = delta
                            best_move      = move
                            best_cand_obj  = cand_obj
                            best_cand_tour = candidate

            if best_move is None:
                break   # no admissible move found (rare)

            # Materialise tour for plain TSP if not yet built
            if best_cand_tour is None:
                best_cand_tour = self._apply_2opt(current_tour, *best_move)

            current_tour = best_cand_tour
            current_obj  = best_cand_obj

            # Update tabu list (FIFO, evict oldest when full)
            if len(tabu_list) == self.tabu_tenure:
                tabu_set.discard(tabu_list[0])
            tabu_list.append(best_move)
            tabu_set.add(best_move)

            # Update global best
            if current_obj < best_obj:
                best_tour  = current_tour[:]
                best_obj   = current_obj
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_obj)

            if no_improve >= self.patience:
                break

        return best_tour, best_obj, history

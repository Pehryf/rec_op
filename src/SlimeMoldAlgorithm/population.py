from agent import Agent
from operators import *
import numpy as np
import random

class Population:
    def __init__(self, graph, nb_agents):
        self.graph = graph
        self.agents = [Agent.random(graph) for _ in range(nb_agents)]

    @property
    def best(self):
        return min(self.agents, key=lambda agent: agent.cost)

    @property
    def worst(self):
        return max(self.agents, key=lambda agent: agent.cost)
    
    def update_weights(self):
        n = len(self.agents)
        self.agents.sort(key=lambda a: a.cost)
        bF = self.best.cost
        wF = self.worst.cost
        if bF == wF:
            return
        for i in range (0, n):
            s = self.agents[i].cost
            calc = np.log((bF - s) / (bF - wF) + 1)
            if i < n/2:
                self.agents[i].weight = 1 + random.uniform(0, 1) * calc
            else:
                self.agents[i].weight = 1 - random.uniform(0, 1) * calc

    def step(self, t=0, phase_length=20, opt_ratio=0.25, max_opt_passes=5):
        self.update_weights()
        X_b = self.best.turn
        bF = self.best.cost

        for agent in self.agents:
            p = np.tanh(np.abs(agent.cost - bF))
            if random.uniform(0, 1) < p:
                X_A = random.choices(self.agents, weights=[a.weight for a in self.agents], k=1)[0]
                remaining = [a for a in self.agents if a is not X_A]
                remaining_weights = [a.weight for a in remaining]
                X_B = random.choices(remaining, weights=remaining_weights, k=1)[0]
                new_turn = ox_crossover(X_b, ox_crossover(X_A.turn, X_B.turn))
            else:
                new_turn = random_swap(agent.turn)

            new_cost = self.graph.turn_cost(new_turn)
            if new_cost < agent.cost:
                agent.turn = new_turn
                agent.cost = new_cost

            if t % phase_length >= phase_length * (1 - opt_ratio):
                agent.turn = two_opt(agent.turn, self.graph, max_passes=max_opt_passes)
                agent.cost = self.graph.turn_cost(agent.turn)

    def run(self, T, phase_length=15, opt_ratio=0.25, max_opt_passes=5, on_step=None):
        opt_phase_start = int(phase_length * (1 - opt_ratio))
        best_before_opt = None
        best_after_opt = None
        opt_was_stale = False
        history = []

        for t in range(T):
            cycle_pos = t % phase_length
            is_opt_phase = cycle_pos >= opt_phase_start
            phase = "2-opt" if is_opt_phase else "exploration"

            self.step(t, phase_length=phase_length, opt_ratio=opt_ratio, max_opt_passes=max_opt_passes)
            best_cost = self.best.cost
            history.append(best_cost)

            if on_step:
                on_step(t, self, history, phase, stopped=False)

            # Entrée en 2-opt : snapshot du best avant optimisation
            if cycle_pos == opt_phase_start:
                best_before_opt = best_cost

            # Pendant le 2-opt : mettre à jour stale en continu
            if is_opt_phase:
                opt_was_stale = best_before_opt is not None and best_cost >= best_before_opt
                best_after_opt = best_cost

            # Pendant l'exploration : si le 2-opt précédent était stale
            # et le best n'a pas bougé depuis → stop
            if not is_opt_phase and opt_was_stale:
                if best_after_opt is not None and best_cost >= best_after_opt:
                    if on_step:
                        on_step(t, self, history, phase, stopped=True)
                    break

        return self.best.cost, len(history), history, self.best.turn


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

    def step(self, t=0, phase_length=20, opt_ratio=0.25):
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
                agent.turn = two_opt(agent.turn, self.graph)
                agent.cost = self.graph.turn_cost(agent.turn)


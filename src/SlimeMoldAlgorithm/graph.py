import math
import random
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, cities):
        self.cities = cities  # List of tuples (x, y)

    def distance(self, i, j):
        x1, y1 = self.cities[i]
        x2, y2 = self.cities[j]
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def turn_cost(self, turn):
        cost = 0
        for i in range(len(turn) - 1):
            cost += self.distance(turn[i], turn[i + 1])
        cost += self.distance(turn[-1], turn[0])
        return cost

    @classmethod
    def random(cls, nb_cities, x_range=(0, 100), y_range=(0, 100)):
        cities = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(nb_cities)]
        return cls(cities)

    def plot(self, turn=None, optimal_turn=None, history=None, best_known=None):
        if not hasattr(self, '_fig'):
            self._fig, (self._ax_graph, self._ax_cost) = plt.subplots(1, 2, figsize=(14, 6))

        self._ax_graph.cla()
        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        self._ax_graph.scatter(x, y, c='blue', zorder=3)

        if optimal_turn is not None:
            opt_x = [self.cities[i][0] for i in optimal_turn] + [self.cities[optimal_turn[0]][0]]
            opt_y = [self.cities[i][1] for i in optimal_turn] + [self.cities[optimal_turn[0]][1]]
            self._ax_graph.plot(opt_x, opt_y, c='green', zorder=1, linewidth=1, label='optimal')

        if turn is not None:
            turn_x = [self.cities[i][0] for i in turn] + [self.cities[turn[0]][0]]
            turn_y = [self.cities[i][1] for i in turn] + [self.cities[turn[0]][1]]
            self._ax_graph.plot(turn_x, turn_y, c='red', zorder=2, linewidth=1, label='SMA')

        self._ax_graph.legend()
        self._ax_graph.set_title('Tournée')

        self._ax_cost.cla()
        if history is not None:
            self._ax_cost.plot(history, c='red', label='SMA')
        if best_known is not None:
            self._ax_cost.axhline(y=best_known, c='green', linestyle='--', label=f'optimal ({best_known})')
        self._ax_cost.set_title('Convergence')
        self._ax_cost.set_xlabel('Itération')
        self._ax_cost.set_ylabel('Coût')
        self._ax_cost.legend()

        plt.pause(0.01)

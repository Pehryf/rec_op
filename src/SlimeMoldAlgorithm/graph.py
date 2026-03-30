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

    def tour_cost(self, tour):
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.distance(tour[i], tour[i + 1])
        cost += self.distance(tour[-1], tour[0])
        return cost

    @classmethod
    def random(cls, nb_cities, x_range=(0, 100), y_range=(0, 100)):
        cities = [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(nb_cities)]
        return cls(cities)

    def plot(self, tour=None):
        plt.clf()

        x = [city[0] for city in self.cities]
        y = [city[1] for city in self.cities]
        plt.scatter(x, y, c='blue', zorder=2)

        if tour is not None:
            tour_x = [self.cities[i][0] for i in tour] + [self.cities[tour[0]][0]]
            tour_y = [self.cities[i][1] for i in tour] + [self.cities[tour[0]][1]]
            plt.plot(tour_x, tour_y, c='red', zorder=1)

        plt.title('TSP Graph')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.pause(0.01)

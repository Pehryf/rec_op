from graph import Graph
from population import Population
import matplotlib.pyplot as plt

n = 200
T = 70
graph = Graph.random(n)
population = Population(graph, 20)
plt.ion()
for t in range(T):
    population.step(t, T)
    graph.plot(population.best.turn)
    print(f"iter {t} - best cost: {population.best.cost:.2f}")

plt.show(block=True)
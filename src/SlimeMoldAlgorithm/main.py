from graph import Graph
from population import Population
import matplotlib.pyplot as plt

n = 50
graph = Graph.random(n)
population = Population(graph, 20)
plt.ion()
for i in range(0, 500):
    population.step()
    graph.plot(population.best.turn)
    print(f"iter {i} - best cost: {population.best.cost:.2f}")

plt.show(block=True)
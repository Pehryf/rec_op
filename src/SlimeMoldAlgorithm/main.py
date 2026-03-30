import matplotlib.pyplot as plt
from graph import Graph

graph = Graph.random(20)
tour = list(range(len(graph.cities)))
print("Tour cost:", graph.tour_cost(tour))
graph.plot(tour)
plt.show(block=True)
from graph import Graph
from population import Population
import matplotlib.pyplot as plt

# berlin52 - solution optimale certifiée : 7542
cities = [
    (565,575),(25,185),(345,750),(945,685),(845,655),(880,660),(25,230),
    (525,1000),(580,1175),(650,1130),(1605,620),(1220,580),(1465,200),
    (1530,5),(845,680),(725,370),(145,665),(415,635),(510,875),(560,365),
    (300,465),(520,585),(480,415),(835,625),(975,580),(1215,245),(1320,315),
    (1250,400),(660,180),(410,250),(420,555),(575,665),(1150,1160),(700,580),
    (685,595),(685,610),(770,610),(795,645),(720,635),(760,650),(475,960),
    (95,260),(875,920),(700,500),(555,815),(830,485),(1170,65),(830,610),
    (605,625),(595,360),(1340,725),(1740,245)
]
best_known_distance = 7542

graph = Graph(cities)

T = 50
population = Population(graph, 40)
history = []

plt.ion()
for t in range(T):
    population.step(t, phase_length=20, opt_ratio=0.25)
    best_cost = population.best.cost
    history.append(best_cost)
    graph.plot(population.best.turn, history=history, best_known=best_known_distance)
    print(f"iter {t} - best cost: {best_cost:.2f} | optimal: {best_known_distance} | gap: {100 * (best_cost - best_known_distance) / best_known_distance:.1f}%")

plt.show(block=True)

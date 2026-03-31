import random

def random_swap(turn):
    tmp = turn[:]
    a, b = random.randint(0, len(turn)-1), random.randint(0, len(turn)-1)
    tmp[a], tmp[b] = tmp[b], tmp[a]
    return tmp


def ox_crossover(parent1, parent2):
    n = len(parent1)
    a = random.randint(0, n - 1)
    b = random.randint(0, n - 1)
    a, b = min(a, b), max(a, b)

    enfant = [None] * n
    enfant[a:b] = parent1[a:b]
    placed = set(parent1[a:b])

    remainging_cities = [v for v in parent2 if v not in placed]

    j = 0
    for i in range(0, n):
        if enfant[i] is None:
            enfant[i] = remainging_cities[j]
            j += 1
    
    return enfant

def two_opt(turn, graph, max_passes=5):
    n = len(turn)
    for _ in range(max_passes):
        improved = False
        for i in range(0, n-2):
            for j in range(i+2, n-1):
                gain = (graph.distance(turn[i], turn[i+1]) + graph.distance(turn[j], turn[j+1])
                    - graph.distance(turn[i], turn[j]) - graph.distance(turn[i+1], turn[j+1]))
                if gain > 0:
                    turn[i+1 : j+1] = turn[i+1 : j+1][::-1]
                    improved = True
        if not improved:
            break
    return turn
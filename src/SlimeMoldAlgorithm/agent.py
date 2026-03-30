import random

class Agent:
    def __init__(self, graph, turn):
        self.graph = graph
        self.turn = turn
        self.weight = 1.0
        self.cost = self.graph.turn_cost(turn)

    @classmethod
    def random(cls, graph):
        turn = list(range(len(graph.cities)))
        random.shuffle(turn)
        return cls(graph, turn)

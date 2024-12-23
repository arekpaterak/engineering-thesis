from typing import Optional

import matplotlib
import numpy as np
import torch_geometric as pyg

import matplotlib.pyplot as plt
import networkx as nx
from gymnasium.spaces import MultiBinary


def draw_graph(
    graph: pyg.data.Data,
    with_labels: bool = True,
) -> matplotlib.pyplot:
    G = pyg.utils.to_networkx(graph, node_attrs=['x', 'pos'], to_undirected=True)

    # Extract positions as a dictionary for NetworkX visualization
    # NetworkX expects a dictionary {node_id: (x, y)}
    fixed_pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.pos.size(0))}

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=fixed_pos, with_labels=with_labels, node_color="skyblue", node_size=500, edge_color="black", font_size=10)
    return plt


class MultiBinaryWithLimitedSampling(MultiBinary):
    """
    An extension of `gymnasium.spaces.MultiBinary` with a possible sampling of k elements from all as 1s.
    """

    def __init__(self, n, seed = None) -> None:
        super().__init__(n, seed)

    def sample_limited(self, k: int) -> np.ndarray:
        size = self.shape[0]
        result = np.zeros(size, dtype="int")
        choices = np.random.choice(size, size=k, replace=False)
        result[choices] = 1
        return result


def route_from_circuit(circuit: list[int]):
    route = [0]
    next_node = circuit[0]
    while next_node != 0:
        route.append(next_node)
        next_node = circuit[next_node]
    return route

def minizinc_circuit_to_python(minizinc_circuit: list[int]) -> list[int]:
    return [n-1 if n != 0 else None for n in minizinc_circuit]
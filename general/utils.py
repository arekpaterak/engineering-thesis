from typing import Optional

import numpy as np
import torch_geometric as pyg

import matplotlib.pyplot as plt
import networkx as nx
from gymnasium.spaces import MultiBinary


def draw_graph(graph: pyg.data.Data) -> None:
    G = pyg.utils.to_networkx(graph, node_attrs=['x', 'pos'], to_undirected=True)

    # Extract positions as a dictionary for NetworkX visualization
    # NetworkX expects a dictionary {node_id: (x, y)}
    fixed_pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.pos.size(0))}

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos=fixed_pos, with_labels=True, node_color="skyblue", node_size=500, edge_color="gray", font_size=10)
    plt.show()


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

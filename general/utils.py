from typing import Optional

import matplotlib
import numpy as np
import torch_geometric as pyg

import matplotlib.pyplot as plt
import networkx as nx
from gymnasium.spaces import MultiBinary


def draw_tsp_graph(
    graph: pyg.data.Data,
    with_labels: bool = True,
    node_color: str = "skyblue",
    node_border_color: str = "black",
    border_width: float = 1.5
) -> matplotlib.pyplot:
    G = pyg.utils.to_networkx(graph, node_attrs=['x', 'pos'], to_undirected=True)

    # Extract positions as a dictionary for NetworkX visualization
    # NetworkX expects a dictionary {node_id: (x, y)}
    fixed_pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.pos.size(0))}

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos=fixed_pos,
        with_labels=with_labels,
        node_color=node_color,
        edgecolors=node_border_color,
        linewidths=border_width,
        node_size=500,
        edge_color="black",
        font_size=10,
        width=2
    )
    return plt


def draw_cvrp_graph(
    graph: pyg.data.Data,
    with_labels: bool = True,
    node_color: str = "skyblue",
    node_border_color: str = "black",
    border_width: float = 1.5,
    edge_palette: list = None
) -> matplotlib.pyplot:
    # Convert the PyG graph to a NetworkX graph
    G = pyg.utils.to_networkx(graph, node_attrs=["x", "pos"], edge_attrs=["edge_attr"], to_undirected=True)

    # Extract positions as a dictionary for NetworkX visualization
    fixed_pos = {i: (graph.pos[i][0].item(), graph.pos[i][1].item()) for i in range(graph.pos.size(0))}

    # Create a color map for edges based on edge_attr
    if edge_palette is None:
        edge_palette = plt.cm.tab10.colors  # Use a predefined colormap if not provided

    node_colors = [None] * G.number_of_nodes()
    G.nodes(data=True)

    edge_colors = []
    for u, v, edge_data in G.edges(data=True):
        attr_value = edge_data["edge_attr"]
        color_idx = int(attr_value % len(edge_palette))

        color = edge_palette[color_idx]
        edge_colors.append(color)

        node_colors[u] = color
        node_colors[v] = color

    node_colors[0] = "grey"

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos=fixed_pos,
        with_labels=with_labels,
        node_color=node_colors,
        edge_color=edge_colors,
        edgecolors=node_border_color,
        linewidths=border_width,
        node_size=500,
        font_size=10,
        width=2
    )
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


def minizinc_list_to_python(minizinc_list: list[int]) -> list[int]:
    return [n-1 if n != 0 else None for n in minizinc_list]

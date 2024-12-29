from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric as pyg
from torch_geometric.nn import GATConv, GATv2Conv, Linear, Sequential

from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary


class GraphFeaturesExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 3,
        hidden_channels: int = 64,
        out_channels: int = 512,
        edge_dim: int = None,
        num_heads: int = 1,
        v2: bool = False,
        activation: Callable = F.relu,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        conv = GATConv if not v2 else GATv2Conv
        conv = partial(conv, edge_dim=edge_dim, dropout=dropout)

        self.convs = nn.ModuleList([
            conv(in_channels, hidden_channels, heads=num_heads, concat=True)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
        self.convs.append(conv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False))
        self.fc = Linear(hidden_channels, out_channels)

        self.activation = activation
        self.features_dim = out_channels

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        for layer in self.convs:
            x = self.activation(layer(x, edge_index, edge_attr))
        # x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc(x))
        return x


if __name__ == "__main__":
    import os

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated", "train")
    problem_path = os.path.join(TSP_DATA_DIR, "50_1000_0.json")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

    env = TSPEnvironmentMultiBinary(problem_path, TSP_INIT_SOLVER_PATH, TSP_REPAIR_SOLVER_PATH)

    obs, _ = env.reset()

    graph_data = TSPEnvironmentMultiBinary.preprocess(obs)

    net = GraphFeaturesExtractor(in_channels=2, num_heads=8, edge_dim=1, num_layers=5)
    print(pyg.nn.summary(net, graph_data.x, graph_data.edge_index, graph_data.edge_attr))

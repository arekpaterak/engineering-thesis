from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric as pyg
from torch_geometric.nn import GATConv, GATv2Conv, Linear, Sequential

from config import *
from problems.cvrp.cvrp_env import CVRPEnvironment
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
        residual: bool = False,
    ) -> None:
        super().__init__()

        conv = GATConv if not v2 else GATv2Conv
        conv = partial(conv, edge_dim=edge_dim, residual=residual)

        self.convs = nn.ModuleList([
            conv(in_channels, hidden_channels, heads=num_heads, concat=True)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True))
        self.convs.append(conv(hidden_channels * num_heads, out_channels, heads=1, concat=False))

        self.activation = activation
        self.dropout = dropout

        self.features_dim = out_channels

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        for layer in self.convs[:-1]:
            x = self.activation(layer(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        return x


if __name__ == "__main__":
    import os

    # problem_path = os.path.join(TSP_DATA_DIR, "generated", "train", "50_1000_0.json")
    # env = TSPEnvironmentMultiBinary(problem_path, TSP_INIT_SOLVER_PATH, TSP_REPAIR_SOLVER_PATH)

    problem_path = os.path.join(CVRP_DATA_DIR, "generated", "train", "XML20_1123_00.json")
    env = CVRPEnvironment(problem_path, CVRP_INIT_SOLVER_PATH, CVRP_REPAIR_SOLVER_PATH)

    obs, _ = env.reset()

    graph_data = env.preprocess(obs)

    net = GraphFeaturesExtractor(in_channels=4, num_heads=8, edge_dim=1, num_layers=3, out_channels=1, residual=False)
    print(pyg.nn.summary(net, graph_data.x, graph_data.edge_index, graph_data.edge_attr))

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
    ) -> None:
        super().__init__()

        conv = GATConv if not v2 else GATv2Conv

        self.convs = nn.ModuleList([
            conv(in_channels, hidden_channels, heads=num_heads, concat=True, edge_dim=edge_dim)
        ])
        for _ in range(num_layers - 2):
            self.convs.append(conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, edge_dim=edge_dim))
        self.convs.append(conv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False, edge_dim=edge_dim))

        self.fc = Linear(hidden_channels, out_channels)
        self.activation = activation

        self.features_dim = out_channels

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        for layer in self.convs:
            x = self.activation(layer(x, edge_index, edge_attr))
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc(x))
        return x


class SequentialGraphFeaturesExtractor(nn.Module):
    """
    The GraphFeatureExtractor implemented with the PyTorch Geometric `nn.Sequential`.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 512,
        edge_dim: int = None,
        num_heads: int = 1,
        v2: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        conv = GATConv if not v2 else GATv2Conv
        self.net = Sequential(
            "x, edge_index, edge_attr",
            [
                (
                    conv(
                        in_channels, 32, heads=num_heads, concat=True, edge_dim=edge_dim
                    ),
                    "x, edge_index, edge_attr -> x"
                ),
                activation(),
                (
                    conv(32 * num_heads,64, heads=num_heads, concat=True, edge_dim=edge_dim),
                    "x, edge_index, edge_attr -> x"
                ),
                activation(),
                (
                    conv(64 * num_heads, 64, heads=1, concat=False, edge_dim=edge_dim),
                    "x, edge_index, edge_attr -> x"
                ),
                activation(),
                Linear(64, out_channels),
            ],
        )

    def forward(self, x, edge_index, edge_attr):
        self.net(x, edge_index, edge_attr)


if __name__ == "__main__":
    import os

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated", "train")
    problem_path = os.path.join(TSP_DATA_DIR, "20_1000_0.json")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

    env = TSPEnvironmentMultiBinary(problem_path, TSP_INIT_SOLVER_PATH, TSP_REPAIR_SOLVER_PATH)

    obs, _ = env.reset()

    graph_data = TSPEnvironmentMultiBinary.preprocess(obs)

    net = GraphFeaturesExtractor(in_channels=2, num_heads=8, edge_dim=1, num_layers=5)
    print(pyg.nn.summary(net, graph_data.x, graph_data.edge_index, graph_data.edge_attr))

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
        out_channels: int = 512,
        edge_dim: int = None,
        num_heads: int = 1,
        v2: bool = False,
        activation: Callable = F.relu,
    ) -> None:
        super().__init__()

        conv = GATConv if not v2 else GATv2Conv
        self.convs = nn.ModuleList(
            [
                conv(in_channels, 32, heads=num_heads, concat=True, edge_dim=edge_dim),
                conv(
                    32 * num_heads, 64, heads=num_heads, concat=True, edge_dim=edge_dim
                ),
                conv(64 * num_heads, 64, heads=1, concat=False, edge_dim=edge_dim),
            ]
        )
        self.fc = Linear(64, out_channels)
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
    observation = {'problem': {'node_positions': [{'x': 74, 'y': 528}, {'x': 658, 'y': 280}, {'x': 314, 'y': 534}, {'x': 160, 'y': 915}, {'x': 756, 'y': 153}, {'x': 841, 'y': 843}, {'x': 748, 'y': 954}, {'x': 995, 'y': 922}, {'x': 75, 'y': 1}, {'x': 139, 'y': 470}, {'x': 338, 'y': 176}, {'x': 973, 'y': 586}, {'x': 296, 'y': 844}, {'x': 820, 'y': 770}, {'x': 438, 'y': 229}, {'x': 742, 'y': 866}, {'x': 244, 'y': 638}, {'x': 962, 'y': 942}, {'x': 149, 'y': 403}, {'x': 412, 'y': 11}]}, 'solution': {'route': [1, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]}}

    graph_data = TSPEnvironmentMultiBinary.preprocess(observation)

    net = GraphFeaturesExtractor(in_channels=2, num_heads=8, edge_dim=1)
    print(pyg.nn.summary(net, graph_data.x, graph_data.edge_index, graph_data.edge_attr))

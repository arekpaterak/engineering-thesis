from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric as pyg
from torch.cuda import graph
from torch_geometric import edge_index
from torch_geometric.nn import global_mean_pool

from general.ml.features_extractor import GraphFeaturesExtractor
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary


class Policy(nn.Module):
    def __init__(
        self,
        graph_features_extractor_kwargs: dict,
    ) -> None:
        super().__init__()

        # ==== Shared Network ====
        self.features_extractor = GraphFeaturesExtractor(
            **graph_features_extractor_kwargs
        )
        features_dim = self.features_extractor.features_dim

        # ==== Policy Head ====
        self.head = nn.Linear(features_dim, 1)

        # ==== Value Estimation Head ====
        self.value_estimator = nn.Linear(features_dim, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features_extractor(x, edge_index, edge_attr)
        logits = self.head(features)

        value = F.tanh(self.value_estimator(global_mean_pool(features, batch=batch)))
        
        return torch.flatten(logits), torch.flatten(value)

    def get_action(self, graph_data: pyg.data.Data, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr

        logits = self.forward(x, edge_index, edge_attr)

        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action_idx = prob.multinomial(num_samples=k).detach()

        entropy = -(log_prob * prob).sum(-1, keepdim=True)

        action = torch.zeros(logits.shape[0])
        action[action_idx] = 1

        log_prob = log_prob.gather(0, action_idx)

        return action, log_prob.sum(-1, keepdim=True), entropy

    def get_action_and_value(self, graph_data: pyg.data.Data, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr

        logits, value = self.forward(x, edge_index, edge_attr)

        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action_idx = prob.multinomial(num_samples=k).detach()

        print(prob)
        print(log_prob)

        entropy = -(log_prob * prob).sum(-1, keepdim=True)

        action = torch.zeros(logits.shape[0])
        action[action_idx] = 1

        log_prob = log_prob.gather(0, action_idx)

        return action, log_prob.sum(-1, keepdim=True), entropy, value


if __name__ == '__main__':
    import os

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated")
    problem_path = os.path.join(TSP_DATA_DIR, "10_1000_0.json")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

    env = TSPEnvironmentMultiBinary(problem_path, TSP_INIT_SOLVER_PATH, TSP_REPAIR_SOLVER_PATH)

    obs, _ = env.reset()

    graph_data = TSPEnvironmentMultiBinary.preprocess(obs)

    policy = Policy(
        graph_features_extractor_kwargs=dict(
            in_channels = 2, num_heads = 8, edge_dim = 1, num_layers = 5
        )
    )

    action, log_prob, entropy, value = policy.get_action_and_value(graph_data, 5)

    print(action)
    print(log_prob)
    print(entropy)
    print(value)

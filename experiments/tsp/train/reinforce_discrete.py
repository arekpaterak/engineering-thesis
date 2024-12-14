import os
import time
from collections import deque
from dataclasses import dataclass
import random
from datetime import datetime
from typing import Callable, Tuple, Optional

import gymnasium
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

import torch_geometric as pyg

import torchmetrics

import tyro
import wandb

from general.ml.features_extractor import GraphFeaturesExtractor
from problems.tsp.tsp_env_discrete import TSPEnvironmentDiscrete


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "engineering-thesis"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    debug: bool = True
    """if toggled, extra logs will be printed to the console"""
    representative_instance_idx: int = 0
    """the id of the representative instance for logging"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    n_instances: Optional[int] = None
    """how many problem instances to train on"""

    # Algorithm specific arguments
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    n_epochs: int = 1
    """the number of epochs"""


class Agent(nn.Module):
    def __init__(
        self,
        graph_features_extractor_kwargs: dict,
    ) -> None:
        super().__init__()

        # ==== Policy Network ====
        self.features_extractor = GraphFeaturesExtractor(
            **graph_features_extractor_kwargs
        )
        features_dim = self.features_extractor.features_dim
        self.head = nn.Linear(features_dim, 4845)

        # ==== Optimizer ====
        self.optimizer = None

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        features = self.features_extractor(x, edge_index, edge_attr)
        logits = self.head(features.mean(dim=0, keepdim=True))
        return logits

    def get_action_with_log_prob(self, graph_data: pyg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr

        logits = self.forward(x, edge_index, edge_attr)
        probs = torch.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action)


if __name__ == "__main__":
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair.mzn")

    run_name = f"TSP__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            save_code=True,
            config={
                "env": {
                    "init_solver": TSP_INIT_SOLVER_PATH,
                    "repair_solver": TSP_REPAIR_SOLVER_PATH,
                    "max_t": args.max_t,
                    "n_envs": args.n_instances
                },
                "algorithm": {
                    "learning_rate": args.learning_rate,
                    "gamma": args.gamma,
                    "n_epochs": args.n_epochs,
                }
            },
            job_type="train"
        )

    # ==== Seeding ====
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.debug:
        print(f"Device: {device}")

    # ==== Environment Creation ====
    problem_instances_paths = [os.path.join(TSP_DATA_DIR, path) for path in os.listdir(TSP_DATA_DIR) if path.endswith(".json")]

    envs = TSPEnvironmentDiscrete.create_multiple(
        problem_instances_paths[:args.n_instances] if args.n_instances else problem_instances_paths,
        init_model_path=TSP_INIT_SOLVER_PATH,
        repair_model_path=TSP_REPAIR_SOLVER_PATH,
        solver_name="gecode",
        max_episode_length=args.max_t,
        k=4,
    )

    graph_features_extractor_kwargs = dict(
        in_channels=2,
        num_heads=8,
        edge_dim=1,
    )
    agent = Agent(
        graph_features_extractor_kwargs=graph_features_extractor_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate)

    # Line 3 of pseudocode
    agent.train()
    for epoch in range(args.n_epochs):
        if args.debug:
            print(f"Epoch {epoch}")
            epoch_start_time = time.time()

        avg_action_expense = torchmetrics.aggregation.MeanMetric()
        avg_ignored_actions = torchmetrics.aggregation.MeanMetric()
        avg_total_reward = torchmetrics.aggregation.MeanMetric()
        representative_instance_best_objective_value = None

        for env_idx, env in enumerate(envs):
            if args.debug:
                print(f"Env {env_idx}")

            saved_log_probs = []
            saved_rewards = []

            observation, info = env.reset()

            # Line 4 of pseudocode
            for t in range(args.max_t):
                graph_data = env.preprocess(observation).to(device)

                action, log_prob = agent.get_action_with_log_prob(graph_data)

                observation, reward, terminated, truncated, info = env.step(action.cpu().item())

                saved_log_probs.append(log_prob)
                saved_rewards.append(reward)

                # ==== Logging ====
                if args.debug and env_idx == args.representative_instance_idx:
                    print(f"Action: {action.cpu().item():4} / {env.action_to_vector(action.cpu().item())}, Prob: {torch.exp(log_prob).item():.4f}, Step objective value: {info['step_objective_value']}")

                if terminated or truncated:
                    break

            avg_total_reward.update(sum(saved_rewards))

            # Line 6 of pseudocode: calculate the return
            returns = deque(maxlen=args.max_t)
            n_steps = len(saved_rewards)
            # Compute the discounted returns at each timestep,
            # as
            #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
            #
            # In O(N) time, where N is the number of time steps
            # (this definition of the discounted return G_t follows the definition of this quantity
            # shown at page 44 of Sutton&Barto 2017 2nd draft)
            # G_t = r_(t+1) + r_(t+2) + ...

            # Given this formulation, the returns at each timestep t can be computed
            # by re-using the computed future returns G_(t+1) to compute the current return G_t
            # G_t = r_(t+1) + gamma*G_(t+1)
            # G_(t-1) = r_t + gamma* G_t
            # (this follows a dynamic programming approach, with which we memorize solutions in order
            # to avoid computing them multiple times)

            # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
            # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

            ## Given the above, we calculate the returns at timestep t as:
            #               gamma[t] * return[t] + reward[t]
            #
            ## We compute this starting from the last timestep to the first, in order
            ## to employ the formula presented above and avoid redundant computations that would be needed
            ## if we were to do it from first to last.

            ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
            ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
            ## a normal python list would instead require O(N) to do this.
            for t in range(n_steps)[::-1]:
                discounted_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(args.gamma * discounted_return_t + saved_rewards[t])

            ## normalization of the returns is employed to make training more stable
            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            eps = np.finfo(np.float32).eps.item()
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # ==== Loss Calculation ====
            loss = []
            for log_prob, discounted_return in zip(saved_log_probs, returns):
                loss.append(-log_prob * discounted_return)
            loss = torch.cat(loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if env_idx == args.representative_instance_idx:
                representative_instance_best_objective_value = info["best_objective_value"]
            if args.debug and env_idx == args.representative_instance_idx:
                print(f"Loss: {loss.item():.4f}")

        # ==== Logging ====
        logs = {
            "avg_total_reward": avg_total_reward.compute(),
            "best_objective_value_for_representative_instance": representative_instance_best_objective_value,
        }
        if args.debug:
            print(logs)
            print(f"Time: {(time.time() - epoch_start_time):.2f} s")
        if args.track:
            wandb.log(logs)

    # ==== Saving the model ====
    if args.track:
        model_path = os.path.join(wandb.run.dir, "model.pt")
        torch.save(agent.state_dict(), model_path)

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        wandb.finish()

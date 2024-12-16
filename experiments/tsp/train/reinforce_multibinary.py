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
from torch.distributions import Bernoulli

import torch_geometric as pyg

import torchmetrics

import tyro
import wandb

from general.ml.features_extractor import GraphFeaturesExtractor
from problems.tsp.tsp_env_multibinary import TSPEnvironmentMultiBinary


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
    low_action_bound: float = 0.1
    """the threshold below which actions are ignored"""
    high_action_bound: float = 0.5
    """the threshold above which actions are ignored"""
    n_instances: Optional[int] = None
    """how many problem instances to train on"""
    action_penalty: Optional[float] = -100.0
    """the penalty for an expensive action"""
    solver: str = "gecode"
    """the solver to use to find an initial solution and repair the subsequent"""

    # Algorithm specific arguments
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    n_epochs: int = 1
    """the number of epochs"""
    entropy_coefficient: float = 0.01
    """the entropy coefficient for the entropy regularization"""
    proportion: float = 0.2
    """the proportion of the nodes in the problem to destroy in one step"""
    max_grad_norm: Optional[float] = None
    """the maximum norm for gradient clipping"""


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
        self.head = nn.Linear(features_dim, 1)

        # ==== Optimizer ====
        self.optimizer = None

    def forward(self, x, edge_index, edge_attr=None) -> torch.Tensor:
        features = self.features_extractor(x, edge_index, edge_attr)
        logits = self.head(features)
        return torch.flatten(logits)

    def get_action(self, graph_data: pyg.data.Data, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr

        logits = self.forward(x, edge_index, edge_attr)

        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action_idx = prob.multinomial(num_samples=k).detach()

        action = torch.zeros(logits.shape[0])
        action[action_idx] = 1

        log_prob = log_prob.gather(0, action_idx)

        return action, log_prob.sum(-1, keepdim=True)


if __name__ == "__main__":
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair.mzn")

    run_name = f"TSP__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y%m%d_%H%M')}"

    # ==== Tracking Initialization ====
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
                    "n_envs": args.n_instances,
                    "low_action_bound": args.low_action_bound,
                    "high_action_bound": args.high_action_bound,
                    "action_penalty": args.action_penalty,
                    "solver": args.solver,
                },
                "algorithm": {
                    "learning_rate": args.learning_rate,
                    "gamma": args.gamma,
                    "n_epochs": args.n_epochs,
                    "entropy_coefficient": args.entropy_coefficient,
                    "proportion": args.proportion,
                    "max_grad_norm": args.max_grad_norm,
                }
            },
            job_type="train"
        )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.debug:
        print(f"Device: {device}")

    # ==== Seeding ====
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # ==== Environments Creation ====
    problem_instances_paths = [os.path.join(TSP_DATA_DIR, path) for path in os.listdir(TSP_DATA_DIR) if path.endswith(".json")]

    envs = TSPEnvironmentMultiBinary.create_multiple(
        problem_instances_paths[:args.n_instances] if args.n_instances else problem_instances_paths,
        init_model_path=TSP_INIT_SOLVER_PATH,
        repair_model_path=TSP_REPAIR_SOLVER_PATH,
        solver_name=args.solver,
        max_episode_length=args.max_t,
        action_bounds=(args.low_action_bound, args.high_action_bound),
        action_penalty=args.action_penalty,
    )

    # ==== Model Creation ====
    graph_features_extractor_kwargs = dict(
        in_channels=2,
        num_heads=8,
        edge_dim=1,
    )
    model = Agent(
        graph_features_extractor_kwargs=graph_features_extractor_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==== Training ====
    model.train()
    for epoch in range(1, args.n_epochs + 1):
        if args.debug:
            print(f"Epoch {epoch:4}")
            epoch_start_time = time.time()

        avg_total_reward = torchmetrics.aggregation.MeanMetric()
        avg_action_expense = torchmetrics.aggregation.MeanMetric()
        avg_ignored_actions = torchmetrics.aggregation.MeanMetric()
        representative_instance_best_objective_value = None
        avg_policy_loss = torchmetrics.aggregation.MeanMetric()
        avg_entropy_loss = torchmetrics.aggregation.MeanMetric()
        avg_total_loss = torchmetrics.aggregation.MeanMetric()

        for env_idx, env in enumerate(envs):
            if args.debug:
                print(f"Env {env_idx}")

            k = int(env.problem.num_nodes * args.proportion)

            log_probs = []
            rewards = []
            entropies = []

            n_ignored_actions = 0

            # ==== One Trajectory Acquisition ====
            observation, info = env.reset()
            for t in range(args.max_t):
                graph_data = env.preprocess(observation).to(device)

                action, log_prob = model.get_action(graph_data, k=k)
                action = action.cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action)

                prob = torch.exp(log_prob)
                entropy_loss = -(log_prob * prob).sum(-1, keepdim=True)

                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy_loss)

                # ==== Logging ====
                if info["is_action_ignored"]:
                    n_ignored_actions += 1
                avg_action_expense.update(sum(action))

                if args.debug and env_idx == args.representative_instance_idx:
                    log_prob = log_prob.detach().cpu()
                    prob = torch.exp(log_prob).item()
                    print(f"Step {t:3}, Action: {action}, Log(Prob): {log_prob.item():.4f}, Prob: {prob:.4f}, Step objective value: {info['step_objective_value']}, Action expense: {sum(action)}")

                if terminated or truncated:
                    break

            # ==== Returns Calculation ====
            returns = deque(maxlen=args.max_t)
            for t in reversed(range(len(rewards))):
                discounted_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(args.gamma * discounted_return_t + rewards[t])

            # ==== Returns Normalization ====
            eps = np.finfo(np.float32).eps.item()
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # ==== Loss Calculation ====
            policy_loss = []
            for log_prob, discounted_return in zip(log_probs, returns):
                policy_loss.append(-log_prob * discounted_return)

            policy_loss = -torch.cat(policy_loss).sum()
            entropy_loss = -args.entropy_coefficient * torch.cat(entropies).sum()
            total_loss = policy_loss + entropy_loss

            # ==== Policy Update ====
            optimizer.zero_grad()
            total_loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # ==== Logging ====
            avg_total_reward.update(sum(rewards))
            avg_ignored_actions.update(n_ignored_actions)
            avg_policy_loss.update(policy_loss.item())
            avg_entropy_loss.update(entropy_loss.item())
            avg_total_loss.update(total_loss.item())
            if env_idx == args.representative_instance_idx:
                representative_instance_best_objective_value = info["best_objective_value"]
            if args.debug and env_idx == args.representative_instance_idx:
                print(f"Policy Loss: {policy_loss.item():.4f}")
                print(f"Entropy Loss: {entropy_loss.item():.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")

        # ==== Logging ====
        logs = {
            "avg_total_reward": avg_total_reward.compute(),
            "best_objective_value_for_representative_instance": representative_instance_best_objective_value,
            # "avg_ignored_actions": avg_ignored_actions.compute(),
            # "avg_action_expense": avg_action_expense.compute(),
            "avg_policy_loss": avg_policy_loss.compute(),
            "avg_entropy_loss": avg_entropy_loss.compute(),
            "avg_total_loss": avg_total_loss.compute(),
        }
        if args.debug:
            print(logs)
            print(f"Time: {(time.time() - epoch_start_time):.2f} s")
        if args.track:
            wandb.log(logs)

    # ==== Saving the model ====
    if args.track:
        model_path = os.path.join(wandb.run.dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        wandb.finish()

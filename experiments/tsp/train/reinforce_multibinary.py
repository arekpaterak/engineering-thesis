import os
import time
from collections import deque
from dataclasses import dataclass, field
import random
from datetime import datetime
from dis import disco
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.xavier_normal_(layer.weight, 0.1)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
    save_every_n_epochs: int = 10
    """how often to save the model"""

    # Environment specific arguments
    max_t: Optional[int] = None
    """the maximum number of steps during one episode"""
    problem_sizes: list[int] = field(default_factory=lambda: [20])
    """"""
    instances: int | Tuple[int, int] = 0
    """the instance index or a range"""
    solver: str = "gecode"
    """the solver to use to find an initial solution and repair the subsequent"""
    processes: int = 1
    """the number of processes to use in MiniZinc"""

    # Neural Network specific arguments
    num_layers: int = 5
    num_heads: int = 8
    hidden_channels: int = 64
    out_channels: int = 512
    gat_v2: bool = False

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
    use_value_baseline: bool = False


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

        value = F.tanh(self.value_estimator(pyg.nn.global_mean_pool(features, batch=batch)))

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

    def get_action_and_value(self, graph_data: pyg.data.Data, k: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr

        logits, value = self.forward(x, edge_index, edge_attr)

        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        action_idx = prob.multinomial(num_samples=k).detach()

        entropy = -(log_prob * prob).sum(-1, keepdim=True)

        action = torch.zeros(logits.shape[0])
        action[action_idx] = 1

        log_prob = log_prob.gather(0, action_idx)

        return action, log_prob.sum(-1, keepdim=True), entropy, value


if __name__ == "__main__":
    args = tyro.cli(Args)

    BASE_PATH = "D:\\Coding\\University\\S7\\engineering-thesis"

    TSP_DATA_DIR = os.path.join(BASE_PATH, "problems", "tsp", "data", "generated", "train")

    TSP_SOLVERS_DIR = os.path.join(BASE_PATH, "problems", "tsp", "minizinc")
    TSP_INIT_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_init_circuit.mzn")
    TSP_REPAIR_SOLVER_PATH = os.path.join(TSP_SOLVERS_DIR, "tsp_repair_circuit.mzn")

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

    # Filter instances
    training_instances_paths = []
    match args.instances:
        case int():
            for path in problem_instances_paths:
                size, _, idx = os.path.basename(path).rstrip(".json").strip().split("_")
                size, idx = int(size), int(idx)
                if size in args.problem_sizes and idx == args.instances:
                    training_instances_paths.append(path)
        case _:
            index_range = range(args.instances[0], args.instances[1]+1)
            for path in problem_instances_paths:
                size, _, idx = os.path.basename(path).rstrip(".json").strip().split("_")
                size, idx = int(size), int(idx)
                if size in args.problem_sizes and idx in index_range:
                    training_instances_paths.append(path)

    if args.debug:
        print("Training instances:", training_instances_paths)

    envs = TSPEnvironmentMultiBinary.create_multiple(
        training_instances_paths,
        init_model_path=TSP_INIT_SOLVER_PATH,
        repair_model_path=TSP_REPAIR_SOLVER_PATH,
        solver_name=args.solver,
        max_episode_length=args.max_t,
        processes=args.processes
    )

    # ==== Model Creation ====
    graph_features_extractor_kwargs = dict(
        in_channels=2,
        num_heads=args.num_heads,
        edge_dim=1,
        num_layers=args.num_layers,
        v2=args.gat_v2,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels,
    )
    model = Policy(
        graph_features_extractor_kwargs=graph_features_extractor_kwargs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==== Tracking Initialization ====
    run_name = f"TSP{args.problem_sizes}x{len(envs)}__{args.exp_name}__{args.seed}__{datetime.now().strftime('%Y%m%d_%H%M')}"
    print(run_name)
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            save_code=True,
            config={
                "env": {
                    "n_instances": len(envs),
                    "init_solver": TSP_INIT_SOLVER_PATH,
                    "repair_solver": TSP_REPAIR_SOLVER_PATH,
                    "max_t": args.max_t,
                    "instances": args.instances,
                    "solver": args.solver,
                    "training_instances_paths": training_instances_paths,
                },
                "neural_network": {
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "hidden_channels": args.hidden_channels,
                    "out_channels": args.out_channels,
                },
                "algorithm": {
                    "learning_rate": args.learning_rate,
                    "gamma": args.gamma,
                    "n_epochs": args.n_epochs,
                    "entropy_coefficient": args.entropy_coefficient,
                    "proportion": args.proportion,
                    "max_grad_norm": args.max_grad_norm,
                    "use_value_baseline": args.use_value_baseline,
                }
            },
            job_type="train"
        )

    # ==== Training ====
    model.train()
    for epoch in range(0, args.n_epochs):
        if args.debug:
            print(f"Epoch {epoch:4}")
            epoch_start_time = time.time()

        # ==== Training Metrics ====
        avg_total_reward = torchmetrics.aggregation.MeanMetric()
        representative_instance_best_objective_value = None
        avg_policy_loss = torchmetrics.aggregation.MeanMetric()
        avg_entropy = torchmetrics.aggregation.MeanMetric()
        avg_value_loss = torchmetrics.aggregation.MeanMetric()
        avg_total_loss = torchmetrics.aggregation.MeanMetric()
        avg_best_objective_value = torchmetrics.aggregation.MeanMetric()

        for env_idx, env in enumerate(envs):
            if args.debug:
                print(f"Env {env_idx}")

            k = int(env.problem.num_nodes * args.proportion)

            log_probs = []
            rewards = []
            entropies = []
            values = []

            # ==== One Trajectory Acquisition ====
            observation, info = env.reset()
            for t in range(args.max_t):
                graph_data = env.preprocess(observation).to(device)

                action, log_prob, entropy, value = model.get_action_and_value(graph_data, k=k)
                action = action.cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                values.append(value)

                # ==== Logging ====
                if args.debug and env_idx == args.representative_instance_idx:
                    log_prob = log_prob.detach().cpu()
                    prob = torch.exp(log_prob).item()
                    print(f"Step {t:3}, Action: {action}, Log(Prob): {log_prob.item():.4f}, Prob: {prob:.4f}, Entropy: {entropy.item():.4f}, Reward: {reward:.4f}, Value: {value.item():.4f}, Step objective value: {info['step_objective_value']}, k: {int(sum(action))}")

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
            value_loss = []
            for log_prob, discounted_return, value in zip(log_probs, returns, values):
                if args.use_value_baseline:
                    advantage = discounted_return - value
                    policy_loss.append(-log_prob * advantage.detach())
                    value_loss.append(advantage)
                else:
                    policy_loss.append(-log_prob * discounted_return)

            policy_loss = torch.cat(policy_loss).sum()
            # As the regularization, the entropy should be maximized (which equals minimizing its negative)
            entropy = torch.cat(entropies).mean()
            if args.use_value_baseline:
                value_loss = torch.cat(value_loss).pow(2).mean()
                total_loss = policy_loss - args.entropy_coefficient * entropy + 0.5 * value_loss
            else:
                total_loss = policy_loss - args.entropy_coefficient * entropy

            # ==== Policy Update ====
            optimizer.zero_grad()
            total_loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # ==== Logging ====
            avg_total_reward.update(sum(rewards))
            avg_policy_loss.update(policy_loss.item())
            avg_entropy.update(entropy.item())
            if args.use_value_baseline:
                avg_value_loss.update(value_loss.item())
            avg_total_loss.update(total_loss.item())
            avg_best_objective_value.update(info["best_objective_value"])
            if env_idx == args.representative_instance_idx:
                representative_instance_best_objective_value = info["best_objective_value"]
            if args.debug:
                print(f"Policy Loss: {policy_loss.item():.4f}")
                print(f"Avg. Entropy: {entropy.item():.4f}")
                if args.use_value_baseline:
                    print(f"Value Loss: {value_loss:.4f}")
                print(f"Total Loss: {total_loss.item():.4f}")

        # ==== Logging ====
        logs = {
            "avg_total_reward": avg_total_reward.compute(),
            "avg_best_objective_value": avg_best_objective_value.compute(),
        "best_objective_value_for_representative_instance": representative_instance_best_objective_value,
            "avg_policy_loss": avg_policy_loss.compute(),
            "avg_entropy": avg_entropy.compute(),
            "avg_value_loss": avg_value_loss.compute(),
            "avg_total_loss": avg_total_loss.compute(),
        }
        if args.debug:
            print(logs)
            print(f"Time: {(time.time() - epoch_start_time):.2f} s")
        if args.track:
            wandb.log(logs)

        # ==== Saving the model ====
        if args.track and ((epoch + 1) % args.save_every_n_epochs == 0 or (epoch + 1) == args.n_epochs):
            print("Saving the model...")
            model_path = os.path.join(wandb.run.dir, "model.pt")
            torch.save(model.state_dict(), model_path)

            artifact = wandb.Artifact(f"model-{args.proportion}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    if args.track:
        wandb.finish()

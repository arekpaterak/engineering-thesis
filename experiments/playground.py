import gymnasium as gym
import numpy as np
from itertools import combinations
import torch

prob = torch.tensor([0.25, 0.75])
log_prob = torch.log(prob)

print(prob)
print(log_prob)

print(-(log_prob*prob).sum())

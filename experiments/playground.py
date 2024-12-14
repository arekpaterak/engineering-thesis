import gymnasium as gym
import numpy as np
from itertools import combinations
import torch

probs = torch.tensor([0, 0.25, 0.25, 0.25])
action = probs.multinomial(num_samples=2, replacement=False)
print(action)

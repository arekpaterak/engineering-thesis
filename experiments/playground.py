import torch
import torch.nn.functional as F

def top_k_bernoulli(logits, k):
    # Convert logits to probabilities
    probs = torch.sigmoid(logits)
    # Sample from Bernoulli
    actions = torch.bernoulli(probs)
    # Force exactly k ones
    if actions.sum() > k:
        # Keep top-k logits where actions were 1
        topk_indices = torch.topk(logits * actions, k).indices
        mask = torch.zeros_like(actions)
        mask[topk_indices] = 1
        actions = mask
    return actions

# Example logits and k
logits = torch.tensor([0.2, -1.5, 0.8, 0.5, -0.3])
k = 3

actions = top_k_bernoulli(logits, k)
print("Sampled Actions:", actions)

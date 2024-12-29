import torch
import torch.nn.functional as F


def multinomial_sample_log_prob(probs, k):
    """
    Sample k items using multinomial and compute differentiable log probability
    Args:
        probs: tensor of probabilities [n]
        k: number of items to sample
    Returns:
        indices: selected indices [k]
        log_prob: differentiable log probability of the selection
    """
    # Sample without replacement
    indices = torch.multinomial(probs, k, replacement=False)

    # Compute log probability of this specific sequence
    log_prob = 0.0
    remaining_probs = probs.clone()
    total_prob = remaining_probs.sum()

    for i in range(k):
        idx = indices[i]
        # P(selecting this item) = prob[idx] / sum(remaining_probs)
        step_prob = remaining_probs[idx] / total_prob
        log_prob = log_prob + torch.log(step_prob)

        # Update for next selection
        total_prob = total_prob - remaining_probs[idx]
        remaining_probs[idx] = 0

    return indices, log_prob


def gumbel_sample_log_prob(probs, k, epsilon=1e-10):
    """
    Sample k items using Gumbel-top-k and compute differentiable log probability
    Args:
        probs: tensor of probabilities [n]
        k: number of items to sample
        epsilon: small constant for numerical stability
    Returns:
        indices: selected indices [k]
        log_prob: differentiable log probability of the selection
    """
    n = len(probs)

    # Sample Gumbel noise
    z = -torch.log(-torch.log(torch.rand_like(probs) + epsilon) + epsilon)

    # Get top-k indices
    gumbel_scores = torch.log(probs + epsilon) + z
    _, indices = torch.topk(gumbel_scores, k)

    # Compute log probability using order statistics of Gumbel
    log_prob = 0.0
    remaining_probs = probs.clone()
    total_prob = remaining_probs.sum()

    for i in range(k):
        idx = indices[i]
        step_prob = remaining_probs[idx] / total_prob
        log_prob = log_prob + torch.log(step_prob)

        total_prob = total_prob - remaining_probs[idx]
        remaining_probs[idx] = 0

    return indices, log_prob


# Example usage in REINFORCE
def reinforce_step(policy_net, state, k):
    # Get probabilities from policy network
    logits = policy_net(state)
    probs = F.softmax(logits, dim=-1)

    # Sample using either method
    indices, log_prob = multinomial_sample_log_prob(probs, k)
    # OR
    # indices, log_prob = gumbel_sample_log_prob(probs, k)

    # Now log_prob is differentiable and can be used in REINFORCE
    return indices, log_prob


# Test gradient flow
def test_gradient_flow():
    n = 5
    k = 2
    logits = torch.randn(n, requires_grad=True)
    probs = F.softmax(logits, dim=-1)

    # Test multinomial
    indices, log_prob_multi = multinomial_sample_log_prob(probs, k)
    loss_multi = -log_prob_multi  # Minimize negative log probability
    loss_multi.backward()
    grad_multi = logits.grad.clone()

    # Reset gradients
    logits.grad = None

    # Test Gumbel
    indices, log_prob_gumbel = gumbel_sample_log_prob(probs, k)
    loss_gumbel = -log_prob_gumbel
    loss_gumbel.backward()
    grad_gumbel = logits.grad.clone()

    return grad_multi, grad_gumbel


# Example of verifying log probabilities sum to 1
def verify_log_probs(n=5, k=2, num_samples=1000):
    probs = torch.ones(n) / n
    total_prob = 0.0

    for _ in range(num_samples):
        _, log_prob = multinomial_sample_log_prob(probs, k)
        total_prob += torch.exp(log_prob).item()

    # Should be close to 1
    print(f"Total probability: {total_prob / num_samples}")


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)
    # Run tests
    grad_multi, grad_gumbel = test_gradient_flow()
    print("Multinomial gradients:", grad_multi)
    print("Gumbel gradients:", grad_gumbel)
    verify_log_probs()

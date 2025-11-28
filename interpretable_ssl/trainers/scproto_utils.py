import torch
import math

def get_hard_assign_cnts(logits):
    assignments = logits.argmax(dim=1)
    num_prototypes = logits.size(1)
    counts = torch.bincount(assignments, minlength=num_prototypes)
    return counts

def get_assignment_metrics(assignments, label):
    res = {}

    row_sums = assignments.sum(dim=1, keepdim=True)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2):
        assignments_prob = assignments / (row_sums + 1e-8)
    else:
        assignments_prob = assignments

    # Entropy
    entropy = (
        -torch.sum(assignments_prob * torch.log(assignments_prob + 1e-8), dim=1)
        .mean()
        .item()
    )
    res[f"{label}_entropy"] = entropy

    # Hard assignments
    counts = get_hard_assign_cnts(assignments)

    # Number of empty prototypes
    num_prototypes = assignments.size(1)
    res[f"{label}_empty_protos_ratio"] = (counts == 0).sum().item() / num_prototypes
    res[f'{label}_proto_usage'] = (counts != 0).sum().item() / min(num_prototypes, assignments.size(0))
    
    # Uniformity metric (L2 distance from uniform distribution, normalized)
    
    expected = counts.sum() / num_prototypes
    uniform_metric = (
        torch.norm(counts.float() - expected, p=2).item() / counts.sum().item()
    )
    res[f"{label}_uniformity_distance"] = uniform_metric
    return res

def get_matched_pairs_ratio(pt, ps):
    # Assign each sample to the most similar prototype (argmax over prototypes)
    pt_assign = pt.argmax(dim=1)
    ps_assign = ps.argmax(dim=1)

    # Count number of matched prototype assignments
    matched = (pt_assign == ps_assign).sum().item() / len(pt_assign)
    return matched


def kl_scheduler(epoch, kl_start_epoch, n_epochs, warmup_ratio=0.3, max_lambda=1.0):
    """
    Linear warm-up for KL coefficient (λ_kl), with a start epoch.
    KL stays 0 until kl_start_epoch, then warms up linearly.
    """
    # If KL never warms up, return final value immediately after start
    warmup_epochs = int(n_epochs * warmup_ratio)

    # Before KL starts → λ = 0
    if epoch < kl_start_epoch:
        return 0.0

    # If warmup is 0 → jump to max immediately
    if warmup_epochs == 0:
        return max_lambda

    # How many epochs have passed since KL started
    passed = epoch - kl_start_epoch

    # During warmup → linear increase
    if passed < warmup_epochs:
        return max_lambda * (passed / warmup_epochs)

    # After warmup → fixed max
    return max_lambda


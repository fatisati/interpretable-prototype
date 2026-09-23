"""[D31] Age-weighted assignment memory for the cohesion terms.

Extends [D29] (scproto.py, cohesion_global=2). [D29] remembers every cell's
last-known assignment (N x K) and swaps in the current batch's rows live, so the
per-prototype profiles and sizes are exact full-graph sums. Its one approximation
is STALENESS: a non-batch row is as old as the last batch that contained that
cell (up to ~1 epoch), yet counts at full weight.

[D31] discounts each remembered row by its age:

    age[j]  = current_step - step_when_cell_j_was_last_in_a_batch
    w[j]    = 0.5 ** (age[j] / half_life)        # half_life in steps
    w[j]    = 1                                  # for this batch's cells (fresh, live)

and uses  w[j] * assign[j][k]  wherever a membership appears -- in the
similarity numerators AND in the global size -- so every downstream quantity
stays a proper weighted mean (numerator and denominator discounted together;
same both-sides rule as CLAUDE.md 9b).

Limits, all covered in tests/test_cohesion_memory.py:
  half_life = 0 (default, off)  ->  w = 1 everywhere, bit-identical to [D29]
  batch = whole dataset         ->  exact objective, any half_life
  never-visited cells (last_seen = -1) age from step -1, so the uniform-init
  prior fades automatically instead of persisting until first visit.

This module is deliberately torch-only and trainer-free so it can be unit-tested
locally (the trainer chain cannot be imported outside Colab -- Data_Handler).
"""
import torch


def age_weights(last_seen, step, half_life):
    """Per-cell decay weight  w[j] = 0.5 ** (age[j] / half_life).

    Args:
        last_seen: (N,) long -- step index of each cell's last batch visit;
                   -1 means never visited (the uniform-init row).
        step:      int -- current memory step counter.
        half_life: float > 0 -- steps of absence after which a row counts half.

    Returns:
        (N,) weights in (0, 1], in the current default float dtype.
    """
    age = (float(step) - last_seen.to(torch.get_default_dtype())).clamp(min=0.0)
    return torch.pow(0.5, age / float(half_life))


def assemble_membership(S_mem, S_live, batch_idx, last_seen=None, step=None,
                        half_life=0.0):
    """Build the effective membership matrix and global size for one step.

        S_eff[j][k] = w[j] * assign[j][k]
        assign[j]   = S_live row  (gradient-carrying)  if j is in the batch
                      S_mem row   (detached memory)    otherwise
        w[j]        = 1 for batch rows, 0.5**(age/half_life) for the rest
        size_g[k]   = sum over ALL cells j of S_eff[j][k]   (unclamped)

    half_life <= 0 disables the weighting (w = None returned) and reproduces
    [D29] exactly: plain clone + row swap + column sum.

    Returns:
        (S_eff, size_g, w) -- w is None when the decay is off.
    """
    S_eff = S_mem.clone()
    S_eff[batch_idx] = S_live               # batch rows carry the gradient
    w = None
    if half_life and half_life > 0:
        w = age_weights(last_seen, step, half_life)
        w[batch_idx] = 1.0                  # fresh rows at full weight
        S_eff = S_eff * w.unsqueeze(1)      # batch rows * 1.0 -- still live
    return S_eff, S_eff.sum(0), w


@torch.no_grad()
def batch_drift(S_mem, S_live_det, batch_idx, last_seen, step):
    """Staleness diagnostic, to be called BEFORE the memory overwrite.

    For the batch's previously-seen cells, compares the remembered assignment
    row against the freshly computed one:

        drift = mean over seen batch cells of  sum_k |fresh[k] - remembered[k]|

    (L1 per cell, range [0, 2]: 0 = memory still exact, 2 = cell fully moved to
    a disjoint set of prototypes). Also returns the mean age (in steps) of those
    rows, i.e. how long the drift took to accumulate. Never-visited cells
    (last_seen = -1) are excluded -- their memory row is the uniform init, not a
    real past assignment. Returns (0.0, 0.0) if no batch cell was seen before.
    """
    seen = last_seen[batch_idx] >= 0
    if not bool(seen.any()):
        return 0.0, 0.0
    idx = batch_idx[seen]
    drift = (S_live_det[seen] - S_mem[idx]).abs().sum(dim=1).mean().item()
    age = (float(step) - last_seen[idx].to(torch.float32)).mean().item()
    return drift, age

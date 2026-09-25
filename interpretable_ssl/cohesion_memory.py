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


def bio_batch_degrees(A, bcode, n_bio):
    """[D47] Per-bio-batch filtered-subgraph degrees, precomputed once.

    Batch b's graph keeps every edge with AT LEAST ONE endpoint in b (both
    stored directions) -- the same edge rule as [D41]'s V[b] and as
    metacell_scoreboard.modularity_per_batch, so the training term and the
    eval metric agree on what "batch b's graph" means.

        deg_b[b][j] = full degree of j            if bio(j) == b
                      weight of j's edges into b  otherwise
        twom[b]     = sum over j of deg_b[b][j]   (total degree of b's graph)

    With n_bio == 1 this is the plain global degree vector and twom = 2m, so
    the single-batch path is bit-identical to the batch-agnostic formula.

    Args:
        A:     scipy.sparse CSR, symmetric, zero diagonal (the [D26] _coh_A).
        bcode: (N,) int64 bio-batch code per cell.
        n_bio: int number of bio batches.

    Returns:
        (deg_b, twom): float32 ndarray (n_bio, N) and float64 ndarray (n_bio,).
    """
    import numpy as np
    import scipy.sparse as sp
    N = A.shape[0]
    full_deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float32)
    onehot = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), bcode)),
                           shape=(N, n_bio))
    into_b = np.asarray((A @ onehot).todense()).astype(np.float32)  # (N, n_bio)
    deg_b = into_b.T.copy()                                         # (n_bio, N)
    for b in range(n_bio):
        sel = bcode == b
        deg_b[b, sel] = full_deg[sel]
    twom = deg_b.sum(axis=1).astype(np.float64)
    return deg_b, twom


def memory_modularity(GS_raw, S_live, S_eff, deg_b, twom, bcode_cells,
                      cell_idx, n_all):
    """[D47] Age-weighted soft modularity from the [D29]/[D31] memory.

    Per minibatch cell c, against its OWN bio batch's filtered subgraph
    (b = bcode_cells[c]; c is in b, so every edge of c is in that graph and
    GS_raw's full-affinity row is already the right reward):

        reward(c,p)   = sum over ALL j of  w[c,j] * S_eff[j,p]     (= GS_raw)
        vol_eff[b,p]  = sum over ALL j of  deg_b[b,j] * S_eff[j,p]
        luck(c,p)     = deg_b[b,c] * vol_eff[b,p] / twom[b]
        q(c)          = sum over p of  S_live[c,p] * (reward - luck) / twom[b]

    Returns  n_all * mean over minibatch cells of q(c)  -- a minibatch
    estimate of the all-cells total sum_c q(c). In the full-batch / fresh /
    hard / single-bio-batch limit that total IS Newman's Q exactly
    (tests/test_memory_modularity.py), so the value lives on the familiar
    modularity scale (trivial partition 0, max 1, min -0.5).

    Age handling: S_eff already carries w_age[j] on the MEMBERSHIP (the
    claim), so reward and luck are discounted together; deg_b and twom are
    graph EVIDENCE and are never age-weighted. That both-sides placement is
    what keeps the trivial/uniform anchors at exactly 0 for any age vector
    (CLAUDE.md 9b, modularity flavour). Unlike cohesion there is NO
    per-prototype size division -- modularity is a luck-corrected TOTAL, and
    dividing by membership mass would flatten the size^2 merge penalty and
    turn this into a second cohesion.

    Args:
        GS_raw:      (n_batch, K) spmm(A[batch rows], S_eff), raw affinity,
                     zero diagonal -- scproto's _GS_aff_raw under
                     cohesion_global=2. No cohesion_diag added: w[c,c]=0 is
                     the Newman convention (the self-pair enters only through
                     vol_eff, matching igraph's d_c^2 term).
        S_live:      (n_batch, K) live gradient-carrying batch assignments.
        S_eff:       (N, K) from assemble_membership (age weights included).
        deg_b:       (n_bio, N) torch tensor from bio_batch_degrees.
        twom:        (n_bio,) torch tensor from bio_batch_degrees.
        bcode_cells: (n_batch,) long, bio batch of each minibatch cell.
        cell_idx:    (n_batch,) long, global index of each minibatch cell.
        n_all:       int, N.

    Returns:
        scalar tensor (keeps gradient through S_live and S_eff's live rows).
    """
    vol = deg_b @ S_eff                                    # (n_bio, K)
    twom_c = twom[bcode_cells].clamp(min=1e-8)             # (n_batch,)
    deg_c = deg_b[bcode_cells, cell_idx]                   # (n_batch,)
    luck = deg_c.unsqueeze(1) * vol[bcode_cells] / twom_c.unsqueeze(1)
    q = (S_live * (GS_raw - luck)).sum(dim=1) / twom_c     # (n_batch,)
    return float(n_all) * q.mean()


def batch_sizes(S_eff, batch_code, n_bio):
    """[D41] Per-bio-batch prototype size, for multi-batch cohesion.

        size_in_batch[b][p] = sum over cells j with batch_code[j] == b
                              of  S_eff[j][p]

    S_eff already carries the [D31] age weight w[j], so this is the
    age-weighted size and numerator/denominator stay on the same footing
    (both-sides rule, CLAUDE.md 9b).

    Why this exists: with cross-batch similarity masked to zero, a cell can only
    ever accumulate numerator from members of its OWN batch. Dividing that by a
    GLOBAL size would score a batch-mixed prototype far below an identical
    batch-pure one (measured 0.169x on pancreas -- a ~5.9x standing reward for
    batch purity, produced purely by the scale mismatch). Restricting the
    denominator to the same population removes it.

    With one bio batch this returns a single row equal to S_eff.sum(0), so the
    caller's per-cell gather reproduces the global path exactly.

    Args:
        S_eff:      (N, K) effective membership from assemble_membership.
        batch_code: (N,) long -- bio-batch index of every cell, in [0, n_bio).
        n_bio:      int -- number of bio batches.

    Returns:
        (n_bio, K) tensor. UNCLAMPED -- the caller applies the founder-gate
        clamp after gathering, so the gate is applied once, per cell.
    """
    out = torch.zeros(n_bio, S_eff.shape[1],
                      dtype=S_eff.dtype, device=S_eff.device)
    return out.index_add(0, batch_code, S_eff)

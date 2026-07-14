# `sim_recon` sparsity: what we tried, and what we landed on

## The problem

`sim_recon_loss` trains a decoder to reconstruct each cell's row of the
affinity graph (`aff_csr`) from its soft prototype assignment. The affinity
graph is a kNN graph — mean degree ~75 out of ~58,000 cells, so each target
row is **>99.8% zero**.

Training with a plain `F.mse_loss(predicted, target)` collapsed `sim_recon`
to near-zero within 1-2 epochs. This was initially suspected to be a
`lambda_sim_recon` scaling issue, but it isn't: the population of zero-target
entries dominates the averaged gradient, so "predict ~0 everywhere" is a
strong trivial minimum regardless of `lambda`. Sum-vs-mean reduction doesn't
fix it either — it's a uniform rescale of the whole loss, not a change to the
internal balance between positive-vote and negative-vote entries.

## What SEACells does (the reference point)

SEACells optimizes `SRE = ||M - MBA||^2` over the *full* n×n kernel matrix
`M` (also a sparse adaptive-Gaussian kNN kernel — same sparsity class as our
affinity graphs), via Frank-Wolfe, with `A`/`B` constrained to the simplex
(columns sum to 1). Confirmed directly against `SEACells/core.py`,
`cpu.py`/`cpu_dense.py` (`compute_RSS`), and the paper
(`literature/s41587-023-01716-9.pdf`).

Key finding: **`M` itself is never row-normalized to sum to 1.** The simplex
constraint is on `A`/`B` (the assignment/archetype matrices), not on the
target similarity values. So SEACells' own robustness to sparsity comes from
the simplex constraint on its convex-combination weights, not from
normalizing the reconstruction target — which ruled out one of the options
below.

## Options discussed, and why each was rejected or kept

1. **Sum instead of mean reduction.**
   Proven algebraically to be `sum = mean × count` — a content-independent
   global rescale. Does not change the relative gradient contribution of
   positive vs. zero entries, so it does not fix the collapse. Rejected as a
   fix (though still relevant separately for `lambda_sim_recon` tuning).

2. **Softmax + cross-entropy / KL**, row-normalizing both predicted and
   target into distributions (`sum = 1`), matching t-SNE/UMAP's `P_{j|i}`
   form. Structurally immune to collapse (a distribution can't collapse to
   all-zero). **Rejected**: real affinity values in SEACells' own
   construction are not normalized to 1, and forcing that here would make
   the reconstruction target diverge from what SEACells actually optimizes
   against. User's explicit call: "i dont like sum 1 - because real
   similarity not like this."

3. **BCE with negative sampling** — classify each (cell, sampled-neighbor)
   and (cell, sampled-non-neighbor) pair independently via sigmoid + BCE,
   avoiding any row-sum constraint. **Rejected**: user does not want a
   sampling-based mechanism (adds stochasticity/variance, moves away from
   deterministic full-row optimization). "i prefer not have neg sampling
   thing."

4. **Zero-inflated two-head (existence + magnitude)** — a classification
   head predicting whether a pair is a real edge, and a regression head
   predicting the magnitude conditional on existence, in the style of
   ZINB-type losses. Discussed as the most literature-grounded option for
   "many zeros, few positive continuous values," but adds real architectural
   complexity (two heads, gating) for a marginal gain over option 5. Not
   pursued further once option 5 was agreed on.

5. **Class-balanced weighted MSE — chosen.**
   Per anchor cell (per row), weight positive (true-neighbor) entries by
   `n_neg / n_pos` and negative (zero-target) entries by `1`, computed
   independently from that row's own real neighbor count — no sampling, no
   row normalization, no assumption that anything sums to 1. Directly
   analogous to `pos_weight` in `BCEWithLogitsLoss`, applied to squared
   error instead of log-loss. This upweights the rare positive signal so it
   isn't drowned out by the zero population, while still training on the
   raw, unnormalized affinity values SEACells itself uses.

   ```python
   pos_mask = target > 0
   n_pos = pos_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
   n_neg = (target.shape[1] - n_pos).clamp(min=1)
   pos_weight_row = n_neg / n_pos
   weight = torch.where(pos_mask, pos_weight_row.expand_as(target), torch.ones_like(target))
   loss = (weight * (predicted - target) ** 2).mean()
   ```

## Reconstruction scope: what each row is compared against

Separately from the weighting scheme, we also fixed *what* each cell's row
is reconstructed against:

- **Rejected**: reconstructing only against the current random minibatch
  (`predicted`/`target` both `(n_b, n_b)`) — this dilutes the true neighbor
  pool with whatever happened to land in the same minibatch, which is a much
  smaller and noisier context than SEACells' full-matrix objective.
- **Kept as-is (this was the important constraint)**: the per-data-batch /
  per-section loop. Each cell's row is reconstructed against **all cells in
  its own batch/section** (`decode_sim_profiles(protos, col_idx=batch_cols)`
  where `batch_cols` = every cell sharing that section), not the full
  cross-dataset `n`, and not the diluted minibatch block. This is required
  because cross-section affinity entries in `aff_csr` are *structurally
  absent* (never computed), not genuine negatives — treating them as
  negatives in the loss would be a batch-effect confound. This mirrors
  exactly how `nassoc` already handles multi-batch data, and `nassoc`
  itself was left untouched.

For a single-batch dataset (current training setup), this loop naturally
degenerates to "reconstruct against all n cells" with no extra machinery
needed — the multi-batch case is handled for free without special-casing it.

## Decoder output range: the follow-up issue

Once the weighting scheme was settled, a separate issue surfaced: the
target (raw affinity/kernel values) is always non-negative, but
`sim_decoder_out` was a plain `nn.Linear` with no output activation, so it
could predict negative "similarities" — meaningless, and corrupting the
downstream `predicted = soft_assign @ decoded` weighted sum.

Fix (implemented in `decode_sim_profiles`,
`interpretable_ssl/models/swav.py`): pass the decoder's raw output through
`softplus` before returning it.

- **ReLU** was considered and rejected: given the target is >99.8% zero,
  clipped units risk reintroducing the exact same dead-gradient collapse
  the class-balanced weighting was added to fix.
- **exp** was also considered (echoing the earlier "unnormalized softmax
  numerator" idea) but risks blow-up for large pre-activations.
- **softplus** was chosen: smooth everywhere, strictly positive, no dead
  zone.

## Batch conditioning of the decoder itself — explicitly not done

Separately discussed and explicitly decided against (for now): making
`sim_decoder` itself batch-conditional, mirroring how the gene-expression
decoder (`self.scpoli_cvae.decoder`, via `CondLayers` + per-condition
`nn.Embedding` lookups) is conditioned on batch in `swav.py`'s `decode`
method. Two structural mismatches make this not a good fit even if desired
later:

- Output space: the scpoli decoder's output width is fixed to `n_genes`;
  `sim_decoder`'s output width is `n_cells`.
- Conditioning axis: scpoli conditions the *input row* (each cell knows its
  own batch). `sim_decoder`'s input rows are *prototypes*, which aren't
  batch-specific — the batch effect lives on the *output column* (which
  cell), not the input row. The batch confound is already fully handled at
  the trainer level (the per-batch loop described above), so no decoder-side
  conditioning is needed.

Current status: `sim_decoder` remains a plain (non-conditional) function of
the prototype embedding, unchanged, per explicit instruction not to
over-engineer this while training is still single-batch.

# `sim_recon` investigation — index

Entry point for the `lambda_sim_recon` line of work: giving scProto
per-cell resolution pressure that `nassoc` structurally can't provide, and
figuring out how close that gets scProto to SEACells' archetypal-analysis
fidelity. Read in this order — each doc builds on the previous one's
conclusion.

## 1. What `sim_recon` is and its first bug: collapse to near-zero

**[`sim_recon_diffusion_coordinates.md`](sim_recon_diffusion_coordinates.md)**

What a diffusion coordinate is, why it's a reasonable compression target for
the affinity graph, and the first bug found: `sim_recon_loss` collapsed to
near-zero within 1-2 epochs. Root cause was a **target-scale artifact**
(`eigsh`'s unit-L2-norm convention shrinks raw entries as `~1/√N_batch`), not
a real vanishing-gradient problem. Fixed by rescaling to O(1) RMS entry,
dropping the trivial leading eigenvector, and removing an unwired
`diffusion_t` config knob entirely (see doc #4 below for why removal, not a
fix, was the right call).

## 2. The `full`-target version of the same collapse, and how it was fixed

**[`sim_recon_sparsity_decision.md`](sim_recon_sparsity_decision.md)**

`sim_recon_target='full'` reconstructs each cell's actual (>99.8%-zero) row
of the affinity graph — a different collapse mechanism (zero-flood
dominating plain MSE, not a scale artifact). Walks through every option
considered (sum-vs-mean reduction, softmax/KL, BCE+negative-sampling,
zero-inflated two-head) and why each was rejected, landing on
**class-balanced weighted MSE** (weight each row's positive/true-neighbor
entries by that row's own `n_neg/n_pos`, no sampling, no row-sum
constraint). Also covers the per-batch/per-section reconstruction scope
(mirrors `nassoc`'s own batch handling) and the decoder's `softplus` output
activation.

## 3. Measuring whether `diffusion`'s compression is actually safe

**[`diffusion_eigenvalue_decay.ipynb`](../notebooks/diffusion_eigenvalue_decay.ipynb)**
(notebook, not a decision doc — referenced here because #4 depends on it)

Measures this dataset's actual diffusion-map eigenvalue spectrum instead of
guessing `n_eigs`. Finding: the spectrum decays **slowly** — 90% of spectral
energy needs `n_eigs≈734`, 95%/99% need *more* than `num_prototypes=800`.
This graph's real structure is spread out enough that no practical `n_eigs`
avoids discarding meaningful signal.

## 4. Why that pushes us toward `full` (+ compaction), not toward tuning `n_eigs`

**[`sim_recon_full_column_subsampling.md`](sim_recon_full_column_subsampling.md)**

Direct consequence of #3: since `diffusion` can't cheaply match SEACells-
level fidelity on this dataset, `full` is the safer target going forward.
Covers `full`'s real cost problem (reconstructing every cell against every
other cell in its batch, expensive at 58k cells) and the fix that already
exists in the codebase: `sim_recon_neg_sample` — column-subsampling that
always keeps each cell's real neighbor columns and freshly samples a random
set of zero columns each step. Flags the subsample size itself as still
unmeasured (the same "guessed instead of measured" gap `n_eigs` had before
doc #3).

## 5. Why we don't fix `diffusion`'s uniform-weighting problem by weighting it

**[`sim_recon_global_vs_local_compaction.md`](sim_recon_global_vs_local_compaction.md)**

Follow-up question: since every kept diffusion dimension is rescaled to
equal weight regardless of how much real structure it carries, why not
weight the loss by each dimension's eigenvalue (variance explained)? Answer:
that's exactly the `diffusion_t` knob from doc #1, already removed, because
it deprioritizes the fine/local directions rare cell types are most likely
to live in. Deeper point: reweighting the loss can't fix this anyway,
because the bias is baked in one stage earlier, at which eigenvectors get
selected into the top-`n_eigs` cutoff in the first place — a rare pattern
that doesn't move population-wide variance has weak influence on survival
regardless of downstream weighting. Lands on the same conclusion as doc #4
from a different angle: compact **locally** (per-cell neighbor-subsampling,
`full`'s existing `sim_recon_neg_sample`), not **globally** (a shared
eigenbasis, however weighted) — a rare cell type's real neighbors are kept
with certainty either way, independent of dataset-wide rarity.

**Refinement added later in the same doc**: `sqrt(eigenvalue)` weighting
(`diffusion_t=0.5`) turns out to be *exactly* (Eckart-Young-optimal, not
approximately) equivalent to MSE on a rank-`n_eigs` reconstruction of the
affinity matrix — i.e. the precise knob that makes `diffusion` behave like
`full`/SEACells, at the precise cost of the fine/rare-pattern sensitivity
`t=0` protects. Implemented as `sim_recon_diffusion_t`
(`configs/defaults.py`, `trainers/scproto.py`), default `0.0` (unchanged
behavior). `notebooks/train_eval_sim_recon_diffusion.py` now trains both
`t=0` and `t=0.5` alongside `arbf` and SEACells so this trade-off is
empirically checkable, not just theoretical.

## 6. Which other loss terms fight `sim_recon` for the same `soft_assign`

**[`sim_recon_competing_losses.md`](sim_recon_competing_losses.md)**

Direct answer to "why does scProto+sim_recon still underperform SEACells" —
`sim_recon` never trains alone; it's one of five active loss terms (`umap`,
`proto_recon`, `nassoc`, `proto_usage`, `sim_recon`) simultaneously
reshaping the same `soft_assign`/prototype/encoder parameters each step.
Walks through each term's purpose and fight-risk with `sim_recon`: `umap`
and `proto_recon` assessed low-risk (aligned or too small a weight, with one
flagged edge case — two rare-but-graph-similar cell types whose real
expression differs more than the kernel's density correction implies);
`nassoc` aligned in principle (same affinity graph) but can give
short-term-conflicting gradients due to a coarse-vs-fine resolution
mismatch; `proto_usage` flagged as the most plausible direct antagonist,
since it rewards even prototype usage regardless of whether real
population sizes support an even split. Also notes the weight imbalance:
three λ=1 graph-topology losses (`umap`, `nassoc`, `sim_recon`) versus
`proto_recon`'s λ=0.01 as the only real-expression grounding force.

## 7. Correction — what SEACells' objective actually is, and what that implies about `proto_usage`

**[`seacells_kernel_archetypal_vs_scproto_losses.md`](seacells_kernel_archetypal_vs_scproto_losses.md)**

Earlier framing in this investigation (including in this notebook's own
intro comments) assumed SEACells' archetypal analysis reconstructs raw/PCA
*expression*. It doesn't — confirmed from the SEACells paper in
`literature/`: it reconstructs the cell-cell **kernel/affinity matrix**
`M ≈ MBA` under hard simplex constraints on `A`/`B`. That makes `nassoc` +
`sim_recon` (both operating on the affinity graph) the real analogue of
SEACells' objective, not `proto_recon` (raw expression, no SEACells
equivalent at all). It also means SEACells gets purity *and* no-dead-
archetypes for free from one structurally-constrained convex optimization,
where scProto spends two separate competing soft losses (`nassoc`,
`proto_usage`) trying to approximate the same two properties. Raises the
open question below about whether `proto_decoupled`+`gmm_resurrect` is a
closer structural analogue than `proto_usage_loss`.

## 8. `proto_usage_loss` vs. two position-anchoring alternatives

**[`proto_anchoring_vs_proto_usage.md`](proto_anchoring_vs_proto_usage.md)**

Works out that `proto_usage_loss` regularizes aggregate softmax *mass*
only — no term for prototype *position* — and actively fights
purity-seeking losses whenever real population sizes are uneven. Compares
two replacements: (A) a soft MSE anchor pulling the free trainable
prototype toward a column-softmax-weighted combination of this batch's cell
embeddings (a latent-space analogue of SEACells' `B` matrix), versus (B)
the already-implemented `proto_decoupled`+`_update_protos_ema` EMA
mechanism. Shows both use the *same* underlying combination rule
(`Σ_i weight[i,k]·z_i / Σ_i weight[i,k]`) — the difference is that (A) is a
single-minibatch, gradient-competing soft penalty (needs a stop-gradient on
the weight matrix to avoid a moving-target feedback loop), while (B) is a
cross-minibatch EMA, fully decoupled from the shared gradient budget, with
an adaptive forgetting factor that specifically protects rare/low-usage
prototypes from single-batch noise. Recommended trying (B) first — **(A) is
now also implemented**, as `lambda_proto_anchor` (`configs/defaults.py`,
`trainers/scproto.py`), mutually exclusive with `proto_decoupled`. Neither
has been run yet.

## Open questions not yet resolved anywhere above

- `sim_recon_neg_sample`'s actual value — not yet chosen empirically.
  `notebooks/train_eval_sim_recon_negsample.ipynb` is built to test a first
  guess (1000) against SEACells (loaded eval-only, no retrain) — not yet run.
- Whether `full` + neighbor-subsampling, run in isolation (competing losses
  `lambda_umap`/`lambda_nassoc`/`lambda_proto_recon` zeroed), can close the
  purity gap to SEACells, or whether that gap is structural (representation
  capacity, joint-objective training) rather than something any `sim_recon`
  variant fixes. `notebooks/train_eval_sim_recon.ipynb` is where that
  ablation belongs once run.
- The per-row (rarity-based) reweighting alternative mentioned at the end of
  doc #5 — noted, not implemented.
- Whether `sim_recon_diffusion_t=0.5` (doc #5's refinement) actually trades
  SEACells-closeness for rare-cell-type purity the way the Eckart-Young
  argument predicts, or does better/worse in practice — implemented and
  wired into `notebooks/train_eval_sim_recon_diffusion.py`, not yet run.
- Whether `proto_usage_loss` (soft, gradient-competing) can be dropped in
  favor of `proto_decoupled=True` + `gmm_resurrect=True` (EMA-of-cluster-mean
  position update, hard-anchored to real cell embeddings, non-gradient
  anti-collapse via explicit resurrect) — see doc #8. Not yet run.
- Same question, cheaper variant: `lambda_proto_anchor>0` (option A from
  doc #8, now implemented, prototype stays a free gradient param — no
  `proto_decoupled`) vs. `proto_usage`-only vs. option B, three-way. Not yet
  run.

## Notebooks referenced across these docs

- `notebooks/train_eval_sim_recon.ipynb` / `.py` — `arbf` baseline vs
  `+full` vs `+diffusion` vs SEACells, `full`-target scope.
- `notebooks/train_eval_sim_recon_diffusion.ipynb` / `.py` — dedicated
  `diffusion`-target notebook (kept separate so the two targets' runs don't
  clobber each other); includes the between-prototype-variance-fraction
  diagnostic added while investigating doc #1's collapse, and (doc #5's
  refinement) an unweighted (`t=0`) vs `sqrt(eigenvalue)`-weighted (`t=0.5`)
  comparison.
- `notebooks/diffusion_eigenvalue_decay.ipynb` / `.py` — doc #3's
  measurement.
- `notebooks/train_eval_sim_recon_negsample.ipynb` / `.py` — `full`-target
  `sim_recon` with `sim_recon_neg_sample` column-subsampling vs SEACells
  (SEACells loaded eval-only). Companion to doc #4.

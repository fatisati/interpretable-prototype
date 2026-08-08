# Why `full` (+ column-subsampling) over tuning `diffusion`'s `n_eigs`

## The question this resolves

We'd been guessing at `sim_recon_target='diffusion'`'s `n_eigs` (10, then 128,
then 1024) without ever measuring the one thing that actually determines
whether a given `n_eigs` is enough: how fast this specific graph's diffusion-
map eigenvalue spectrum decays. `notebooks/diffusion_eigenvalue_decay.ipynb`
measured it directly instead of guessing further.

## What it measured

Same construction as `_compute_sim_recon_diffusion_targets`
(`trainers/scproto.py`): symmetric-normalize the `s28nsc`/`arbf` affinity
graph (`L_sym = D^{-1/2} A D^{-1/2}`), `eigsh` out to 900 eigenvectors, drop
the trivial leading one, then read cumulative spectral "energy"
(`Σeigenvalue²`, the diffusion-map analog of PCA's explained-variance-ratio)
against `num_prototypes = NP = 800`.

## What it found — the spectrum decays slowly, not quickly

- Top 10 eigenvalues (after dropping the trivial leading one):
  `1.0, 0.99, 0.98, 0.97, 0.97, 0.97, 0.96, 0.96, 0.95, 0.94` — barely moved
  in 10 components.
- Eigenvalues 190-199 (near the cheap 200-eigenvector probe's cutoff): still
  `~0.69-0.70` — nowhere near zero at rank ~200.
- Cumulative energy thresholds (deep probe, `k=900`, took 16.2 min):
  - **90%** → `n_eigs=734` (just inside `NP=800`)
  - **95%** → `n_eigs=813` (**exceeds** `NP=800`)
  - **99%** → `n_eigs=882` (**exceeds** `NP=800`)

This graph's real structure isn't concentrated in a handful of coarse
directions — it takes nearly as many eigenvectors as there are prototypes
just to capture 90% of it, and *more than* `NP` to reach 95-99%. That's a
high-effective-rank / "spread out" graph, not a small-number-of-communities
graph.

**This overturned our own earlier guess.** We had been assuming the tail
past a few hundred eigenvectors was mostly noise and recommending `n_eigs`
be cut down (to 128, or even 32-64) to avoid asking the decoder to fit
unlearnable dimensions. This notebook shows that's wrong for this dataset:
those dimensions carry real, substantial eigenvalue energy, not noise. The
existing run's `n_eigs=1024` was *not* the wastefully oversized choice we'd
assumed — it sits just above the 99%-energy point (882), i.e. roughly
appropriately sized.

## The verdict (from the notebook's own closing markdown)

> If it takes close to or more than `num_prototypes` eigenvectors to reach
> 90%, the graph's real structure is spread out enough that any `n_eigs`
> you'd consider is discarding a meaningful amount — `full` (+
> column-subsampling for speed) is the safer match to SEACells-level
> fidelity.

That's the case here (734 ≈ 800). So going forward: stop tuning
`diffusion`'s `n_eigs` as the fidelity lever for this dataset — reach for
`sim_recon_target='full'` instead, which reconstructs each cell's actual
affinity row rather than a fixed-size spectral compression of it, and is the
more literal match to what SEACells' own RSS objective (`||X - XBA||²`,
`SEACells/cpu_dense.py`) optimizes.

## The cost problem `full` has, and the fix that already exists

`full` mode's naive cost: for a single-batch dataset like `s28nsc`
(58,423 cells, one section), each cell is reconstructed against *every*
other cell in its batch (`trainers/scproto.py`'s per-batch `sim_recon_loss`
block) — i.e. `decode_sim_profiles` producing a `(K, n_batch_cells)` slice
and a matching dense target slice, every single training step. For 58k
cells that's expensive per step, which is exactly why `n_eigs`-tuned
`diffusion` looked appealing in the first place — but per the measurement
above, that appeal was based on a wrong assumption about this graph's
spectrum.

**The fix already exists in the codebase**: `sim_recon_neg_sample`
(`configs/defaults.py:221`, abbreviated `srns` in `constants.py:142`,
implemented at `trainers/scproto.py:1596-1643`). When set `>0`:

- Always keeps every column that's a real neighbor of the current row
  (`pos_cols = np.unique(aff_csr[idx_b_np].indices)`) — the informative
  "yes" cases, cheap since true degree is tiny (~74.5 mean here).
- Adds a random sample of `sim_recon_neg_sample` zero-target columns
  (`neg_cols`), **freshly drawn every step** — nothing is permanently
  discarded the way a fixed diffusion-map basis would discard high-index
  eigenvectors. Over many steps the model still sees the full negative
  population, the same guarantee ordinary minibatching already relies on.
- The class-balanced positive/negative weighting (from
  `files/sim_recon_sparsity_decision.md`, option 5) is computed from
  whatever ends up in this reduced column set, so it self-adjusts to the
  smaller column count — no separate reweighting needed.
- Default is `0` (off — reconstruct against every column in the batch, the
  original, most expensive behavior).

## What's still open

`sim_recon_neg_sample`'s actual value hasn't been chosen yet — this is the
same kind of "we guessed instead of measuring" gap that `n_eigs` had before
`diffusion_eigenvalue_decay.ipynb`. Reasonable next step: pick a value (e.g.
a small multiple of mean degree, ~74.5 here) empirically — training-speed
vs. `sim_recon` loss-quality tradeoff — rather than guessing a round number,
the same lesson this whole investigation keeps teaching.

## Where this is tracked

- `notebooks/diffusion_eigenvalue_decay.ipynb` — the eigenvalue-decay
  measurement itself.
- `notebooks/train_eval_sim_recon.py` (`.ipynb`) — trains and compares
  `arbf`, `arbf+full`, `arbf+diffusion`, and SEACells; `full` is the
  variant this doc recommends prioritizing going forward.
- `files/sim_recon_diffusion_coordinates.md` — what a diffusion coordinate
  is and the earlier (target-scale) collapse fix for `diffusion` mode.
- `files/sim_recon_sparsity_decision.md` — the class-balanced weighting
  `sim_recon_neg_sample` reuses, and why `full` mode's zeros can't just be
  ignored outright.

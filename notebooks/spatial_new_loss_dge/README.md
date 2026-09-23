# spatial_new_loss_dge

**What it answers.** Does the corrected `nassoc` diagonal remove metacell collapse on the
spatial dataset, and does removing it close the niche-program-recovery (DGE) gap that the
plan1 rebuttal experiment found against SEACells?

The plan1 result had SEACells ahead on both DGE branches, with the diagnosis that
scProto's niche-positive metacells were built from too few real cells to give a stable
pseudobulk fold change — the fragmentation half of the over-merging failure. The new loss
targets exactly that, so this is the first experiment worth running with it.

**Dataset.** `ss28nsc` (NSCLC CosMx, one slide) and only that one, run fully, before any
second tissue.

**Affinity.** `bk08` — BANKSY at lambda=0.8 plus the real SEACells RBF kernel.

## Arms

| Arm | Source | Why |
|---|---|---|
| `scProto (new loss, pair)` | trained here | the candidate |
| `scProto (old loss, volume)` | trained here | one-variable control — only `nassoc_diag_norm` differs |
| `SEACells (PCA)` | loaded | transcriptomics-only baseline, zero spatial signal |
| `scProto (plan1 run, ctx)` | loaded | the run the already-reported rebuttal numbers came from |

The two trained arms share every hyperparameter except `nassoc_diag_norm`, so any
difference between them is attributable to the loss.

## Running it

Top to bottom. Every step is load-if-exists — finished steps print `loaded from ...` and
cost nothing, so after a Colab disconnect just re-run from the top.

- `LOAD_UMAP = True` — reload trained checkpoints instead of retraining.
- `FORCE_RERUN = True` — deliberately recompute the eval tables. Leave `False` otherwise.

**Restart the runtime after the pip cell.** Not optional — numpy/scipy/anndata are
C-extension linked.

**Check `X_bk08 precomputed: True`** in the dataset cell. If it is False and `ss28nsc` is a
filtered subset, BANKSY's neighbour-mean term will only see the surviving cells, which is
not the real spatial context. Do not report numbers from a run that warned about this.

## Outputs

```
tables/ss28nsc/collapse_summary.csv        largest_share, dip_p, joint purity per arm
tables/ss28nsc/<arm>_per_pair.csv          per (cell type, niche): pearson_r, tau, tpr, sizes
tables/ss28nsc/<arm>_summary.csv           macro means + stds
tables/ss28nsc/dge_summary.csv             all arms stacked
tables/ss28nsc/dge_paired.csv              paired per-pair gaps vs the new-loss arm
tables/ss28nsc/dge_size_floor_sweep.csv    gap vs minimum metacell size
tables/ss28nsc/discovery_candidates.csv    pairs whose genes pooled cell-type DE cannot see
figures/ss28nsc_collapse.png
figures/ss28nsc_dge_summary.png
figures/ss28nsc_size_floor_sweep.png
```

Shared across notebooks (written to `files/`, not here):
`celltype_niches_full.csv` (ground-truth niche DE) and `celltype_pooled_de.csv`
(pooled cell-type DE) — both method-independent, so they are computed once.

## Reading the result

1. `collapse_summary.csv` is the gate. Healthy = `dip_p > 0.05` (one smooth size hump, not
   a cliff) and a lower `largest_share` than the old-loss arm. If the new loss did not
   change these, nothing downstream is meaningful.
2. `dge_summary.csv` — did `pearson_r` close the gap to SEACells?
3. `dge_size_floor_sweep.csv` — if a gap remains, does it shrink as the minimum metacell
   size rises? Shrinking = small-N pseudobulk noise. Flat at floor 30–50 = the signal
   really is not there.
4. `discovery_candidates.csv` — the shortlist for the paper's R4 figure. Everything else in
   this notebook measures *recovery* of programs already visible in the single cells, which
   is a validity check and not a discovery claim. This table is the non-circular version.

## Where the code lives

All logic is in `interpretable_ssl/evaluation/niche_dge.py` (a library port of
`neurips_manuscript/rebuttle/notebooks/plan1_niche_recovery_eval.ipynb`, so numbers stay
comparable with what was already reported) and
`interpretable_ssl/evaluation/collapse_diagnostics.py`. The notebook only configures and
calls them.

# s28nsc_dge_lite

**What it answers.** Same niche-DGE recovery comparison as
`notebooks/new_features/nassoc_density_d1_d2.ipynb`'s DGE section (`A_baseline`,
`B_d1_pair`, `C_d1d2_g0.01/0.03/0.1`, `SEACells_bk08`, all K=200 on `s28nsc`) -- but this
notebook never constructs a model. It reads each arm's already-saved per-cell metacell
labels (`cell_assignments.csv` / `umap_cells.csv`) via `nd.labels_from_run`, which only needs
`pandas`. No `torch`, no `scarches`, no SEACells package, no `faiss`, no `cupy`, no GPU.

**Dataset.** `s28nsc` only.

**When to use this vs. the full notebook.** Use this one to re-slice or re-run DGE metrics
(new size floors, a different `DGE_REFERENCE` arm, rebuilding a figure) against arms that
have *already* finished training and eval. If you need a new gamma value, a fresh
reproducibility run, or anything else that requires training, that has to happen first in
`nassoc_density_d1_d2.ipynb` -- this notebook is read-only over what that one already wrote
to disk.

**Known gap: `B_d1_pair` is currently SKIPPED here.** Checked directly against
`MODEL_DIR/s28nsc/proto_bk08_np200_ema15_d1pair_.../` on 2026-09-05: that folder has
`metacells.h5ad`, `metrics.json`, `umap_checkpoint.pth` (the model is real and trained), but
no `cell_assignments.csv` / `umap_cells.csv` / `clusters.npz` -- i.e. its eval step never
finished writing per-cell exports (most likely a Colab disconnect after the checkpoint saved).
`nd.labels_from_run` has nothing to read for it, so it prints `SKIP B_d1_pair: no saved cell
assignments` and it's absent from every table here. Its real collapse numbers (the healthy
`largest_share≈0.034`, `dip_p≈0.986` reported earlier) come from `labels_from_trainer` in
`nassoc_density_d1_d2.ipynb`, which forward-passes the loaded checkpoint directly instead of
reading a CSV -- that's the thing this lite notebook trades away for the lighter install.
**Fix:** in Colab, load that checkpoint with `load_umap=True, skip_eval=False` (forces the
missing eval/export step without retraining) so `cell_assignments.csv` gets written, then
`B_d1_pair` will show up here on the next run.

## Running it

Top to bottom, load-if-exists throughout. `FORCE_RERUN = True` only to deliberately
recompute ground truth or an arm's DGE table.

**Restart the runtime after the pip cell.** Still needed: numpy/scipy/anndata are
C-extension linked, same as every other notebook in this project, even though the install
list here is much shorter.

## Outputs

```
tables/collapse_summary.csv        size/dip/purity per arm (no tightness -- no affinity loaded)
tables/dge_summary.csv             macro means + stds per arm
tables/dge_paired.csv              per (cell type, niche) gaps vs DGE_REFERENCE
tables/dge_size_floor_sweep.csv    gap vs minimum metacell size
figures/dge_summary.png
figures/dge_size_floor_sweep.png
```

Shared across notebooks (written to `files/`, not here, and s28nsc-scoped so they never
collide with `ss28nsc`'s cached ground truth):
`celltype_niches_full_s28nsc.csv`, `celltype_pooled_de_s28nsc.csv`.

## Where the code lives

All logic is in `interpretable_ssl/evaluation/niche_dge.py` and
`interpretable_ssl/evaluation/collapse_diagnostics.py` -- the same libraries
`nassoc_density_d1_d2.ipynb` and `spatial_new_loss_dge.ipynb` use. This notebook only
configures and calls them, using `labels_from_run` instead of `labels_from_trainer`.

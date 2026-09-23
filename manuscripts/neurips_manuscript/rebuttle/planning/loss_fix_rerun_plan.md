# nassoc loss fix: what needs rerunning, and in what order

Context: while building the spatial DGE (niche-recovery) rebuttal experiment on `s28nsc`,
we found `L_nassoc`'s diagonal term (`nassoc_diag_norm='volume'`, the current default)
structurally favors merging prototypes into a few giants plus a tail of near-empty ones,
because "containment" (edges_inside/volume) is additive under merging. This note inventories
every experiment that could be touched by that fix, what we already know, and what's still
open, so the rerun decision is made once, deliberately, instead of dataset-by-dataset as
Colab time allows.

## What we've actually measured so far (not guesses)

| dataset | evidence | shape |
|---|---|---|
| `s28nsc` (spatial), old loss (`volume`/`sqrt`, the default) | `spatial_new_loss_dge` table + `scproto_niche_advantage_notes.md` (independent, earlier notebook) | **collapsed**: median metacell size 2, `dip_p=0.0` (cliff, not one hump), `largest_share=0.177`, effective_n≈7-9/200 in the earlier check. Only 8-25/146 (cell type, niche) pairs even testable. |
| `s28nsc`, D1 fix only (`nassoc_diag_norm='pair'`) | same table | **fixed the cliff** (`dip_p=0.212`, healthy) but **fragments resolution**: only 48 metacells survive (was up to 300), so testable pairs drop 25→8. DGE recovery (`pearson_r`, `tpr`) improves a lot but still well below SEACells. |
| `s28nsc`, D1+D4 (`nassoc_offdiag_norm='min'`, added on top) | `modal_scripts/train_bk02_d1_offdiag_min.py` | **queued, not yet run.** This is the candidate for recovering resolution without reintroducing the cliff. |
| Pancreas / Lung / PBMC-immune (main table), old loss (default) | `kbet_ilisi_metacell_results.md` | **indirect, partial reassurance only**: metacell counts land close to target K (pancreas 219/220, lung 298/300) — rules out *gross* collapse into a handful of prototypes, but nobody has run the actual `dip_p`/size-distribution check on them. A milder version of the same pathology is still possible and untested. |

The honest state: **we know the spatial loss was broken and have a partial fix. We do not
know whether the three main-table datasets have the same problem**, because the diagnostic
that would tell us (`collapse_diagnostics.compare_runs`, added this session) has only ever
been pointed at `s28nsc`.

## Why this matters for scope

If Pancreas/Lung/PBMC turn out healthy: the fix is a spatial-only footnote, one notebook's
numbers get updated, nothing else moves. If they turn out to have the same pathology: the
**main paper's headline table (Table 1: modularity/batch-entropy/purity) and rare-cell table
(Table 2) were computed on a suboptimal checkpoint**, and every downstream rebuttal
experiment that reads from those same checkpoints inherits the same question mark. That is
the difference between "footnote" and "retrain three datasets, redo their ablations,
significance tests, and heldout-modularity results before resubmission."

## Experiment inventory, split by whether it can even be affected

**Not affected — uses no scProto-trained checkpoint at all.** Nothing here depends on
`nassoc`/`sim_recon`; these numbers stand regardless of the outcome below.
- scVI baselines: `scvi_default_stage2_results.md`, `scvi_gauss_matched_dim_results.md`,
  `scvi_stage2_summary.json`, `scvi_stage2_pancreas_summary.json`
- Harmony baselines: `harmony_default_dim_results.md`, `harmony_matched_dim_results.md`
- BBKNN baseline: `bbknn_matched_dim_results.md`
- SEACells-on-PCA arm of every table (it's the fixed reference point, not a scProto run)

**Affected only if the main-table datasets turn out unhealthy** — these read scProto
checkpoints for Pancreas/Lung/PBMC-immune trained with the current default loss:
- Main paper Table 1 (`results.tex`, community structure + batch integration) and Table 2
  (rare-cell coverage/homogeneity)
- `ablation_results.md` (nassoc on/off, stop-gradient — all 3 RNA-seq datasets)
- `node_heldout_modularity_results.md` (all 3 RNA-seq datasets)
- `headline_significance_test_results.md` (paired significance on Table 1/2, same checkpoints)
- `kbet_ilisi_metacell_results.md`, Part A (scProto rows only; SEACells/Leiden rows unaffected)

**Already known affected, partial fix in hand, work in progress** — spatial only:
- `scproto_niche_advantage_notes.md` (original collapse diagnosis)
- Plan 1 / `dge_story.md` (the rebuttal DGE-recovery claim vs. SEACells)
- `spatial_new_loss_dge` notebook (the fix-in-progress; D1 done, D4 queued)
- `nassoc_density_d1_d2` notebook (D1/D2/gamma sweep, s28nsc only)

## Recommended order

1. **Cheap diagnostic first, no retraining, no GPU.** Load the existing Pancreas/Lung/PBMC
   `scProto` checkpoints (`load_umap=True`) and run
   `collapse_diagnostics.compare_runs` (already used for `s28nsc`) on each. This needs
   only the checkpoint + its cached affinity, both already on disk — minutes, not a
   training run. Output: `dip_p`, `largest_share`, `size_tightness_corr` per dataset.
   This is the fact that decides everything below, so it goes first regardless of what
   else is queued.
2. **Finish the D4 test already queued** (`train_bk02_d1_offdiag_min.py`, ~12-13 min on a
   T4) — decides whether D1+D4 is the final spatial loss config or D1 alone is what we
   report, before writing up Plan 1's numbers a second time.
3. **Branch on step 1's result:**
   - Main-table datasets healthy → update only the spatial rebuttal writeup with the
     D1(+D4) numbers; main paper and its other rebuttal experiments are untouched.
   - Main-table datasets unhealthy → retrain Pancreas/Lung/PBMC with the winning config,
     one dataset at a time per CLAUDE.md §6, then re-run (in this order, each cheap
     relative to training): Table 1/2 eval → `headline_significance_test` →
     `ablation_results` → `node_heldout_modularity` → `kbet_ilisi` Part A. Baselines
     (scVI/Harmony/BBKNN/SEACells) are never rerun — they don't use this loss.

## Open decisions (yours, not mine)

- Whether to greenlight step 1 (the diagnostic) now, given it's cheap and removes the
  biggest unknown before any GPU time is spent.
- Whether "D1 alone" is an acceptable spatial result to report if D4 doesn't clearly beat
  it, or whether the rebuttal should wait for D4 either way.
- If step 1 comes back unhealthy: whether a full main-table retrain is even feasible in
  the remaining time, versus reporting the known limitation and prioritizing the spatial
  story only.

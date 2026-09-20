# SEACells(PCA), s28nsc, K=200 — the information-matched baseline

Run 2026-09-20. Script: `modal_scripts/train_seacell_pca_k200.py` (Modal, CPU).
Artifacts: `models/s28nsc/seacell_K200/`, `files/tables/s28nsc_seacell_pca_k200/`,
`files/figures/s28nsc_seacell_pca_k200/`.

**This note records measurements only. No conclusion has been drawn — that is Fatemeh's
call.**

## Why it was run

Every s28nsc K=200 comparison on record used `seacell_X_bklam02_K200` as the SEACells
reference: SEACells fitted on the **BANKSY lambda=0.2 embedding**, i.e. on features that
already mix each cell's expression with its spatial neighbours'. `decisions.md` §14
measured that this fact alone accounts for most of the gap — through the same no-bias
linear head fitted to the same targets, `BANKSY X_bklam02` (50-d) reaches
internal_fraction 0.4262 while `pca50` reaches 0.2790.

scProto's encoder receives **expression plus batch key only**. Spatial information reaches
it solely through the affinity graph inside the nassoc loss. So the BANKSY SEACells run is
not information-matched to scProto.

The only SEACells(PCA) run that existed on s28nsc was K=800 (`models/s28nsc/seacell`,
`n_SEACells: 800`), so it was not K-matched either. **Nothing on record was both.** This
run is: same expression matrix, same 50-component PCA, same K=200, same graph and same
ground truth for scoring.

## What is matched and what is not

```
matched across ALL rows of the master table:
  cells                58,423 (s28nsc)
  K                    200 prototypes/metacells
  scoring graph        bk02  (internal_fraction, density, tightness)
  DGE ground truth     files/celltype_niches_full_s28nsc.csv, 146 (cell type, niche) pairs
  metric definitions   identical code path

NOT matched (this is the variable under test):
  features SEACells was FITTED on
      SEACells bk02 -> X_bklam02   expression + spatial neighbours' expression
      SEACells PCA  -> X_pca       expression only  <-- what scProto's encoder sees
```

## Numbers

```
run                  DGE(pearson_r)  n_pairs  density  int_frac  median  max/med  jointP
SEACells bk02            0.849         19     0.02952   0.4465    258      3.6     0.504
SEACells PCA             0.747         16     0.01462   0.2343    249      4.5     0.401
best scProto run         0.834         16     0.02628   0.2926    293      1.3     0.438
  (lf_L16_g015_uOFF)
```

Term definitions, since these get confused:

- `DGE` = `pearson_r` from the niche-DGE evaluation: for each (cell type, niche) state,
  how well the metacell-level differential-expression signal reproduces the cell-level
  one. Higher = the metacells preserve within-cell-type spatial programs.
  `0.7470 +/- 0.0698` (std over pairs); kendall_tau `0.5109 +/- 0.0623`; tpr `0.9573`.
- `n_pairs` = how many distinct (cell type x niche) states the partition actually
  resolved, out of 146 in the ground truth. This is the coverage axis.
- `density` = `aggregate_density` = summed affinity weight of within-metacell edges
  divided by the total number of possible within-metacell pairs.
- `int_frac` = `internal_fraction` = summed affinity weight of edges whose two endpoints
  landed in the SAME metacell, divided by total edge weight in the graph. Placement
  quality on that graph, nothing to do with size.
- `jointP` = `purity_micro` of the joint `(cell type | niche)` label — each CELL counts
  equally, so it is not inflated by fragmentation.
- `max/med` = largest metacell size divided by the median — size unevenness.

internal_fraction of this same partition on the other two graphs, for use against runs
trained on them (`partition_stats.csv`):

```
graph   internal_fraction   aggregate_density
bk02          0.2343             0.014622
bk04          0.1814             0.010884
bk08          0.0743             0.004391
```

Full task1/2/3 metric set, each with its std, in `models/s28nsc/seacell_K200/metrics.json`:

```
mean_cell_type_purity           0.8126 +/- 0.1946   (size-weighted 0.8200 +/- 0.1842)
mean_niche_purity               0.4923 +/- 0.1904
mean_batch_entropy             -1.0e-10 +/- 0.0
coverage                        0.9444
modularity                      0.4777
mean_modularity_batch           0.4777 +/- nan
aff_compactness_mean            0.1112              (measured on bk02)
dge_rbo_avg                     0.3149
dge_kendall_avg                 0.4368
dge_jaccard_avg                 0.2979
scgraph_corr_avg                0.8710 +/- 0.0546
ct_niche_rbo_avg                0.0167 +/- 0.0328
n_unused_protos                 0 / 200
```

Collapse diagnostics on bk02 (`collapse_summary.csv`): `largest_share` 0.0191,
`dip_p` 0.840 (sizes are one hump, no collapse cliff), `tightness_gap` 0.0101,
`size_tightness_corr` -0.786, `purity_micro` 0.4013, `purity_gap` 0.0297.

## Caveats that belong with these numbers

1. **`std_modularity_batch` is NaN and `mean_batch_entropy` is ~0 because s28nsc has a
   single batch** (`section_28`). Same as every other s28nsc run — it is the data, not a
   failed computation.

2. **`ct_niche_rbo_avg` is 0.0167, near zero, and that is explained in the log, not a
   bug.** Four cell types (Cycling immune cells, Regulatory T cells, Mast cells, B cells)
   end up with **no niche holding more than one metacell**, so their metacell-side marker
   list is empty and RBO is 0 by construction. At K=200 with 20 cell types x 9 niches,
   most (cell type, niche) states get 1 metacell or fewer. This is a property of K=200,
   not of PCA — check the same field on the other K=200 arms before comparing.

3. **The niche-DGE pseudobulk spot check printed
   `corr(total counts, n_cells) = 0.247 (want ~1.0)`.** That check tests whether the
   pseudobulk summed counts scale with metacell size, which they only do if the layer
   being summed is raw counts. s28nsc is stored normalized (`"normalized": True` in its
   dataset config), so a low value here is expected for this dataset and applies to
   **every** arm scored on it equally — it does not invalidate the between-arm
   comparison. It does mean the absolute DGE numbers on s28nsc rest on normalized
   pseudobulk. Not investigated further here.

4. **The local `seacell_sc.h5ad` is a delta against
   `models/s28nsc/seacell_sc_base.h5ad`** (see `save_seacell`). The Modal-side base and
   the Drive-side base are not byte-identical: the Modal base additionally carries
   `obsm['X_bklam02']` and `obsm['X_bklam04']`, and their `obsp` hashes differ. Every
   field this run actually needs matches by hash (`X`, `layers['counts']`,
   `layers['SCT']`, `obsm['X_pca']`) and the run's own `obsp` + `obs['SEACell']` are
   stored inside the delta, so reconstructing locally is correct — it just will not carry
   the two BANKSY obsm keys. The CSVs and `metrics.json` are the authoritative outputs
   either way.

5. **`aff_compactness_mean` was measured on bk02**, matching the graph the K=200 scProto
   arms are judged on. The K=800 `models/s28nsc/seacell` run's value (0.3239) was measured
   on a different graph and is **not** comparable to this one.

## Open question for Fatemeh

Section 1 of `EXPERIMENTS.md` currently reads "Zero runs beat SEACells bk02 on both DGE
and coverage", and section 9 / `decisions.md` §14 frame the remaining gap as an
information gap. `SEACells PCA` sits at DGE 0.747 / int_frac 0.2343 / jointP 0.401, which
is below several scProto runs on DGE and int_frac and above them on none of the three at
once. **What, if anything, this row should change in those sections has not been written
in.** Nothing in `EXPERIMENTS.md` was rewritten beyond adding the row and disambiguating
which SEACells each statement refers to.

# Heart Cell Atlas (subsampled)

**Source:** Litviňuková et al. 2020, *Nature* — "Cells of the adult human heart."
Built into scvi-tools specifically for NB-loss models (scVI/scANVI), auto-downloaded by
its own loader — no figshare step, no manual download, nothing blocked.
https://www.heartcellatlas.org/

**Citation required:** cite Litviňuková et al. 2020 when using this data.

## Get it

No manual download needed. Run stage 1 of [`docs/runbook.md`](../docs/runbook.md). The script calls `scvi.data.heart_cell_atlas_subsampled()`, which downloads
`hca_subsampled_20k.h5ad` into `datasets/raw/` itself (cached — subsequent runs skip the
download), then writes `datasets/processed/heart_cell_atlas_hvg.h5ad`.

| | raw file | processed (`heart_cell_atlas_hvg.h5ad`) |
|---|---|---|
| sha256 | `4a54acfa72faf108473f8c2705894eea2194b45a2b5e21b7630bdd32eaf1e808` | `7e686ed33e98f18b3d0f7bff08eb3011b8b5daae00ab9bc67c02cd7fdb96de4f` |

Re-ran the whole pipeline twice on a clean `processed/` output — both runs produced the
**identical** processed checksum above, confirming the pipeline is deterministic (no
unseeded randomness in the `seurat_v3` HVG step).

## Environment (exact versions used to produce the checksums above, 2026-11-16)

```
python        3.13.11
scvi-tools    1.5.0.post1
scanpy        1.12.4
anndata       0.13.2
numpy         2.4.6
torch         2.13.0+cpu
scikit-misc   0.5.3   (needed for scanpy's seurat_v3 HVG flavor)
```
scvi-tools additionally required installing `lightning`, `docrep`, `ml-collections`,
`mudata`, `pyro-ppl`, `sparse>=0.14.0`, `tensorboard`, `xarray` (not pulled in
automatically by the scvi-tools package install in this environment).

## Environment (SEACells baseline, 2026-11-21)

```
SEACells   0.3.3   (editable install from F:\codes\SEACells)
palantir   1.4.6
numba      0.67.0
```
Installed with `--no-deps` and `pyranges` skipped deliberately -- it's only used by
SEACells' ATAC-specific modules (`genescores.py`/`tfactivity.py`/`accessibility.py`),
never imported by the RNA metacell path (`core.py`/`cpu_dense.py`) this baseline uses,
and it needs a C compiler this machine doesn't have. Also needed `ipython` (pure
Python, no compiler) -- `SEACells/utils.py` imports `IPython.display` at module load
time even though this baseline never calls anything that uses it.

## Stats (measured directly, 2026-11-16)

- raw: 18,641 cells × 26,662 genes (default subsample of the full 486,134-cell atlas)
- processed: 18,641 cells × 2,000 genes (HVG-subset)
- batch key: `donor` — 14 donors, sizes range 581–3,009 cells (D1 smallest)
- label key: `cell_type` — 11 broad types; also `cell_states` — 65 fine-grained states

### Rare cell types — confirmed directly, not just from the literature

Broad (`cell_type`), rarest first:

| cell_type | cells | % |
|---|---|---|
| Mesothelial | 17 | 0.09% |
| Adipocytes | 145 | 0.78% |
| **Neuronal** | 153 | 0.82% |
| Smooth_muscle_cells | 641 | 3.44% |

Matches the literature's "<1%" claim for neuronal cells exactly.

Fine-grained (`cell_states`), rarest 10 of 65:

| cell_states | cells | % |
|---|---|---|
| NC6 | 2 | 0.011% |
| NC5 | 3 | 0.016% |
| IL17RA+Mo | 3 | 0.016% |
| Adip4 | 5 | 0.027% |
| aCM5 | 5 | 0.027% |
| NC4 | 5 | 0.027% |
| Adip3 | 6 | 0.032% |
| NC3 | 10 | 0.054% |
| Meso | 17 | 0.091% |
| **vCM5** | **32** | **0.172%** |

`vCM5` — the rare conduction-system-linked cardiomyocyte subtype the literature flags —
is present with 32 cells, enough to work with. **Caveat:** several neuronal sub-states
(NC3–NC6) have only 2–10 cells in this 20k-cell subsample — too few for any reliable
per-type metric. The literature's "~1%" figure for vCM5 is for the full 486k-cell atlas;
subsampling to 20k cells shrinks already-rare populations further and non-uniformly, so
don't assume the full atlas's rare-type proportions carry over unchanged.

## Preprocessing applied (`datasets/scripts/prepare_scvi_builtin.py`)

Different recipe from the scIB trio (`prepare.py`), because this dataset's purpose is
different (NB-loss training, not MSE-loss baselines) and its raw counts are trustworthy:

1. **Confirmed `X` is true raw counts directly**: 0.0000% of nonzero entries are
   non-integer (vs. 42.8%/91.6%/7.9% for pancreas/lung/pbmc_immune). Script raises an
   error and stops if this check ever fails, rather than silently training on bad data.
2. `X` copied into `layers['counts']` for clarity; `X` itself is **not**
   normalized/log-transformed — scVI's own NB likelihood expects raw counts as input,
   and does its own size-factor handling internally.
3. HVG: top 2,000 genes (scvi-tools' own tutorial default, not the scIB trio's 4,000),
   `sc.pp.highly_variable_genes(flavor='seurat_v3', layer='counts', batch_key='donor')`
   — `seurat_v3` operates directly on raw counts, the recipe scvi-tools' own
   preprocessing tutorial recommends for data headed into an NB-loss model.
4. Subset to the HVG set, write to `datasets/processed/heart_cell_atlas_hvg.h5ad`.

## Known caveat

None on raw counts — this is the one dataset among those looked at so far where `X` is
directly confirmed trustworthy for NB-loss training, no workaround needed. The only
caveat is the subsampling effect on already-rare cell states noted above.

# Runbook: the command for each stage

The one place for **how to run** each pipeline stage, in the order you run them.
Stage numbers match `docs/architecture.md`. Run every command from the repo root.

**What was actually run** is recorded automatically. Each script adds a `run_history`
entry to the JSON file it saves (see `evaluation/run_record.py`). That entry holds the
exact command, the code version (git commit), whether there were uncommitted edits
(`git_dirty`), and the start/end time. To see how an output was made, open its JSON
and don't rely on this doc.

Examples use heart_cell_atlas. For other datasets, take the flag values (`--h5ad`,
`--batch-key`, `--label-key`, `--layer`) from the table in `baselines/README.md`.
Run times are measured on the heart atlas (~18.6k cells, CPU); "not measured" means nobody has timed it yet.

---

## 0. Comparing two machines

Background and the full plan: [`reproducibility-plan.md`](reproducibility-plan.md). Evidence
from the Mac and step-by-step instructions: [`../repro/README.md`](../repro/README.md).

```
# hashes + summaries of one dataset's stage outputs -> repro/fingerprints/<dataset>/<machine>.json
python evaluation/fingerprint.py --run-dir runs/rare_cell_preservation/heart_cell_atlas --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --layer counts --machine <machine>
# two fingerprints, stage by stage; names the first input difference
python evaluation/compare_fingerprints.py repro/fingerprints/heart_cell_atlas/<a>.json repro/fingerprints/heart_cell_atlas/<b>.json
# two run folders or evidence packs: gene, split, graph, embedding, cluster and metric differences
python evaluation/compare_runs.py --a <folder> --b <folder>
# small committable copy of a run folder -> repro/evidence/<machine>/<dataset>/
python evaluation/make_evidence_pack.py --run-dir runs/rare_cell_preservation/heart_cell_atlas --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --machine <machine>
```

- **Reads only**; trains nothing. `fingerprint.py` recomputes one PCA (seconds).
- **Time:** under 1 min each on heart.

## 1. Dataset prep: raw file → clean, HVG-selected `.h5ad`

| dataset | command | download first? |
|---|---|---|
| heart_cell_atlas | `python datasets/scripts/prepare_scvi_builtin.py --dataset heart` | no, it downloads itself |
| pancreas | `python datasets/scripts/prepare.py --dataset pancreas` | yes, see `datasets/pancreas.md` |
| lung | `python datasets/scripts/prepare.py --dataset lung` | yes, see `datasets/lung.md` |
| pbmc_immune | `python datasets/scripts/prepare.py --dataset pbmc_immune` | yes, see `datasets/pbmc_immune.md` |
| hlca | `python datasets/scripts/prepare_hlca.py` | yes, see `datasets/hlca.md` |

Task 1 candidates downloaded 2026-11-22. They have no prepare script yet, so these are
download-only commands. Each file goes in its own folder under `datasets/raw/`:

```
B=https://ftp.ncbi.nlm.nih.gov/geo/series
# NeurIPS 2021 bone marrow, CITE-seq part (GEO GSE194122), 616 MB
curl -L -o datasets/raw/neurips2021_bmmc/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz \
  $B/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
# Haber 2017 mouse intestine (GEO GSE92332), 16 MB + 18 MB
curl -L -o datasets/raw/haber2017_intestine/GSE92332_atlas_UMIcounts.txt.gz \
  $B/GSE92nnn/GSE92332/suppl/GSE92332_atlas_UMIcounts.txt.gz
curl -L -o datasets/raw/haber2017_intestine/GSE92332_SalmHelm_UMIcounts.txt.gz \
  $B/GSE92nnn/GSE92332/suppl/GSE92332_SalmHelm_UMIcounts.txt.gz
```

Moffitt 2018 MERFISH hypothalamus (Dryad doi:10.5061/dryad.8t8s248, 1.03 GB): the download is
**manual**, because Dryad blocks scripted downloads (it returns 401 or 403). Open
https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248 in a browser, download
`Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv`, and save it to
`datasets/raw/moffitt2018_merfish/`.

- **Writes:** `datasets/processed/<name>_hvg.h5ad`, plus `<name>_hvg.run.json` (sha256 + run record).
- **Gene list:** if `datasets/frozen/<name>/genes.txt` exists (committed), it is used as is and
  gene selection is skipped, so every machine gets the same genes. The first run for a new
  dataset writes it: commit it. To recompute on purpose, add `--refresh-frozen`; it prints the
  added/removed genes, and the git diff is the change to review. Every later stage is then stale.
- **Re-run when:** the preprocessing code changes. Everything after it must then be re-run too.
- **Time:** not measured.

## 2. Evaluation setup: split, affinity graphs, cluster count

```
python evaluation/prepare_eval.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas
```

- **Writes:** `affinity_train.npz`, `affinity_full.npz`, `affinity_config.json`, `cluster_config.json` into `--out-dir`.
- **Split:** read from the committed `datasets/frozen/<dataset>/split.csv` (`split.csv.gz` above
  100k cells). Only written if missing (commit it), or with `--refresh-split`, which also
  rebuilds the graphs. Baselines read the split from there too.
- **Skips** if those files already exist. Add `--force` to rebuild the graphs and cluster count (the split is kept). A rebuild makes every later stage stale.
- **Time:** not measured.

## 3. Baselines: one folder per method, `runs/<task>/<dataset>/<baseline>/`

**scVI** (local CPU/GPU):
```
python baselines/scvi_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/scvi
```
On an Apple Silicon Mac, add `--accelerator mps` to train on the Metal GPU. Without it the run
uses CPU: scvi-tools turns `auto` into CPU on a Mac on purpose. `config.json` records
`accelerator` and `device_used`. Timings and the environment are in
[`docs/local-mac.md`](local-mac.md).

To run on Modal GPU instead, see the docstring of `baselines/scvi_modal.py`. Modal runs get
**no** run record, because the remote machine has no git.

**PCA + Leiden** (unintegrated control, ~5 min):
```
python baselines/pca_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/pca
```

**SEACells** (reads stage 2's split, affinity graphs and cluster count; run `prepare_eval.py` first; ~25 min):
```
python baselines/seacells_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/seacells
```

**Re-cluster only** (no retraining, e.g. after changing the cluster-count search; ~9 min):
```
python evaluation/recluster.py --run-dir runs/rare_cell_preservation/heart_cell_atlas/scvi
```

- **Writes:** `embedding.csv` (or `assignments.csv` for SEACells) and `config.json`.
- **Re-run when:** the method's code or its flags change. After any re-run or re-cluster, stage 4 is stale.

## 4. Metrics: score each baseline (~45 s per baseline)

```
python evaluation/compute_metrics.py --baseline pca \
  --embedding runs/rare_cell_preservation/heart_cell_atlas/pca/embedding.csv \
  --affinity-dir runs/rare_cell_preservation/heart_cell_atlas \
  --label-key cell_type --batch-key donor
```
Change `--baseline` and `--embedding` for scvi, seacells, and so on.

- **Writes:** `<run>/metrics/` (official rare rule).
- **What-if copy:** add `--rare-rule std` (or `fixed`). This writes to `<run>/metrics__rare-<rule>/` and leaves the official numbers alone.
- **Other label set** (no retraining, same embedding and clusters), e.g. heart's 65 fine `cell_states`:
  add `--label-key cell_states --labels-from datasets/processed/heart_cell_atlas_hvg.h5ad`.
  This reads the labels from the processed file by cell ID and writes to `<run>/metrics__label-cell_states/`.
  Tables: `python evaluation/build_results.py --task rare_cell_preservation --metrics-subdir metrics__label-cell_states`
  writes `results/rare_cell_preservation/whatif_label-cell_states/`.
- **Re-run when:** stage 3 output or the metric code changes.

## 5. Results: tables + figures (~20 s)

```
python evaluation/build_results.py --task rare_cell_preservation
# what-if version:
python evaluation/build_results.py --task rare_cell_preservation --metrics-subdir metrics__rare-std
```

- **Writes:** `results/<task>/official/` (or `whatif_rare-<rule>/`): `tables.md`, `figures/`, `data/*.csv`, `run.json`.
- `tables.md` is generated, so never hand-edit it. The verdict is written by hand in `results/<task>.md`.
- **Sanity test** (synthetic data, no real runs needed): `python tests/check_build_results.py`

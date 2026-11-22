# Cross-machine reproducibility: evidence and how to investigate

The Windows machine and the Mac Studio give different rare-cell numbers for the same
pipeline on the same raw file. This folder holds everything from the Mac side needed to
find out why, from the Windows side, without the Mac user being available.

- The fix plan (files, tests, tasks, owners): [`docs/reproducibility-plan.md`](../docs/reproducibility-plan.md)
- This file: what was measured, what is in this folder, and the steps to locate the cause.

Written 2026-11-22 on the Mac. All Mac outputs here were produced at clean-v2 commit
`e11debf` plus the changes on branch `clean-v2-local-mac` (none of which change stage 1,
stage 2, PCA or the metrics).

## What is in this folder

```
repro/
  README.md                                  this file
  fingerprints/heart_cell_atlas/
    mac-studio.json                          hashes + summaries of every Mac stage output
  evidence/mac-studio/
    pip_freeze.txt                           every installed package, exact version
    heart_cell_atlas/                        laid out like runs/rare_cell_preservation/heart_cell_atlas/
      hvg_genes.txt                          the 2,000 genes the Mac kept, in matrix order
      split.csv, split.json                  the Mac's train/test split
      cluster_config.json, affinity_config.json
      affinity_full.npz                      the Mac's affinity graph (all cells)
      pca/        embedding.csv.gz, config.json, metrics/
      scvi_cpu/   embedding.csv.gz, config.json, metrics/     (--accelerator cpu)
      scvi_mps/   embedding.csv.gz, config.json, metrics/     (--accelerator mps, Apple GPU)
      seacells/   assignments.csv.gz, config.json, metrics/   (UNSEEDED run: see below)
```

The processed `.h5ad` (58 MB) is not included. Its content is covered by the hashes in
the fingerprint and by `hvg_genes.txt`. Total size of the pack: about 14 MB.

Tools added in `evaluation/` (read each file's docstring for details):

| Script | What it does |
|---|---|
| `fingerprint.py` | hashes and summaries of one dataset's stage outputs -> one JSON |
| `compare_fingerprints.py` | two fingerprint JSONs, stage by stage; names the first input difference |
| `compare_runs.py` | two run folders (or evidence packs): gene overlap, split overlap, graph overlap, embedding neighbour overlap, cluster agreement, metric differences |
| `make_evidence_pack.py` | builds a pack like this one for the machine it runs on |

They were tested on the Mac only (Python 3.11, pandas 2.3, anndata 0.12). They have not
been run on the Windows versions (Python 3.13, pandas 3.0, anndata 0.13). If one fails
there, that is a tool bug, not a finding.

## What was measured on the Mac

| # | Finding | Numbers |
|---|---|---|
| 1 | Raw file is the same on both machines | sha256 `4a54acfa...` |
| 2 | Processed file checksum differs | Mac `36949136...`, Windows `7e686ed3...` (from `datasets/heart_cell_atlas.md`) |
| 3 | All-cell metrics agree with the committed official table | within 0.005 (PCA and scVI) |
| 4 | Rare-cell metrics do not | PCA train rare F1: 0.696 official, 0.846 Mac |
| 5 | On the Mac, every stage is byte-repeatable with a fixed seed | second run identical for: processed h5ad, split, both affinity graphs, PCA embedding, scVI on CPU, scVI on Apple GPU; cluster agreement 1.000 |
| 6 | SEACells was the exception, and is now fixed | unseeded: agreement 0.58 between two identical runs. With `np.random.seed(seed)` before each fit (now in `baselines/seacells_baseline.py`): agreement 1.000 |
| 7 | Two scVI runs that differ only in device are very different embeddings | CPU vs Apple GPU, same seed: each cell shares on average 40% of its 15 nearest neighbours; cluster agreement 0.35; all-cell metrics within 0.002; test rare F1 0.846 vs 0.924 |
| 8 | Stage 2 on the Mac gives | 14,913 train / 3,728 test, `k_train` 198, `k_full` 248, affinity_full 415,131 stored entries |

"Agreement" is the adjusted Rand index between two clusterings: 1 = same grouping,
0 = no better than chance.

Reading of 5 and 4 together: PCA has no training randomness and is repeatable on one
machine, yet its rare F1 differs by 0.15 between machines. So the two machines are not
running the same computation on the same input. The cause is one of:

- **H1** a different set of 2,000 genes from stage 1 (HVG selection under scanpy 1.11.5 vs 1.12.4);
- **H2** a different train/test split (scikit-learn / numpy / pandas versions);
- **H3** same genes and split, but slightly different PCA or neighbour numerics
  (different linear-algebra library on Windows and macOS), amplified by Leiden at K = 248;
- **H4** only the file format differs and the content is equal, in which case the
  difference in finding 4 comes from H3 alone.

Finding 7 matters separately: even with identical inputs, a single scVI run cannot be
compared across machines on rare metrics. That needs several seeds (plan, Phase D).

The SEACells files in the pack come from a run made **before** the seeding fix. Do not
use them to judge cross-machine agreement.

## Steps on the Windows machine (about 15 minutes, no retraining)

Run from the repo root, in the environment that produced the official results. The
commands are single lines so they work the same in PowerShell, cmd and bash. If your
scVI folder is named `scvi`, keep `--map scvi=scvi_cpu` as written.

**Step 1. Fingerprint the existing Windows outputs.**
```
python evaluation/fingerprint.py --run-dir runs/rare_cell_preservation/heart_cell_atlas --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --layer counts --machine win-main
```

**Step 2. Compare with the Mac fingerprint.** The last lines name the first input difference.
```
python evaluation/compare_fingerprints.py repro/fingerprints/heart_cell_atlas/win-main.json repro/fingerprints/heart_cell_atlas/mac-studio.json
```

**Step 3. Compare the actual files.** This gives sizes: how many genes differ, how many
cells changed side, how many graph edges and neighbours are shared.
```
python evaluation/compare_runs.py --a runs/rare_cell_preservation/heart_cell_atlas --b repro/evidence/mac-studio/heart_cell_atlas --map scvi=scvi_cpu
```

**Step 4. Exchange test: score the Mac's PCA embedding with the Windows metric code.**
Work on a copy, because metrics are written next to the embedding and would overwrite
the Mac's numbers in the pack.
```
python -c "import shutil; shutil.copytree('repro/evidence/mac-studio/heart_cell_atlas', 'runs/exchange_mac/heart_cell_atlas')"
python evaluation/compute_metrics.py --baseline pca --embedding runs/exchange_mac/heart_cell_atlas/pca/embedding.csv.gz --affinity-dir runs/exchange_mac/heart_cell_atlas --label-key cell_type --batch-key donor
python evaluation/compare_runs.py --a repro/evidence/mac-studio/heart_cell_atlas --b runs/exchange_mac/heart_cell_atlas
```
In the last command's `baseline: pca vs pca` block, every metric `|diff|` should be about
1e-9 or smaller (on the Mac this procedure gives exactly 0). If not, the metric code
itself behaves differently on the two machines, and that must be fixed before anything
else. This test does not depend on the gene list or the split of the Windows machine: it
uses the Mac's embedding, split and graph throughout.

**Step 5. Build the Windows pack and commit it**, so the Mac side can run the same comparisons.
```
python evaluation/make_evidence_pack.py --run-dir runs/rare_cell_preservation/heart_cell_atlas --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --machine win-main
```
Commit `repro/fingerprints/heart_cell_atlas/win-main.json` and `repro/evidence/win-main/`.

## How to read the result

| What Step 2 / Step 3 show | Cause | Fix |
|---|---|---|
| `processed.genes_hash_sorted` differs; gene overlap below 1 | H1: HVG selection is version-dependent | Commit one gene list (`repro/inputs/<dataset>/hvg_genes.txt`) and make `prepare*.py` subset to it instead of selecting again. Plan 4.5, task C1/C2 |
| genes same, `processed.matrix_hash` differs | values changed in stage 1 (normalisation or dtype) | diff the two stage-1 code paths; `X` and `layers/counts` are hashed separately to show which |
| `processed.cells_hash` differs | cell order or ids differ | stage 1 must not reorder; everything downstream assumes the processed file's order |
| processed same, `split.hash` differs | H2: the split is version-dependent | Commit `split.csv`; `prepare_eval.py` copies it instead of creating one. Plan 4.5 |
| all L0 lines SAME; `pca` or `affinity_full` lines DIFF, graph overlap near but below 1 | H3: platform numerics | Record the measured overlap as the tolerance (plan 4.7). Try a fixed solver (`svd_solver="full"`) on both sides. If still unequal, build the graph once and share the file |
| all L0 SAME, PCA and graph SAME, PCA clusters differ | Leiden / igraph version | pin leidenalg and igraph; pass the seed explicitly |
| everything through the PCA baseline SAME, only scVI / SEACells differ | expected | compare through seeds, not single runs (plan Phase D) |
| Step 4 metrics differ on the same embedding | metric code is platform- or version-dependent | find the metric; pandas 2 vs 3 group ordering and string handling are the first suspects |

## If you have more time: experiments that would settle open points

Each is independent. Write the result into section 2 of `docs/reproducibility-plan.md`.

1. **Is HVG selection the version-sensitive step?** In a scratch environment, install
   scanpy 1.11.5 (the Mac's version) next to your 1.12.4 one, run stage 1 in both, and
   compare the gene lists with `compare_runs.py --genes-a ... --genes-b ...`. Same test
   for `scikit-misc` if the versions differ.
2. **Is the split version-sensitive?** Run `prepare_eval.py --force` into a scratch
   `--out-dir` on Windows and compare `split.csv` with the Mac's in the pack. If the gene
   lists already differ this still works: the split depends only on cell ids, labels and batch.
3. **How large is pure platform numerics?** Copy the Mac's `hvg_genes.txt` order onto the
   Windows processed file (subset and reorder genes), rebuild stage 2 in a scratch folder,
   and run `compare_runs.py` against the pack. With inputs forced equal, whatever remains
   in the PCA and graph lines is the platform effect. This gives the real tolerance
   numbers for the plan's gate table (4.7).
4. **Does a fixed PCA solver remove it?** Repeat 3 with `svd_solver="full"` in
   `common.compute_pca` and `pca_baseline.py`.
5. **Seed spread on Windows.** Run scVI (`--accelerator cpu`) with 5 seeds and report the
   std of each rare metric. The Mac will do the same. This is the number every later
   comparison is judged against. Needs the `--seed` / `seed<k>/` change from plan task D1
   first, or five separate `--out-dir` folders as a stopgap.
6. **Pin the environment.** `pip freeze` from Windows becomes
   `environment/requirements.txt` (plan Phase B). The Mac's full list is in
   `evidence/mac-studio/pip_freeze.txt` for spotting packages that differ.

## Environment on the Mac (the versions behind this pack)

```
macOS 26.5.1 arm64, python 3.11.18
scvi-tools 1.4.2    torch 2.15.0     lightning 2.6.6
scanpy 1.11.5       anndata 0.12.19  numpy 2.2.6      scipy 1.13.1   pandas 2.3.3
scikit-learn 1.9.1  scikit-misc 0.5.3  leidenalg 0.12.0  igraph 1.0.0
SEACells 0.3.3 (--no-deps)  palantir 1.4.5  numba 0.68.1
```
The same list, machine-readable, is the `environment` block of the fingerprint.

## Commands that produced the Mac outputs

Stage 1, stage 2 and the baselines exactly as in `docs/runbook.md` (heart examples), with
two scVI runs: `--accelerator cpu --out-dir .../scvi_cpu` and
`--accelerator mps --out-dir .../scvi_mps`. Metrics with `evaluation/compute_metrics.py`
per baseline. Each `config.json` in the pack carries its own `run_history` with the
exact command and commit.

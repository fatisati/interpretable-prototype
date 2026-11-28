# Reproducibility plan: comparable results on the Windows and Mac machines

Status: PROPOSED, 2026-11-22. Done so far (Mac side): task A1 (the fingerprint and
compare tools), task A3 (Mac fingerprint and evidence pack, in `repro/`), and the SEACells
seeding part of task D1. Everything else is open. To investigate from the Windows side,
start with [`repro/README.md`](../repro/README.md).
Written so that Claude Code on either machine can execute it: every task has an ID,
an owner, the files it touches, and a "done when" check.

Owners: **WIN** = the Windows machine (produced the committed `official` results).
**MAC** = the Mac Studio. **ANY** = do once, on either machine, and push.

---

## 1. Goal, and what "comparable" means here

Two people must be able to run the same stage on different machines and put the numbers
in one table. That needs three different guarantees, because the stages are different in
kind. Do not aim for byte-identical output everywhere: it is not achievable across
operating systems for the trained models, and it is not needed.

| Level | Applies to | Guarantee | How it is checked |
|---|---|---|---|
| **L0 identical inputs** | gene list, counts matrix, cell order, split, cluster count K | exactly equal on both machines | content hash equal |
| **L1 deterministic numerics** | PCA, affinity graph, PCA baseline embedding, every metric given a fixed embedding | equal within a small, measured tolerance | array comparison + metric difference |
| **L2 stochastic methods** | scVI, SEACells, Leiden clustering, later scProto | equal **in distribution**: same mean within the seed-to-seed spread | N seeds per machine, compare mean and std |

A cross-machine comparison is valid only when L0 passes. A claimed improvement is real
only when it exceeds the L2 seed spread.

---

## 2. What is known (measured on the Mac, 2026-11-22)

Facts, each from a run on this branch. Heart Cell Atlas, 18,641 cells.

| Fact | Evidence |
|---|---|
| Raw file is identical on both machines | sha256 `4a54acfa...` matches `datasets/heart_cell_atlas.md` |
| Processed file differs between machines | file sha256 `36949136...` on Mac, `7e686ed3...` on Windows |
| All-cell metrics agree across machines | PCA and scVI within 0.005 of the committed official table |
| Rare-cell metrics do not | PCA train rare F1: 0.696 official, 0.846 on Mac. PCA has no training randomness, so this is an input or numerics difference, not seed noise |
| **On one machine, every stage except SEACells is exactly repeatable** | Second run with the same seed: processed h5ad, `split.csv`, both affinity files, PCA embedding, scVI embedding on CPU, scVI embedding on Metal are all byte-identical; cluster labels identical (ARI 1.000) |
| **SEACells was not repeatable even on one machine; fixed** | Two unseeded runs, same command: cluster agreement ARI 0.58. Cause: `SEACells/cpu.py` draws from the global `np.random` and the baseline script never seeded it. With `np.random.seed(seed)` before each fit (now in `baselines/seacells_baseline.py`): ARI 1.000 between two runs |
| scVI on CPU and scVI on Metal are different runs | Same seed, different device: each cell shares on average 40% of its 15 nearest neighbours, cluster agreement ARI 0.35, all-cell metrics within 0.002, test rare F1 0.846 vs 0.924 |
| Package versions differ | WIN: python 3.13.11, scanpy 1.12.4, anndata 0.13.2, numpy 2.4.6, pandas 3.0.5, scvi-tools 1.5.0.post1, torch 2.13.0+cpu. MAC: python 3.11.18, scanpy 1.11.5, anndata 0.12.19, numpy 2.2.6, pandas 2.3.3, scvi-tools 1.4.2, torch 2.15.0 |
| The committed official results came from uncommitted code | `results/rare_cell_preservation/official/run.json` has `"git_dirty": true` |
| No environment is pinned anywhere | no requirements or lock file; `baselines/scvi_modal.py` installs unpinned packages |

Consequence of the bold rows: the pipeline is deterministic given (code, inputs,
environment, device). So the cross-machine gap comes from inputs, environment or
platform numerics, and it can be located stage by stage.

**Not known yet** (needs files from WIN; task A1 answers it): at which stage the two
machines first diverge. Candidates, most likely first:

1. **HVG selection** picks a different 2,000 genes under a different scanpy version.
   Everything downstream changes.
2. **The split** differs (different scikit-learn / numpy / pandas versions or group order).
3. Same genes and split, but **PCA / kNN numerics** differ slightly (different linear
   algebra library on Windows and macOS), which flips a few graph edges and then many
   Leiden cluster boundaries at K = 248.
4. Only the **file encoding** differs (anndata 0.12 vs 0.13) and the content is the same.
   Then the file checksum was never a valid identity check.

Also true regardless of the answer: pandas writes CSV with `\r\n` line endings on
Windows, so file checksums of CSVs can never match across the two machines even when the
content does.

---

## 3. Design decisions

1. **Identity is checked on content, not on file bytes.** One function hashes what a
   file means (cell order, gene order, count values), so file format versions and line
   endings stop mattering.
2. **Shared inputs are exchanged, not rebuilt.** The gene list, split and cluster count
   are small: commit them. Each machine verifies its own build against them, and uses the
   committed copy. The step most sensitive to package versions (HVG selection) then runs
   once, ever, per dataset.
3. **One pinned environment for both machines.** Pin to the Windows versions: they are
   newer for the data-shaping libraries and they produced the official results.
4. **Official numbers use CPU for training** on both machines. Windows has CPU torch
   only, so this is the one device setting both can share. Metal stays available for
   development runs and is recorded in `config.json` (already done on this branch).
5. **Every stochastic method runs with several seeds.** Results report mean and std
   across seeds. This replaces single-run comparisons, which cannot rank methods on rare
   cells (the measured swing is as large as the differences being interpreted).
6. **A gate decides whether a machine may contribute to `official`.** A script compares
   the two machines' fingerprints and prints pass or fail per stage.
7. **Every result carries its machine tag and input hash**, so a mismatch is visible in
   the table instead of being discovered later.

Not chosen, and why: Docker (no Metal GPU inside a container on the Mac, and awkward on
Windows); running everything on Modal (works, but costs money per run and the Mac has no
Modal CLI installed) - it remains a fallback if Phase C fails, see section 8.

---

## 4. Files to add or change

> **Update 2026-11-24.** The committed inputs described below were implemented under a different
> path: `datasets/frozen/<dataset>/` (`genes.txt`, `split.csv`, see `evaluation/frozen.py`), not
> `repro/inputs/<dataset>/`. Not implemented: `manifest.json`, the content-hash check and
> `tests/check_inputs.py`. Added instead: committed baseline outputs
> (`datasets/frozen/<dataset>/baselines/<baseline>/`, `evaluation/frozen_run.py`), because the
> same genes and split still gave different Leiden clusters on the two machines.

New files are marked NEW. Paths are final names; keep them so both sides match.

### 4.1 Environment

| File | What it holds |
|---|---|
| NEW `environment/requirements.txt` | Exact `==` pins for: scanpy, anndata, numpy, scipy, pandas, scikit-learn, scikit-misc, leidenalg, igraph, scvi-tools, lightning, torch, palantir, numba, threadpoolctl, matplotlib. Versions = what WIN has today (`pip freeze`), not ranges |
| NEW `environment/README.md` | Python version (3.13.x), the install commands per OS, the SEACells install line (`pip install --no-deps git+https://github.com/dpeerlab/SEACells.git@<commit sha>`), and the torch line per OS (CPU wheel on Windows, default wheel on macOS) |
| NEW `evaluation/check_env.py` | Compares installed versions to `requirements.txt`. Prints every mismatch. Exit code 1 on any mismatch. Called at the start of every stage script through one helper, as a warning by default and an error with `--strict-env` |
| CHANGE `evaluation/common.py::environment_versions` | Add: scipy, scikit-learn, scikit-misc, lightning, SEACells, platform string, linear-algebra library name and thread count (from `threadpoolctl.threadpool_info()`), and a `machine` tag read from the env var `SCPROTO_MACHINE` (e.g. `win-main`, `mac-studio`) |
| CHANGE `baselines/scvi_modal.py` | Install from `environment/requirements.txt` instead of unpinned names |

### 4.2 Canonical inputs (committed, small)

```
repro/inputs/<dataset>/
  hvg_genes.txt          one gene name per line, in matrix order
  split.csv              cell_id,split   (written with "\n" line endings)
  split.json             seed, test_size, keys, n_train, n_test
  cluster_config.json    k_train, k_full, cells_per_cluster
  manifest.json          content hashes (see 4.3) of: raw file, processed matrix,
                         obs_names, var_names, split; summary of each affinity graph
                         (n, nnz, weight sum, row-degree min/median/max)
```

`runs/` is git-ignored, so these cannot live there. Heart sizes: split 0.7 MB, the rest
a few KB. The affinity files (3-4 MB each) and the processed h5ad (58 MB) are not
committed; see 4.4.

### 4.3 Fingerprints and the gate

| File | What it does |
|---|---|
| NEW `evaluation/fingerprint.py` | `content_hash(adata)`: sha256 over `obs_names`, `var_names`, and the counts matrix in canonical form (CSR, sorted indices, int32 values). Also `fingerprint_dataset(dataset)`: runs through every stage output that exists and writes one JSON (below). No training; reads files only |
| NEW `repro/fingerprints/<dataset>/<machine>.json` | Output of the above, committed by each machine. Fields per stage: `raw.sha256`; `processed.content_hash`, `processed.n_cells`, `processed.n_genes`, `processed.genes_hash`; `split.hash` (hash of sorted `cell_id:split` pairs), `split.n_test`; `pca.shape`, `pca.explained_variance[:10]`, `pca.abs_column_sums[:10]`; `affinity_full.{n,nnz,weight_sum,edge_hash}` (edge_hash = hash of the sorted (row,col) pairs); `cluster_config.{k_train,k_full}`; per baseline: `embedding.shape`, `embedding.column_means`, `n_clusters`, and the full `metrics_summary`; plus `environment_versions()` |
| NEW `evaluation/compare_fingerprints.py` | `--dataset X --a <machine> --b <machine>`. Walks the stages in pipeline order and prints one line per stage: `PASS`, `WITHIN TOL`, or `FAIL` with the two values. Stops interpreting at the first L0 failure ("everything below is expected to differ"). Exit code 1 on any L0 failure |
| NEW `evaluation/compare_runs.py` | For L1 arrays, which a hash cannot compare: takes two files of the same kind (two `embedding.csv`, two `affinity_*.npz`) and prints: max absolute difference, k=15 neighbor overlap per cell (mean, 1st percentile), edge-set Jaccard, and cluster agreement (ARI) |

### 4.4 Large shared artifacts

Needed for L1 checks and for the exchange test (4.6): each machine's processed h5ad,
`affinity_*.npz`, and `embedding.csv` / `assignments.csv` per baseline. They are
re-creatable but must be compared directly at least once.

**Decision needed (section 9, Q1):** where they live. Options: a shared Google Drive
folder, the existing Modal volume `scproto-data`, or GitHub release assets. The plan only
needs a folder both sides can read and write, laid out as
`<store>/<machine>/<task>/<dataset>/...` mirroring `runs/`.

| File | What it does |
|---|---|
| NEW `evaluation/share.py` | `push --dataset X` / `pull --dataset X --machine M`: copies the files listed above between `runs/` and the store path given by env var `SCPROTO_SHARE`. Plain file copy; no cloud API in the script |

### 4.5 Changes to existing stages

| File | Change | Why |
|---|---|---|
| `datasets/scripts/prepare*.py` | If `repro/inputs/<dataset>/hvg_genes.txt` exists: subset to exactly those genes in that order and skip HVG selection. If it does not exist: select, then write it. After writing the h5ad, compute `content_hash` and compare with `manifest.json`; error on mismatch. Store the content hash in the `.run.json` next to the file sha | Removes the most version-sensitive step from every later run; replaces a check that cannot work across machines |
| `evaluation/prepare_eval.py` | If `repro/inputs/<dataset>/split.csv` exists: copy it and `cluster_config.json` into the run folder instead of creating them, after checking the cell ids match. `--force` no longer regenerates a committed split; a new flag `--new-canonical` does. Always build the affinity graphs locally, then compare their summary with the manifest and print PASS / WITHIN TOL / FAIL | Split and K become shared data, as `docs/architecture.md` already intends |
| `baselines/seacells_baseline.py` | Call `np.random.seed(seed)` (and `random.seed(seed)`) immediately before each `run_seacells(...)`; write `seed` and "numpy global seed set" into `config.json` | Measured: unseeded, ARI 0.58 between two identical runs |
| `baselines/*.py`, `evaluation/recluster.py` | Accept `--seed` and write outputs to `runs/<task>/<dataset>/<baseline>/seed<k>/`. The split seed stays a separate, fixed value (`--split-seed`, default 0) so changing the model seed never changes the split | Enables L2. Today one `--seed` flag means both things |
| `evaluation/common.py::sweep_leiden_to_k` | Pass `random_state=seed` to `sc.tl.leiden` explicitly and record it | It is currently the scanpy default; make it visible |
| `evaluation/compute_metrics.py`, `build_results.py` | Read every `seed*/metrics/` folder of a baseline. Report mean ± std **across seeds** as the headline, keep the across-cells std as a second column. Add columns `machine` and `input_hash` to `summary.csv`; refuse to put rows with different `input_hash` in one table unless `--allow-mixed-inputs` | Makes mismatches visible and stops single-run comparisons |
| All CSV / JSON writers | `to_csv(..., lineterminator="\n")`, `open(..., newline="\n")`; store paths in configs as repo-relative POSIX strings (`Path.as_posix()`), never absolute | Windows writes `\r\n` and backslashes; both break diffs |
| NEW `.gitattributes` | `* text=auto eol=lf` and `*.csv text eol=lf`, `*.json text eol=lf`, `*.md text eol=lf`; `*.npz binary`, `*.h5ad binary`, `*.png binary` | Same reason, for files in git |
| `evaluation/run_record.py` | Add `machine` (from `SCPROTO_MACHINE`) and `platform` to every `run_history` entry. In `build_results.py`, refuse to write into `results/<task>/official/` when `git_dirty` is true unless `--allow-dirty` | Official numbers must map to a commit |

### 4.6 Tests (all runnable as `python tests/<name>.py`, like the existing one)

| File | Checks | Level |
|---|---|---|
| NEW `tests/check_env.py` | installed versions == `environment/requirements.txt` | - |
| NEW `tests/check_inputs.py` | for each dataset present locally: processed content hash, gene list, split hash and K equal `repro/inputs/<dataset>/manifest.json` | L0 |
| NEW `tests/check_metrics_golden.py` | a committed fixture (`tests/fixtures/golden_embedding.csv`, about 600 cells, plus a small affinity npz) scored by `compute_metrics` gives exactly the committed `tests/fixtures/golden_metrics.json` (tolerance 1e-9). Extends the synthetic data already built in `check_build_results.py` | L1 |
| NEW `tests/check_determinism.py` | runs the PCA baseline twice on the fixture and requires identical output; runs SEACells twice on the fixture and requires identical assignments (this fails today and passes after the seeding fix) | within-machine |
| NEW `tests/check_cross_machine.py` | runs `compare_fingerprints` for every pair of committed fingerprint files; fails on any L0 mismatch | L0 |
| existing `tests/check_build_results.py` | keep | - |
| NEW `tests/run_all.py` | runs the above in order and prints one PASS/FAIL line each | - |

### 4.7 Reproducibility metrics (what the gate reports)

| Stage | Metric | Pass |
|---|---|---|
| Processed data | content hash, gene-set overlap (Jaccard) | equal, 1.000 |
| Split | hash; share of cells with the same side | equal, 1.000 |
| Cluster count | `k_train`, `k_full` | equal |
| PCA | k=15 neighbor overlap per cell between machines | mean ≥ 0.99 (first guess; replace with the measured value from task C3) |
| Affinity graph | edge-set Jaccard; max weight difference on shared edges | ≥ 0.99; ≤ 1e-6 (first guess, as above) |
| PCA baseline | cluster agreement (ARI) between machines; all-cell metric difference | reported; ≤ 0.005 |
| Metric code | other machine's `embedding.csv` scored locally vs their `metrics_summary.json` (the **exchange test**) | difference ≤ 1e-9 |
| scVI, SEACells | per metric: difference of the two machines' seed means, against the pooled seed std | difference ≤ 2 x standard error |

The thresholds marked "first guess" are placeholders, not measurements. Task C3 measures
the real cross-machine values once inputs are identical, and that task ends by writing
the measured tolerances into this table.

---

## 5. Tasks, in order

### Phase A - locate the divergence (half a day, no code changes to stages)

| ID | Owner | Task | Done when |
|---|---|---|---|
| A1 | ANY | Write `evaluation/fingerprint.py`, `compare_fingerprints.py`, `compare_runs.py` (4.3). Push | scripts run on the machine that wrote them |
| A2 | WIN | Pull. Run `python evaluation/fingerprint.py --dataset heart_cell_atlas --machine win` on the **existing** outputs (no re-run). Commit `repro/fingerprints/heart_cell_atlas/win.json`. Also save `pip freeze > environment/win-freeze.txt` and commit it | both files pushed |
| A3 | MAC | Same, `--machine mac`, plus `environment/mac-freeze.txt` | both files pushed |
| A4 | ANY | Run `compare_fingerprints.py --dataset heart_cell_atlas --a win --b mac`. Write the first failing stage into section 2 of this file ("Not known yet" becomes a fact) | the cause is named, with the two values |

Expected: the first FAIL is at processed data (gene list) or at PCA numerics.
If it is the gene list or the split, Phases B and C fix it directly. If L0 passes and
only numerics differ, Phase C3 measures how much and Phase D carries the comparison.

### Phase B - one environment (half a day)

| ID | Owner | Task | Done when |
|---|---|---|---|
| B1 | WIN | Turn `win-freeze.txt` into `environment/requirements.txt` (only the packages in 4.1) and write `environment/README.md` with the Windows install commands that are known to work | pushed |
| B2 | MAC | Create a new conda env `scproto-clean` with Python 3.13 and install from `requirements.txt`. Add the macOS commands to `environment/README.md`. If a pinned version has no macOS arm64 build, do not substitute silently: list it under "Version conflicts" in that README and tell WIN, so both sides move to a version that exists on both | `python tests/check_env.py` passes on MAC, or the conflict list is pushed |
| B3 | WIN | `python tests/check_env.py` in the existing environment | passes |
| B4 | ANY | Write `check_env.py`, extend `environment_versions`, pin `scvi_modal.py` (4.1) | merged |

### Phase C - shared inputs (1 day)

| ID | Owner | Task | Done when |
|---|---|---|---|
| C1 | ANY | Implement 4.5 rows for `prepare*.py` and `prepare_eval.py`, the line-ending and path changes, `.gitattributes`, and `tests/check_inputs.py` | `tests/check_build_results.py` still passes |
| C2 | WIN | Generate canonical inputs for heart from the existing Windows files: `hvg_genes.txt` from the processed h5ad, plus `split.csv`, `split.json`, `cluster_config.json`, `manifest.json`. Commit under `repro/inputs/heart_cell_atlas/`. Repeat for HLCA and any other prepared dataset | pushed; `tests/check_inputs.py` passes on WIN |
| C3 | MAC | Pull. Re-run stage 1 and stage 2 for heart in the pinned env. `tests/check_inputs.py` must pass. Then run the PCA baseline, push artifacts with `share.py`, and run `compare_runs.py` against the Windows PCA embedding and affinity graph. Write the measured values into the table in 4.7 | L0 passes on MAC; 4.7 thresholds are measured numbers |
| C4 | ANY | Exchange test: score the other machine's PCA `embedding.csv` locally and compare with their `metrics_summary.json` | difference ≤ 1e-9, or the differing metric is named and fixed |

### Phase D - seeds and the noise floor (1 to 2 days, mostly run time)

| ID | Owner | Task | Done when |
|---|---|---|---|
| D1 | ANY | Seed SEACells; add `--seed` / `--split-seed` and the `seed<k>/` layout; make `compute_metrics` and `build_results` aggregate across seeds (4.5). Add `tests/check_determinism.py` and `check_metrics_golden.py` | all tests pass on the machine that wrote them |
| D2 | WIN + MAC | Each runs heart with seeds 0-4: scVI with `--accelerator cpu`, SEACells, and PCA (PCA once; it has no model seed, only the Leiden seed). Push fingerprints | 5 seed folders per stochastic baseline per machine |
| D3 | ANY | Run the gate. Write `repro/noise_floor/heart_cell_atlas.csv`: per metric, per machine, mean and std across seeds, and the cross-machine difference of means | file committed; gate prints PASS or names the failing metric |
| D4 | ANY | Rebuild `results/rare_cell_preservation/official/` from seeded runs made on a clean commit (`git_dirty` false), with the `machine` and `input_hash` columns | new official tables committed |

Approximate run time on the Mac for D2 (measured single runs): scVI on CPU about 3 min
per seed, SEACells about 8.5 min per seed, PCA under 1 min. About 1 hour total.

### Phase E - keep it fixed (ongoing)

| ID | Owner | Task |
|---|---|---|
| E1 | ANY | Add to `CLAUDE.md`: the rules in section 6 below |
| E2 | ANY | Add the gate and `tests/run_all.py` to `docs/runbook.md` as stage 0 |
| E3 | both | New dataset: the person who prepares it commits `repro/inputs/<dataset>/`; the other runs `tests/check_inputs.py` before any baseline |

---

## 6. Rules after the fix (to copy into `CLAUDE.md`)

1. Set `SCPROTO_MACHINE` once per machine. Run in the pinned environment;
   `tests/check_env.py` must pass before a run that will be reported.
2. Never rebuild a committed gene list or split. Changing one is a deliberate act
   (`--new-canonical`), announced to the other person, and makes all earlier runs stale.
3. Official numbers: CPU training, at least 5 seeds, clean commit, gate passing.
   Anything else goes in a `whatif_*` folder.
4. Compare two results only if their `input_hash` is equal.
5. A difference counts only if it is larger than twice the seed std in
   `repro/noise_floor/<dataset>.csv`. State the std next to every mean.
6. Before reporting a cross-machine comparison, run `compare_fingerprints.py`.

---

## 7. What each person does, in short

**Windows collaborator**
1. A2: fingerprint the existing outputs and commit `pip freeze`.
2. B1, B3: publish the pinned requirements; check the environment.
3. C2: commit the canonical inputs (gene list, split, cluster count, manifest).
4. D2: run 5 seeds of scVI (CPU) and SEACells; push fingerprints and artifacts.
5. Stop producing `official` results from a dirty working tree.

**Mac user**
1. A3: fingerprint and commit `pip freeze`.
2. B2: build the pinned Python 3.13 environment; report version conflicts.
3. C3, C4: rebuild from canonical inputs, measure the L1 tolerances, run the exchange test.
4. D2: run 5 seeds with `--accelerator cpu`; push fingerprints and artifacts.
5. Use `--accelerator mps` only for development runs.

**Either (code, one pull request each, in this order)**
A1 fingerprint tools -> B4 environment check -> C1 canonical inputs -> D1 seeds and
aggregation -> D3/D4 gate and new official tables -> E1/E2 docs.

---

## 8. If it still does not match

| Symptom after Phase C | Meaning | Action |
|---|---|---|
| L0 fails on the Mac with canonical inputs present | a stage is not reading the canonical files | fix that stage; it is a bug |
| L0 passes, PCA neighbor overlap is well below 1 | platform numerics (different linear-algebra library) | set PCA to a fixed solver (`svd_solver="full"`) in `compute_pca` and `pca_baseline.py` on both sides, re-measure. If still low, build the affinity graph once and share the file instead of rebuilding it |
| L1 passes, clusters differ a lot (low ARI) at K = 248 | Leiden at high resolution amplifies tiny graph differences | expected; compare through seeds (L2), not single clusterings |
| L2 fails: seed means differ by more than 2 standard errors | a real environment effect on training | run both machines' seeds through the pinned Modal image as the common reference and compare each machine to it |
| Exchange test fails | the metric code depends on platform or package version | fix the metric; this must be exact |

---

## 9. Decisions needed from the two of you

| ID | Decision | Recommendation |
|---|---|---|
| Q1 | Where the large shared artifacts live (4.4) | whichever folder both already use; the code only needs a path |
| Q2 | Which versions are the reference | Windows versions, Python 3.13 |
| Q3 | Number of seeds for official results | 5 |
| Q4 | Whether the committed heart results are regenerated | yes, in task D4: they came from one seed and uncommitted code |
| Q5 | Machine tag names | `win-main`, `mac-studio` |

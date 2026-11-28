# scVI in the reference container (2026-11-28)

**Question:** does training scVI inside the container give the same embedding every time,
and on both CPU types?

**Answer on the Mac:**
- Same CPU type: **yes, byte for byte.** Two runs in the Linux ARM container wrote the
  same `embedding.csv` (same sha256), clusters included.
- ARM vs x86-64: **no.** The embeddings differ, and so do the clusters. The difference is
  much smaller than changing the training seed, and the neighbour metrics barely move.

The x86-64 container ran under emulation on the Mac. One seed per comparison, so the
sizes below are single measurements, not averages.

## What was run

`baselines/scvi_baseline.py` on heart, CPU, seed 0, 100 epochs, in the image from
`repro/docker/Dockerfile` (threads set to 1), then `evaluation/compute_metrics.py`.

| Run | Platform | Training seed | Time |
|---|---|---|---|
| `arm64_a`, `arm64_b` | Linux ARM | 0 | 5.5 min each (run side by side) |
| `amd64` | Linux x86-64, emulated | 0 | 10.7 min |
| `arm64_seed1` | Linux ARM | 1 | about 5 min |

Input: the Mac's `datasets/processed/heart_cell_atlas_hvg.h5ad` (the Mac gene list, as in
`../../heart_cell_atlas/`) and the split and cluster counts of the Mac run folder. All
four runs read the same files, so they compare with each other. They do not compare with
runs on the frozen gene list.

`arm64_seed1` uses the same split as the others. `--seed` on the baseline also has to
match the split's seed, so `other_seed_runner.py` changes the training seed only.

## Result: the embeddings

| Comparison | Median coordinate difference | Cells with the same 15 neighbours | Clusters, ARI (train / full) |
|---|---|---|---|
| ARM vs ARM, same seed | 0 | 100% | 1.00 / 1.00 |
| ARM vs x86-64, same seed | 0.025 | 6.5% | 0.63 / 0.59 |
| ARM seed 0 vs ARM seed 1 | 0.80 | 0.01% | 0.34 / 0.31 |

The coordinates have a standard deviation of about 1. ARI: 1 = same grouping, 0 = chance.
Each run finds its own Leiden resolution for the target cluster count, so the cluster
ARI includes that.

## Result: the metrics (rare cells, train job)

| Metric | ARM | x86-64 | ARM, seed 1 | Gap from CPU type | Gap from seed |
|---|---|---|---|---|---|
| `knn_same_type_global` | 0.8202 | 0.8194 | 0.8291 | 0.0008 | 0.0089 |
| `knn_same_type_cross` | 0.8019 | 0.8000 | 0.8167 | 0.0019 | 0.0147 |
| `agreement_global` | 0.835 | 0.880 | 0.893 | 0.045 | 0.058 |
| `rare_f1` | 0.900 | 0.918 | 0.901 | 0.017 | 0.000 |
| `cohesion` | 0.063 | 0.073 | 0.062 | 0.010 | 0.001 |

- **Neighbour metrics:** the CPU type moves them by 0.001–0.003 (train and test), about
  a tenth of what one seed change does (0.009–0.024).
- **Cluster metrics:** the CPU type moves them about as much as a seed change (rare
  agreement 0.045 vs 0.058 on train, 0.050 vs 0.120 on test).

Full numbers: `metrics_summary.json` in each run's folder.

## What this means

- The container makes scVI exactly repeatable on one CPU type. Two Intel machines running
  the image should then agree; that is not tested here.
- Across ARM and Intel, a trained embedding has to be frozen (committed) to be scored
  identically, or compared through neighbour metrics, where the CPU effect is small next
  to the seed effect.

## Files

- `<run>/config.json`: the run's settings, environment and command (`arm64_seed1` has no
  command entry, because it was started through `other_seed_runner.py`).
- `<run>/metrics_summary.json`: stage 4 output (not for `arm64_b`, which is the same file
  as `arm64_a`).
- `<run>/embedding.sha256`: sha256 of the run's `embedding.csv`. The embeddings are not
  stored (3 MB each).
- `compare_embeddings.py`: prints the embedding table from run folders.
- `other_seed_runner.py`: the seed-1 run.

## Environment

```
Host: Mac Studio (Apple Silicon), macOS 26.5.1, Docker 29.9.3
Image: python:3.13.11-slim, linux/arm64 and linux/amd64 (emulated), 2.5 GB
torch 2.13.0+cpu   scvi-tools 1.5.0.post1   lightning 2.6.6   pyro-ppl 1.9.2
scanpy 1.12.4   anndata 0.13.2   leidenalg 0.12.0   igraph 1.0.0   numpy 2.4.6   pandas 3.0.5
Threads: OMP, OPENBLAS, MKL and NUMBA all set to 1
```

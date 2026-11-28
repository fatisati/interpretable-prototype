# Reference container check on the Mac (2026-11-28)

**Question:** inside one Linux image with the Windows machine's package versions, do two
different CPU types give the same Leiden clusters?

**Answer on the Mac: yes.** The Linux ARM container and the Linux x86-64 container give
identical labels for every cell, in all four columns. The x86-64 container ran under
emulation on the Mac, so a run on a real Intel machine is still needed (see "To do").

## What was run

`repro/container_check.py` in the image built from `repro/docker/Dockerfile`, once per
platform. Input: the Mac's committed PCA embedding of heart
(`../heart_cell_atlas/pca/embedding.csv.gz`) at the resolutions in its `config.json`.
Leiden seed 0, train and full job, on the graph built in the container (`own_graph`) and
on the graph saved by Windows (`windows_graph`).

## Result

| Comparison | Train job | Full job |
|---|---|---|
| Linux ARM vs Linux x86-64, graph built in the container | identical labels | identical labels |
| Linux ARM vs Linux x86-64, Windows graph | identical labels | identical labels |
| Either container vs Windows native (`clusters_seed0.csv`) | ARI 0.7992, 200 vs 204 clusters | ARI 0.7777, 246 vs 249 clusters |

Both label files have the same sha256: `40a5eea3de63474dce64bda39f949550f47f8ac5c08554702663ee9b261b611b`.

Two more things seen on the way (not stored here, see PR #7 comments):
- The Mac outside the container, with the same package versions, gives the same labels as
  the containers on the Windows graph. So macOS ARM, Linux ARM and Linux x86-64 agree with
  each other on an identical graph, and all three differ from Windows native. The cause of
  the Leiden difference is the Windows build of the libraries, not the CPU. Which part of
  the build is not known.
- The Mac outside the container, building its own graph, differs from the containers on
  the train job (ARI 0.83). Matching package versions alone is not enough; the container is.

## What this does not show

- Whether Docker on the Windows machine gives these labels. That is the next step.
- Anything about PCA, scVI or SEACells inside the image. Only the neighbour graph and
  Leiden were run, on a fixed embedding.

## To do on Windows (about 10 minutes, mostly the image build)

From the repo root, in PowerShell:
```
docker build -f repro/docker/Dockerfile -t clean-v2-ref repro/evidence/windows
docker run --rm -v "${PWD}:/repo" clean-v2-ref python repro/container_check.py --expect repro/evidence/mac-studio/container_check/labels_linux_arm64.csv.gz
```
The last line printed is `SAME CLUSTERS in every column` or `DIFFERENT clusters in at
least one column`. Please commit `runs/container_check/labels.json` as
`repro/evidence/windows/container_check/labels_linux_amd64.json`.

## Files

- `labels_linux_arm64.csv.gz`: the labels from the Linux ARM container (the x86-64 file
  is the same, byte for byte, so only one is stored).
- `labels_linux_arm64.json`, `labels_linux_amd64.json`: each run's platform, package
  versions, thread settings, sha256 of its labels, comparisons and command. `git_commit`
  is empty because the image has no git; both runs used the script as committed with
  this folder.

## Environment

```
Host: Mac Studio (Apple Silicon), macOS 26.5.1, Docker 29.9.3
Image: python:3.13.11-slim, Linux (glibc 2.41), linux/arm64 and linux/amd64 (emulated)
scanpy 1.12.4   anndata 0.13.2   pynndescent 0.6.0   umap-learn 0.5.12
leidenalg 0.12.0   igraph 1.0.0   numpy 2.4.6   scipy 1.18.0
scikit-learn 1.9.0   pandas 3.0.5   numba 0.67.0   llvmlite 0.49.0
Threads: OMP, OPENBLAS, MKL and NUMBA all set to 1
```

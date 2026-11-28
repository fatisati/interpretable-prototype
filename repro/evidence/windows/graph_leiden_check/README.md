# Graph or Leiden? Cross-machine check (2026-11-28)

**Question:** the Mac's own PCA embedding, clustered on Windows with the same Leiden seed (0)
at the same resolution, gives different clusters (train: 204 vs 198 clusters, ARI 0.78).
Is that the neighbour graph, Leiden, or both?

**What is here** (built on Windows from `repro/evidence/mac-studio/heart_cell_atlas/pca/embedding.csv.gz`
at the resolutions in its `config.json`):
- `graph_train.npz`, `graph_full.npz`: neighbour graph (connectivities + distances + cell order),
  built exactly as `evaluation/common.py` builds it (`sc.pp.neighbors` defaults, graph seed 0).
- `clusters_seed0.csv`: Leiden seed 0 on those graphs. Train 204 clusters, full 249.

**Run on the Mac** (from repo root, about 2 min):
```
E=repro/evidence/mac-studio/heart_cell_atlas/pca
python repro/graph_leiden_check.py compare $E/embedding.csv.gz $E/config.json repro/evidence/windows/graph_leiden_check
```
It prints, per job: (1) share of cells whose 15-neighbour list matches ours, and the largest
edge-weight difference; (2) Leiden seed 0 on **our** graph vs our clusters; (3) Leiden seed 0
on **your own** graph vs our clusters.

| (1) graph | (2) Leiden on our graph | Meaning |
|---|---|---|
| differs | ARI 1.000 | the graph is the source; Leiden is portable |
| identical | ARI < 1 | Leiden is the source |
| differs | ARI < 1 | both |

Self-check on Windows (compare against its own files): 100% same lists, weight difference 0,
ARI 1.000 for (2) and (3), both jobs.

Environment: see `../leiden_seed_test/README.md`. Note pynndescent (the approximate-neighbour
library behind `sc.pp.neighbors`) is 0.6.0 here and 0.7.0 on the Mac; leidenalg and igraph match.

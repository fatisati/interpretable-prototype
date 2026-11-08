# Committed train/test splits

One text file per split: the sorted `obs_names` of the HELD-OUT (test) cells, one per line.
Every other cell of the dataset is a training cell. Committed so that every machine (Modal,
Colab, local CPU/MPS) scores against the same held-out set without regenerating it from a
seed -- the generating code (`modal_scripts/make_ctb_split.py`, `notebooks/pbmc_immune_local/
prepare_pbmc_immune.py`) is deterministic, but a committed file is a check, not a hope.

| file | dataset | protocol | n_test / n_total |
|---|---|---|---|
| `pbmc-immune_arbf_ctb20_seed0_test_ids.txt` | pbmc-immune (33,506 cells, `study` x `final_annotation`) | 20% stratified by (cell type x batch), seed 0, groups with <2 cells kept in train | 6,700 / 33,506 |

Verified 2026-11-01: identical to the `test_mask` stored in Modal's
`files/volume_runs/pbmc-immune/d63_mse-rec/clusters_full.npz` and to the local
`data/node_heldout/pbmc-immune_arbf_ctb20_seed0_split.npz`.

`prepare_pbmc_immune.py` loads this file when it exists and only falls back to the seeded
draw when it does not (and then asserts the draw reproduces the file if both are present).
Each run's `provenance.json` records the md5 of the split it actually trained on.

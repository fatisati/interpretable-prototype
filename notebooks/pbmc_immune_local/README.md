# pbmc_immune_local — reproduce the D63/D64 arms on the Mac Studio (CPU, no CUDA)

What it answers: does the D63 recipe (cohesion gram+aff with prototypes detached, GM-KL,
CVAE recon, proto_recon mix3 with scores detached; two-budget gradient-share lambdas)
reproduce locally on pbmc-immune / arbf_ctb20_seed0, scored with the D63 metric set.

Run order (from repo root, env `interpretable-ssl`; every step is load-if-exists):

```
python notebooks/pbmc_immune_local/prepare_pbmc_immune.py   # HVG file, split, train graph
python notebooks/pbmc_immune_local/probe_gradnorms.py        # 50-ep pretrain + frozen probe -> tables/gradshares_local.json
ARM=d63_mse-rec EPOCHS=10 python notebooks/pbmc_immune_local/train_d63_arm.py
python notebooks/pbmc_immune_local/score_local.py            # local arms + pulled Modal runs, one table
```

Inputs: `data/raw_figshare/Immune_ALL_human.h5ad` (figshare 12420968, file 25717328).
Outputs: `tables/<arm>/{epoch0,final}/clusters_full.npz ...`, `tables/scoreboard_local.csv`,
run checkpoints under `models/pbmc-immune_arbf_ctb20_seed0_train/`.

Known differences from the Modal runs (2026-10-31):
- HVG recipe of the original `Immune_ALL_human_hvg.h5ad` is unrecorded; local file uses
  seurat-flavor 4,000 HVGs on the shipped X. Density matches (7.27%), X scale does not
  (sum 338 vs 267). The split (seed 0, type x batch) matches Modal's test mask exactly.
- Lambdas are re-measured locally (probe), not copied from Modal.
- Device: CPU by default; `USE_MPS=1` opts into the Apple GPU.

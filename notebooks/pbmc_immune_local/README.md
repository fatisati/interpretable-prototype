# pbmc_immune_local — reproduce the D63/D64 arms on the Mac Studio (CPU or Apple MPS, no CUDA)

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
- Device: CPU by default; `USE_MPS=1` opts into the Apple GPU (M3 Ultra). Measured per
  training step of d63_mse-rec (2,003 steps/epoch), 2026-10-31:

  | device | s/step | per epoch | notes |
  |---|---|---|---|
  | CPU, legacy scipy/CSR cohesion | 1.78 | 59 min | the first local run (`tables/d63_mse-rec`) |
  | CPU, device-resident cohesion | 1.55 | 52 min | same numbers, bit-identical losses |
  | MPS, legacy path via CPU sparse fallback | 8.3 | 4.6 h | host<->GPU copies per step |
  | MPS, device-resident cohesion, index_put writes | 3.8 | 2.1 h | `tensor[idx] = rows` costs 1.8 s/call on MPS |
  | **MPS, device-resident cohesion + index_copy_** | **0.19** | **6.5 min** | same speed as Modal's L4 |

  `check_device_equivalence.py` shows one step is numerically identical on mps and cpu
  (worst rel diff 2e-6) once dropout and the reparameterisation sample are held fixed;
  `bench_device.py` times 40 real steps (`USE_MPS=0|1 STEPS=40`). Full MPS replicate of the
  arm: `RUN_TAG=d63_mse-rec_mps` (pre-registered in `notes/d63_mse-rec_mps_expectation.md`).

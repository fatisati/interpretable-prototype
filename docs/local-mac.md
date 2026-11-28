# Running on an Apple Silicon Mac (Metal GPU)

Checked 2026-11-22 on a Mac Studio M3 Ultra, heart_cell_atlas, stages 1-5 of
[`runbook.md`](runbook.md). Every stage runs with no code change except the one flag below.

## The one Mac-specific flag

```
python baselines/scvi_baseline.py ... --accelerator mps
```

scVI is the only stage that uses a GPU. Its default `--accelerator auto` trains on **CPU** on a
Mac: scvi-tools 1.4.2 replaces `auto` with `cpu` when the only GPU is Metal ("mps"), and uses
Metal only when asked by name. `config.json` records both `accelerator` (what was asked) and
`device_used` (what training ran on, e.g. `mps:0`).

Stages 1, 2, 4, 5 and the PCA and SEACells baselines are CPU-only code (scanpy, scikit-learn,
scipy), so there is nothing to move to the GPU.

## Measured wall time (whole script, heart_cell_atlas, 18,641 cells)

| stage | time |
|---|---|
| 1 dataset prep (includes the 66 MB download) | 9 s |
| 2 `prepare_eval.py` | 5 s |
| 3 scVI, `--accelerator cpu`, 100 epochs | 3 min 02 s |
| 3 scVI, `--accelerator mps`, 100 epochs | 1 min 47 s |
| 3 PCA + Leiden | 44 s |
| 3 SEACells (train-only run + full run) | 8 min 23 s |

The scVI times include the Leiden search for K that runs after training on CPU; training alone
was not timed separately.

## Numbers from this Mac are not interchangeable with `results/.../official/`

- Raw file: sha256 identical to `datasets/heart_cell_atlas.md` (`4a54acfa...`).
- Processed file: sha256 **differs** (`36949136...` here, `7e686ed3...` there). The package
  versions differ (table below). Whether the cell-by-gene content differs or only the file
  encoding was not checked, because the other machine's processed file is not available here.
- Metrics over all cells agree with the committed official table to within 0.005 (PCA and scVI).
- Metrics over rare cells do not: PCA train `rare_f1` is 0.846 here against 0.696 official, and
  PCA is deterministic given its input. So a Mac run and an official run must not share a table.
- scVI on CPU and on Metal are two different training runs (different random streams): all-cell
  metrics agree within 0.002, rare-cell metrics differ by up to 0.08 (test `rare_f1` 0.846 CPU,
  0.924 Metal). One run each, so this says nothing about which device is better.

Nothing from these Mac runs is committed under `results/`.

## Environment (2026-11-22)

```
macOS 26.5.1 arm64, python 3.11.18   (conda env `interpretable-ssl`)
scvi-tools 1.4.2    torch 2.15.0     lightning 2.6.6
scanpy 1.11.5       anndata 0.12.19  numpy 2.2.6      scipy 1.13.1   pandas 2.3.3
scikit-learn 1.9.1  scikit-misc 0.5.3  leidenalg 0.12.0  igraph 1.0.0
SEACells 0.3.3 (--no-deps)  palantir 1.4.5  numba 0.68.1
```

## What was NOT carried over from the `local-mac-mps` branch

That branch's Metal work (device-resident COO cohesion graphs, `index_copy_` memory writes,
`get_device()` with `USE_MPS=1`, the scPoli trainer device pin) all patches the scProto trainer
in `interpretable_ssl/`. This branch has no scProto code yet (`scproto/` is empty), so there is
nothing to apply those changes to. They are needed again when the trainer is added here; the
write-up is section 12 of `CLAUDE.md` on `local-mac-mps`.

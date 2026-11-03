# files/repro — reproducibility snapshot (immune dataset, arm `d63_mse-rec`)

Goal: redo the train/test split, the current lambda config, and read the
current metrics, from git alone — no GPU run needed just to see what we have.

## what's in this folder

- `immune_split.csv` — every cell ID in the pbmc-immune dataset, labeled
  `train` or `test`. 26,806 train / 6,700 test (80/20 held-out split, source
  tag `arbf_ctb20_seed0`). Pulled from the Modal volume file
  `data/node_heldout/pbmc-immune_arbf_ctb20_seed0_split.npz` on 2026-11-03 —
  that npz (not this csv) is the source of truth if the split is ever
  regenerated; re-pull with:
  `modal volume get scproto-data data/node_heldout/pbmc-immune_arbf_ctb20_seed0_split.npz <local>`
- `immune_latest_lambdas.json` — the exact loss config for arm `d63_mse-rec`
  (see below). Verified against the run's own saved summary
  (`files/volume_runs/pbmc-immune/d63_mse-rec/d63_arm_summary.json`) —
  every lambda matches to 6 decimal places, this is not a guess.
- `immune_scoreboard.csv` — `d63_mse-rec`'s row from
  `files/tables/d63_budgets/pbmc-immune_d63_full.csv`, trimmed to the
  columns described below.

## which run this is, and why

You asked for "the one with cvae loss, proto recon, cohesions, gmkl, all
same ratio, mse not nb" — that is arm **`d63_mse-rec`** (script:
`modal_scripts/train_d63_budgets.py`). Four loss terms are active:

| term | what it does | lambda |
|---|---|---|
| `lambda_cohesion` (gram) | pulls each prototype's members toward each other, in CVAE-latent dot-product similarity | 2.902454 |
| `lambda_cohesion_aff` | same, but using the affinity-graph similarity instead | 5.073052 |
| `lambda_recon` | the CVAE's own reconstruction loss ("cvae loss") | 0.021126 |
| `lambda_gmkl` | the GM-VAE KL term — pulls the encoder's output toward its assigned prototype's gaussian | 0.022276 |
| `lambda_proto_recon` | moves the prototype itself (and the decoder) to better reconstruct its members; cell assignment scores are DETACHED here so this term can't fight cohesion for where cells go | 0.00021671 |

"all same ratio" = the four terms are not weighted arbitrarily: the
ENCODER-side budget is split 1/8 : 1/8 : 1/4 : 1/4 (cohesion-gram :
cohesion-aff : recon : gmkl), and the PROTOTYPE-side budget (gmkl vs
proto_recon — the only two terms that can move a prototype) is split 50/50.
Every lambda above is *solved from* that share plus a measured gradient
norm, not picked by hand (method: CLAUDE.md §9d-1).

Reproduce this exact arm:
```
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach \
  modal_scripts/train_d63_budgets.py --arms d63_mse-rec
```

## the scoreboard columns, in plain words

- `within_*_all` — for each cell: does its cell type match the majority
  type of its OWN (metacell, batch) group? (computed separately within the
  train half or the test half)
- `cross_*_all` — for each cell: does its (metacell, batch)'s majority type
  match that metacell's GLOBAL majority type (pooled across all batches)?
  **this is the cross-batch alignment check** — low means a metacell's
  batches disagree about what cell type it is.
- `coh_aff` / `coh_gram` — cohesion score itself (mean similarity of a
  metacell's members to each other), in the affinity-graph space / the
  CVAE-latent space.
- `rare_homogeneity` / `rare_f1` / `rare_coverage` — same idea restricted to
  rare cell types only (bottom quartile by frequency).

## measured values: current arm vs the baseline it has to beat

`immune_scoreboard.csv` has two rows: `leiden_st1` (the baseline — Leiden
clustering on the Stage-1 CVAE latent, i.e. what scPoli's own pretraining
gives you for free, before any of scProto's cohesion/gmkl/proto_recon loss
touches it) and `d63_mse-rec` (the current arm).

| metric | leiden_st1 (baseline) | d63_mse-rec (current) |
|---|---|---|
| coh_aff (train / test) | 0.0591 / 0.0535 | 0.1305 / 0.0773 |
| coh_gram (train / test) | 0.1106 / 0.1049 | 0.2236 / 0.1494 |
| within (train / test) | 0.8675 / 0.8708 | 0.8960 / 0.8537 |
| **cross (train / test)** | **0.9665 / 0.9827** | **0.8750 / 0.9091** |
| rare_f1 (train / test) | 0.6214 / 0.6662 | 0.7808 / 0.7605 |

## the challenge

**Cross-batch alignment (the `cross` row) is worse than the baseline, on
both train and test** — 0.9665→0.8750 (train, −0.091) and 0.9827→0.9091
(test, −0.074). That's the real problem: adding scProto's loss (cohesion +
gmkl + proto_recon + cvae-recon) buys a large cohesion gain and a large
rare-f1 gain over plain scpoli-stage-1 Leiden, but it costs cross-batch
alignment relative to that same baseline. (Earlier draft of this section
compared this arm's own train vs test numbers to each other instead of to
this baseline, and read the `cross` row as fine — that was the wrong
comparison; corrected here.)

This matches the ADR's own verdict for this arm
(`my-docs/adr/D63-two-budget-detached-mse-nb.md`, P3): cross-batch "FAILS...
preserved, not improved" against its own chosen baseline
(`d60_leid_noc`) — same direction of result, measured a different way.

reproduce the comparison: run
`modal_scripts/score_d63_budgets.py` (or re-read
`files/tables/d63_budgets/pbmc-immune_d63_full.csv`, rows `leiden_st1` and
`d63_mse-rec`) — both already include everything above.

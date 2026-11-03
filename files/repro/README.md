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

## measured values for this run

| metric | train | test |
|---|---|---|
| coh_aff | 0.1305 | 0.0773 |
| coh_gram | 0.2236 | 0.1494 |
| within (purity) | 0.8960 | 0.8537 |
| cross (cross-batch agreement) | 0.8750 | **0.9091** |
| rare_homogeneity | 0.6919 | 0.6598 |
| rare_f1 | 0.7808 | 0.7605 |
| rare_coverage | 1.0 | 1.0 |

## the challenge — flagging a discrepancy, not papering over it

You described the current challenge as: "with gmkl, generalization works
but cross-batch alignment does not." What's actually measured on THIS run
doesn't match that framing: `cross` (the cross-batch agreement column) is
*higher* on test than train (0.909 vs 0.875) — it generalizes fine here.
What DOES drop train-to-test is cohesion itself (`coh_aff` 0.130→0.077,
`coh_gram` 0.224→0.149) and, to a smaller extent, `within` (0.896→0.854).

So on this specific run, the gap is in cohesion/within-batch purity
generalizing, not in cross-batch alignment. If "cross-batch alignment
failing" is based on a different run, a different metric, or a visual/DGE
check rather than this `cross` column, say which — I don't want to guess
and silently build the rest of this doc on the wrong problem statement.

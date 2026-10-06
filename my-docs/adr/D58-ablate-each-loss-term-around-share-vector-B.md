---
id:             D58
title:          ablate each loss term around share vector B, immune, dose 0/1x/2x
status:         running
date_opened:    2026-10-06
date_closed:
supersedes:
superseded_by:
tags:           [ablation, dose-sweep, loss-weighting, rare-cells]
datasets:       [pbmc-immune]
protocol:       NH-heldout (arbf_nodeheldout20_seed0, 20% test, frozen-pass placement)
K:              300
comparable_to:  the immune NH scoreboard in files/tables/node_heldout_scoreboard/
runs:           [d58_base_s31, d58_base_s32, d58_gram0, d58_aff0, d58_comm0, d58_recon0]
lit:            [chen2018gradnorm]
code:           fa04afd
cost:           ~$0.79/arm measured on L4; 6 arms ~= $4.1-4.7 (hard cap $5)
paper:          ablation table (main text)
---

# D58 — ablate each loss term around share vector B, immune, dose 0/1x/2x

> In the context of **a 4-term loss whose terms had never been individually
> tested**, facing that **we could not say whether community earns its place when
> cohesion is already there, nor which cohesion kernel does the work**, we decided
> for **one dose curve per term around the D56-calibrated share vector B, where
> dose=0 IS that term's ablation arm** and against **running ablation and sweep as
> two separate experiments**, to achieve **both results from one set of runs inside
> a $5 cap**, accepting **one term varied at a time (no interactions), a single
> dataset, and one seed per arm against a 2-seed noise floor**.

## Context and Problem Statement

`D56` produced calibrated base lambdas but its own Consequences section predicted
that the calibrated point is **not** the good point: on immune, candidate A's
community lambda (0.0323) sat essentially on top of `d59_v3`'s 0.0345, the weakest
arm on the scoreboard, while `d59` at 0.124 did best. So the weights needed a
dose response, not just a calibration.

Separately, no term had ever been removed individually. The three existing immune
"ablation-looking" rows in `RESULTS.md` differ in **all four** lambdas *and* in
prototype init (they predate the `D57` fix), so they are not an ablation.

## Decision Drivers

- `$5` hard budget cap (user decision). Measured ~$0.79/arm on L4 → ~6 arms.
- The user's own open question: *"why do we need community when we have cohesion?"*
- The user wants `gram` and `aff` separable, not just "cohesion" as a block.
- `D57` fixed the nondeterministic prototype init, so a noise floor is finally
  meaningful — before that, run-to-run variance was inflated by ~65%-different
  prototype draws.

## Prior Art

- `D56` — the calibrated base lambdas and the share-vector parameterisation this
  sweep moves around. Share vector B = equal share per **subsystem**.
- `D46` — community at 0.25x cohesion's gradient on s28nsc: fails. Evidence that
  community is dose-critical, and that its useful dose may be dataset-dependent.
- `D39` — on s28nsc, gram+aff beat either kernel alone (joint purity 0.4576 vs
  0.4472 aff-only / 0.4558 gram-only), and gram-only left 5/200 prototypes dead
  while aff-only left 0. So if one kernel matters more, expect `aff`.
- `D48` — routed community (`umap_proto_sg=1`): prototypes detached, so the
  community gradient reaches the encoder only. This is what makes the
  held-out-vs-full contrast below a meaningful test.

## Considered Options

- **Ablation and sweep as two experiments** — clearest to describe; but ~2x the
  runs, and the budget does not allow it.
- **CHOSEN: one dose curve per term, dose=0 = the ablation arm** — both results
  from one set of runs. Describable in a sentence: *"each term's calibrated weight
  was scaled by {0, 1x, 2x}, one term at a time."*
- **Sweep the 3 subsystems instead of the 4 terms** — one fewer axis; rejected
  because it cannot say whether `gram` or `aff` does the work.
- **Do nothing / keep share vector B as given** — rejected: `D56` already predicts
  the calibrated point is not the good one.

## Decision Outcome

Base = share vector B on pbmc-immune from `D56` (run tag `d57`, post-prototype-fix):

```
gram 3.794    aff 6.399    community 0.04307    recon 0.02042
```

Each of the four terms scaled by `{0, 1x, 2x}` individually, everything else at
base. `train_epochs=10`, `seed=31` unless stated, `K=300`, NH-heldout protocol.

**Arms actually launched inside the cap** (dose-2x points deferred):

```
base_s31   reference, ratio 1x everywhere
base_s32   seed 32 -- NOISE FLOOR
gram0      gram -> 0
aff0       aff -> 0
comm0      community -> 0
recon0     recon -> 0
```

**Cost explicitly accepted:** one seed per arm (noise floor from 2 seeds only, so
it bounds scale but is not a proper std), one dataset, one term at a time.

### Confirmation

> Written BEFORE the run. Never edited afterwards.

- **Expect** —
  - `gram0` / `aff0`: test cohesion drops, rare metrics drop. If one kernel
    dominates, expect `aff` to matter more (`D39`).
  - `recon0`: batch entropy **and** cross-batch homogeneity drop; purity and rare
    metrics drop. With recon off nothing pushes batch information out of `z` — the
    batch embedding has no other consumer, and the affinity graph is built from PCA
    of `X`, which is batch-confounded.
  - `comm0`: held-out rare metrics and purity drop, and the **held-out penalty is
    larger than the full-dataset penalty**. This is the discriminating test:
    cohesion can lower itself by moving prototypes, which does nothing for an
    unseen cell, whereas routed community has prototypes detached and so can only
    change the encoder's input→z map, which a new cell inherits.
- **Would falsify this** — any arm with `n_unused_protos > 0` (invalid, not
  "worse"); a `comm0` arm whose held-out penalty is no larger than its full-dataset
  penalty (then community is not earning its place and should be dropped); any
  delta smaller than the base-seed spread (then it is noise, not an effect).
- **Scale/range check that must pass first** — every arm prints its four lambdas
  in-run (`[D58 CONFIG VERIFIED IN-RUN]`) and asserts every non-design term is
  exactly 0 and at least 2 terms remain active; `cohesion` must report in `[0,1]`.

### Setup and Provenance

- **Run script** — `modal_scripts/train_d58_ablation.py` (`--arms` selects a wave);
  collect with `collect_d58_arms.py`; score with `score_d58_arms.py`.
- **Model folder** — `models/pbmc-immune_arbf_nodeheldout20_seed0_train/d58_<arm>_...`
- **Data** — `pbmc-immune_arbf_nodeheldout20_seed0_train.h5ad`; metrics against the
  **FULL** affinity `affinity_pbmc-immune33506_ncomp50_kneighbors50_arbf.pkl`
  (held-out cells' true neighbours exist only there, CLAUDE.md §7a).
- **Seed, versions** — seed 31 (32 for the floor arm); Modal image pinned in script.
- **Deviations from CLAUDE.md §9d** — `coh_mem_hlife=0.0` (no age decay; §9d wants
  `c*N/uniq` but no `coh_mem_drift` measurement exists, and `d59` used 0 so we
  match it); `epsilon` falls back to `effk=5.0` on every dataset (known-broken
  calibration, CLAUDE.md §9b) — identical in every arm, so neither can confound.
- **Known arm-specific caveat** — in `recon0`, `freeze_batch_embedding=True` is a
  no-op because nothing else consumes the batch embedding. That arm is therefore
  "no recon AND effectively no batch mechanism".

### Consequences

> Every number with the path it was read from. Means with their std.

Reference arm, from `files/tables/d58_ablation/pbmc-immune_d58_ablation.csv`:

| metric | base_s31 |
|---|---|
| rare homogeneity | 0.5849 |
| rare F1 | 0.7046 |
| rare coverage | 0.9333 |
| cross-batch homogeneity | 0.2599 |
| batch entropy (test, wtd) | 0.8699 |
| purity (test, wtd) | 0.7594 |
| purity (full, wtd) | 0.7798 |
| cohesion aff (test) | 0.0788 |
| cohesion gram (test) | 0.1541 |
| modularity (test) | 0.1506 |
| metacells used | 300 / 300 |
| train time | 3553 s on L4 |

*Remaining arms running; table to be completed before this ADR closes.*

- **Good** — pending
- **Bad** — pending
- **Surprised me** — `gram0` trained in **1125 s** against base's **3553 s**, a
  3.2x speedup, because `scproto.py:2943` skips the 2-hop Gram similarity entirely
  when `lambda_cohesion=0`. So the gram kernel is ~70% of this loss's step cost,
  and ablation arms that remove a term are materially cheaper than the base.
- **Deviations from Confirmation** — pending

## Verdict

*pending — 1 of 6 arms scored.*

## More Information

- Plan, in plain text: `my-docs/doing/ablation-plan.txt`
- Base lambdas and how they were measured: `my-docs/experiments/base-lambdas-gradnorm.md`
- `D56` (the calibration this moves around), `D57` (prototype-init fix that makes a
  noise floor meaningful), `D46`, `D39`, `D48`.
- **Selection rule, declared before any result:**
  ```
  GATE       n_unused_protos == 0 -- dead prototypes make an arm INVALID
  PRIMARY    rare homogeneity, held-out
  GUARD      held-out purity must not fall more than 1 noise-floor std below base
  SECONDARY  rare F1, rare coverage, cross-batch homogeneity, batch entropy,
             test cohesion, test modularity -- reported, NOT selected on
  TIE-BREAK  within 1 noise-floor std on primary -> prefer higher batch entropy
  ```
  Primary is rare homogeneity on a precision argument, measured on immune: SE of
  the mean is 0.272/sqrt(4386 cells) = **0.0041**, against purity's
  0.180/sqrt(300 metacells) = 0.0104 and coverage/F1's 0.267/sqrt(5 batches) =
  **0.1194**. The ±0.25-0.40 figures in `RESULTS.md` are the spread of the
  distribution, not the error on the mean — so rare homogeneity is the *most*
  precisely estimated metric in the set, and coverage/F1 cannot carry a selection.
- **Follow-ups this ADR creates:**
  - `seed` is missing from `interpretable_ssl/constants.py` ABBREVIATIONS, so runs
    differing only in seed share one model folder and clobber each other
    (CLAUDE.md §9). Worked around here with distinct `experiment_name`s; the real
    fix is behaviour-preserving (default 31 stays omitted from names).
  - `score_all_baselines.load_labels()` reads only `clusters.npz`, never
    `clusters_full.npz`, so NH runs that save only the latter are silently skipped
    — this is why `d59_v2`/`d59_v3` never reached the immune scoreboard.
  - Immune run-dir naming is ambiguous: both `d59`/`d59_v2`/`d59_v3` and
    `d59_pbmc-immune`/`d59_pbmc-immune-v3` exist, and `RESULTS.md`'s "4-term
    share25" row comes from `d59_pbmc-immune`. Needs cleaning before publication.

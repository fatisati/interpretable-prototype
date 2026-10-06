---
id:             D56
title:          lambda from a FROZEN full-pass gradnorm median, not one minibatch
status:         works
date_opened:    2026-10-06
date_closed:    2026-10-06
supersedes:     D55
superseded_by:
tags:           [lambda-calibration, gradnorm, loss-weighting, measurement-protocol]
datasets:       [pancreas, lung, pbmc-immune, s28nsc, crcx]
protocol:       NH-heldout (pancreas, lung, pbmc-immune) + full-dataset (s28nsc, crcx)
K:              220 / 300 / 300 / 800 / 670 (dataset defaults, CLAUDE.md 9d)
comparable_to:  n/a — this is a measurement, not a scoreboard run
runs:           [d60_pancreas, d60_lung, d60_pbmc-immune, d60_s28nsc, d60_crcx]
lit:            [chen2018gradnorm, dhillon2004kernel]
code:           fc896e5
cost:           ~2.5 GPU-h T4 (attempt 1, ~$1.70) + ~0.8 GPU-h (attempt 2, ~$0.55)
paper:          method section (loss weighting) + appendix table (per-dataset lambdas)
---

# D56 — lambda from a FROZEN full-pass gradnorm median, not one minibatch

> In the context of **choosing the four loss weights**, facing that **a single-minibatch
> gradnorm probe varies 2.0x on the very same frozen checkpoint, and loss-value balancing
> disagrees with gradient balancing by 34x on this loss**, we decided for **a FROZEN
> full-pass gradnorm MEDIAN, `lambda_i = share_i / median_i`** and against **learnable
> GradNorm weights and value-based grid search**, to achieve **a reproducible calibration
> that is portable across datasets as a share vector and statable in one sentence**,
> accepting **~5x the per-step cost on probed steps, one extra epoch per dataset per
> calibration, and a ~30% run-to-run spread on the cohesion terms that is still
> unexplained**.

**Why it matters:** choosing these weights was the single most consequential unmeasured
decision in the model. The outcome is uncomfortable and useful — the correctly calibrated
a-priori weights reproduce the *worst*-performing arm we have, which turns the dose sweep
from a robustness check into the actual experiment.

## Context and Problem Statement

I did not know how to choose the four lambdas, and the choice is very effective — it
changes the reported metrics more than most architectural changes do. Every lambda in
use had been set by a single-minibatch measurement, or by a process nobody wrote down.

Three measured facts made this untenable:

- **Uncalibrated weights delete the model's own contribution.** At all lambdas = 1 on
  pbmc-immune, cohesion receives **0.30%** of the encoder gradient (recon 72.9%, community
  26.8%). We would be training a CVAE with a community loss and calling it scProto.
  *(read from `files/volume_runs/pbmc-immune/d58b/grad_probe.jsonl`, step 0)*
- **A raw lambda is not portable across datasets.** Measured per-unit `gnorm_enc` spreads
  **6–14x** between datasets (`cohesion_gram` 0.0123 on s28nsc vs 0.1063 on pancreas), so
  the same number means a very different influence depending on the dataset.
- **A single minibatch is not a measurement.** Three step-0 probes of the *same* frozen
  s28nsc checkpoint gave `cohesion_gram` = 0.0183 / 0.0234 / 0.0362 — a **2.0x spread**
  from batch sampling alone.
  *(`models/s28nsc/{d46_probe,d50rt_gmA}/grad_probe.jsonl`)*

## Decision Drivers

- `[D55]` left the target share vector **PENDING** and recorded that no dataset except
  s28nsc had ever been calibrated by a documented method. Everything downstream of
  "the 4-term loss" rested on that gap.
- The ablation and dose sweep cannot start until the base config exists and is defensible.
- Reviewer-facing: the method section has to state how the weights were chosen in one
  sentence that survives "why not just grid search?".
- Cost is not the constraint (T4, ~$2 for all five datasets). **Wall-clock and
  reproducibility are.**

## Prior Art

- **GradNorm** — Chen, Badrinarayanan, Lee & Rabinovich, *GradNorm: Gradient
  Normalization for Adaptive Loss Balancing in Deep Multitask Networks*, ICML 2018
  (`chen2018gradnorm`, in `references.bib`).
  *Claims:* balance multi-task losses by driving each task's **gradient norm** toward a
  target, rather than tuning weights by hand.
  *What we reuse:* the central premise — **gradient magnitude, not loss value, is what
  balances terms.**
  *What we do NOT reuse:* its mechanism. GradNorm makes the weights **learnable
  parameters** with their own optimizer and auxiliary objective
  `L_grad = sum_i |G_i - G_avg * r_i^alpha|`, updated every step
  (`r_i` = task *i*'s loss relative to its own initial value, i.e. how far behind it is).
  *Does not cover here:* an unsupervised clustering objective with no task-wise labels,
  where "which task is learning slower" has no clean meaning.
- **Uncertainty weighting** — Kendall, Gal & Cipolla, CVPR 2018. The other standard
  learnable-weight route (one log-variance per term). Same reason for rejection as
  GradNorm's, plus the homoscedastic-noise interpretation does not transfer to a
  clustering loss. *(not yet in `references.bib`)*
- **Kernel k-means = normalized cut** — Dhillon, Guan & Kulis, KDD 2004
  (`dhillon2004kernel`, in `references.bib`). Why cohesion is scale-free and therefore
  has structurally *small* gradients: its value is bounded in `[0,1]` and its reachable
  span on these graphs is only ~0.05–0.38.
- **MinCutPool** (Bianchi et al., ICML 2020) and **DMoN** (Tsitsulin et al., JMLR 2023)
  normalize every term to a comparable bounded range *by design*, which is why they can
  use weights near 1. Our loss mixes one bounded ratio with two unbounded terms, so we
  cannot. **Neither is currently in `references.bib` — add both before submission**; a
  reviewer who knows MinCutPool will ask.

## Considered Options

- **Hand-picked round numbers** — free, and what most papers do; but measured to give
  cohesion 0.30% of the gradient. Rejected on evidence.
- **Learnable weights (GradNorm proper / uncertainty weighting)** — principled and
  adaptive; but it is not what papers in this area do, it adds a second optimizer and an
  `alpha` to defend, and `[D46]` already measured that realized shares drift only mildly,
  so the adaptivity buys little. **Rejected: not standard for work like this, and the
  problem it solves is small here.**
- **Grid search on the loss VALUE (the field norm)** — cheap, needs no extra backward
  passes, and is what comparable single-cell / deep-clustering papers report. **Rejected
  on measurement:** for this loss, value-balancing and gradient-balancing disagree by
  **34x**.
  ```
  step 0, pbmc-immune        value     gnorm_enc
  umap (community)            3.706      7.238
  cohesion_gram               0.8901     0.0511

  normalize by VALUE     lambda_umap/lambda_gram = 0.8901/3.706 = 0.240
  normalize by GRADIENT  lambda_umap/lambda_gram = 0.0511/7.238 = 0.00706   -> 34x apart
  ```
  Mechanistic reason: cohesion is `1 - mean_similarity`, **bounded in [0,1]** with a
  reachable span of only ~0.05–0.38, so it can never be steep; community is an unbounded
  log-loss summed over 5 negatives per edge; recon is an MSE **summed over 4,000 genes**.
  `value/gnorm` ranges over **52x** across the four terms, so value carries no information
  about influence.
- **Static single-snapshot gradnorm, measured at step 0 on ONE minibatch** — the `[D55]`
  recipe; one forward/backward. Rejected: the 2.0x same-checkpoint spread above.
- **CHOSEN: static gradnorm from a FROZEN FULL-PASS MEDIAN** — see below.
- **Do nothing** — keep the `d59` lambdas. Rejected: `[D55]` showed they were never
  calibrated by any documented method, and on immune they were reconstructed as being
  3.6–7.6x off their intended share.

## Decision Outcome

**Measure each term's per-unit encoder-gradient norm over one full pass with the weights
frozen, take the MEDIAN across iterations, and set `lambda_i = share_i / median_i`.**

```
1. load the pretrained CVAE checkpoint                    (identical to real training)
2. build the edge dataset at the DEPLOYMENT threshold, train_epochs=10
3. freeze the weights, then PROVE they did not move
4. loop over every batch of edges exactly once            (= one real epoch)
5. at each probed step, measure EACH TERM's gradient norm SEPARATELY
6. median / mean / std per term over the pass
7. lambda_i = share_i / median(gnorm_i)        4 significant digits
```

Why the median and not the mean: gradient norms are positive and right-skewed, so one
outlier batch should not move the calibration. Both are reported; they agree to <1% here.

Why a **separate** backward pass per term, and why "no weight update" does **not** make
this cheap: real training computes `total.backward()` once, after which every parameter
holds the four terms' contributions **already summed** — they cannot be separated again
(`a+b+c+d = 300` does not give you `a`). Per-term norms therefore require per-term
backward passes. Skipping `optimizer.step()` saves almost nothing; it is the cheapest part
of a step.

**Cost explicitly accepted:** ~5x the per-step cost of training for the probed steps, and
one extra epoch per dataset per calibration. ~$2 per full sweep of five datasets on T4.

**Two share vectors are carried forward, both chosen a priori, neither tuned on any
metric.** This ADR decides the *measurement*, not the share vector; the share vector is
`D57`'s job.

```
A  flat            0.25 / 0.25 / 0.25 / 0.25   -- cohesion gets HALF the budget, purely
                                                  because it is WRITTEN as two kernels
B  per-subsystem   1/6 / 1/6 / 1/3 / 1/3       -- three MECHANISMS weighted equally:
                                                  who belongs together / does it
                                                  generalize / is z still about genes
exact conversion:  lambda_B = lambda_A * 2/3 (gram, aff) ; * 4/3 (community, recon)
```

### Confirmation

> Written BEFORE the run. Never edited afterwards.

- **Expect** — the frozen full-pass median lands near the step-0 values (sampling noise
  only), *not* near `d58b`'s run-averaged values, because `d58b` averaged over a pass in
  which the model was actually training.
- **Would falsify this** — any non-prototype parameter differing before vs after the pass;
  or `recon`'s gnorm trending across the pass (it never touches the cohesion memory, so
  with frozen weights it can only vary by batch composition); or the probe logging far
  fewer rows than `ceil(len(loader)/grad_probe)`.
- **Scale/range check that must pass first** (CLAUDE.md §9b) — `cohesion` must report in
  `[0,1]`; every term's `lambda * gnorm_enc` printed side by side at step 0 and checked
  against its intended share before any lambda is deployed.

### Run

- **Run script** — `modal_scripts/measure_grad_shares_d60.py` (run tag `d60`; the tag
  predates this ADR and does **not** match the ADR id — `d58`/`d59` were already consumed
  as script tags. ADR ids and run tags are separate namespaces here; the artifacts are
  under `d60_*`).
- **Model folder** — `models/<ds>[_arbf_nodeheldout20_seed0_train]/d60_<ds>_..._blr0.0_gprb5_.../grad_probe.jsonl`
- **Data** — `arbf_nodeheldout20_seed0` train split (pancreas/lung/immune); `bk02`
  (s28nsc), `bk05` (crcx). All three NH h5ads are log-normalized, HVG 4,000.
- **Seed, package versions** — `seed=31` (default); Modal image pinned in the script.
- **Deviations from CLAUDE.md §9d** — all four lambdas set to **1.0** (this is a
  measurement, not a training run); `base_lr=0.0`; `grad_probe=5`; `train_epochs` used
  only to set the edge threshold, with the loop exited after one pass.
- **Freeze, three independent mechanisms** — patch `_umap_state['optimizer'].step`
  (`scproto.py:1294` is what the stage-2 loop actually calls); zero every
  `param_group['lr']`; `base_lr=0.0` so the optimizer is *built* with `lr=0`.
- **Deliberately left ON** — `normalize_prototypes()` and the cohesion memory
  (`coh_mem_init='full'`). Both run during real training's first pass; the probe is meant
  to look like real training.

### Consequences

**Verification (attempt 2, the valid one).** Every non-prototype tensor bit-identical;
prototype change bounded and measured.

| check | value | read from |
|---|---|---|
| datasets closed | **5 / 5**, zero failures | run log |
| non-prototype tensors bit-identical | **12 / 12 on every dataset** | run log, `[D60] VERIFIED` |
| prototype relative drift | 1.15 / 1.62 / 2.62 / 1.27 / 2.00 **e-08** (tol 1e-4) | run log |
| probe rows vs expected | 180/180, 379/379, 401/401, 897/897, 733/733 | run log |
| probe errors | 0 on every dataset | run log |
| `recon` last50/first50 | 0.979–1.003, all `OK` | run log |

**Measured per-unit `gnorm_enc`** (median over the pass, all lambdas = 1):

| dataset | rows | gram | aff | community | recon | CV range |
|---|---|---|---|---|---|---|
| pancreas | 180 | 0.1063 | 0.06707 | 1.321 | 74.00 | 0.02–0.08 |
| lung | 379 | 0.09124 | 0.04737 | 4.876 | 11.79 | 0.02–0.06 |
| pbmc-immune | 401 | 0.03969 | 0.02301 | 7.431 | 16.33 | 0.04–0.11 |
| s28nsc | 897 | 0.01311 | 0.009133 | 3.118 | 30.95 | 0.03–0.07 |
| crcx | 733 | 0.01713 | 0.01540 | 5.661 | 5.121 | 0.03–0.10 |

All read from `models/<ds>/d60_<ds>_..._blr0.0_gprb5_.../grad_probe.jsonl`.
Full per-dataset tables with mean/std/step0/first50/last50: run log, Modal app
`ap-WqH1Xf9GtC91YyRPp0nTu6`.

Within-run CV is **0.02–0.11** on every term, so the median is well determined by a few
hundred samples.

**Base lambdas, `lambda_i = share_i / median_i`, 4 significant digits.** Four significant
digits and not 2 decimal places: CLAUDE.md §9d's rule exists for *filename length*
(`lcoh4.03750478377344` blew the 255-byte limit and killed a run after training finished),
but blind 2 dp turns `recon = 0.003378` into `0.00`.

**A — flat 0.25 each**

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 2.352 | 3.728 | 0.1893 | 0.003378 |
| lung | 2.740 | 5.278 | 0.05128 | 0.02121 |
| pbmc-immune | 6.299 | 10.87 | 0.03364 | 0.01531 |
| s28nsc | 19.08 | 27.37 | 0.08018 | 0.008077 |
| crcx | 14.59 | 16.23 | 0.04416 | 0.04882 |

**B — per-subsystem 1/6, 1/6, 1/3, 1/3**

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.568 | 2.485 | 0.2524 | 0.004505 |
| lung | 1.827 | 3.519 | 0.06837 | 0.02828 |
| pbmc-immune | 4.199 | 7.245 | 0.04486 | 0.02042 |
| s28nsc | 12.72 | 18.25 | 0.1069 | 0.01077 |
| crcx | 9.729 | 10.82 | 0.05888 | 0.06509 |

- **Good** — the protocol is sound and self-verifying. The freeze is proven rather than
  assumed; the row-count and `recon`-flatness checks both passed; and the measurement
  reproduces the only other uncontaminated probe (`d58c`, 10 frozen single steps) within
  0.72–1.84x while disagreeing with the contaminated `d58d` by up to 5.2x.

- **Bad / the decisive consequence** — **the correctly calibrated a-priori base reproduces
  the worst arm we have.** The pre-registered question was whether the frozen median lands
  near step-0 or near `d58b`'s averages. It lands near step-0:
  ```
  immune cohesion_gram:  d60 frozen median 0.0397 - 0.0480   (two runs)
                         step-0            0.0498 - 0.0577
                         d58b average      0.1328
  ```
  so candidate A on immune gives community `0.0336`, essentially `d59_v3`'s `0.0345`
  — and `d59_v3` is the weakest immune arm on the scoreboard (modularity 0.3032, purity
  0.779) against `d59`'s 0.3833 / 0.8264 at community `0.124`. Candidate B only reaches
  `0.0449`, still ~3x below `d59`.
  **Therefore the dose sweep is the experiment, not a robustness check.** Carried to `D57`.

- **Surprised me — run-to-run spread on the cohesion terms is far larger than within-run
  noise. ROOT CAUSE NOW FOUND (see below).** Two frozen runs of the same checkpoint and the same
  code, one at `grad_probe=1` and one at `grad_probe=5`:
  ```
  term          pancreas   lung    immune     (ratio gprb5 median / gprb1 median)
  gram            1.064    1.102    0.827
  aff             1.015    1.096    0.657   <- 34% apart
  community       1.003    0.896    0.872
  recon           1.001    1.002    0.992   <- stable everywhere
  ```
  Within-run CV is only 0.04–0.05, so this is **not** sampling. `recon` is reproducible to
  <1%; the cohesion terms are not. Both candidate causes touch cohesion and not recon:
  The variability tracks exactly how strongly each term depends on the prototype vectors,
  which is what located the cause.

  **DIAGNOSED: `eigsh` in the waypoint prototype init is not seeded.**
  `_setup_umap_edges()` defaults to `init_prototypes=True`, so every run DISCARDS the
  prototypes it just loaded from the pretrain checkpoint and re-derives them. That init
  does

  ```python
  # scproto.py:409
  _, vecs = eigsh(L_sym, k=n_eigs, which="LM", tol=1e-2)      # no v0 -> RANDOM start
  ```

  ARPACK starts from a random vector, and `tol=1e-2` is loose enough that different starts
  return materially different eigenvectors. The greedy MaxMin that follows is deterministic
  *given* `vecs` — which is exactly what the comment at `scproto.py:414-421` asserts
  ("Everything after this first pick is already deterministic given it") — but `vecs` is
  not deterministic, so seeding the first pick does not save it.

  Reproduced on a synthetic graph of the same shape, running the identical MaxMin loop:

  ```
  tol=1e-2, no v0  (current code):   seed-cell overlap across 3 calls = 35.5%, 31.4%
  tol=1e-10, no v0:                  100%, 100%
  tol=1e-2, fixed v0:                100%, 100%
  ```

  **~65% of prototype seed cells differ between runs.** Which is why the spread tracks
  prototype-dependence: `recon` never touches prototypes (stable <1%), `community` touches
  them only through the softmax (0.87-1.00x), `cohesion` is computed directly against their
  member sets (0.66-1.10x).

  **Epsilon is exonerated as an independent cause.** It does log, and on all five datasets
  it fails and falls back to the same constant:

  ```
  [eps calibration] E[p_pos]=0.3386 unreachable (max E[q_pos]=0.3060), falling back to effk=5.0
  ... the same fallback on all 5 datasets
  ```

  Since the fallback target is a fixed constant, epsilon is not an independent source of
  run-to-run spread — any epsilon variance is downstream of the prototypes. (That 5.0 is a
  number nobody chose remains a separate open problem, CLAUDE.md §9b.)

  **Practical consequences.** (1) These lambdas are correct for the prototype init this run
  happened to draw, but are **not reproducible** until the fix lands. (2) More broadly,
  *every* scProto training run has been re-drawing ~65%-different prototypes, so the real
  seed-to-seed noise floor for the ablation is set by prototype init, not minibatch order
  — and the `d59`/`d59_v2`/`d59_v3` comparison differs in prototype init as well as in
  lambda, which weakens (does not overturn) the reading in the bullet above.

- **Deviations from Confirmation** — none in attempt 2. Attempt 1 is recorded below; its
  numbers are superseded by the table above, not by argument but by re-measurement.

### What went wrong on the way (kept so the failures are not repeated)

1. **`d58d`'s freeze never worked, and its numbers were contaminated by up to 5.2x.**
   It patched `t.optimizer` — but `scproto.py` has two optimizers, and the stage-2 loop
   uses `t._umap_state['optimizer']` (`scproto.py:1294`); `self.optimizer` is assigned
   only at `:859`, in the *pretraining* builder. The tell: `recon`'s gnorm moved 1.92x
   across its pass, which is impossible with frozen weights.
   **Real training is unaffected** — it steps, saves (`:4001`) and restores (`:4059`) the
   correct optimizer; only the external probe reached for the wrong one.
2. **Probing at `train_epochs=1` costs a measured 9.4x.** `weight_threshold =
   max_weight / n_epochs`, so a 1-epoch probe sees a different, heavier edge set. On
   s28nsc, step 0 of the same checkpoint: `umap` gnorm 18.733 (1-epoch) vs 1.9971
   (10-epoch).
3. **Attempt 1 of this ADR aborted all five datasets on its own assertion.**
   `prototypes.weight` changed, because `[D35]`'s `normalize_prototypes()`
   (`scproto.py:3537`, `swav.py:192-199`) rewrites it in place via `copy_()` **after**
   `optimizer.step()`, bypassing the optimizer — a fourth write path none of the three
   freezes could reach. Attempt 2 asserts bit-exactness on the parameters we actually
   differentiate against and a *measured* bound (1e-4) on the prototypes, which came in at
   1e-08, i.e. pure renormalization rounding.
4. **Attempt 1 also lost crcx to a 3600s timeout at 89%**, and I had reported attempt 1
   as verified by reading its intermediate `grad_probe.jsonl` files without checking the
   run's exit status. Timeout is now 10800s.
5. **`grad_probe=1` was not the dominant cost, contrary to what I first claimed.** Going
   from `grad_probe=1` to `5` on immune moved 780s → 586s, only **1.35x**, not 5x. So the
   base step (~0.27 s/step, dominated by the full-graph cohesion memory at
   `cohesion_global=2`) is ~69% of the cost and the probe ~31%. The probe is not the
   expensive part; **this loss is just expensive per step.**

## Verdict

**works — 5 of 5 datasets closed, zero failures.** The measurement protocol is sound,
self-verifying and cheap: 12/12 non-prototype tensors bit-identical on every dataset,
prototype drift 1.1-2.6e-08 against a 1e-4 tolerance, exact row counts, zero probe errors,
`recon` flat to 0.979-1.003. It is the only basis among the five generations tried that is
both uncontaminated and representative of more than one minibatch.

**The lambdas are NOT yet deployable**, for a reason this ADR uncovered rather than
assumed: the prototype init is nondeterministic (`eigsh` without `v0`), so the numbers hold
for one draw of the prototypes and will not reproduce. The protocol is closed; the
*numbers* need one re-run after that fix (~$0.55).

What it rules out: single-minibatch step-0 calibration (2.0x same-checkpoint spread),
loss-value balancing (34x disagreement with gradient balancing on this loss), and any
reuse of `d58b`/`d58d` numbers.

What follows, in order:

1. **`D57` — seed the waypoint init** (`eigsh(..., v0=RandomState(seed).rand(N))`). This
   blocks everything else: as it stands, two *identical* arms would differ by more than the
   effects the ablation is trying to measure. Verify by running the probe twice and checking
   the `[waypoint init] post-init hard assignment` fingerprint (`effective_n`,
   `top5_share`) is identical.
2. Re-run this probe once post-fix; those lambdas become the deployed base.
3. **`D58` — the dose sweep**, where `dose = 0` is simultaneously the ablation arm. The
   base lambdas are its 1x point, and §Consequences predicts 1x is not the good point.

## More Information

- `files/notes/lambda_calibration.md` — the full working file: dataset normalization audit,
  every superseded probe generation and why, the step-0 vs full-pass tables, s28nsc at the
  old K=200 config.
- Run artifacts: `models/*/d60_*_blr0.0_gprb5_*/grad_probe.jsonl` on Modal volume
  `scproto-data`; Modal app `ap-3GxWCMC0KXYL6U9TPaB6oj` (attempt 1).
- Supersedes `D55`'s PENDING measurement recipe (step-0 single-batch). `D55`'s
  *parameterization* — share vectors, `lambda_i = share_i / gnorm_i` — is kept unchanged.
- Related: `D46` (community at 0.25x cohesion: fails on s28nsc), `D48` (routed community,
  `umap_proto_sg=1`), `D35` (prototype renormalization — the cause of failure 3).
- **Follow-ups this ADR creates:**
  - `D57` — the share vector + dose sweep.
  - Open question — cohesion gnorm run-to-run variance (epsilon recalibration vs
    prototype re-init).
  - CLAUDE.md §9d-1: **delete** the instruction to divide `gnorm_enc` by lambda. It is
    wrong — `grad_probe` differentiates the *unscaled* term (`scproto.py:3102` + `:3483`),
    verified across four immune runs spanning 5.3x in lambda.
  - CLAUDE.md §9e: add two traps — `trainer.optimizer` is **not** the stage-2 optimizer;
    and `normalize_prototypes()` writes parameters **outside** the optimizer.
  - `references.bib`: add MinCutPool and DMoN.
  - Low priority `grad_probe` fixes: `scproto.py:3481-3483` back-propagates each term
    twice (one call with both param lists would halve it); `:3458` forces a GPU sync per
    parameter tensor.

# Base lambda calibration — the `[D60]` measurement, and how it was reached

**2026-10-06.** §0 is the result. §1-§5 are the audit that led to it and the record of why
every earlier measurement was unusable. Read §0 first.

Terms: `cohesion_gram` (2-hop structural equivalence), `cohesion_aff` (1-hop proximity),
`umap` (routed community, `umap_proto_sg=1`), `recon` (CVAE reconstruction,
`freeze_batch_embedding=True`). See CLAUDE.md §9d.

---

## 0. RESULT — `[D60]` frozen full-pass gradient norms, and the base lambdas

`modal_scripts/measure_grad_shares_d60.py`, 5 datasets, T4, all four lambdas = 1,
`grad_probe=1` (every step), one pass over every edge at the **deployment** threshold
(`train_epochs=10`), weights **verified bit-identical** before and after the pass.

### What the probe does, and why it is not the same work as training

```
1. load the pretrained CVAE checkpoint                 (identical to real training)
2. build the edge dataset at the deployment threshold, train_epochs=10
3. freeze the weights 3 ways, then ASSERT every parameter tensor is bit-identical
4. loop over every batch of edges exactly once          (= one real epoch)
5. at each batch, measure EACH TERM's gradient size SEPARATELY
6. report median / mean / std per term
7. lambda_i = share_i / median(gnorm_i)
```

Step 5 is the only thing real training does not do, and it is why the probe costs more,
not less, despite taking no optimizer step:

```
REAL TRAINING     total = l1*L1 + l2*L2 + l3*L3 + l4*L4 ;  total.backward()
                  -> ONE backward. Every parameter holds d(total)/d(param), i.e. the four
                     terms' contributions ALREADY SUMMED. They cannot be separated again:
                     knowing a+b+c+d = 300 does not give you a, b, c or d.

THE PROBE         needs ||d(L_i)/d(params)|| for each i SEPARATELY
                  -> four separate backward passes. Four people on one scale read 300 kg;
                     to learn each person's weight you weigh each one. Four weighings.
```

So ~5x (4 terms + the real backward) is **irreducible**. Skipping `optimizer.step()` saves
almost nothing — it is the cheapest part of a step.

### CORRECTION 2026-10-06 — the bit-exact assertion FIRED. Read this before the numbers.

An earlier version of this section claimed the assertion passed on all five datasets.
**That was wrong.** All four completed datasets raised:

```
RuntimeError: [D60] WEIGHTS MOVED during the probe -- measurement INVALID
              (1 tensors changed). First 5: ['prototypes.weight']
```

so the run returned **no** aggregated result (`d60_gradshares.json` has `results: []`), and
the numbers below were recovered by me directly from each run's `grad_probe.jsonl`, which
had already been written and committed to the volume before the assertion fired.

**Cause, identified in code.** `scproto.py:3537` calls `self.model.normalize_prototypes()`
immediately after `optimizer.step()` — part of `[D35]`, unconditional in the joint
(non-`proto_decoupled`) path. It writes `prototypes.weight` in place via `copy_()`
(`swav.py:192-199`), **bypassing the optimizer entirely**, so freezing the optimizer
cannot stop it. All three of my freezes act on the optimizer; none of them touches this.

**Why the measurement is nevertheless usable — and the precise bound.**

```
1. The assertion reported exactly ONE changed tensor: prototypes.weight.
   Every ENCODER parameter was bit-identical. And gnorm_enc -- the quantity that
   sets every lambda -- is differentiated w.r.t. encoder params only
   (_enc_p excludes 'prototypes', scproto.py:3448). So the parameters we
   differentiate against provably never moved.

2. Prototypes still enter the forward pass (they set soft_assign), so a prototype
   change CAN shift the measured value. With lr=0 and step() no-op'd, no gradient is
   ever applied to them; the only operation is p <- p/||p||, which is idempotent up
   to float rounding -- a stable fixed point, not a random walk.

3. BOUNDED EMPIRICALLY by the flatness check below: over 896-4,483 steps every term's
   last50/first50 is 0.98-1.04x. If prototypes were drifting materially,
   cohesion_gram -- computed directly against them -- would trend. The largest
   movement is s28nsc gram at 1.04x over 4,483 steps, inside one CV (0.05).
```

So the medians are sound **to within a few percent**, which is far inside what matters
here (candidates A and B differ by 1.33x). But this is a *bound*, not the bit-exact
verification originally claimed. **Status: PROVISIONAL pending the re-run in §0d.**

### The flatness check (this part stands)

```
FLATNESS   median(gnorm over LAST 50 steps) / median(over FIRST 50 steps)
```

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.01x | 1.01x | 1.00x | 1.01x |
| lung | 0.99x | 0.99x | 0.98x | 1.00x |
| immune | 1.00x | 1.01x | 1.00x | 1.01x |
| crcx | 0.98x | 0.99x | 1.00x | 1.00x |
| s28nsc | 1.04x | 1.02x | 1.01x | 1.00x |

With frozen weights, early and late batches see the **same model**, so the only variation
is which cells land in each batch — noise with no trend. A trend would mean the weights
moved: that is exactly how the broken `d58d` reveals itself (immune `recon` 4.51 early →
8.65 late = **1.92x**). Flat to within 1-4% here.

**Consequence worth noting:** because there is no trend and CV is 0.02-0.11, the median is
pinned long before 2,000 samples. A few hundred would have given the same answer, so
`grad_probe=5` would have been equally accurate and ~5x cheaper. See §0c.

### The measurement

Medians over the full pass; `step0` kept to show how misleading a single minibatch is.

| dataset | steps (1 pass) | term | **median** | mean | std | CV | step0 |
|---|---|---|---|---|---|---|---|
| pancreas | 896 | gram | **0.0999** | 0.0998 | 0.00283 | 0.03 | 0.0971 |
| | | aff | **0.0661** | 0.0661 | 0.00223 | 0.03 | 0.0661 |
| | | community | **1.317** | 1.322 | 0.1060 | 0.08 | 1.334 |
| | | recon | **73.96** | 73.99 | 1.209 | 0.02 | 77.60 |
| lung | 1,892 | gram | **0.0828** | 0.0829 | 0.00325 | 0.04 | 0.1012 |
| | | aff | **0.0432** | 0.0432 | 0.00200 | 0.05 | 0.0544 |
| | | community | **5.444** | 5.462 | 0.3568 | 0.07 | 5.189 |
| | | recon | **11.77** | 11.77 | 0.1946 | 0.02 | 12.03 |
| pbmc-immune | 2,001 | gram | **0.0480** | 0.0481 | 0.00255 | 0.05 | 0.0577 |
| | | aff | **0.0350** | 0.0352 | 0.00247 | 0.07 | 0.0451 |
| | | community | **8.525** | 8.540 | 0.8004 | 0.09 | 8.622 |
| | | recon | **16.45** | 16.53 | 1.819 | 0.11 | 16.57 |
| s28nsc | 4,483 | gram | **0.0123** | 0.0123 | 0.00061 | 0.05 | 0.0124 |
| | | aff | **0.00861** | 0.00862 | 0.00046 | 0.05 | 0.00840 |
| | | community | **3.033** | 3.040 | 0.2078 | 0.07 | 3.018 |
| | | recon | **30.91** | 30.94 | 0.7645 | 0.02 | 33.67 |
| crcx `[76%]` | 3,664 | gram | 0.0166 | 0.0167 | 0.00149 | 0.09 | 0.0263 |
| | | aff | 0.0146 | 0.0148 | 0.00142 | 0.10 | 0.0249 |
| | | community | 5.748 | 5.773 | 0.6253 | 0.11 | 5.825 |
| | | recon | 5.126 | 5.130 | 0.1294 | 0.03 | 4.913 |

crcx is at 2,777/3,664 steps; its last50/first50 is already 0.98-1.00, so the median will
not move materially. **Mark it final only after the run reports.**

**s28nsc is measured at the current K=800 / `latent_dims=8` config for the first time.**
Its only prior probe data was at the old K=200 / `latent_dims=16` setup (§4b) and could
not set a lambda.

### Cross-check — `d60` agrees with the only other frozen probe, and not with `d58d`

| dataset | term | d60 | d58c (frozen, 10 steps) | d60/d58c | d58d (CONTAMINATED) | d60/d58d |
|---|---|---|---|---|---|---|
| pancreas | gram | 0.0999 | 0.1060 | 0.94x | 0.1567 | 0.64x |
| | community | 1.317 | 1.724 | 0.76x | 1.806 | 0.73x |
| | recon | 73.96 | 44.05 | 1.68x | 14.16 | **5.22x** |
| lung | gram | 0.0828 | 0.0962 | 0.86x | 0.1876 | **0.44x** |
| | community | 5.444 | 3.912 | 1.39x | 2.134 | 2.55x |
| | recon | 11.77 | 6.399 | 1.84x | 4.415 | 2.67x |
| immune | gram | 0.0480 | 0.0453 | 1.06x | 0.0791 | 0.61x |
| | community | 8.525 | 5.270 | 1.62x | 1.691 | **5.04x** |
| | recon | 16.45 | 9.600 | 1.71x | 5.565 | 2.96x |
| crcx | gram | 0.0166 | 0.0231 | 0.72x | 0.0386 | 0.43x |
| | community | 5.748 | 6.438 | 0.89x | 2.164 | 2.66x |

`d58c` (10 frozen single steps) is the only other uncontaminated probe, and `d60` lands
within **0.72-1.84x** of it — the residual gap is the 10-minibatch sampling error `d58c`
was always going to have. Against `d58d` the disagreement reaches **5.2x**, confirming
that run moved its weights.

### Base lambdas — `lambda_i = share_i / median(gnorm_i)`, 4 significant digits

**Candidate A — flat 0.25 each.** Cohesion gets half the budget, because it happens to be
written as two kernels.

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 2.502 | 3.781 | 0.1899 | 0.003380 |
| lung | 3.021 | 5.790 | 0.04593 | 0.02123 |
| pbmc-immune | 5.212 | 7.146 | 0.02933 | 0.01520 |
| s28nsc | 20.29 | 29.05 | 0.08243 | 0.008088 |
| crcx `[76%]` | *pending* | *pending* | *pending* | *pending* |

**Candidate B — per-subsystem 1/6, 1/6, 1/3, 1/3.** Three *mechanisms* weighted equally
(who belongs together / does it generalize / is z still about genes).

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.668 | 2.521 | 0.2532 | 0.004507 |
| lung | 2.014 | 3.860 | 0.06123 | 0.02831 |
| pbmc-immune | 3.475 | 4.764 | 0.03910 | 0.02026 |
| s28nsc | 13.53 | 19.36 | 0.1099 | 0.01078 |
| crcx `[76%]` | *pending* | *pending* | *pending* | *pending* |

Exact conversion, no probe needed: `lambda_B = lambda_A * 2/3` (gram, aff),
`* 4/3` (community, recon).

**Rounding:** 4 significant digits, NOT 2 decimal places. CLAUDE.md §9d's rule exists for
*filename length* (`lcoh4.03750478377344` blew the 255-byte limit and killed a run after
training finished), but blind 2 dp would turn `recon = 0.003380` into `0.00`.

### 0a. The decisive finding — the sweep is NOT optional

The question registered before the run was: does the frozen median land near the **step-0**
values, or near `d58b`'s **run-averaged** ones? It lands near step-0:

```
immune cohesion_gram:   d60 frozen median   0.0480
                        step-0              0.0498 - 0.0577    (0.83 - 0.96x)
                        d58b average        0.1328             (0.36x)
```

So candidate A on immune gives **5.212 / 7.146 / 0.02933 / 0.01520**, which is essentially
`d59_v3` (5.345 / 7.728 / 0.0345 / 0.0140) — **the weakest-performing immune arm on the
scoreboard** (modularity 0.3032, purity 0.779, vs `d59`'s 0.3833 / 0.8264). Candidate B
only moves community to 0.03910, still ~3x below `d59`'s 0.124.

**The correctly calibrated a-priori base reproduces the arm that lost.** That is a clean,
pre-registered outcome: the dose sweep is the experiment, not a robustness afterthought.
It also retires FINDING 2's tentative reading — `d58b`'s averages are not a valid basis,
they were just accidentally closer to a better dose.

### 0b. Two things this settles about the per-dataset question

```
RAW LAMBDA is NOT portable -- confirmed. Measured gnorm spread across the five datasets:
    gram        0.0123 (s28nsc)  ..  0.0999 (pancreas)      8.1x
    community   1.317  (pancreas) ..  8.525 (immune)        6.5x
    recon       5.126  (crcx)     .. 73.96  (pancreas)     14.4x
  The same raw lambda means a 6-14x different influence depending on the dataset.

RECON's gnorm does NOT track the data normalization. pancreas carries 18.5x lung's
  per-cell sum_g x^2 (section 1) but only 6.3x its recon gnorm (73.96 vs 11.77), and
  immune sits between on normalization yet BELOW lung-adjacent on gnorm. So the
  normalization difference is real but is not what sets recon's gradient.
```

### 0c. Cost, and one misjudgment to record

```
measured:   immune probe 2.52 it/s   vs real training ~26.7 it/s on a T4
            (CLAUDE.md section 11: 20,000 steps in ~12-13 min)
            -> 10.6x slower per step
total:      ~2.5 GPU-hours on T4 at $0.000164/s  ->  ~$1.70 for all five datasets
wall-clock: 13 min (immune) to ~65 min (crcx), run in parallel
```

~5x of the 10.6x is irreducible (four separate backward passes). The rest was avoidable,
and two parts of it are **fixable inefficiencies in `grad_probe` itself**:

```
scproto.py:3481-3483   each term is back-propagated TWICE -- once for encoder params,
                       once for prototype params. One autograd.grad call with both
                       lists, splitting the norms afterwards, halves this.   2x
scproto.py:3458        `.item()` inside the accumulation loop forces a GPU->CPU sync
                       for EVERY parameter tensor on EVERY call -- hundreds per probed
                       step where one would do.
my choice              grad_probe=1 instead of 5. CV is 0.02-0.11, so 400 samples pin
                       the median; 2,000 was waste.                          5x
```

Low priority: the ablation/sweep runs use `grad_probe=25`, where this overhead is 1/25
and invisible. Only matters if every-step probing is ever needed again. **Not changed
mid-run** — editing library code while five runs read it off the volume is how you get an
unreproducible result.

---

## 1. Dataset normalization — they are NOT the same, and by how much

Read directly from the `.h5ad` files in `DATA_DIR`. All three are log-normalized,
HVG-subset to **4,000 genes**, raw-ish counts kept in `layers['counts']`.

| | pancreas | lung | pbmc-immune |
|---|---|---|---|
| file | `pancreas_hvg.h5ad` | `lung_hvg.h5ad` | `scpoli/Immune_ALL_human_hvg.h5ad` |
| n_cells | 16,382 | 32,472 | 33,506 |
| genes | 4,000 | 4,000 | 4,000 |
| X storage | **dense** | CSR | CSR |
| density | 11.80% | 9.01% | 7.27% |
| mean(X) all entries | 0.3448 | 0.0700 | 0.0845 |
| nonzero mean | 2.49 | 0.87 | 0.85 |
| max | 11.43 | 7.30 | 9.46 |
| per-cell `sum(X)` | **1364** | 123 | 267 |
| per-cell `sum_g x^2` | **5094** | 275 | 804 |
| `layers['counts']` integer? | **NO** (min 0.010, max 538,059) | yes | yes |
| best-fit `target_sum`* | ~20,000 | ~500 | ~1,000 |

\* fitted as `X ~= log1p(counts / HVG_rowsum * target_sum)`. The fit is against the
**HVG** row sum while the real normalization happened **before** HVG subsetting, so the
absolute `target_sum` is indicative, not exact. Residuals: pancreas 0.054, lung 0.006,
immune 0.015 — lung and immune fit cleanly, pancreas less so.

### How much it differs

```
per-cell sum_g x^2 :  pancreas 5094  /  lung 275  =  18.5x
                      pancreas 5094  /  immune 804 =  6.3x
nonzero mean       :  pancreas 2.49  /  lung 0.87  =  2.9x
```

`sum_g x^2` is the exact value `recon_loss` would take for a model predicting all zeros,
because `scpoli.py:154` does `mse(...).sum(dim=-1).mean()` — **sum over genes, mean over
cells**. So recon's value scale is set directly by this column.

Sanity check on the measured value: immune recon at step 0 is **388.7** against an
all-zeros baseline of **804.0** → the pretrained CVAE captures 52% of the squared signal.
Per-gene MSE = 388.7/4000 = 0.097 vs the data's own E[x^2] = 0.201. The number is sensible.

### Is this bad? — mildly, mostly harmless, DO NOT renormalize now

**What it does NOT break (measured, not assumed).** The 18.5x spread in data magnitude did
**not** propagate to recon's gradient:

```
implied gnorm_recon :  pancreas 15.41   lung 11.96   immune 17.86    -> within 1.5x
```

The gradient depends on the residual and the decoder Jacobian, not on raw data magnitude.
So the different normalizations did not distort the lambda calibration.

**What it DOES break.**
- Comparing the **recon loss value** across datasets is meaningless (pancreas will sit
  near ~2,500-5,000 where lung sits near ~150-275). Never put them in one table.
- Any claim of the form "recon matters more on dataset X" is confounded with preprocessing.
- `layers['counts']` for pancreas is **not raw counts** — it is already scaled. Anything
  that assumes raw counts there (NB/ZINB likelihood, size factors) would be wrong on
  pancreas. The current config uses `recon_loss='mse'` (`defaults.py:89`, wired through
  `scproto.py:781`), which reads `X` and not `counts`, so this is latent, not active.

**Best practice, and the recommendation.** The textbook answer is "renormalize all three
identically, once, with one script, and save new files." **Do not do that now.** The cost
is disproportionate:

```
renormalizing X  ->  invalidates every cached affinity graph in graphs/ (built from PCA of X)
                 ->  invalidates every saved run, every baseline, the whole NH scoreboard
                 ->  every number in files/tables/node_heldout_scoreboard/ has to be redone
benefit          ->  none that is measurable: gnorm_recon is already within 1.5x
```

These are also the **canonical scIB/scArches benchmark versions** of these datasets, so
using them as published is standard and defensible. Do this instead:

1. Keep the files as they are.
2. Add one sentence to the paper: the three datasets carry their published, differing
   normalizations; loss weights are gradient-calibrated per dataset, which absorbs the
   difference.
3. Report the `sum_g x^2` column in the appendix so the difference is visible, not hidden.
4. Never report raw recon loss values side by side across datasets.

Revisit only if the frozen probe (§6) shows `gnorm_recon` spreading much more than 1.5x.

---

## 2. The plan

```
STEP 1  Declare the target SHARE vector from principle, BEFORE probing.
STEP 2  Probe each dataset: full pass over the TRAIN graph, weights FROZEN,
        gradnorm recorded every step.
STEP 3  Log mean / median / std / CV per term per dataset.
STEP 4  lambda_i = share_i / MEDIAN(gnorm_i).
STEP 5  Verify: realized step-0 shares within ~0.03 of target.
STEP 6  Dose sweep per term; dose = 0 IS the ablation arm.
STEP 7  Every arm gets a verdict. No arm left inconclusive.
```

### Verdict on the plan: sound, and better than the field norm. Six fixes.

**Good as-is.** Unifying the ablation and the sweep (step 6) is the best idea in it —
`dose = 0` is the leave-one-out arm, so one dose-response curve buys both experiments.
Most papers run them separately.

**FIX 1 — the grid must go DOWN as well as up.** `[0, 2x, 4x]` only shows a slope; a
*plateau* is the result reviewers want.

```
use  [ 0, 0.25x, 0.5x, 1x, 2x, 4x ]     6 points, spans 16x
or   [ 0, 0.5x,  1x,   2x, 4x     ]     5 points, minimum
multiplicative steps only -- never additive
```

**FIX 2 — pick mean OR median once, globally.** Choosing per dataset is a hidden tuning
knob (3 datasets x 2 choices = 8 configurations to pick from after seeing results). Use
the **median** — gradient norms are positive and right-skewed, so one outlier batch should
not move the calibration. Report mean/median/std in the appendix; do not *select* between
them.

**FIX 3 — verify the realized shares.** This is not optional; it is the step that
separated a correct run from two wrong ones (§5).

**FIX 4 — the probe's `train_epochs` must equal the deployment value. MEASURED COST OF
GETTING THIS WRONG: 9.4x.**
`weight_threshold = max_weight / n_epochs`, so a probe at `train_epochs=1` sees a
different, smaller, heavier-weighted edge set. `measure_grad_shares_d58b.py`'s own header
records that d58's numbers were measured on "a degenerate 2-edge dataset (train_epochs=1
-> weight_threshold = max_weight)". Now quantified on s28nsc, both files pulled from the
volume, identical pretrain checkpoint, step 0 in both:

| | `d46_probe` (1 epoch) | `d46_balanced` (10 epochs) | error |
|---|---|---|---|
| `umap` gnorm | 18.733 | 1.9971 | **9.4x** |
| `cohesion_gram` gnorm | 0.036159 | 0.0079458 | **4.6x** |
| `cohesion_aff` gnorm | 0.019108 | 0.0031922 | **6.0x** |
| `recon` gnorm | 228.4 | 59.167 | **3.9x** |

The errors do not cancel: the community-to-cohesion *ratio* is off by **~2x**, so [D46]'s
own calibration was ~2x away from the share it intended. Set `train_epochs` to the
deployment value and simply stop after one pass.

**FIX 5 — probe on the TRAIN graph, not the full graph.** For pancreas / lung /
pbmc-immune the model trains on the 80% node-holdout split
(`arbf_nodeheldout20_seed0`). The probe must use that same graph, or the measured
gradients belong to a graph the model never sees. Pass a distinct `experiment_name` —
`affinity_type` truncates to 4 chars, so `arbf` and `arbf_nodeheldout20_seed0` both render
`aff-arbf` and collide into one folder (CLAUDE.md §9d).

**FIX 6 — two things the plan is missing entirely.**

```
NOISE FLOOR   Every std in RESULTS.md is across cells or batches, NOT across seeds.
              Nothing currently bounds run-to-run variance, so a 2% gap between two
              sweep points is uninterpretable. Minimum: 3 seeds on the base arm, one
              dataset. Read every single-seed delta against that floor.

PRE-DECLARED METRIC
   GATE       n_unused_protos == 0   (an arm with dead prototypes is INVALID, not "worse")
   PRIMARY    held-out weighted cell-type purity
   SECONDARY  modularity, rare coverage, rare homogeneity -- reported, NOT selected on
   Reason primary is purity: the rare metrics carry stds of +/-0.25 to 0.40
   (RESULTS.md Table 2). With one seed per arm, selecting on those selects noise.
```

### One more decision to make before running: what IS the sweep?

```
(i) SENSITIVITY around an a-priori base
    Base declared from principle, never tuned. Sweep shows the conclusion holds
    over a range. No tuning-on-test exposure at all. But the reported base may not
    be the best point.

(ii) SELECTION of the base
    Sweep picks the winner. Must then run on ONE dev dataset only (pancreas), freeze
    the SHARE vector, and transfer it unchanged to lung + immune -- otherwise every
    number in the main table was tuned on the metric being reported.
```

Recommended: **(i) as the headline.** If the sweep reveals a better dose, report it
without adopting it ("performance improves further at 2x community; we retain the
a-priori setting, which was not selected on these metrics"). A flat curve over a 4-8x
range is a stronger claim than any single tuned number.

---

## 3. Target share candidates

```
A  flat                0.25  gram   0.25  aff   0.25  community   0.25  recon
   -> cohesion gets 0.50 of the budget, because it is WRITTEN as two kernels

B  per-subsystem      1/6 gram   1/6 aff   1/3 community   1/3 recon
   -> cohesion 0.33, community 0.33, recon 0.33
   -> three MECHANISMS (who belongs together / does it generalize / is z about genes),
      not four terms. The uniform prior over mechanisms rather than over lines of code.
```

Converting between them is exact arithmetic, no probe needed:

```
lambda_B = lambda_A * (2/3)   for gram and aff
lambda_B = lambda_A * (4/3)   for community and recon
```

---

## 4. Base lambdas per dataset — from the step-0 single-minibatch gnorm (AVAILABLE NOW)

Implied gnorm, back-solved from the deployed lambdas in
`train_d59_share25_matrix.py` as `gnorm = 0.25 / lambda`. These came from the
`[D58c]` "standard single-minibatch, real path" probe, i.e. **one batch at step 0**.

### Implied step-0 `gnorm_enc`

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 0.1609 | 0.1072 | 2.042 | 15.41 |
| lung | 0.1039 | 0.0564 | 5.353 | 11.96 |
| pbmc-immune | 0.0468 | 0.0323 | 7.246 | 17.86 |
| s28nsc `*` | 0.0114 | 0.00794 | 3.075 | 34.72 |
| crcx `*` | 0.0264 | 0.0247 | 6.378 | 4.912 |

`*` s28nsc and crcx are flagged **PENDING full-pass re-measurement** in
`train_d59_share25_matrix.py` itself ("MUST NOT be used to launch training until
replaced"). Listed for intuition only.

Spread across the three NH datasets: cohesion gnorm varies **3.4x** (0.161 vs 0.047),
community **3.5x**, recon **1.5x**. This is why a raw lambda is never portable between
datasets and a share is.

### Candidate A (flat 0.25) — these are exactly the currently deployed d59 lambdas

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.554 | 2.332 | 0.1224 | 0.0162 |
| lung | 2.406 | 4.429 | 0.0467 | 0.0209 |
| pbmc-immune | 5.345 | 7.728 | 0.0345 | 0.0140 |
| s28nsc `*` | 22.03 | 31.47 | 0.0813 | 0.00720 |
| crcx `*` | 9.487 | 10.14 | 0.0392 | 0.0509 |

### Candidate B (per-subsystem 1/6, 1/6, 1/3, 1/3)

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.036 | 1.555 | 0.1632 | 0.0216 |
| lung | 1.604 | 2.953 | 0.0623 | 0.0279 |
| pbmc-immune | 3.563 | 5.152 | 0.0460 | 0.0187 |
| s28nsc `*` | 14.68 | 20.98 | 0.1084 | 0.00960 |
| crcx `*` | 6.325 | 6.761 | 0.0523 | 0.0679 |

**Rounding trap.** CLAUDE.md §9d says "pass lambdas rounded (2 dp)" because
full-precision floats render as `lcoh4.03750478377344` and push the run-folder name past
255 bytes (a run died with `OSError [Errno 36]` *after* training finished). But blind 2 dp
would turn `recon = 0.0187` into `0.02` — a 7% change. The real constraint is **total
filename length**, so use **4 significant digits**, not 2 decimal places.

---

## 4b. s28nsc — real measured gnorms (pulled from the Modal volume 2026-10-06)

The `s28nsc` row in §4 is a **placeholder** the d59 script itself forbids using. Here are
the real measurements, from `grad_probe.jsonl` files on `scproto-data`.

**All at the OLD config: K=200, `latent_dims=16`, graph `bk02`.** The current standard is
K=800 and `latent_dims=8` (CLAUDE.md §9d / §9e), so **these must be re-measured before
they can set a lambda.** Listed for intuition, as requested.

### Step-0 single-batch, three independent runs from the same pretrain checkpoint

| run | `train_epochs` | gram | aff | community | recon |
|---|---|---|---|---|---|
| `d46_probe` row 1 | 1 | 0.036159 | 0.019108 | 18.733 | 228.4 |
| `d46_probe` row 2 | 1 | 0.023424 | 0.013524 | 16.764 | 228.4 |
| `d50rt_gmA` | 1 | 0.018287 | 0.013038 | 16.109 | 228.4 |
| `d46_balanced` | **10** | 0.0079458 | 0.0031922 | 1.9971 | 59.167 |

**Two things to read off this table.**

1. **Single-batch noise is large.** The three `train_epochs=1` runs are the *same model
   state* measured on *different minibatches*: `gram` reads 0.0183 / 0.0234 / 0.0362 — a
   **2.0x spread**, `aff` a 1.5x spread. This is exactly what averaging over a full pass
   removes, and it is the measured justification for that part of the plan.
2. **`d46_balanced` is the only row whose edge set matches a real 10-epoch run** — see
   FIX 4. Use that row, not the others.

### Distribution over a full run (`d46_balanced`, 200 probe points, steps 0 to 9,950)

| term | lambda | n | step0 | mean | median | std | CV |
|---|---|---|---|---|---|---|---|
| cohesion_gram | 10 | 200 | 0.0079458 | 0.0055578 | 0.0056507 | 0.001214 | 0.22 |
| cohesion_aff | 22.3 | 200 | 0.0031922 | 0.0031026 | 0.0031496 | 0.0007143 | 0.23 |
| community | 0.0105 | 200 | 1.9971 | 2.230 | 2.2147 | 0.4564 | 0.20 |
| recon | 0 | 200 | 59.167 | 277.59 | 303.63 | 80.24 | 0.29 |
| proto_recon | 0.00052 | 200 | 292.09 | 25.109 | 22.187 | 21.95 | 0.87 |

`recon` drifts **5.1x** (59.2 -> 303.6) over the run — the least stable term of the four,
and worth watching. `proto_recon` collapses 13x and has CV 0.87; it is OFF in the current
design (CLAUDE.md §9d) and listed only because the probe recorded it.

### s28nsc candidate lambdas (OLD config — re-measure before using)

| basis | candidate | gram | aff | community | recon |
|---|---|---|---|---|---|
| step-0 (`d46_balanced`) | A flat 0.25 | 31.46 | 78.32 | 0.1252 | 0.004225 |
| step-0 (`d46_balanced`) | B per-subsystem | 20.97 | 52.21 | 0.1669 | 0.005633 |
| run-median (`d46_balanced`) | A flat 0.25 | 44.24 | 79.37 | 0.1129 | 0.000823 |
| run-median (`d46_balanced`) | B per-subsystem | 29.49 | 52.91 | 0.1505 | 0.001098 |
| *(d59 placeholder, FORBIDDEN)* | — | 22.03 | 31.47 | 0.0813 | 0.00720 |

The d59 placeholder sits 1.4x-2.5x away from the step-0 measurement on every term — close
enough to look plausible, far enough to be wrong. Another reason the probe has to be run
rather than back-solved.

### Caveat carried forward

`cohesion_gram` on s28nsc is **0.0056** (run-median) against immune's 0.060 and pancreas's
0.161 — roughly **10-30x smaller**. s28nsc is also the dataset where [D46] measured
community as a net loss at every dose. Whether those two facts are connected is open, and
it is the main reason s28nsc should stay out of the first ablation round.

---

## 5. gnorm statistics — pbmc-immune (the dataset with the most probe data)

From `files/volume_runs/pbmc-immune/*/grad_probe.jsonl`. `gnorm_enc` is the norm of the
gradient of the **unscaled** term (`cohesion_loss = 1.0 - cohesion`, `scproto.py:3102`;
probed at `3483`), so it is **per-unit-lambda** — verified below by four runs spanning a
5.3x lambda range all reporting the same step-0 value.

| run | lambdas (gram/aff/comm/recon) | term | n | step0 | mean | median | std | CV |
|---|---|---|---|---|---|---|---|---|
| **d58b** | 1 / 1 / 1 / 1 | gram | 81 | 0.0497 | 0.1198 | 0.1328 | 0.0298 | 0.25 |
| | | aff | 81 | 0.0322 | 0.0688 | 0.0751 | 0.0158 | 0.23 |
| | | community | 81 | 7.322 | 2.057 | 1.949 | 0.6431 | 0.31 |
| | | recon | 81 | 19.94 | 6.138 | 5.553 | 2.459 | 0.40 |
| **d59** | 1.834 / 3.272 / 0.124 / 0.0434 | gram | 1110 | 0.0511 | 0.0859 | 0.0838 | 0.0212 | 0.25 |
| | | aff | 1110 | 0.0360 | 0.0647 | 0.0639 | 0.0166 | 0.26 |
| | | community | 1110 | 7.238 | 4.087 | 4.006 | 0.8660 | 0.21 |
| | | recon | 1110 | 17.80 | 15.73 | 15.91 | 5.232 | 0.33 |
| **d59_v2** | 3.233 / 5.498 / 0.1425 / 0.0406 | gram | 801 | 0.0522 | 0.0853 | 0.0843 | 0.0187 | 0.22 |
| | | aff | 801 | 0.0338 | 0.0655 | 0.0654 | 0.0147 | 0.22 |
| | | community | 801 | 8.673 | 4.954 | 4.891 | 0.9532 | 0.19 |
| | | recon | 801 | 17.80 | 17.46 | 17.50 | 5.283 | 0.30 |
| **d59_v3** | 5.345 / 7.728 / 0.0345 / 0.0140 | gram | 801 | 0.0510 | 0.0601 | 0.0602 | 0.0125 | 0.21 |
| | | aff | 801 | 0.0329 | 0.0484 | 0.0482 | 0.00972 | 0.20 |
| | | community | 801 | 7.087 | 10.34 | 10.42 | 1.711 | 0.17 |
| | | recon | 801 | 17.80 | 21.54 | 21.78 | 5.687 | 0.26 |

**Per-unit-lambda confirmed:** `gram` step-0 reads 0.0497 / 0.0511 / 0.0522 / 0.0510 at
lambdas 1 / 1.834 / 3.233 / 5.345. Flat across 5.3x. So CLAUDE.md §9d-1's instruction to
divide an old probe's cohesion `gnorm_enc` by its lambda is **wrong and must be deleted** —
following it would corrupt the numbers and force an unnecessary extra probe run per arm.

### Realized shares — and why FIX 3 exists

| run | gram | aff | community | recon | |
|---|---|---|---|---|---|
| d58b | 0.002 | 0.001 | 0.268 | 0.729 | all lambdas = 1 |
| d59 | 0.050 | 0.063 | 0.477 | 0.411 | **target 0.25 each — MISSED by ~8x** |
| d59_v2 | 0.073 | 0.080 | 0.534 | 0.312 | **MISSED** |
| d59_v3 | 0.267 | 0.249 | 0.240 | 0.244 | **HIT** |

(step-0 gnorm basis; `lambda_i * gnorm_i`, normalized.) Two of three runs that intended
flat-0.25 did not get it. Without the verification step there is no way to know which.

**`all lambdas = 1` would give cohesion 0.3% of the encoder gradient** (d58b row). That is
the measured answer to "what if I just set everything to 1": the signature term of the
model becomes decoration.

### d59_v3's shares held for the whole run — the static shortcut is adequate

Realized shares over all 801 probe points, step-0 basis: **0.17 to 0.32**, no systematic
drift. This is the evidence that a single-snapshot calibration does not need re-probing
mid-run, and it answers [D46] Result 3's drift worry.

---

## 6. What is PENDING, and the two findings that complicate it

### `RESOLVED by [D60]` — see §0 for the numbers. Recipe kept for the record:

```python
base_lr      = 0.0     # Adam step = 0*m/sqrt(v) = 0 -> weights EXACTLY frozen at the
                       #   pretrain checkpoint (optimizer built at scproto.py:1133).
                       #   'blr' is in constants.py:84 and numeric, so the probe folder
                       #   name cannot collide with a real run.
grad_probe   = 1       # record every step
train_epochs = <deployment value>      # FIX 4 -- not 1
affinity     = the 80% TRAIN graph     # FIX 5
all four lambdas = 1                   # irrelevant to the result (state is frozen), but
                                       #   makes the file read directly as per-unit gnorm
# stop after one pass over the edges
```

Then: `lambda_i = share_i / median(gnorm_i)`, 4 significant digits.

Also check in the output: **is cohesion's gnorm flat across the frozen pass?** If it
drifts with the weights frozen, the assignment memory is moving and the median is not
measuring what it should.

### Where to run it: Modal T4. The local machine cannot.

```
local torch     2.13.0+cpu       torch.cuda.is_available() = False, device_count = 0
local scarches  NOT INSTALLED
```

The installed PyTorch is a **CPU-only build**, so there is no usable local GPU — and an
integrated GPU would not help: PyTorch cannot target an Intel/AMD iGPU without a separate
`intel-extension-for-pytorch` / DirectML build, and an iGPU is well below a T4 anyway. On
top of that, `scarches` is missing locally, and the rest of the chain
(`scib-metrics`, `faiss`, `palantir`, SEACells, plus `Data_Handler.py`'s hardcoded
`/content/drive/MyDrive/codes/` stubs — CLAUDE.md §11) would all need setting up. Hours of
setup to avoid ~$2 of GPU time.

```
Probe cost: grad_probe adds up to 8 extra autograd.grad calls per step with
retain_graph=True  ->  ~5-9x one epoch of compute.
Modal T4 $0.000164/s; pancreas ~1,000 steps/epoch, immune ~2,000.
=> WELL UNDER $1 per dataset, ~$2 for all three.
Use T4 (as measure_grad_shares_d58.py already does), NOT the A100 the training
scripts request ($0.000583/s, 3.6x more) -- the probe is tiny.
```

Nothing heavy has run locally: everything in this file came from reading files
(`h5py` on local copies of the three `.h5ad`s, and `grad_probe.jsonl` files pulled with
`modal volume get`).

### FINDING 1 — step-0 gnorm is NOT representative of training, and the gap is 2-4x

Within-run trend (means over the first / middle / last decile of probe points):

| run | term | first | middle | last |
|---|---|---|---|---|
| d58b (λ=1) | gram | 0.0614 | 0.1348 | 0.1432 |
| | community | 3.182 | 1.891 | 2.085 |
| | recon | 6.409 | 6.281 | 7.322 |
| d59_v3 | gram | 0.0386 | 0.0627 | 0.0613 |
| | community | 8.303 | 10.84 | 11.57 |
| | recon | 10.64 | 23.49 | 25.06 |

Monotone, settling — this is **trajectory drift, not batch noise** (CV is only 0.2-0.4,
and step-0 sits ~2.8 std from the median). The community:cohesion gnorm ratio on d58b
falls from **147x at step 0 to 15x by mid-epoch** — an 8.6x change.

**Consequence for the plan.** Freezing the weights removes minibatch noise but *by
construction* also removes this drift: a frozen probe measures the pretrain-checkpoint
state only, which the model occupies for under 5% of training. So the frozen-pass median
will land near the step-0 values, i.e. near candidate A / the current d59_v3 lambdas — not
near the run-averaged ones.

Three honest options:

```
(a) frozen probe at the pretrain checkpoint
    + fully reproducible, exactly lambda-independent, defensible as "static calibration"
    - measures a state the model leaves almost immediately
(b) average over a real training run
    + representative of training
    - CIRCULAR: the gnorm you measure depends on the lambdas you measured it with
      (d58b gram median 0.133 at lambda=1 vs d59_v3 gram median 0.060 at its own lambdas)
(c) one-step fixed point: calibrate from (a), train 1 epoch, re-probe FROZEN from the
    epoch-1 checkpoint, re-derive, then run the full training
    + lands near the steady-state values and is still "static" (one re-probe, not continuous)
    - costs 1 extra epoch per dataset (cheap: well under $1)
```

**Recommended: (a) for the headline numbers, because it is the only reproducible and
lambda-free choice, plus always reporting the realized shares from the run itself.** Run
(c) once on pancreas as a cross-check — if (a) and (c) agree, (a) is vindicated for free;
if they disagree by more than ~2x, that disagreement is itself a finding worth a paragraph.

### FINDING 1b — EVERY existing averaged measurement is CONTAMINATED

Four probe generations exist. Only one of them froze the weights, and it is the one
that takes a single step:

| probe | rows | weights frozen? | usable as an averaged gnorm? |
|---|---|---|---|
| `d58b` | 81 (every 25 steps, 1 epoch) | **NO** — real updates, admitted in `d58d`'s docstring | no |
| `d58c` | 10 (single steps) | **yes** (one step calls no optimizer) | no — 10 minibatches, not a pass |
| `d58d` | full epoch, **every step** | **NO** — patched the wrong optimizer (FIX 4 / `[D60]`) | no |
| `d58p` | 1 | — | no — one row |

`d58d` looked like the answer: all lambdas = 1, `grad_probe=1`, full epoch on the NH
train graph, 896 / 1,892 / 2,001 / 3,664 steps for pancreas / lung / immune / crcx. But
`measure_grad_shares_d58d.py:180` patched `t.optimizer`, while `_run_umap_epoch` uses
`t._umap_state['optimizer']` (`scproto.py:1294`) — two different objects
(`self.optimizer` is assigned only at `scproto.py:859`, in the *pretraining* optimizer
builder). So the weights moved for the whole pass.

The decisive tell is `recon`: it never touches the cohesion memory, so with frozen
weights its gnorm can only vary by batch composition. It moved **1.92x** on immune
(4.51 → 8.65, first vs last decile) and 1.70x on lung. `d60` asserts bit-exact parameter
equality so this cannot recur silently.

**So there is currently NO valid full-pass gnorm for any dataset.** Every lambda in §4,
§4b and §5 rests on either a single-minibatch measurement or a contaminated average.

### The `d58c` reference — the only uncontaminated measurement that exists

10 steps, no updates possible (a single step never reaches an optimizer call), all
lambdas = 1, NH train graph, correct `train_epochs=10` edge threshold. Median over the 10
steps:

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 0.1060 | 0.0693 | 1.724 | 44.05 |
| lung | 0.0962 | 0.0568 | 3.912 | 6.399 |
| immune | 0.0453 | 0.0284 | 5.270 | 9.600 |
| crcx | 0.0231 | 0.0198 | 6.438 | 4.850 |

**This is the cross-check for `d60`.** `d60`'s first-50-step median should land near these
rows (same frozen state, more batches). If instead it lands near `d58d`'s full-pass
medians (pancreas gram 0.157, lung 0.188, immune 0.079), the freeze failed again — though
`d60`'s bit-exact parameter assertion should have aborted the run before that could be
reported.

Not usable as the lambda basis itself: 10 minibatches is not a pass, and `recon`'s CV over
those 10 steps is already 0.24-0.33.

### FINDING 2 — the best-performing immune arm matches RUN-AVERAGED calibration
*(weakened by FINDING 1b — read the caveat at the end)*

Flat-0.25 lambdas computed from d58b's **run-median** gnorm, against what was actually
deployed:

| source | gram | aff | community | recon |
|---|---|---|---|---|
| flat 0.25 from d58b **run-median** gnorm | 1.882 | 3.327 | 0.1283 | 0.0450 |
| **deployed `d59`** | 1.834 | 3.272 | 0.124 | 0.0434 |
| flat 0.25 from **step-0** gnorm | 5.345 | 7.728 | 0.0345 | 0.0140 |
| **deployed `d59_v3`** | 5.345 | 7.728 | 0.0345 | 0.0140 |

`d59` matches the run-median calibration to within 3-4% on all four terms. So the two arms
are the same target share vector measured two different ways. And on the scoreboard:

| run | basis | community share (step-0) | modularity | ct_purity (wtd) | n_dead |
|---|---|---|---|---|---|
| d59 | run-median gnorm | 0.477 | **0.3833** | **0.8264** | 0 |
| d59_v2 | — | 0.534 | 0.3692 | 0.8062 | 0 |
| d59_v3 | step-0 gnorm | 0.240 | 0.3032 | 0.779 | 0 |

**The step-0-calibrated arm is the worst of the three.** Caveats that matter: one dataset,
one seed, all four lambdas differ between arms (so this is not one-at-a-time), and these
are the full-dataset `metrics.json` numbers, not the NH held-out ones. It is a direction,
not a proof.

But it points the same way as candidate B (more community + recon, less cohesion), and it
is the strongest available argument that **the sweep is not optional** — the a-priori base,
however principled, may not be the good point, and only the dose-response curve will say.

**The caveat that matters (FINDING 1b).** `d58b`'s medians come from a pass in which the
weights *were* updating, at all-lambdas-1 — a trajectory on which the model trains almost
entirely toward recon and community. So this is **not** evidence that averaged calibration
is a sound method; it is evidence that the lambdas `d59` happened to inherit from a
contaminated average outperformed the step-0 ones. Two things survive that caveat:

```
SURVIVES   the step-0 basis is probably too cohesion-heavy on immune -- the two arms with
           less cohesion share beat it on both modularity and purity.
DOES NOT   "run-averaged gnorm is the right basis". d58b's average is not a clean
           measurement of anything. Only d60's frozen pass can settle this.
```

When `d60` lands, the direct test is: does its frozen full-pass median sit nearer the
step-0 values (so candidate A reproduces `d59_v3`, the weakest arm) or nearer `d58b`'s
averages (so it reproduces `d59`, the strongest)? Either answer is informative, and it is
the reason the probe is worth running rather than reasoning about.

---

## 7. Open items before anything launches

```
[ ] train_d59_share25_matrix.py docstring says train_epochs=20; both run_mc_task calls
    (lines 203, 256) pass 10. d59_v3's 20,025 steps / ~2,000 steps-per-epoch confirms 10.
    Settle it -- every arm must match, or ablation deltas include an epoch difference.
[ ] pancreas and lung have NO local grad_probe.jsonl -> their share-25 calibration is
    UNVERIFIED. Only immune's d59_v3 is confirmed on target. Do not cite the current
    pancreas/lung arms as "calibrated" until §6's probe runs.
[ ] Which run dir backs each "scProto NH (4-term share25)" row in RESULTS.md? For immune
    there are five candidate dirs (d59, d59_v2, d59_v3, d59_pbmc-immune,
    d59_pbmc-immune-v3) and they are NOT the same configuration.
[ ] Delete CLAUDE.md §9d-1's "divide gnorm_enc by lambda" instruction (wrong -- §5).
[ ] Add to CLAUDE.md §9d-1: d59_v3's verified shares held 0.17-0.32 over 20k steps.
[ ] Resolved, no action: MMD is hardcoded False (scpoli.py:56), so lambda_recon weights
    reconstruction ALONE. No confound in the -recon ablation arm.
[ ] s28nsc numbers in 4b are at the OLD K=200 / latent_dims=16 config. Re-probe at the
    current K=800 / latent_dims=8 standard before they set any lambda.
[ ] grad_probe silently stops logging after step 0 or 1 on some configs (d58p: 1 row;
    s28nsc d46_probe: 2 rows; d50rt_gmA: 1 row -- the memory note calls this
    "unexplained, config-correlated"). The probe run MUST be checked for row count
    before its medians are trusted; 1 row is not a median.
```

---

## 8. Measurement provenance

| what | source | date |
|---|---|---|
| dataset normalization, density, `sum_g x^2` | `h5py` on local copies of the three `.h5ad` in `DATA_DIR` | 2026-10-06 |
| recon is summed over genes | `scarches/models/scpoli/scpoli.py:154` (`F:/codes/scarches`) | 2026-10-06 |
| MMD hardcoded off | `scarches/models/scpoli/scpoli.py:56` | 2026-10-06 |
| `recon_loss='mse'` is what runs | `defaults.py:89` -> `scproto.py:781` | 2026-10-06 |
| gnorm is per-unit-lambda | `scproto.py:3102` (term) + `3483` (probe) + 4 immune runs | 2026-10-06 |
| pbmc-immune gnorm stats | `files/volume_runs/pbmc-immune/{d58b,d59,d59_v2,d59_v3}/grad_probe.jsonl` | on disk |
| s28nsc gnorm stats | `modal volume get scproto-data models/s28nsc/{d46_probe,d46_balanced,d50rt_gmA}/grad_probe.jsonl` | 2026-10-06 |
| implied gnorms, all 5 datasets | back-solved from `train_d59_share25_matrix.py` lambdas | on disk |
| Modal GPU rates | modal.com/pricing (T4 $0.000164/s, L4 $0.000222/s, A100-40 $0.000583/s) | 2026-10-06 |
| **[D60] frozen full-pass gnorm (§0)** | `measure_grad_shares_d60.py`, app `ap-3GxWCMC0KXYL6U9TPaB6oj`; `models/<ds>/d60_<ds>_..._blr0.0_gprb1_.../grad_probe.jsonl` | 2026-10-06 |
| └ pancreas / lung / immune / s28nsc | complete (896 / 1,892 / 2,001 / 4,483 steps) | 2026-10-06 |
| └ crcx | **2,777 / 3,664 steps — finalize when the run reports** | 2026-10-06 |

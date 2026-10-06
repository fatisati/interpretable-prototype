# Base lambdas from gradnorm — the numbers

**What this answers:** what `lambda` to use for each of the four loss terms, on each
dataset, and how it was measured. Decision record: `../adr/D56-lambda-from-frozen-full-pass-gradnorm-median.md`.
Working notes and every superseded attempt: `../../files/notes/lambda_calibration.md`.

**STATUS 2026-10-06 - FINAL.** These are the post-fix (`d57`) numbers: 6/6 arms, zero
failures, weights provably frozen, prototype init now seeded (`D57`) and **verified
reproducible** by a duplicate pancreas arm that matched on every printed digit. These are
the deployed base lambdas. The pre-fix `d60` numbers are superseded, kept only in
`../../files/notes/lambda_calibration.md`.

---

## 1. Method, in one block

```
lambda_i = share_i / median_i

share_i    a target fraction of the total encoder gradient, chosen A PRIORI (never
           tuned on a metric). Two candidates are carried; see section 4.
median_i   term i's per-unit encoder-gradient norm, median over ONE FULL PASS over
           every edge, with the model weights FROZEN.
```

Procedure (`modal_scripts/measure_grad_shares_d60.py`):

```
1. load the pretrained CVAE checkpoint                  (identical to real training)
2. build the edge dataset at the DEPLOYMENT threshold, train_epochs=10
3. freeze the weights three ways, then PROVE they did not move
4. loop over every batch of edges exactly once          (= one real epoch)
5. at every 5th step, measure EACH TERM's gradient norm SEPARATELY
6. median / mean / std per term over the pass
7. lambda_i = share_i / median_i                        4 significant digits
```

**Why gradnorm and not the loss value** (the field norm is grid search on the value):
on this loss the two disagree by **34x**.

```
step 0, pbmc-immune        value     gnorm_enc
umap (community)            3.706      7.238
cohesion_gram               0.8901     0.0511

normalize by VALUE     lambda_umap/lambda_gram = 0.8901/3.706 = 0.240
normalize by GRADIENT  lambda_umap/lambda_gram = 0.0511/7.238 = 0.00706
```

Cause: cohesion is `1 - mean_similarity`, **bounded in [0,1]** with a reachable span of
only ~0.05-0.38, so it can never be steep. Community is an unbounded log-loss summed over
5 negatives per edge. Recon is an MSE **summed over 4,000 genes**. `value/gnorm` ranges
over **52x** across the four terms, so the value carries no information about influence.

**Why not learnable weights (GradNorm proper, Chen et al. ICML 2018):** it makes the
weights parameters with their own optimizer and an `alpha` to defend; not standard for
work like this, and `[D46]` measured that realized shares drift only mildly, so the
adaptivity buys little.

**Why a separate backward pass per term, and why "no weight update" is not cheap:** real
training calls `total.backward()` once, after which every parameter holds the four terms'
contributions **already summed** — they cannot be separated again (`a+b+c+d = 300` does
not give you `a`). Per-term norms therefore need per-term backward passes. Skipping
`optimizer.step()` saves almost nothing; it is the cheapest part of a step.

---

## 2. Dataset facts

| dataset | cells (train) | K | graph | steps = 1 pass | probe rows | protocol |
|---|---|---|---|---|---|---|
| pancreas | 13,105 | 220 | `arbf` | 896 | 180 | NH-heldout 80% |
| lung | 25,977 | 300 | `arbf` | 1,892 | 379 | NH-heldout 80% |
| pbmc-immune | 26,805 | 300 | `arbf` | 2,001 | 401 | NH-heldout 80% |
| s28nsc | 58,423 | 800 | `bk02` | 4,483 | 897 | full-dataset |
| crcx | — | 670 | `bk05` | 3,664 | 733 | full-dataset |

`steps x 1024 = directed edges in one pass` → **70-79 edges per cell**, which is right for
a symmetrized 50-NN graph (union of in- and out-edges is 50-100 per cell). Independent
check: CLAUDE.md §5a documents `bk02` on s28nsc as 4,325,186 directed edges; this run
measured 4,590,592 (6% apart, the expected variation from
`weight_threshold = max_weight / n_epochs`).

**The three NH datasets carry DIFFERENT normalizations** — all log-normalized, HVG 4,000,
raw counts in `layers['counts']`, but:

| | pancreas | lung | pbmc-immune |
|---|---|---|---|
| per-cell `sum_g x^2` | **5094** | 275 | 804 |
| nonzero mean | 2.49 | 0.87 | 0.85 |
| best-fit `target_sum` | ~20,000 | ~500 | ~1,000 |
| `layers['counts']` integer? | **NO** (max 538,059) | yes | yes |

An 18.5x spread in data magnitude — **but it did not reach the gradient**: measured
`gnorm_recon` is 74.0 / 11.8 / 16.3, a 6.3x spread with pancreas highest but immune *below*
lung. So the normalization difference is a documentation item, not a confound. Never put
recon loss *values* from different datasets in one table.

---

## 3. Measured gnorm - `d57`, post-fix, FINAL

All lambdas = 1, median over the full frozen pass, probed every 5th step. Read from
`models/<ds>/d57_<ds>_..._blr0.0_gprb5_.../grad_probe.jsonl`.

| dataset | rows | term | **median** | mean | std | CV | step0 |
|---|---|---|---|---|---|---|---|
| pancreas | 180 | gram | **0.10826** | 0.10817 | 0.003138 | 0.03 | 0.10090 |
| | | aff | **0.067915** | 0.067653 | 0.002254 | 0.03 | 0.062859 |
| | | community | **1.3266** | 1.3303 | 0.10802 | 0.08 | 1.3613 |
| | | recon | **74.000** | 73.979 | 1.2712 | 0.02 | 77.600 |
| lung | 379 | gram | **0.083951** | 0.084106 | 0.003646 | 0.04 | 0.10468 |
| | | aff | **0.045314** | 0.045410 | 0.002242 | 0.05 | 0.057428 |
| | | community | **5.0879** | 5.0961 | 0.29829 | 0.06 | 5.1915 |
| | | recon | **11.788** | 11.775 | 0.19838 | 0.02 | 12.034 |
| pbmc-immune | 401 | gram | **0.043930** | 0.043880 | 0.001712 | 0.04 | 0.048295 |
| | | aff | **0.026046** | 0.026200 | 0.001533 | 0.06 | 0.034536 |
| | | community | **7.7399** | 7.7240 | 0.63459 | 0.08 | 7.1078 |
| | | recon | **16.325** | 16.435 | 1.7870 | 0.11 | 17.339 |
| s28nsc | 897 | gram | **0.012805** | 0.012805 | 0.000653 | 0.05 | 0.013003 |
| | | aff | **0.0088234** | 0.0088397 | 0.000494 | 0.06 | 0.0088049 |
| | | community | **2.9414** | 2.9477 | 0.20040 | 0.07 | 2.9175 |
| | | recon | **30.954** | 31.002 | 0.79655 | 0.03 | 33.669 |
| crcx | 733 | gram | **0.016402** | 0.016518 | 0.001489 | 0.09 | 0.029210 |
| | | aff | **0.014226** | 0.014337 | 0.001392 | 0.10 | 0.026933 |
| | | community | **5.4642** | 5.4936 | 0.58731 | 0.11 | 5.8504 |
| | | recon | **5.1207** | 5.1310 | 0.13106 | 0.03 | 4.9131 |

**Mean and median agree to <1% on every term**, so the choice between them does not matter
in practice. Median is used because gradient norms are positive and right-skewed, so one
outlier batch should not move the calibration.

**`step0` is kept to show why one minibatch is not a measurement** - it sits up to 78% away
from the median (crcx gram 0.0292 vs 0.0164; crcx aff 0.0269 vs 0.0142).

### Per-unit gnorm spreads 6-14x across datasets

```
gram        0.012805 (s28nsc)  ..  0.10826 (pancreas)    8.5x
aff         0.0088234 (s28nsc) ..  0.067915 (pancreas)   7.7x
community   1.3266 (pancreas)  ..  7.7399 (immune)       5.8x
recon       5.1207 (crcx)      ..  74.000 (pancreas)    14.5x
```

**So a raw lambda is never portable between datasets; the SHARE is the portable
quantity.** This is the central practical claim of the method.

## 4. The base lambdas

Two share vectors, both a priori, neither tuned on a metric. Which one to deploy is
`D58`'s decision, not this file's.

```
A  flat            0.25 / 0.25 / 0.25 / 0.25   -- cohesion gets HALF the budget, purely
                                                  because it is WRITTEN as two kernels
B  per-subsystem   1/6 / 1/6 / 1/3 / 1/3       -- three MECHANISMS weighted equally:
                                                  who belongs together / does it
                                                  generalize / is z still about genes
exact conversion:  lambda_B = lambda_A * 2/3 (gram, aff) ; * 4/3 (community, recon)
```

**A - flat 0.25 each**

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 2.309 | 3.681 | 0.1884 | 0.003378 |
| lung | 2.978 | 5.517 | 0.04914 | 0.02121 |
| pbmc-immune | 5.691 | 9.598 | 0.03230 | 0.01531 |
| s28nsc | 19.52 | 28.33 | 0.08499 | 0.008077 |
| crcx | 15.24 | 17.57 | 0.04575 | 0.04882 |

**B - per-subsystem 1/6, 1/6, 1/3, 1/3**

| dataset | gram | aff | community | recon |
|---|---|---|---|---|
| pancreas | 1.540 | 2.454 | 0.2513 | 0.004505 |
| lung | 1.985 | 3.678 | 0.06551 | 0.02828 |
| pbmc-immune | 3.794 | 6.399 | 0.04307 | 0.02042 |
| s28nsc | 13.02 | 18.89 | 0.1133 | 0.01077 |
| crcx | 10.16 | 11.72 | 0.06100 | 0.06509 |

### How these compare to what was ACTUALLY TRAINED (`d59` deployed lambdas)

| dataset | term | d59 trained | A flat | A/d59 | B | B/d59 |
|---|---|---|---|---|---|---|
| pancreas | gram | 1.554 | 2.309 | 1.49x | 1.540 | **0.99x** |
| | aff | 2.332 | 3.681 | 1.58x | 2.454 | **1.05x** |
| | community | 0.1224 | 0.1884 | 1.54x | 0.2513 | 2.05x |
| | recon | 0.01622 | 0.003378 | **0.21x** | 0.004505 | **0.28x** |
| lung | gram | 2.406 | 2.978 | 1.24x | 1.985 | 0.83x |
| | aff | 4.429 | 5.517 | 1.25x | 3.678 | 0.83x |
| | community | 0.0467 | 0.04914 | **1.05x** | 0.06551 | 1.40x |
| | recon | 0.0209 | 0.02121 | **1.01x** | 0.02828 | 1.35x |
| pbmc-immune | gram | 5.345 | 5.691 | **1.06x** | 3.794 | 0.71x |
| | aff | 7.728 | 9.598 | 1.24x | 6.399 | 0.83x |
| | community | 0.0345 | 0.03230 | **0.94x** | 0.04307 | 1.25x |
| | recon | 0.0140 | 0.01531 | **1.09x** | 0.02042 | 1.46x |
| s28nsc | gram | 22.03 | 19.52 | 0.89x | 13.02 | 0.59x |
| | aff | 31.47 | 28.33 | 0.90x | 18.89 | 0.60x |
| | community | 0.0813 | 0.08499 | **1.05x** | 0.1133 | 1.39x |
| | recon | 0.0072 | 0.008077 | 1.12x | 0.01077 | 1.50x |

(s28nsc and crcx `d59` values were flagged in the script itself as forbidden
placeholders; shown for orientation only.)

Three readings:

1. **lung, immune and s28nsc were already close to candidate A** - 0.89-1.25x on every
   term. Those three were effectively already at flat-0.25, so this measurement confirms
   rather than overturns them.
2. **pancreas `recon` is the outlier: the deployed lambda is ~4.8x too large** (0.01622 vs
   0.003378). Its recon gnorm is 74.0, 6.3x lung's, which traces to its different
   normalization (per-cell `sum_g x^2` = 5094 vs lung 275). Pancreas is the one dataset
   whose recon weight was materially wrong.
3. **Candidate B lands almost exactly on pancreas's trained gram/aff** (0.99x / 1.05x) -
   a coincidence on one dataset, not evidence.

**Rounding: 4 significant digits, NOT 2 decimal places.** CLAUDE.md §9d's rule exists for
*filename length* (`lcoh4.03750478377344` blew the 255-byte limit and killed a run after
training had finished), but blind 2 dp turns `recon = 0.003378` into `0.00`.

---

## 5. Verification — why these numbers are trustworthy

| check | result |
|---|---|
| datasets closed | 5 / 5, zero failures |
| non-prototype tensors bit-identical before vs after the pass | **12 / 12 on every dataset** |
| prototype relative drift (tol 1e-4) | 1.15 / 1.62 / 2.62 / 1.27 / 2.00 **e-08** |
| probe rows vs expected | 180/180, 379/379, 401/401, 897/897, 733/733 |
| probe errors | 0 |
| `recon` last50 / first50 | 0.979 - 1.003, all `OK` |

The `recon` flatness check is the sharp one: `recon` never touches the cohesion memory, so
with frozen weights its gnorm can only vary by batch composition. A trend there means the
freeze failed — which is exactly how the superseded `d58d` revealed itself (immune `recon`
4.51 → 8.65 = **1.92x**).

Prototype drift is nonzero because `[D35]`'s `normalize_prototypes()` (`scproto.py:3537`)
rewrites `prototypes.weight` in place after every step, bypassing the optimizer. At 1e-08
that is float rounding of an already-unit-norm tensor, four orders of magnitude inside
tolerance. It is deliberately left running: real training renormalizes too.

---

## 6. Cost and timing

```
attempt 1 (grad_probe=1)  ~2.5 GPU-h T4  ~$1.70   13-65 min per dataset
attempt 2 (grad_probe=5)  ~0.8 GPU-h T4  ~$0.55   200s (pancreas) - 955s (lung)
```

T4, not A100 — the probe needs no A100 (`d58d` requested one for a job that doesn't).

**`grad_probe` is NOT the dominant cost.** Going from `grad_probe=1` to `5` on immune moved
780s → 586s, only **1.35x**, not 5x. So the base step (~0.27 s/step, dominated by the
full-graph cohesion memory at `cohesion_global=2`) is ~69% of the cost and the probe ~31%.
This loss is simply expensive per step; `grad_probe=5` is nevertheless the right setting
because CV 0.02-0.11 means a few hundred samples already pin the median.

---

## 7. `D57` - the prototype init was not reproducible. FIXED and VERIFIED.

`_setup_umap_edges()` defaults to `init_prototypes=True`, so every run discarded the
prototypes it had just loaded from the pretrain checkpoint and re-derived them via
`eigsh(L_sym, k, which="LM", tol=1e-2)` with **no `v0`**. ARPACK starts from a random
vector; at `tol=1e-2` different starts return materially different eigenvectors, so the
greedy MaxMin picked different seed cells. The comment at `scproto.py:414-421` claimed
"everything after this first pick is already deterministic" - true *given* `vecs`, false
because `vecs` was not.

Validated locally before spending, mirroring the exact code path on a synthetic graph at
immune's shape, in **separate processes** (the real scenario):

```
unfixed, 3 processes          -> 3 different seed-cell hashes (overlap 31-36%)
fixed v0, 3 processes         -> identical hash x3
fixed v0, OMP threads 1/2/8   -> identical  (BLAS threading does not reintroduce it,
                                             so tol does not need tightening)
```

The first pick was identical in all six runs - the already-seeded draw worked; everything
after it diverged. Fixed at `scproto.py:409` with `v0 = RandomState(seed).rand(N)`.

**Verified end-to-end on Modal** by a duplicate `pancreas_rep2` arm, identical config:

| | pancreas | pancreas_rep2 |
|---|---|---|
| prototype init fingerprint | `effective_n=82.9  top5_share=16.0%` | **identical** |
| prototype drift | 6.758e-09 | **identical** |
| median gram / aff / community / recon | 0.10826 / 0.067915 / 1.3266 / 74.000 | **identical, every printed digit** |

Before the fix the same term varied up to **34%** between runs (immune `aff` 0.0350 vs
0.0230). The residual difference now is ~1e-09, i.e. GPU float-reduction noise.

### Two consequences that outlive this measurement

- **Every scProto run before this fix re-drew ~65%-different prototypes.** So the real
  seed-to-seed noise floor was set by prototype init, not minibatch order - and the
  `d59`/`d59_v2`/`d59_v3` comparison differs in prototype init as well as in lambda. That
  weakens, but does not overturn, the reading in section 8.
- **Epsilon is exonerated as an independent cause.** It does log, and on **all five**
  datasets it fails and falls back to the same constant:
  ```
  [eps calibration] E[p_pos]=0.3386 unreachable (max E[q_pos]=0.3060), falling back to effk=5.0
  ```
  A fixed fallback adds no run-to-run variance. That 5.0 is a number nobody chose remains
  open (CLAUDE.md section 9b).

---

## 8. What follows - the sweep is the experiment, not a robustness check

Three existing immune scoreboard points already sketch a dose-response curve, and its peak
is **~4x above candidate A**:

| arm | lambda_comm | relative to A | modularity | purity (wtd) |
|---|---|---|---|---|
| `d59_v3` | 0.0345 | 1.07x | 0.3032 | 0.779 |
| **`d59`** | **0.124** | **3.84x** | **0.3833** | **0.8264** |
| `d59_v2` | 0.1425 | 4.41x | 0.3692 | 0.8062 |
| candidate A | 0.0323 | 1.00x | - | - |

So candidate A sits essentially on top of the *weakest* arm. Proposed design, to be
pre-registered as `D58` before any arm runs:

```
dev dataset: pbmc-immune -- three existing scoreboard points anchor the curve, and it is
             not the preprocessing outlier that pancreas is

phase 1  NOISE FLOOR, 3 runs, seeds {31,32,33}, same config.  DO THIS FIRST --
         nothing in RESULTS.md currently bounds run-to-run spread, so without it no
         sweep delta is interpretable.
phase 2  community dose  [0, 0.5, 1, 2, 4, 8] x A    <- dose 0 = the -community arm
phase 3  recon dose      [0, 0.5, 1, 2, 4]   x A     <- dose 0 = the -recon arm
phase 4  gram : aff      [1:0, 3:1, 1:1, 1:3, 0:1]   <- 1:0 = -aff, 0:1 = -gram
phase 5  TRANSFER: freeze the winner, apply UNCHANGED to pancreas, lung, s28nsc, crcx

~20 runs, ~$17 on T4 plus eval.  Exactly 3 free axes (4 lambdas minus the global
constant Adam absorbs), and phases 2-4 deliver all four leave-one-out ablation arms
as grid endpoints -- one curve buys both experiments.
```

Candidate B is **not a rival** to A under this design; it is one point on the grid
(community x1.33, recon x1.33, cohesion x0.67), so the sweep subsumes the A-vs-B question.

Pre-declare before launching:

```
GATE       n_unused_protos == 0   (dead prototypes = INVALID arm, not "worse")
PRIMARY    held-out weighted cell-type purity
SECONDARY  modularity, rare coverage, rare homogeneity -- reported, NOT selected on
TIE-BREAK  within 1 noise-floor std on primary -> prefer higher modularity
```

Primary is purity because the rare metrics carry stds of +/-0.25-0.40 (`RESULTS.md`
Table 2); with one seed per arm, selecting on those selects noise.

Two caveats for the caption: one-at-a-time means **interactions are unexplored** (the full
grid is 6x5x5 = 150 arms), and `cohesion` is both a training term and a reported metric, so
the non-circular claims are that it is measured on **held-out** cells and that removing it
hurts purity and rare homogeneity, which it does not optimize.

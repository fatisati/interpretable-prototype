# EXPERIMENTS — what was run and what came out

> **Which file is which.**
> `main-idea.txt` — Fatemeh's own core notes: the main idea, nothing else.
> `decisions.md` — the chronological record of *decisions*: what we changed and why.
> **`EXPERIMENTS.md` (this file)** — the experiments and their **results**. Numbers live here.
> `CURRENT_RUN.md` — only what is executing right now.
>
> Every number below is `s28nsc` / affinity `bk02` / K=200 / 58,423 cells, one seed,
> the same `run_niche_dge` pipeline with majority-vote labels for every method.
> Standard deviations are 0.08–0.25, so **differences under ~0.1 DGE are not
> distinguishable from noise on a single seed.**

---

## The reference graph (measured once, used everywhere below)

```
total edge weight  m (upper triangle, self-loops removed) = 770,419
mean weighted degree                                      = 26.374
cells N = 58,423      prototypes K = 200      N/K = 292 cells per metacell
```

Every cell is assigned (softmax memberships sum to 1, there is no "unassigned" slot), so
**mean metacell size is pinned at 292 for every method, including SEACells.**

---

## 1. Master comparison — every run with a DGE number

`DGE` = `pearson_r` of the niche-DGE evaluation. `n_pairs` = how many distinct
(cell type x niche) states the method resolved — the coverage metric.
`int_frac` = share of all graph edge weight kept inside metacells.
`max/med` = max metacell size divided by median — the size-unevenness measure.

```
run                    latent  gamma  usage    DGE  n_pairs   density  int_frac  med  max/med  jointP
SEACells bk02             --     --    --    0.849     19     0.02952   0.4465   258    3.6     0.504
SEACells PCA              --     --    --    0.747     16     0.01462   0.2343   249    4.5     0.401
lf_L16_g015_uOFF          16   0.15   OFF    0.834     16     0.02628   0.2926   293    1.3     0.438
lf_j_nousage_g024          8   0.024  OFF    0.821     13     0.02449   0.3000   268    2.4      --
lf_L16_g015_uON           16   0.15   ON     0.819     13     0.02614   0.2914   293    1.2     0.441
lf_L16_g02_uON            16   0.20   ON     0.785     12     0.02577   0.2868   293    1.2     0.436
H1_exact_g025              8   0.025  ON     0.781     12        --        --    272     --      --
lf_k_hi_g046               8   0.046  OFF    0.763     15        --        --    183     --     0.356
d7fix                      8   0.15   ON     0.758     14     0.02381   0.2661   294    1.2     0.423
lf_j_nousage_g05           8   0.05   OFF    0.757     12        --        --    289     --      --
lf_j_nousage_g01           8   0.01   OFF    0.746     12     0.01908   0.3788   172    6.7      --
H4_g05                     8   0.5    ON     0.712     15        --        --    188     --      --
H4_g07                     8   0.7    ON     0.647     18        --        --    204     --      --
E5_gful_bk02 (lam=1)       8   0.15   ON     0.496     27        --        --      7     --      --
lf_L16_g02_uOFF           16   0.20   OFF    0.493     20     0.01011   0.2255   178    9.3     0.363
```

**Zero runs beat SEACells bk02 on both DGE and coverage.**

**The two SEACells rows are different baselines and the distinction matters.** `SEACells
bk02` is SEACells fitted on the **BANKSY lambda=0.2 embedding**, which mixes each cell's
expression with its spatial neighbours' (`seacell_X_bklam02_K200`). `SEACells PCA` is
SEACells fitted on the **PCA of expression alone**, same 58,423 cells, same K=200
(`models/s28nsc/seacell_K200`, run 2026-09-20 by `modal_scripts/train_seacell_pca_k200.py`).
scProto's encoder receives expression only -- spatial information reaches it solely
through the affinity graph in the nassoc loss (section 9) -- so `SEACells PCA` is the
information-matched comparator and `SEACells bk02` is not. Both are scored here on the
**same bk02 graph** and against the **same niche-DGE ground truth**, so the DGE, n_pairs
and jointP columns are directly comparable across all rows; only the features each
SEACells run was *fitted on* differ. `SEACells PCA`'s int_frac on the other graphs:
0.1814 (bk04), 0.0743 (bk08) -- `files/tables/s28nsc_seacell_pca_k200/partition_stats.csv`.
Full metric set (purity, niche purity, modularity, coverage, DGE consistency, scGraph,
ct-niche RBO, each with std) is in `models/s28nsc/seacell_K200/metrics.json`.

*What this row implies for the section's conclusions has NOT been written in -- see
`files/notes/s28nsc_seacell_pca_k200.md` for the raw comparison, undecided.*

---

## 2. The DGE / coverage frontier — and the one time it moved

Best DGE achieved at each coverage level:

```
n_pairs      13      14      15      16       17      18      19      20
before     0.846   0.839   0.800    --      0.718   0.647   0.253    --
after L16    "       "       "    0.834      "       "       "     0.493
SEACells                                                    0.849
```

Before `latent_dims=16`, every change **slid along** this curve — more coverage always
cost quality. `lf_L16_g015_uOFF` (0.834 at 16 states) is the first point to sit **above**
the old 15→0.800 / 17→0.718 line, by roughly +0.07. **The frontier lifted for the first
time, and latent dimension is what lifted it.**

---

## 3. Gamma sweeps — tested thoroughly, did NOT solve it

### 3a. Usage loss ON (gful)

```
gamma     DGE    n_pairs   note
0.025   0.781      12
0.15    0.758      14      the long-standing default
0.5     0.712      15
0.7     0.647      18      most coverage, worst quality
```

### 3b. Usage loss OFF (gful)

```
gamma     DGE    n_pairs   density   int_frac   med    max/med
0.01    0.746      12      0.01908    0.3788    172      6.7
0.024   0.821      13      0.02449    0.3000    268      2.4
0.046   0.763      15        --         --      183       --
0.05    0.757      12        --         --      289       --
```

**Non-monotonic, peaking at gamma=0.024.** Raising gamma trades quality for coverage and
never buys both.

### 3c. Why higher gamma cannot work — two independent reasons, both measured

**(i) Above gamma ~0.0265 the request is unsatisfiable.** Gamma implicitly names a metacell
size:

```
density        = mean_weighted_degree * internal_fraction / (size - 1)
requested_size = mean_weighted_degree * internal_fraction / gamma + 1

gamma = 0.024  ->  322 cells   satisfiable at K=200
gamma = 0.15   ->   52 cells   would need K ~ 1,120
gamma = 0.20   ->   40 cells   would need K ~ 1,460
```

Since size is pinned at 292 by `N/K`, anything above

```
gamma_max = mean_weighted_degree * internal_fraction / (N/K - 1)
          = 26.374 * 0.2926 / 291  =  0.0265
```

is impossible to satisfy. Every prototype then scores negative, the CPM gradient points
the same way everywhere, and the term stops discriminating. That is why gamma 0.15 and
0.20 produce essentially the same density (0.02614 vs 0.02577) — an 8x change in gamma
moved density 5%.

**(ii) Above gamma = 0.0386 the loss prefers the WRONG partition.** See section 6.

### 3d. The CORRECTED gamma window — mechanism fixed, outcome unchanged (2026-09-15)

First runs inside **both** the satisfiable bound (0.0265) and the correctly-specified
regime (< 0.0386). `latent_dims=16`, usage OFF, gful.

```
run                gamma    DGE  n_pairs   density  int_frac  med  max/med  jointP   asked/got
lf_L16_cg02_uOFF   0.020  0.842     13     0.02593   0.3338   259    2.7     0.437     1.30
lf_L16_cg026_uOFF  0.026  0.812     15     0.02648   0.3183   272    2.2     0.440     1.02
lf_L16_cg032_uOFF  0.032  0.830     13     0.02679   0.3149   271    2.0     0.442     0.84
--- references ---
lf_L16_g015_uOFF   0.15   0.834     16     0.02628   0.2926   293    1.3     0.438     0.18
SEACells bk02       --    0.849     19     0.02952   0.4465   258    3.6     0.504      --
```

**What improved — the mechanism.**

1. **Density finally tracks gamma, and the satisfiable bound is confirmed exactly.**
   `asked/got` is `achieved_density / gamma`. At gamma=0.026 — which is the predicted
   ceiling 0.0265 — the ratio is **1.02**, essentially exact. Below it the model overshoots
   (0.020 -> ratio 1.30, it cannot go that low); above it the model undershoots (0.032 ->
   ratio 0.84, saturating as predicted). This is the *real* validation of the size formula;
   the earlier "density converged to gamma" claim at gamma=0.024 was a coincidence (§3d).
2. **internal_fraction rose for all three** vs the gamma=0.15 run: 0.3338 / 0.3183 / 0.3149
   vs 0.2926. Lower gamma gives better containment. +0.041 at gamma=0.020 is the largest
   internal_fraction gain from any loss-side change tested.
3. **Size distribution became SEACells-like.** median 259 vs SEACells' 258; `max/median`
   2.7 vs 3.6, up from a rigid 1.3.

**What did NOT improve — the outcome.**

- DGE 0.842 / 0.812 / 0.830 vs 0.834 before: **all within single-seed noise** (std 0.08-0.25).
- Coverage 13 / 15 / 13 vs 16 before, and vs SEACells' 19. No gain.
- Frontier: at 13 states the best was already 0.846 (so 0.842 is not a lift); at 15 states
  0.800 -> 0.812 is a lift of +0.012, far inside noise. **The frontier did not move.**
- Joint purity 0.437-0.442, flat, vs SEACells' 0.504.

**Conclusion: putting gamma in the correct window fixes the loss's internal behaviour but
does not close the gap to SEACells.** Combined with §8, this is strong evidence the gap is
not in the loss at all.

### 3e. Density is not a free parameter — the formula holds to <1%

```
run                int_frac  size   predicted   measured   error
lf_L16_g015_uOFF    0.2926   293     0.02643    0.02628    0.6%
lf_L16_g015_uON     0.2914   293     0.02632    0.02614    0.7%
lf_L16_g02_uON      0.2868   293     0.02590    0.02577    0.5%
d7fix               0.2661   294     0.02395    0.02381    0.6%
```

Density is fixed by `mean_weighted_degree` (the graph), `size` (= N/K), and
`internal_fraction` (placement quality). **Gamma is not one of them.** The formula holds
only while sizes are uniform; it overestimates when they are not (SEACells: predicts
0.0458 vs 0.0295 actual; `lf_j_nousage_g024` at max/med 2.4: predicts 0.0296 vs 0.0245).

**Correction to an earlier claim.** `CURRENT_RUN.md` recorded "density converged to ~=
gamma, the framework is validated" from the single point gamma=0.024 -> density 0.0245.
That was a **coincidence**: 0.024 is simply where density sits regardless, so that one
point happened to cross the `density = gamma` line. The other points refute it
(gamma=0.01 -> 0.0191, gamma=0.15 -> 0.0261).

---

## 4. The usage loss OFF is better — controlled evidence

The only *matched-gamma* ON/OFF pairs, both at `latent_dims=16`:

```
gamma   usage    DGE   n_pairs   density   int_frac   med   max/median
0.15     ON     0.819     13     0.02614    0.2914    293      1.2
0.15     OFF    0.834     16     0.02628    0.2926    293      1.3     <- +0.015 DGE, +3 states
0.20     ON     0.785     12     0.02577    0.2868    293      1.2
0.20     OFF    0.493     20     0.01011    0.2255    178      9.3     <- blew up
```

Plus one matched pair at 8-d: gamma 0.025 ON (0.781, 12 states) vs 0.024 OFF (0.821, 13).

**Turning the usage loss off is worth roughly +0.015 to +0.040 DGE and +1 to +3 states,
at gamma 0.15 and below.** It is NOT safe at gamma 0.20, where it produced a degenerate
partition (density halved, joint purity 0.363, `max/median` 9.3). The pseudobulk spot
check flagged that run independently: `corr(total_counts, n_cells) = 0.592`, want ~1.0.

**All 200 prototypes stay alive with the usage loss off** — in every usage-OFF run, at
every gamma from 0.01 to 0.20. Nothing dies. See `decisions.md` §12 for why.

---

## 5. Size uniformity is caused by CPM, not by the usage loss

At **matched gamma = 0.15**, `max/median` is 1.2 with the usage loss ON and 1.3 with it
OFF. Density and internal_fraction are likewise unchanged (0.02614/0.2914 vs
0.02628/0.2926). **Removing the usage loss does not change the size distribution.**

The equalizing force is CPM's quadratic charge:

```
score[p]          = edges_inside[p] - gamma * possible_pairs[p]
possible_pairs[p] ~ size[p]^2 / 2
```

Because the penalty grows with `size^2`, moving one cell from a small prototype to a big
one costs an extra `gamma * (size_big - size_small)` — strictly increasing in the size
gap. CPM therefore always pushes cells from big prototypes toward small ones.

**Uneven sizes never helped scProto.** Sorted by unevenness: max/med 1.3 -> DGE 0.834,
2.4 -> 0.821, 6.7 -> 0.746, 9.3 -> 0.493. Every time scProto's sizes spread out, quality
fell and aggregate density fell with it — meaning its big metacells are low-density
garbage collectors, not coherent regions. SEACells is uneven (3.6) *and* good, but at
matched unevenness scProto is ~35% less dense, so unevenness is not the explanatory
variable.

Uniformity does cap **coverage** (uniform 292-cell metacells permit ~20 states at K=200),
but that ceiling is **not yet binding** — we are at 16.

---

## 6. `latent_dims`: 8 -> 16 (2026-09-15)

The only fully controlled pair — identical config, only `latent_dims` changed:

```
             DGE    n_pairs   int_frac   density
8-d        0.758      14       0.2661    0.02381
16-d       0.819      13       0.2914    0.02614
```

**+0.061 DGE and +0.025 internal_fraction from latent dimension alone.** `decisions.md`
§9c had predicted "at most +0.014, not the fix" — that estimate was off by ~4x. This is
the single largest gain of any change tested, and the only one that lifted the frontier.

---

## 7. Diagnostics: scoring SEACells' partition under OUR OWN loss

Computed from raw label vectors (`diag_and_train_correct_gamma.py`):

```
partition              int_frac   edges_inside   total_possible_pairs   median   max
SEACells                0.44647      343,971         11,650,407          258     925
lf_L16_g015_uOFF        0.29257      225,400          8,577,888          293     367
```

```
cpm_total = edges_inside - gamma * total_possible_pairs

   gamma         scProto        SEACells    our loss prefers
   0.010         139,621         227,467    SEACells
   0.020          53,842         110,963    SEACells
   0.026           2,375          41,060    SEACells
   0.032         -49,093         -28,842    SEACells
   0.039        -109,138        -110,395    scProto  (WRONG)
   0.050        -203,495        -238,549    scProto  (WRONG)
   0.150      -1,061,283      -1,403,590    scProto  (WRONG)
   0.200      -1,490,178      -1,986,110    scProto  (WRONG)

EXACT CROSSOVER gamma = 0.03859
```

SEACells captures more edges (343,971 vs 225,400) but is charged for more pairs (11.65M
vs 8.58M, because uneven sizes inflate the sum of squares). Which side wins depends
entirely on gamma.

**Above gamma 0.0386 our own objective ranks SEACells' better partition LOWER.** Every
gamma run at latent 16 so far (0.15, 0.20) sat in that regime, as did 0.5 and 0.7. Below
it the loss prefers the right answer and the optimizer simply never found it — a
different failure needing a different fix.

---

## 8. Expressivity probe — **result stands, interpretation was overstated**

Fit a linear head from our frozen 16-d `z` to SEACells' 200 labels (supervised,
in-sample, no competing losses — the most generous possible setting):

```
                                  int_frac   density   n_alive   accuracy   ARI
SEACells (target)                  0.4465    0.02952     200       --        --
best linear softmax on our z       0.3156    0.01675     199      0.404     0.225
what our trained loss achieved     0.2926    0.02628     200       --        --
```

The probe closes only **15%** of the internal_fraction gap.

**Two things this does NOT establish**, both raised and both correct:

1. **The encoder is trainable.** This probe froze `z` at what one run produced. In real
   training the encoder moves, so `z` can reshape itself to make a good partition
   linearly separable. The measured ceiling is "the best linear head on *this* z", not
   "the capacity of the architecture". An earlier claim that ~85% of the gap is
   "unreachable by any loss" **does not survive this** and is withdrawn.
2. Each probe imitates SEACells' specific **label identities**, not internal_fraction
   directly. A different partition in the same family could score higher without
   imitating SEACells. Every probe number is a **lower bound**, not a proven ceiling.

---

### 8b. The feature-space ladder — **the answer** (2026-09-15)

Same probe, but comparing FEATURE SPACES instead of head variants. All linear, **no bias**
(matching our real head exactly), fit to SEACells' 200 labels in-sample, features used
as-is (a no-bias head is origin-sensitive, so re-centring would change what it can
express), weak regularization `C=100` because this is a capacity test.

```
features                dim  int_frac  gap closed     acc     ari   density  max/med
our trained run          16    0.2926          0%      --      --   0.02628      1.3
pca8                      8    0.2402        -34%   0.265   0.129   0.01097      7.3
pca16                    16    0.2624        -20%   0.361   0.186   0.01438      5.8
pca32                    32    0.2745        -12%   0.413   0.220   0.01608      5.1
pca50                    50    0.2790         -9%   0.436   0.237   0.01659      5.0
z16 (OUR ENCODER)        16    0.3016          6%   0.410   0.226   0.01721      5.2
BANKSY X_bklam02         50    0.4262         87%   0.714   0.519   0.02753      3.8
SEACells (target)        --    0.4465        100%      --      --   0.02952      3.6
```

**1. Spatial context is the whole story.** `pca50` and `BANKSY` have the SAME dimension
(50), the SAME linear no-bias head and the SAME target. The only difference is that BANKSY
mixes each cell's expression with its **spatial neighbours'**. internal_fraction
0.2790 -> **0.4262**; accuracy 0.436 -> 0.714; ARI 0.237 -> 0.519. **A trivial linear head
on spatially-aware features closes 87% of the gap.** Nothing built on expression alone
passes 30%.

**2. The encoder is FINE.** z16 (0.3016) beats pca16 (0.2624) at matched dimension — it
concentrates signal, +0.039 over an untrained closed-form baseline. This kills the "the
encoder destroys information" hypothesis, and it was the sharpest available test of it
because PCA-16 requires no trained encoder, sidestepping the objection that a frozen z is
not the architecture's capacity.

**3. Dimension is NOT the fix.** pca8 -> pca50 buys only 0.2402 -> 0.2790, and **pca50
still loses to our 16-d z**. Raising `latent_dims` to 32 or 64 would not have helped. (The
real 8 -> 16 gain in §6 was about representation quality, not raw capacity.)

**The circularity caveat, which is significant.** SEACells was *fit on* `X_bklam02`, and
the `bk02` affinity graph that `internal_fraction` is measured on was *also* built from
`X_bklam02`. So BANKSY features are predicting a BANKSY-built partition scored on a
BANKSY-built graph, and some of that 87% is circular.

**But the circularity is itself the finding.** Our `nassoc` loss optimizes edge containment
on that same `bk02` graph. scProto is therefore asked to maximize a spatially-defined
objective while its encoder is structurally blind to space. That misalignment is real, not
a probe artifact — and it is the same reason §3d's corrected-gamma runs improved the loss's
internal behaviour without moving DGE or coverage.

---

## 9. CONFIRMED: scProto's encoder never sees spatial context

The SEACells run is stored as `seacell_X_bklam02_K200`, i.e. `build_kernel_on =
'X_bklam02'` — the **BANKSY embedding at lambda=0.2**, which mixes each cell's expression
with its **spatial neighbours' expression**.

scProto's `encode_adata` feeds a scPoli dataloader over `adata`: **gene expression plus
batch key, no spatial features.** Spatial information reaches scProto only through the
affinity graph inside the nassoc loss — never through the encoder input.

§8b tested this and it holds: at matched dimension, spatially-aware features reach
internal_fraction 0.4262 while expression-only features cap at 0.2790. **This is an
information gap, not a loss gap, not an architecture gap, not a dimension gap.** Neither
gamma, nor the normalizer, nor latent dims, nor a bias term can close it — confirmed
independently by §3d, where putting gamma in the provably-correct window improved the
loss's internal behaviour and moved neither DGE nor coverage.

It is also the *fairness* point: SEACells is given spatially-aware features and scProto is
not, so the current comparison is not like-for-like.

**Next step (not yet run):** feed `X_bklam02` to the encoder instead of raw expression. It
already exists in `adata.obsm` — no computation needed. This is a change to what the
encoder consumes, i.e. a trainer change, so it needs sign-off before implementing.

---

## 10. What has been ruled out

Each of these was tested and did **not** close the gap to SEACells:

```
gamma, swept 0.01 -> 0.7, usage ON and OFF, 8-d and 16-d      §3
  incl. the provably-correct window 0.020-0.032               §3d
higher gamma specifically (the "more resolution" hypothesis)  §3c -- refuted with mechanism
the usage loss (on vs off)                                    §4  -- worth +0.015-0.040, not the gap
size uniformity / letting sizes go uneven                     §5  -- uneven always made it WORSE
latent dimension 8 -> 16                                      §6  -- best loss-side gain, +0.061, still short
raw feature dimension (PCA 8 -> 50)                           §8b -- barely moves, pca50 < our z16
the encoder destroying information                            §8b -- refuted, z16 beats pca16
K = 200 being too few                                         §7  -- SEACells uses the same K=200
```

**Not ruled out, and now the leading explanation: the encoder receives no spatial
context (§9).**

---

## 11. Linear probe: how much niche information is in gene expression? (2026-09-16)

Scripts: `notebooks/s28nsc_dge_lite/scripts/niche_probe_auc.py`, `niche_mc_coverage.py`
Tables: `tables/s28nsc_niche_probe_auc.csv`, `tables/s28nsc_niche_mc_coverage.csv`
Note: `notes/niche_probe_auc_findings.md`. CPU only, ~5 min, no Modal.

### Method

Standard linear probing (Alain & Bengio 2016 arXiv:1610.01644; control task from Hewitt &
Liang 2019 EMNLP; V-information framing from Xu et al. 2020 ICLR). Within each cell type,
one-vs-rest per niche, 5-fold CV ROC-AUC, plus the same probe on permuted labels.

Probes are valid: control AUC = 0.494 / 0.489 / 0.490 (sd ~0.04) for expr960 / pca50 /
pca50+GBM. So ~0.60 is the meaningful-signal threshold.

### Result A -- niche IS decodable from expression, in most states

```
best-of-probes AUC     mean 0.727   median 0.727
  >= 0.60   91/99 pairs  (92%)
  >= 0.65   72/99  (73%)
  >= 0.70   61/99  (62%)
  >= 0.80   20/99  (20%)
```

`pca50` (0.715) beats `expr960` (0.656): 960 correlated features with a few hundred
positives overfits, PCA denoises. Report best-of-probes, not the raw-matrix probe alone.

### Result B -- DGE is CONSERVATIVE here, not liberal

```
2x2, n_dge_genes >= 20  vs  best AUC >= 0.65
                  not decodable   decodable
no/few DE genes        26            33
many DE genes           1            39

spearman(n_dge_genes, AUC)   = 0.674
spearman(max_abs_logfc, AUC) = 0.720
```

Only **1 of 99** pairs has many DE genes but is not decodable -- and it is the largest
group (`Tumor cells / Tumor surface`, n_pos=2855, 270 DE genes, AUC 0.624), exactly where
"significance grows with sqrt(n_cells) while AUC does not" predicts. The dominant
discordance is the opposite: **33 pairs with few/no DE genes are still decodable**, 9 of
them with ZERO significant DE genes. Per-gene Wilcoxon at n_pos ~30-200 is underpowered;
the multivariate probe pools weak evidence across all 960 genes.

=> Using "has DE genes" as the feasibility screen is safe but misses ~1/3 of recoverable
states.

### Result C -- per-state metacell coverage, scProto vs SEACells on the SAME affinity

Run: `lf_L16_g015_uOFF` (latent 16, gamma 0.15, usage OFF, bk02) vs `seacell_X_bklam02_K200`
-- both on bk02/BANKSY-lambda-0.2, both K=200. Fetched from the Modal volume.
Table: `tables/s28nsc_niche_mc_coverage.csv` (one row per state, with the SIZES of each
method's metacells for that state, plus probe AUC and DGE gene count).

Size distributions (effective n_mc = `1 / sum(share_k^2)`):

```
run                n_mc  eff_n_mc  median   max   min  max/med    cv
L16_g015_uOFF       200     198.3     293    367   209    1.25   0.09
L16_g015_uON        200     198.0     293    357   177    1.22   0.10
L16_cg026_uOFF      200     183.7     272    586    32    2.15   0.30
L16_g02_uOFF        200      99.1     178   1664    17    9.32   1.01
seacell_bklam02     200     146.1     258    925    33    3.59   0.61
```

scProto at gamma 0.15 uses MORE effective metacells than SEACells (198.3 vs 146.1) and is
far more uniform. **The prototype budget is not the constraint.** Note gamma 0.2 halves
effective n_mc to 99.1 -- size uniformity is sharply gamma-dependent in 0.15-0.20.

States covered (a state = a (cell type, niche) pair with >=30 cells; "covered" = at least
one metacell whose majority vote is that state):

```
                             all states      decodable states (AUC>=0.65)
scProto  L16_g015_uOFF      31/99  (31%)          24/72  (33%)
SEACells bklam02            46/99  (46%)          36/72  (50%)
```

So with an equal and better-distributed prototype budget, scProto claims 15 fewer states
than SEACells, and 12 fewer of the states that are provably decodable from expression.
This is a PLACEMENT difference, not a budget or a collapse difference.

### Also corrected while checking

- `epsilon` is NOT fixed: all s28nsc runs carry `ecal1` = `calibrate_eps=1` (p/q matching,
  falling back to effk target 5.0). The `0.05` in `defaults.py` is the uncalibrated SwAV
  value and `sep3.py`'s `.get('epsilon', 0.05)` fallback -- neither is what ran.
- Measured on the bk08 affinity: `mean_weighted_degree = 25.97`, `internal_fraction = 0.628`
  (SEACells) / 0.607 (scProto-old). The two partitions score nearly the SAME internal
  fraction despite wildly different size distributions -- i.e. internal_fraction alone does
  not distinguish a good partition from a degenerate one.
- The gamma formula was REMOVED from `CLAUDE.md` on request pending discussion. It is the
  CPM stability condition (Traag, Van Dooren & Nesterov 2011, Phys Rev E 84:016114):
  a community is stable when `edges_inside / pairs_inside >= gamma`, giving
  `gamma = mean_weighted_degree * internal_fraction / (target_size - 1)`.
  CLAUDE.md had `- 0.5` in the denominator, which does not follow from that derivation.
  Measured values: target 292 cells -> gamma 0.056; target 75 cells -> gamma 0.221.
  Neither equals the 0.15 CLAUDE.md attributed to a 75-cell target. **gamma 0.056 has not
  been tested**; section 3d's tested window was 0.020-0.032 and section 7's crossover is
  0.0386.

---

## 11. The 2026-09-17/18 session: three real loss bugs, none of them the gap

Everything below is `s28nsc` / `bk02` / K=200 unless stated. Scripts:
`modal_scripts/measure_gamma_from_graph.py`, `measure_affinity_distribution.py`,
`measure_estimator_bias.py`, `train_ht_nousage_latent8_16.py`, `train_ht_lowgamma.py`,
`repro_pancreas_paper.py`.

### 11a. gamma, derived from the graph instead of swept

The CPM move rule is `delta = size[p] * (mean affinity from i to p's members - gamma)`,
so **gamma is the minimum AVERAGE affinity a cell needs to a group to be worth joining
it** -- measurable without training anything.

```
best a perfect 292-cell same-state group could give   0.041
SEACells' own metacells, median per-metacell density  0.0449
SEACells' aggregate density                           0.0295
=> usable gamma                                       0.026 - 0.030
```

That is exactly the window already swept in section 3d, now derived rather than found.
**Gamma is settled, and gamma is not the problem.**

### 11b. The affinity WEIGHTS carry almost no state signal

```
                      mean weight of a    fraction of pairs
                      connected pair       connected
same (celltype,niche)     0.3428              0.0159
different                 0.3299              0.00078
ratio                     1.04                20.5
```

98.4% of same-state pairs have no edge, so the median including zeros is 0 and only the
mean -- which IS the density -- is meaningful. All the state information is in **which
pairs are connected**, none in how strongly. Since `density = connected_fraction *
mean_weight` and the second factor is fixed at ~0.34, gamma is a sparsity knob.

### 11c. THE CEILING -- there is almost no headroom above SEACells

```
internal_fraction
  celltype only                            0.8137
  niche only                               0.5514
  celltype x niche (GROUND TRUTH)          0.4549   <- the ceiling
  SEACells                                 0.4465   <- 98% of the ceiling
  scProto, best runs                       0.29 - 0.36   <- 65-79%
```

**The true labels themselves keep only 45.5% of the affinity inside their own classes.**
So the question was never "beat SEACells" -- it is "why are we at two thirds of a
ceiling SEACells already touches".

### 11d. Two minibatch estimator bugs, measured by simulation

Two algebraic attempts failed because the sampling is not uniform: a batch is 1024
EDGES drawn with replacement PROPORTIONAL TO w, and its cells are their deduplicated
endpoints, so a cell's inclusion probability grows with its weighted degree.
Simulating the real sampler against a known partition (estimate / truth):

```
                                 total   <25   50-100  200-300  500-1k
size  [code]                     1.000  0.821   0.856   0.943   0.958
size  [Horvitz-Thompson]         0.997  0.996   0.983   1.005   0.991
pairs [code: c^2 size^2-c sumsq] 1.124  2.386   1.127   1.021   0.982
pairs [Horvitz-Thompson]         0.999  1.108   1.034   1.028   0.996
edges [code: w applied twice]    1.046  1.347   1.042   1.101   1.049
edges [count-only]               1.001  1.025   1.016    --      --
```

(a) edges are sampled proportional to w and then multiplied by w AGAIN, so the objective
was effectively weighted by w^2; the UMAP term does not do this. (b) `possible_pairs` was
rescaled by a uniform `N/n_b`, which cannot correct a non-uniform sample -- neither the
[D7] linear-c form nor the pre-[D7] c^2 form is unbiased. Both fixed by `nassoc_ht=1`.

**In the 200-900 cell band where every prototype sits, the net effect is ~8%.**

### 11e. gamma was on the wrong SCALE -- [D10]

gamma is compared against a SOFT density; every gamma we ever set came from HARD
statistics. Measured by `nassoc_gamma_scale='soft'` on the real data:

```
SOFT density 0.00674   HARD density 0.00990   ratio 0.68   (stable across 4 runs)
=> gamma was ~1.47x too strict in the units the loss uses
```

Root cause, NOT fixed: epsilon calibration **fails on both datasets** and silently falls
back to `umap_proto_effk=5.0`, a hard-coded default nothing chose -- and effk then
drifts 5.0 -> 2.7 because calibration runs once, before training.

```
s28nsc:   E[p_pos]=0.3357 unreachable (max E[q_pos]=0.2051) -> effk=5.0
pancreas: E[p_pos]=0.3397 unreachable (max E[q_pos]=0.2992) -> effk=5.0
```

### 11f. RESULTS -- all three fixes moved nothing

```
run                  gamma      DGE  pairs  jointP  int_frac  max/med  min  n<100
ht_L16_g028_uOFF     0.028    0.833    14   0.436   0.3137     1.7    150     0
ht_L8_g028_uOFF      0.028    0.815    13   0.435   0.2976     1.8     99     1
d10_L16_gh028_uOFF   0.019*   0.840    13   0.440   0.3284     2.2    126     0
d10_L16_gh02_uOFF    0.014*   0.787    15   0.438   0.3604     3.0     77     4
--- references ---
lf_L16_cg02_uOFF     0.020    0.842    13   0.437   0.3338     2.7     --    --
SEACells bk02          --     0.849    19   0.504   0.4465     3.6     --    --
GROUND-TRUTH ceiling   --      --      --     --    0.4549
```
`*` = converted from the hard scale by [D10].

Every difference is inside single-seed noise (std 0.08-0.25). At `n_pairs=15` the
previous best was 0.812, so `d10_gh02`'s 0.787 is ON the frontier, not above it.
**The frontier did not move.**

One result contradicts the hypothesis that the estimator bias caused size rigidity:
correcting it made sizes MORE uniform (max/med 2.7 -> 1.7), consistent with section 5
localizing uniformity to CPM's intrinsic quadratic charge.

### 11g. PANCREAS -- the CPM rewrite is a SPATIAL-specific fix, not a general one

The published objective (`sections/method.tex`) is `volume` normalization +
`(M_kk - 1)^2`, `lambda_nassoc=1`, usage on -- **it has no gamma at all**. Every
pathology above belongs to the CPM rewrite, which exists to fix a spatial collapse
(decisions.md 1-3: 15/200 survivors on s28nsc) that pancreas never had.

Reproduced with today's code (`repro_pancreas_paper.py`, `train_epochs=20`):

```
                       purity            batch_entropy      modularity        coverage
PAPER scProto        0.98 +- 0.07        1.04 +- 0.66      0.61 +- 0.08         --
repro LD20           0.9806 +- 0.0549    0.8319 +- 0.5720  0.6170 +- 0.0757   0.9286
repro LDdefault      0.9724 +- 0.0796    0.8640 +- 0.5486  0.6109 +- 0.0899   1.0000
PAPER SEACells       0.97 +- 0.08        0.19 +- 0.30      0.67 +- 0.05         --
```

Modularity and purity match the paper essentially exactly; batch entropy is ~20% low
but well inside its own std of 0.55-0.67. **The paper's pancreas result reproduces, and
today's commits are behaviour-preserving at their defaults -- verified, not assumed.**

By contrast, the CPM rewrite ported to pancreas (`gamma=0.11`, `lambda_nassoc=200`,
usage off) **regressed badly**: modularity 0.3127 against 0.66-0.70 for the published
objective, early-stopped at epoch 4, `min_size=1`, `max/median=17.2`. That comparison
was not controlled (4 simultaneous changes), but the direction is clear.

**NOTE the table keys.** `TASK1_METRICS` in `result_tables.py` uses
`weighted_mean_batch_entropy` and `weighted_mean_cell_type_purity`, NOT `mean_*`.
Reading the unweighted keys makes scProto's pancreas batch entropy look like 0.40
instead of 0.90 and produces a spurious 2.5x "reproducibility failure". This happened
in-session and cost a false alarm.

### 11i. THE K-MATCHED GAMMA -- CPM optimized properly, and it makes things WORSE

`leiden_cpm_control.py` ran OUR CPM objective with a real optimizer (leidenalg) on the
same graphs, searching the resolution to match our K. This corrected the gamma
derivation twice over -- at the gammas derived from SEACells' density, hard CPM returns
K=1043 (s28nsc) and K=687 (pancreas), not 200/220:

```
                       derived gamma   K it gives   K-MATCHED gamma   int_frac  jointP
s28nsc / bk02  K=201       0.028          1043          0.00686        0.5501   0.529
pancreas/arbf  K=221       0.11            687          0.04209        0.6464     --
```

Leiden-CPM BEATS SEACells on both graphs (0.5501 vs 0.4465; 0.6464 vs 0.5684), so the
objective is not the problem and the graph holds far more structure than anyone has been
extracting. **But running scProto at those gammas shows what optimizing CPM harder
actually buys.**

**s28nsc, latent 16, usage OFF, lambda 200, gamma 0.00686 (only gamma changed):**

```
run                       DGE  pairs  jointP  int_frac  density   K   med   max  n<100
cpmk_L16_gk00686_uOFF   0.605     15   0.425    0.5501  0.01246  195    42  1928   135
Leiden-CPM K=201           --     52   0.529    0.5501  0.03203  201   309   833    --
SEACells bk02           0.849     19   0.504    0.4465  0.02952  200   258   925    --
scProto best (cg02)     0.842     13   0.437    0.3338  0.02593  200   259    --    --
```

TWO CONCLUSIONS, and the second is the important one.

1. **The parameterization is NOT the bottleneck.** scProto reached int_frac 0.5501 --
   Leiden's number exactly, and above SEACells. A linear head on a 16-d expression-only
   `z` CAN express a partition with Leiden-level edge containment. This weakens the
   section 8/9 information-gap reading considerably.
2. **The objective is MISALIGNED with the goal.** At the SAME int_frac 0.5501, Leiden
   gets jointP 0.529 and scProto gets 0.425, and DGE collapsed 0.842 -> 0.605 with a
   wrecked size profile (median 42, max 1928, 135/195 prototypes under 100 cells).
   **Optimizing CPM harder buys containment and loses what we want.**

That inverts the reading of the Leiden control: Leiden is not better because it optimizes
CPM better -- it happens to land on a partition good on BOTH axes, and CPM alone does not
select for that. Every fix this session made the CPM term more faithful, and the more
faithfully it is optimized, the worse DGE gets.

**pancreas, latent 8 (the paper default, confirmed in `appendix/training.tex`: "hidden
dimension 64, latent dimension 8"), usage OFF, gamma 0.04209, lambda from the MEASURED
scale (at lambda=200 nassoc is 2.4x umap here vs 20x on s28nsc):**

```
run                          purity  batch_ent  modularity  coverage
cpmk_panc_gk042_l200_uOFF    0.8828     1.3972     0.3970    0.8571
cpmk_panc_gk042_l50_uOFF     0.9030     1.0907     0.4652    0.9286
--- reference, same keys ---
repro_paper_LDdef            0.9041     0.4220     0.6924    1.0000
```

CPM trades the paper's headline metric for batch mixing: modularity collapses
(0.397 / 0.465 vs 0.692) while batch entropy rises 2.6-3.3x and purity roughly holds.
**lambda is monotone -- 50 beats 200 on every metric -- so lambda -> 0 recovers the
published result.** That is the signature of a term that is actively hurting, not one
that is mis-tuned. Section 17's conclusion now rests on a controlled comparison with
every earlier objection addressed (usage off, paper latent, K-matched gamma, measured
lambda).

**What 11i changes about the diagnosis.** The question is no longer "why can't we reach
SEACells' internal_fraction" -- we can, and we can exceed it. It is
**"internal_fraction is not a sufficient proxy for a good metacell partition"**, and
optimizing it is not the same as optimizing joint purity, coverage or DGE.

### 11h. What this session establishes

```
three genuine loss bugs found and fixed:   w^2 double-count, degree-biased pairs,
                                           gamma scale
frontier movement from any of them:        none
gamma:                                     settled by measurement, not a lever
the ceiling:                               0.455, and SEACells is at 98% of it
still unfixed:                             the epsilon fallback (arbitrary effk=5.0,
                                           fails on both datasets, drifts in training)
still untested:                            leidenalg hard-CPM control -- the only cheap
                                           experiment that DISCRIMINATES between
                                           "objective wrong" and "optimizer weak"
```

**The loss was buggy, but the bugs were not what was holding back coverage and joint
purity.** Section 8's information-gap reading remains a hypothesis, not a fact --
section 8 itself withdraws the strong version, because the probe froze `z` and fit a
linear head to SEACells' label identities, making every number a lower bound.

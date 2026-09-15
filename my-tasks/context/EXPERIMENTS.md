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

**Zero runs beat SEACells on both DGE and coverage.**

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

### 3d. Density is not a free parameter — the formula holds to <1%

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

## 9. Open lead: scProto's encoder never sees spatial context

The SEACells run is stored as `seacell_X_bklam02_K200`, i.e. `build_kernel_on =
'X_bklam02'` — the **BANKSY embedding at lambda=0.2**, which mixes each cell's expression
with its **spatial neighbours' expression**.

scProto's `encode_adata` feeds a scPoli dataloader over `adata`: **gene expression plus
batch key, no spatial features.** Spatial information reaches scProto only through the
affinity graph inside the nassoc loss — never through the encoder input.

If that is the binding constraint it is an **information** gap, not a loss or architecture
gap, and neither latent dims nor gamma nor a bias term would close it. **Not yet tested**
— `probe_expressivity_ladder.py` is written to test it and has not been run.

---

## 10. Running now

`diag_and_train_correct_gamma.py` — latent 16, usage OFF, gamma in
`{0.020, 0.026, 0.032}`: the first runs inside **both** the satisfiable bound (0.0265) and
the correctly-specified regime (< 0.0386). Every previous gamma was outside at least one.

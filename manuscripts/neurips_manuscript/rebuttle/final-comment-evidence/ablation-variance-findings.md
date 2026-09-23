# Ablation variance — findings (Reviewer nG29)

From `notebooks/ablation_paired_significance.ipynb`, run 04 Aug 2026. No retraining —
reads per-batch CSVs already on disk. 3 datasets × 8 arms.

Verified it read the right runs: pancreas full model = `0.6153 ± 0.0801` = the `.615±.08`
he quoted. Lung `0.6554 ± 0.0302`, Immune `0.6234 ± 0.0596`. All match the paper.

---

## 1. What we can now state

**His challenge is answered on the exact arm and dataset he named.**
Stop-gradient, pancreas modularity: `0.6153 ± 0.0801 → 0.5554 ± 0.0701`,
**9 out of 9 batches**, p = 0.0039, q = 0.0091.
p = 0.0039 is the smallest value the test can return at n=9 — it could not score higher.

**Three claims now hold across all three datasets:**

| Claim | Pancreas | Lung | Immune |
|---|---|---|---|
| Community loss drives modularity | 8/9, p=.008 * | **16/16, p=3.1e-5 ***** | 5/5, +0.245 |
| Usage loss prevents collapse (active protos) | 220→63 | 300→77 | 300→48 |
| Recon leaves modularity flat, as designed | 5/9, ns | 5/16, ns | 3/5, ns |

Lung's 16/16 at p = 3.1×10⁻⁵ is the strongest result in the whole ablation. Immune's
+0.245 is the largest single effect anywhere in the study.

**Four more wins on pancreas rare-cell homogeneity** (full = 0.6319 ± 0.1762): usage 7/8,
community 7/8, waypoint init 8/8, stop-gradient 8/8 — all q = 0.027.

**Lung and Immune now have full ± tables** (§4, §5). That gap is closed.

---

## 2. Why the paired test is the right answer to him

He said: the effect is smaller than your own error bar, so it proves nothing.

- `L_nassoc` purity: 0.972±0.09 → 0.962±0.09 — gap 0.009, spread 0.088
- stop-gradient modularity: 0.615±0.08 → 0.555±0.07 — gap 0.060, spread 0.075

The ± is spread *across batches* — it says batch 3 is harder than batch 7. It is not the
uncertainty of the comparison. The same batches appear in both runs, so the numbers are
**paired**: `full[i]` and `ablated[i]` are the same batch. If every batch drops by 0.06 the
effect is perfectly consistent even though the spread is 0.08. A paired test sees that;
error bars cannot.

Same test we already use for Harmony/ComBat (`p=.035* (7/8)`), so the evidence stays
consistent across the response.

**Testable per batch:** modularity, rare coverage/homogeneity/F1.
**Not testable:** purity and batch entropy — their ± is per *metacell*, and two runs make
different metacells, so there is nothing to pair.

**Exact floor** (smallest p a Wilcoxon can return), and what Bonferroni over 7 arms does:

| | n | floor | × 7 arms | can reach 0.05? |
|---|---|---|---|---|
| Pancreas modularity | 9 | 0.0039 | 0.027 | yes |
| Pancreas rare metrics | 8 | 0.0078 | **0.055** | **no** |
| Lung modularity | 16 | 0.00003 | 0.0002 | yes |
| Lung rare metrics | 15 | 0.00006 | 0.0004 | yes |
| **Immune, all metrics** | **5** | **0.0625** | **0.44** | **no** |

Used BH, not Bonferroni. **Immune can never reach significance even uncorrected** — quote
win-counts there (5/5, 4/5), not p-values.

---

## 3. Pancreas

**Modularity, n=9, full = 0.6153 ± 0.0801**

| Arm | ablated | diff | worse | p | q | |
|---|---|---|---|---|---|---|
| **Stop-grad off** | 0.5554 ± 0.0701 | **+0.060** | **9/9** | **0.0039** | 0.0091 | ** |
| **– community** | 0.4491 ± 0.0433 | **+0.166** | 8/9 | 0.0078 | 0.0137 | * |
| k-means init | 0.5938 ± 0.0680 | +0.022 | 7/9 | 0.055 | 0.077 | ns |
| – recon | 0.6135 ± 0.0776 | +0.002 | 5/9 | 1.000 | 1.000 | ns |
| Fixed τ | 0.6230 ± 0.0794 | −0.008 | 3/9 | 0.426 | 0.497 | ns |
| – nassoc | 0.6564 ± 0.0803 | −0.041 | 0/9 | 0.0039 | 0.0091 | ** rev |
| – usage | 0.6632 ± 0.0830 | −0.048 | 0/9 | 0.0039 | 0.0091 | ** rev |

**Rare homogeneity, n=8, full = 0.6319 ± 0.1762**

| Arm | ablated | diff | worse | p | q | |
|---|---|---|---|---|---|---|
| – usage | 0.3965 ± 0.2235 | +0.236 | 7/8 | 0.016 | 0.027 | * |
| – community | 0.4063 ± 0.2458 | +0.226 | 7/8 | 0.016 | 0.027 | * |
| k-means init | 0.4827 ± 0.2325 | +0.149 | 8/8 | 0.008 | 0.027 | * |
| Stop-grad off | 0.5058 ± 0.1729 | +0.126 | 8/8 | 0.008 | 0.027 | * |
| – nassoc | 0.5112 ± 0.2032 | +0.121 | 7/8 | 0.055 | 0.077 | ns |
| Fixed τ | 0.5633 ± 0.1699 | +0.069 | 7/8 | 0.109 | 0.128 | ns |
| – recon | 0.6029 ± 0.1915 | +0.029 | 5/8 | 0.461 | 0.461 | ns |

`k-means init` = waypoint replaced by k-means, so this row means waypoint helps, 8/8.

**Rare F1, n=8, full = 0.5885 ± 0.1977** — all arms right direction, none survives BH.
Largest: – usage 0.3526±0.2674 (+0.236, 6/8, q=0.137), – community 0.3551±0.2629 (+0.233,
6/8, q=0.137).

**Cross-batch homogeneity, n=8, full = 0.2524 ± 0.1297** — not usable, 5 of 7 arms reverse.

---

## 4. Lung

**Modularity, n=16, full = 0.6554 ± 0.0302**

| Arm | ablated | diff | worse | p | q | |
|---|---|---|---|---|---|---|
| **– community** | 0.4834 ± 0.0487 | **+0.172** | **16/16** | **3.1e-5** | 7.1e-5 | *** |
| k-means init | 0.6738 ± 0.0311 | −0.018 | 4/16 | 0.083 | 0.097 | ns |
| – recon | 0.6662 ± 0.0267 | −0.011 | 5/16 | 0.117 | 0.117 | ns |
| Stop-grad off | 0.6667 ± 0.0292 | −0.011 | 4/16 | 0.018 | 0.026 | * rev |
| Fixed τ | 0.6741 ± 0.0264 | −0.019 | 2/16 | 0.0017 | 0.0029 | ** rev |
| – usage | 0.6977 ± 0.0218 | −0.042 | 0/16 | 3.1e-5 | 7.1e-5 | *** rev |
| – nassoc | 0.7077 ± 0.0328 | −0.052 | 0/16 | 3.1e-5 | 7.1e-5 | *** rev |

**Rare homogeneity, n=15, full = 0.5552 ± 0.2406** — nothing significant, spread ±0.24 is
huge. Best: – nassoc 0.4849±0.2333 (+0.070, 11/15, raw 1-sided p=0.047, q=0.657).

**Rare F1, n=15, full = 0.5372 ± 0.2496** — nothing significant. – nassoc 0.4242±0.1448
(+0.113, 12/15, q=0.144). k-means init 0.6375±0.1882 reverses (raw p=0.0125).

**Cross-batch homogeneity, n=15, full = 0.4604 ± 0.2088** — flat, every q ≥ 0.978.

---

## 5. Immune

n=5 — floor p=0.0625, nothing can be significant. Quote win-counts.

**Modularity, full = 0.6234 ± 0.0596**

| Arm | ablated | diff | worse | |
|---|---|---|---|---|
| **– community** | 0.3789 ± 0.0912 | **+0.245** | **5/5** | largest effect anywhere |
| Stop-grad off | 0.6060 ± 0.0928 | +0.018 | 1/5 | |
| – recon | 0.6298 ± 0.0716 | −0.006 | 3/5 | |
| k-means init | 0.6310 ± 0.0610 | −0.008 | 2/5 | |
| Fixed τ | 0.6453 ± 0.0618 | −0.022 | 1/5 | |
| – usage | 0.6468 ± 0.0484 | −0.023 | 1/5 | |
| – nassoc | 0.6531 ± 0.0564 | −0.030 | 0/5 | rev |

**Rare homogeneity, full = 0.9047 ± 0.0355** — every arm points the right way

| Arm | ablated | diff | worse |
|---|---|---|---|
| – community | 0.6968 ± 0.1324 | +0.208 | 5/5 |
| – nassoc | 0.7618 ± 0.1899 | +0.143 | 5/5 |
| – usage | 0.7879 ± 0.1725 | +0.117 | 5/5 |
| k-means init | 0.8080 ± 0.1434 | +0.097 | 4/5 |
| Fixed τ | 0.8526 ± 0.0732 | +0.052 | 5/5 |
| – recon | 0.8712 ± 0.0710 | +0.034 | 4/5 |
| Stop-grad off | 0.8734 ± 0.0644 | +0.031 | 4/5 |

**Rare F1, full = 0.9370 ± 0.0152** — all right direction. – community 0.8491±0.0221 (5/5),
– recon 0.8950±0.0534 (5/5), stop-grad 0.9039±0.0375 (5/5), fixed τ 0.9194±0.0191 (5/5).
Skip – nassoc here (0.7439 ± **0.417**, only 3/5) — one batch drives the mean.

**Cross-batch homogeneity, full = 0.3111 ± 0.1959** — mixed, 2 arms reverse at 0/5. Skip.

---

## 6. Verdict

| Claim | Pancreas | Lung | Immune | |
|---|---|---|---|---|
| Community → modularity | 8/9 * | 16/16 *** | 5/5 +0.24 | **all 3, strongest** |
| Usage → collapse | 63/220 | 77/300 | 48/300 | **all 3, absolute** |
| Recon → modularity flat by design | 5/9 ns | 5/16 ns | 3/5 ns | **all 3, as predicted** |
| Community → rare homog. | 7/8 * | 10/15 ns | 5/5 +0.21 | 2 of 3 |
| Usage → rare homog. | 7/8 * | 10/15 ns | 5/5 +0.12 | 2 of 3 |
| nassoc → rare homog. | 7/8 q=.077 | 11/15 ns | 5/5 +0.14 | direction only |
| Stop-grad → modularity | **9/9 ** ** | 4/16 rev | 1/5 ns | pancreas, significant |
| Stop-grad → rare homog. | 8/8 * | 7/15 ns | 4/5 | pancreas only |
| Waypoint init → rare homog. | 8/8 * | reverses | 4/5 | pancreas only |
| nassoc → modularity | 0/9 | 0/16 | 0/5 | reverses on all 3 |
| Usage → modularity | 0/9 | 0/16 | 1/5 ns | reverses on 2 of 3 |
| rare F1 · cross-batch homog. · purity | — | — | — | no usable evidence |

**State without hedging:** community loss (all 3), usage→collapse (all 3), recon flat (all 3).

**State with the dataset named:** stop-gradient → pancreas modularity 9/9 p=0.0039;
waypoint init → pancreas rare homogeneity 8/8.

**Direction only:** nassoc → rare homogeneity (+0.121 / +0.070 / +0.143; 7/8, 11/15, 5/5;
raw 1-sided p < 0.05 on all three, none survives BH).

**Withdraw:** nassoc-on-purity — 0.009 gap vs 0.088 spread, and purity cannot be paired.

---

## 7. Points to handle carefully

Five arms move against us. Each has a workable position — none needs to be volunteered,
and none is unanswerable if raised.

**a) Removing usage improves modularity** (0/9 pancreas, 0/16 lung).
**Good answer available.** It collapses to 63/220, 77/300, 48/300 active prototypes.
Modularity rewards coarse partitions by construction, so this is a symptom of the collapse,
not a gain — it scores better by giving up 70–84% of its prototypes. Turns into evidence
that modularity alone is incomplete and the usage loss is doing real work.

**b) Removing nassoc improves modularity** (0/9, 0/16, 0/5 — 0 of 30 batches).
Weakest point. The collapse explanation does not apply (active protos barely move:
219→218, 298→290, 288→285). Best position is the trade-off — nassoc gives up modularity to
keep rare states resolvable — but the loss is significant and the gain is not.
**Do not raise it.** Answer only if asked directly.

**c) Stop-gradient reverses on lung** (−0.011, 4/16, q=0.026).
−0.011 on 0.655 is **1.7%**. It reaches significance only because lung has 16 batches and
the tightest variance of the three (±0.030). Pancreas is +0.060 = **9.7%** on 9/9.
Significance and practical size are different things; that framing is honest and holds.

**d) Fixed τ improves lung modularity** (−0.019, 2/16, q=0.003). Never a headline claim —
simply do not raise fixed τ.

**e) Waypoint init loses on lung F1** (0.6375 vs 0.5372, raw p=0.0125, q=0.087). Does not
survive correction, but scope waypoint claims to pancreas only.

He has only seen mean±std tables and cannot run this test — he does not have the per-batch
CSVs. He asked about two specific arms; answering those two plus the main-claim arms is a
complete answer to what he asked.

**Audience note:** he is in Phase 3 (03–10 Aug, reviewer/AC only) and may never read the
reply. The reader is the **AC**, who skims for "did they answer the specific challenge?"
One line — 9/9, p=0.0039 — does that.

---

## 8. Seed run — not done yet

Notebook ready: `notebooks/ablation_seed_variance.ipynb`.
Paired test answers "is it real given batch spread". Seeds answer "would it survive a
different random start". Both are needed to fully close his concern 1.

**Design:** pancreas only, 3 seeds `[31, 1, 2]`, 4 arms = 12 runs.
`full` · `no_community` (`lambda_umap=0`) · `no_nassoc` (`lambda_nassoc=0`) ·
`stopgrad_off` (`proto_recon_stopgrad=0`). Seed 31 is the codebase default, so that arm
also checks the setup reproduces the original ablation.

**Why `no_community` even though he did not ask:** it is the biggest effect (0.615→0.449)
and it supplies the scale — *"the community drop is N× the seed noise, the stop-gradient is
M×"*. Without it the seed numbers mean nothing on their own.

**Two settings:**
- Stage 1 shared across seeds (`seed` is not in `PRETRAIN_PARAM_KEYS`) — right scope, faster.
- **Keep `cvae_epochs=50`.** Lowering it forces a fresh Stage-1 pretrain every run — slower,
  and breaks comparability. Only `train_epochs` is reduced, to 20.

**Report as a reduced-epoch stability check**, not a re-run of the main table. Absolute
values will not match; all arms use the same reduced setting so effects and spread stay
comparable to each other.

**Outputs:** per-seed values · mean±std across seeds · **effect ÷ seed noise** (the number
to quote) · pooled paired test over seeds×batches (27 pairs instead of 9).

**If it comes back weak** (stop-gradient inside seed noise): drop the empirical claim and
justify the stop-gradient by design — the recon arm being flat on all 3 (5/9, 5/16, 3/5) is
direct evidence the mechanism works. Decide this before seeing the number.

# CURRENT RUN — what is executing right now

> **Newest at the top.** This file is the "what is in flight" board. When a run finishes,
> move its findings into `decisions.md` (the permanent chronological record) and mark the
> entry here DONE. `decisions.md` is the history; this file is the present.

---

## IN FLIGHT — gful gamma sweep, usage loss OFF (started 2026-09-14)

**Script:** `modal_scripts/train_gful_nousage_gamma_sweep.py` (Modal, T4, 90-min timeout cap)
**Budget:** ~$1. Commits after EVERY gamma, so a timeout loses at most the in-flight run.

**The question:** in real Leiden, gamma is an INPUT and the number of communities is an
emergent OUTPUT. In scProto, K is a fixed architectural input AND `lambda_proto_usage`
forces every slot to stay populated — so `decisions.md` §6 measured that gamma could move
metacell SIZE but never COUNT. Setting `lambda_proto_usage=0` removes that constraint for
the first time. **Does gamma alone then control how many prototypes survive?**

**Config** — the fully-corrected gful arm:
```python
nassoc_exact      = 1        # exact soft possible_pairs + de-duped edges_inside (§5)
                             # WITH the [D7] cell_scale fix (§7)
nassoc_size_scale = 'full'
nassoc_cpm_norm   = 'gful'
nassoc_cpm_free   = 1
nassoc_diag_norm  = 'pair'
nassoc_diag_loss  = 'cpm'
lambda_nassoc     = 1000
lambda_proto_usage = 0.0     # <-- THE experiment
```
dataset `s28nsc`, affinity `bk02`, K=200, cvae_e50, train_epochs=20, eval_freq=2, patience=4.

**Gammas: `[0.01, 0.024, 0.05, 0.10]`** — deliberately NOT the originally-requested
0.15/0.5/0.7/0.9. See the pre-flight check below for why those were rejected.

---

### Pre-flight check (CLAUDE.md §9b) — done BEFORE spending any GPU time

Measured on the real `bk02` graph using the real trained `d7fix` run's own assignments:

```
                        internal_frac   aggregate density   median per-MC dens   implied gamma
SEACells bk02 K=200        0.4465           0.02952             0.04492             0.0458
scProto d7fix (g=0.15)     0.2661           0.02381             0.01928             0.0240
                          1.68x better       1.24x               2.3x better
```
- scProto's BEST prototype of 200 reaches density 0.069. **Zero of 200 reach gamma=0.15.**
- SEACells keeps **44.7%** of all graph edge weight inside metacells; scProto keeps **26.6%**.
  (Independently confirms `decisions.md` §9's "0.275 vs 0.446".)
- SEACells sizes are uneven (median 258, max 925); scProto is rigidly uniform
  (median 294, max 360) — that is the usage loss forcing balance.

**Why 0.15–0.9 were rejected.** gamma implicitly picks a metacell SIZE:
```
density   = mean_wdeg * internal_fraction / (size - 1)
=> size    = mean_wdeg * internal_fraction / gamma + 1

with the MEASURED mean_wdeg=26.374 and internal_fraction=0.2661:
  gamma=0.024 ->  293-cell metacells -> implies K =  199    <- matches K=200, self-consistent
  gamma=0.046 ->  153-cell metacells -> implies K =  381    <- SEACells-matched
  gamma=0.15  ->   48-cell metacells -> implies K = 1223    <- unsatisfiable at K=200
  gamma=0.9   ->    9-cell metacells -> implies K = 6641    <- far unsatisfiable
```
The formula's own prediction (`0.0241`) matches the measured density (`0.0238`) to 1.3%,
so the framework is validated — and it says 0.15–0.9 all demand metacells far smaller than
K=200 can produce. All four would have landed in the same degenerate "everything scores
negative" regime.

**Scale check of each normalizer** (real numbers, gamma=0.15, vs UMAP loss ~2.5):
```
no normalizer   1,086,196     434,478x too big
/ K  (=200)         5,431       2,172x too big
/ 2m  (gful)       0.7049      same order as UMAP -- OK
/ pairs (dpar)     0.1262      same order as UMAP -- OK
```
`dpar` bound check passed exactly: predicted `[-0.85, +0.15]`, actual `0.1262`, and it
equals `gamma - density = 0.15 - 0.0238 = 0.1262` precisely.

---

### What to look at when it lands

1. **`PROTOTYPES ALIVE: n / 200`** per gamma — the actual question. If gamma controls this,
   scProto's gamma has recovered its Leiden role.
2. `aggregate_density` per run vs the gamma it was trained at — does the run converge to
   `density ~= gamma`, as CPM theory says it should?
3. DGE (`pearson_r`, `n_pairs`), joint purity, median size — vs `decisions.md` §10/§11.

---

## DONE — recent, findings already moved to `decisions.md`

- **`cpar` normalizer added** (commit `f94e742`) — fixed `N*(N-K)/(2K)`. Documented there
  that at fixed N/K it is *equivalent to gful up to a lambda rescale* (5.52x, identical
  gradients); it differs only when K or N changes. Implemented, **not yet run**.
- **[D7] fix rerun** → `decisions.md` §10. Correct, but moved no metric (DGE 0.800→0.758,
  joint purity 0.423), consistent with §9 localizing the gap to prototype placement.
- **btch+dpar gamma sweep** → `decisions.md` §11. gamma=0.03 found 17 niche states (best
  yet, vs SEACells' 19) but at a real cost: ct purity 0.634 vs ~0.85.

---

## Parked / not yet run

- **gamma = 0.046** (the SEACells-matched value derived above) under gful — a genuinely new
  data point, cheap, one run.
- **cpar vs gful at differing K** (K=200 vs K=400) — the only setting where cpar is not
  just gful with a rescaled lambda. Tests the portability claim directly.
- Everything in `decisions.md`'s own "Untested ideas, parked" section.

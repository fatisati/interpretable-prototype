# scProto loss — what we changed and why

**Chronological order: oldest first, newest last.** Each entry:
**what looked wrong → what we found → what we changed → what happened.**

Dataset for all numbers: `s28nsc` (NSCLC CosMx, 58,423 cells), K=200 prototypes,
affinity `bk02` (BANKSY lambda=0.2) unless stated. Metric `DGE` = how well a metacell's
niche-vs-rest gene changes match the same comparison done on real single cells.

---

## The one-paragraph version

The loss had a bug that made **killing a prototype cheaper than keeping a small one**, so
185 of 200 prototypes died. Fixing that took the DGE score from **0.342 to 0.839**, which
is statistically tied with SEACells (0.849). Along the way we found three more bugs in how
the loss counted things, all now fixed and verified. Two gaps remain: scProto finds fewer
cell states than SEACells (14 vs 19) and its metacells are less "one state each"
(joint purity 0.43 vs 0.50). Neither is caused by the resolution parameter gamma.

---

## 1. The original problem: the loss preferred merging

**What looked wrong.** The model never split a cell type into its sub-states.

**What we found.** The old formula was `edges_inside / volume`, where volume = total
connectivity of the members. **Volume doesn't change when you merge two groups** (it's just
a sum), but edges_inside *grows* (it absorbs the connections between them). So merging
always scored better — splits were structurally vetoed.

**What we changed.** `edges_inside / possible_pairs`. When you merge two groups you invent
`size_A x size_B` brand-new possible pairs, nearly all of them empty, so merging now
*lowers* the score.

**What happened.** DGE 0.163 -> 0.654.

---

## 2. The collapse: why prototypes were dying

**What looked wrong.** Only **15 of 200 prototypes survived**. One metacell swallowed
44-73% of all the cells.

**What we found** (from the training logs, not from theory). The loss printed a "density"
each epoch. A density must be between 0 and 1. It was printing **371 to 510**.

Tracing it: when a prototype has under 1 cell's worth of soft membership,
`possible_pairs` clamps to exactly 0, so `density = edges / 0.00000001` explodes. Then
under the old `(density - 1)^2` loss:

```
a prototype that is completely EMPTY   ->  costs 1
a prototype that is merely SMALL       ->  costs ~100,000,000
```

**The loss was paying the model to kill prototypes.** Emptying one was a hundred million
times cheaper than letting it be small.

**What we changed.** `nassoc_size_scale='full'` — rescale counts to full-dataset units so
sizes sit far above the danger zone.

**What happened.** Collapse gone. 200/200 prototypes used.

---

## 3. Switching from MSE to CPM

**What looked wrong.** The old loss asked every metacell for `density = 1`, meaning every
possible pair of members connected. On a kNN graph that is impossible for anything bigger
than a handful of cells — the target could never be reached.

Also, a gamma sweep (0.01, 0.03, 0.1) gave nearly identical results. Gamma had **exactly
zero gradient**: it cancelled out algebraically against the `sum(possible_pairs)`
denominator. Every gamma experiment before this point was measuring nothing.

**What we changed.** Used CPM (Traag et al. 2011, the objective Leiden uses by default):
```
score[p] = edges_inside[p] - gamma * possible_pairs[p]
```
Written as a **subtraction, not a ratio** — which is exactly why Leiden never hits the
divide-by-zero problem. Plus a normalizer that does not cancel gamma.

**What happened.** DGE 0.654 -> 0.780 on bk08, and **0.342 -> 0.760** on bk02, which had
been collapsing completely.

---

## 4. Loss balance: the normalizer — every variant, why, and the numbers

**What looked wrong.** The nassoc term was **2874 out of a total loss of 2888 (99.5%)**.
Everything else — including the main UMAP community loss, at 2.5 — was invisible to the
optimizer.

**What we found.** Two different jobs had been tangled into one number:

```
normalizer      puts the term on a sensible SCALE.      There is a right answer.
lambda_nassoc   says how much you CARE about the term.  This is the tuning dial.
```

Step 2 had rescaled the *numerator* (edges, sizes) to full-dataset units but left the
*denominator* at minibatch units — a ~3900x mismatch. That is not a tuning problem; no
value of `lambda_nassoc` fixes a term whose scale drifts with batch size.

### The four normalizers, in the order they were invented

The CPM diagonal term is always
```
diag_term = - sum_over_prototypes( edges_inside[p] - gamma * possible_pairs[p] ) / DENOM
```
and the variants differ **only** in `DENOM`. Code: `scproto.py:1603-1627` and `1830-1836`.

---

**`pair` — the original. BROKEN. Never use.**
```
DENOM = sum_over_prototypes( possible_pairs[p] )        <- attached to the graph
```
*Why it seemed reasonable:* possible_pairs is what gamma is charged against, so dividing
by its sum looked like the natural unit.

*Why it is broken:* gamma's own contribution is
`gamma * sum(possible_pairs) / sum(possible_pairs)`, which **is just gamma** — a constant
with no dependence on the assignments. **Gamma's gradient is exactly zero.** Every gamma
experiment run under `pair` (the C_d1d2 sweep at 0.01 / 0.03 / 0.1) was measuring nothing;
the three runs came out nearly identical because they *were* the same run.

*Status:* still the `defaults.py` default, purely so old model-folder names don't change.

---

**`graf` — first working fix.**
```
DENOM = summed affinity weight of THIS minibatch's sampled edges   (detached)
```
*Why:* it depends only on **which edges were drawn**, never on the assignments `S`, so
gamma can no longer cancel against it. That restores gamma's gradient.

*Why it is still not ideal:* the denominator is at **minibatch** scale while the numerator
is at **full-dataset** scale (`nassoc_size_scale='full'`). The term is therefore inflated by
roughly `n_edges_total / n_edges_batch`, and it also **jitters with batch composition**.

*Status:* kept only so the E1 runs stay reproducible.

---

**`gful` — CHOSEN. This is the one in the best config.**
```
DENOM = total affinity weight of the WHOLE graph   (constant, computed once)
```
*Why this one:*
1. **Scale-matched.** Numerator is full-dataset, denominator is full-dataset. Nothing
   cancels, nothing is inflated.
2. **Constant across steps.** Same denominator every minibatch, so the term's meaning does
   not change between steps.
3. **Published precedent.** This is exactly DMoN's `1/(2m)` (Tsitsulin et al., JMLR 24,
   2023), where `2m` = total edge weight. We are not inventing a normalizer.
4. **Gamma keeps its gradient** — the denominator has no `S` in it at all.

Code: `self._full_graph_weight = edge_dataset.weights.sum()`, `scproto.py:1208-1211`.

---

**`dpar` — good alternative, kept as a live option.**
```
DENOM = sum_over_prototypes( possible_pairs[p] )   but DETACHED (frozen, no gradient)
```
*Why:* it is `pair` with the denominator's gradient cut. That single change kills the
cancellation while keeping the objective's own natural bound:
```
score[p]  lies in  [ -gamma * pairs[p] , (1-gamma) * pairs[p] ]
=> the whole normalized term lies in  [ -(1-gamma) , +gamma ]
```
So the term is **bounded on a known interval for any dataset or batch size**, with no
reference to an external quantity. `gful` has no such bound.

*Trade-off vs `gful`:* the denominator still moves as the assignments move (it is
recomputed each step, just not differentiated), so the scale is less stable — but because
it is bounded, **`dpar` is far more forgiving of a badly chosen `lambda_nassoc`.**

### Results — every normalizer measured, s28nsc / bk02 / K=200

| normalizer | lambda_nassoc | gamma | DGE | median size | coverage | joint purity | ct purity |
|---|---|---|---|---|---|---|---|
| `pair` (legacy) | 1 | 0.01 / 0.03 / 0.1 | gamma has no gradient — all three runs identical | | | | |
| `graf` | 1 | 0.15 | 0.760 | 294 | 12 | 0.431 | 0.839 |
| `gful` | 1 | 0.15 | 0.496 | **7** | 27 | 0.357 | 0.790 |
| `gful` | 100 | 0.15 | 0.829 | 291 | 13 | 0.429 | 0.846 |
| **`gful`** | **1000** | **0.15** | **0.839** | 296 | 14 | 0.431 | 0.849 |
| `dpar` | 10 | 0.03 | 0.345 | **8** | 20 | 0.389 | 0.808 |
| `dpar` | 100 | 0.03 | 0.776 | 292 | 13 | 0.439 | 0.858 |
| `dpar` | 100 | 0.01 | 0.778 | 282 | 16 | **0.439** | **0.861** |
| `dpar` | 4000 | 0.03 | 0.748 | 192 | 15 | 0.357 | 0.750 |
| *SEACells* | — | — | *0.849* | *258* | *19* | *0.504* | *0.823* |

**Three things this table shows:**

1. **`graf` at lambda=1 and `gful` at lambda=1000 land in the same place**
   (DGE 0.760 vs 0.839, median size 294 vs 296, joint purity 0.431 both). That is the
   scale/weight separation made concrete: `graf`'s denominator is ~1000x smaller than
   `gful`'s, so `graf` at lambda=1 *was* `gful` at lambda~1000 all along. The old setup
   worked by accident, with the scale factor hidden inside the normalizer where it could
   not be tuned.

2. **Getting the scale right made the term too weak, not too strong.** `gful` at
   lambda=1 fragmented badly (median size 7, DGE 0.496). The correct reading is that
   `lambda_nassoc` needed raising — which is exactly what separating the two jobs lets
   you do.

3. **`dpar` is the more robust choice, `gful` the higher-scoring one.** `dpar` reaches
   its best at lambda=100 and only loses 0.03 DGE at lambda=4000 (a 40x overshoot), where
   `gful` loses 0.34 DGE at a 1000x undershoot. `dpar` also gives the **best cell-type
   purity of any run (0.861)**. If lambda has to be re-tuned for a new dataset, start
   from `dpar`.

**What happened.** `gful` + `lambda_nassoc=1000` gave **DGE 0.839** — the best run to date,
and better than the pre-fix brute-force version.

## >>> The result that drove everything below <<<

After steps 1-4:

```
DGE           0.342 -> 0.839    vs SEACells 0.849   -> statistically TIED (p >= 0.19)
coverage      14 states         vs SEACells 19      -> STILL BEHIND
joint purity  0.431             vs SEACells 0.504   -> STILL BEHIND
```

DGE was solved; coverage and joint purity were not. So the question became:

> **Is the remaining coverage / joint-purity gap caused by gamma being badly set?**

That could not be answered while gamma's units were untrustworthy — which is what steps 5
and 6 set out to fix.

---

## 5. Counting things correctly  (`nassoc_exact=1`)

**What looked wrong.** Gamma never behaved like the "density threshold" it is supposed to be.

**What we found.** Two separate counting errors, both measured against brute force:

```
possible_pairs   0.83x TOO LOW
edges_inside     3.21x TOO HIGH
-> density (= edges/pairs) was ~3.9x too high
-> gamma = 0.15 was actually behaving like ~0.038
```

*Why possible_pairs was wrong.* We used `size x (size-1) / 2`. That formula is for HARD
clustering, where a cell is 100% in one cluster. With soft membership it breaks:

```
4 cells, each 25% in prototype p  ->  size = 1.0
  formula says   1.0 x 0.0 / 2          = 0        "no pairs here"
  truth is       6 pairs x 0.25 x 0.25  = 0.375
```
Worse, at `size = 0.5` the formula returns **-0.125** — a negative count of pairs.
(Step 2 stopped this from being *triggered*; it did not make the value *correct*.)

*Why edges_inside was wrong.* The affinity matrix is symmetric AND stores self-similarity,
and the edge list takes every stored entry. So each connection `{i,j}` is listed **twice**
(as i->j and j->i), plus every cell has an `i->i` self-entry:
```
nnz = 4,648,883  =  2 x 2,295,230 (both directions)  +  58,423 (self-loops)
```

**What we changed.**
```
possible_pairs = ( size^2  -  sum_i assign[i,p]^2 ) / 2
edges_inside   = de-double-counted, self-loops removed
```
The new possible_pairs formula is just "count each pair once, don't count a cell with
itself." It gives exactly `n(n-1)/2` when clustering is hard, and it can never be negative.

**What happened.** Both verified exact against brute force (ratio 1.000000).
**DGE was neutral**: 0.839 -> 0.800, inside the noise (stds are 0.11-0.13). This was a
*correctness* fix, not a performance fix.

> **GAP IN THE EVIDENCE.** The `nassoc_exact=1` runs (H1, H4) were only ever evaluated on
> **DGE metrics** — `pearson_r`, `kendall_tau`, `tpr`, `n_pairs`, `n_metacells`,
> `median_mc_size`. **Joint purity and cell-type purity were never computed for them.**
> So "does correct counting change purity?" is an OPEN question, not a settled one. Every
> purity number quoted in this file comes from an `exact=0` run
> (`notebooks/s28nsc_dge_lite/tables/variant_comparison.csv`). Running the purity eval on
> H4_g015 is cheap and should be done before drawing any conclusion about exact math and
> purity.

---

## 6. Sweeping gamma  (most recent)

**What gamma is.** The price of a *missing* connection inside a metacell, measured in units
of one *real* connection. High gamma = "I demand tightly connected metacells."
Low gamma = "loose metacells are fine."

**What we found.** With the units finally correct, gamma trades one thing against the other:

| gamma | DGE | states found | typical metacell size |
|---|---|---|---|
| 0.025 | 0.781 | 12 | 272 cells |
| 0.054 | 0.782 | 11 | 289 cells |
| **0.15** | **0.800** | 15 | 297 cells |
| 0.5 | 0.712 | 15 | 188 cells |
| 0.7 | 0.647 | **18** | 204 cells |

Higher gamma -> smaller metacells -> **more states found** but **worse DGE**.
No single gamma gives both. Joint purity never rose above 0.44 at any gamma.

**Correction to an earlier claim.** We first said "gamma does nothing." That was wrong —
we had only tried 0.025 and 0.054, too narrow a window. Over the full range it moves DGE
by 0.15 and metacell size by 109 cells.

**Still true, though:** gamma cannot change *how many* metacells exist. K is fixed at 200
and the usage loss keeps all 200 alive, so the AVERAGE size is pinned at
`N/K = 58,423/200 = 292` by arithmetic alone. In Leiden, gamma's whole job is to decide
the number of communities; here that decision is already made before gamma is consulted.

What gamma *does* control is the size **distribution**, via `gamma * possible_pairs[p]`.
`possible_pairs` is quadratic in size, so a big prototype is charged much more than two
small ones holding the same cells — that is the size-equalizing force, and it is the CPM
term, **not** the usage loss. Measured: median size 297 at gamma=0.15 -> 188 at gamma=0.5.

**The answer to the question that drove steps 5-6: NO.** Gamma is not the cause of the
coverage / joint-purity gap. Three fixes and ~8 GPU runs went into making gamma
interpretable, and the payoff was ruling it out — a real result, but it moved no metric.

---

## Side finding: BANKSY lambda

The spatial-mixing parameter has **different optima for the two methods**:

| lambda | scProto | SEACells |
|---|---|---|
| 0.2 | 0.839 | **0.849** |
| 0.4 | **0.846** | 0.722 |
| 0.8 | **0.780** | 0.665 |

scProto wins at 0.4 and 0.8, ties at 0.2. **Do not assume an affinity tuned for one method
transfers to the other.**

---

## Where we stand

```
bk02 (BANKSY lambda=0.2), K=200, single seed
                     DGE     states found   joint purity   cell-type purity
  old loss           0.342   2              0.178          0.405      (15/200 survived)
  scProto best       0.839   14             0.431          0.849      (200/200)
  SEACells           0.849   19             0.504          0.823
```

**DGE is statistically tied** — every paired test across pearson_r, kendall_tau and tpr
gives p >= 0.19.

**Two real gaps remain**, and they are *not* gamma problems:
1. **Coverage** — scProto finds 14 states, SEACells 19. The 5 it misses are all rare
   (under 1000 cells).
2. **Joint purity** — 0.43 vs 0.50. A metacell should be one cell type in one niche;
   scProto's are more mixed.

These two are probably the same underlying issue: **scProto is not resolving small states.**

Remaining suspects, all **untested**:
- **K is a fixed input.** All 200 prototypes are kept alive, so mean size is pinned at 292
  by arithmetic. Rare states need small metacells. Note the size-equalizing pressure comes
  from CPM's `gamma * possible_pairs` (quadratic in size), **not** from the usage loss —
  see the note in section 6. So "relax the size constraint" means lowering gamma, which
  the gamma sweep shows costs coverage rather than gaining it.
- **The 8-dimensional latent space.** Assignment is a soft Voronoi partition in 8-d. A state
  not separable there cannot be expressed no matter what the loss says. Consistent with
  scProto's internal-edge-fraction of 0.275 vs SEACells' 0.446.
- **The off-diagonal term does nothing.** It averages over 39,800 prototype pairs that are
  almost all zero, contributing ~0.005 and exerting no pressure to keep rare prototypes
  distinct.

---

## Best config to use

```python
lambda_config = dict(LAMBDA_PROTO_UMAP_PRECON) | {
    'nassoc_diag_norm': 'pair',
    'nassoc_diag_loss': 'cpm',
    'nassoc_cpm_free':  1,
    'nassoc_size_scale':'full',
    'nassoc_cpm_norm':  'gful',
    'nassoc_gamma':     0.15,
    'lambda_nassoc':    1000,
}
```
Also in CLAUDE.md section 9c. `nassoc_exact=1` is correct but performance-neutral.

---

## 7. A second counting bug, found while auditing section 5 (2026-09-14)

**What looked wrong.** Nothing was reported wrong yet -- this was found by re-deriving
section 5's fix from scratch and checking it against real `s28nsc` affinity data before
trusting it, not from a training symptom.

**What we found.** Sections 5's two identities (`possible_pairs = (size^2-sumsq)/2` and
the de-duplicated `edges_inside`) are exact algebraic identities on their own -- re-verified
here against independent brute-force counting on a real `bk08` affinity submatrix
(2000 real cells), both matched to ratio 1.000000. That part of section 5 is solid.

But `nassoc_size_scale='full'` (the config this repo actually runs with -- see "Best
config to use" above) rescales `size` and `sum_i assign^2` to full-dataset units BEFORE
the identity is applied, and the rescaling used two different powers of the same scale
factor for two quantities that should use the same one:

```
cell_scale = n_total_cells / n_batch_cells

size_eff = size_b  * cell_scale        # linear -- correct, size is a sum over cells
sumsq    = sumsq_b * cell_scale**2     # quadratic -- WRONG, sumsq is ALSO a sum over cells
```

`size` and `sumsq` (`sum_i assign[i,p]`, `sum_i assign[i,p]^2`) are both plain sums over
cells -- extensive quantities that combine additively when you add more cells, so both
should rescale the same way, linearly. `size_eff**2` ends up quadratic in `cell_scale`
automatically, because it is the SQUARE of an already-linearly-rescaled number -- that is
not evidence `sumsq` needs its own quadratic factor.

**Two independent checks, both on real numbers:**

1. *Hard-assignment invariant.* For 0/1 assignments, `sumsq == size` always, at any
   population size. `size_eff = size_b*c` is correct, so `sumsq_eff` must equal
   `size_b*c` too -- not `size_b*c^2`. The `c^2` version breaks the very
   "reduces exactly to n(n-1)/2" guarantee section 5 claims for `nassoc_exact`.
2. *Real `s28nsc` `bk08` affinity, real scale.* `n_total=58,423`, `n_batch=1,024`,
   `cell_scale=57.05` (these are the project's actual numbers, not illustrative). Built a
   genuinely rare prototype: 3 real cells own it in the full dataset (true pairs=4.22,
   from the exact identity applied to the full population). One of those 3 cells landed
   in a random minibatch.
   ```
   size_eff (both versions)        = 9.39
   pairs_eff, sumsq*cell_scale**2 (current code) = 0.00   <- collapses to zero again
   pairs_eff, sumsq*cell_scale    (fix)           = 43.34  <- noisy (n=1 sample) but never zero
   ```
   `pairs_eff=0` is exactly the divide-by-zero failure mode section 5 was written to
   eliminate -- the `c^2` bug reintroduces it, specifically for the rare/small
   prototypes this whole fix chain (D5, D6) targets.

**Why the section-5 brute-force check ("ratio 1.000000") didn't catch this.** That check
verified the closed-form identity against combinatorial counting on raw batch values --
it never exercised the `nassoc_size_scale='full'` rescale, which is a separate code path
only entered when that option is on (it is, in "Best config to use"). The identity itself
is still exactly correct; the bug is in what gets fed into it.

**What we changed.** `interpretable_ssl/trainers/scproto.py` (~line 1719, tagged `[D7]`)
and the matching comment in `configs/defaults.py`'s `nassoc_exact` entry:
```
_sumsq = _sumsq * (_cell_scale ** 2)      # before
_sumsq = _sumsq * _cell_scale             # after
```

**What this means for section 5's numbers.** The `nassoc_exact=1` DGE comparison
(0.839 -> 0.800) was run with the `c^2` bug present (config used `nassoc_size_scale='full'`
per "Best config to use"). That comparison should be treated as **not yet valid** until
`H1`/`H4` are rerun with this fix -- the direction (neutral-to-slightly-worse) might hold,
might not, particularly for the rare-prototype / coverage story in section 6's "Where we
stand" (rare states are exactly where this bug bites hardest).

**Status:** code fix applied and logged (this entry). NOT YET RE-RUN on GPU -- ask before
spending Modal budget on it, per this project's cost-awareness rule (see CLAUDE.md
section 11).

---

## 8. Correction to section 5(b): self-loops don't reach training, so they were never
   the mechanism -- but the fix is still correct (2026-09-14)

**What looked wrong.** While confirming section 5 was safe (the user asked whether every
cell being connected to itself with weight 1.0 -- the max possible weight -- means the
WeightedRandomSampler that drives training (`scproto.py:947`, samples edges proportional
to weight) would sample self-loops on almost every draw, wasting the training signal.

**What we found.** The self-loop numbers in section 5(b) (`nnz=4,648,883 = 2*edges +
58,423 self-loops`, diag=1.0) are real, but they describe `self.aff` -- the raw cached
`graphs/*.pkl` file straight out of `generate_affinity()`. They do **not** describe what
training actually uses. `MultiCropsDataset.__init__` (`adata_augmenter.py:124-127`)
builds a SEPARATE copy immediately after loading:
```
_raw = self.aff.tocsr(copy=True)
_raw.setdiag(0)
_raw.eliminate_zeros()
self.aff_raw = _raw
```
and `scproto.py` uses `self.train_ds.aff_raw` (not `.aff`) everywhere that matters --
`_setup_umap_edges` (builds `EdgeDataset` + the `WeightedRandomSampler`, line 911) and
every nassoc computation. Verified directly on the real `bk08` file:
```
self.aff       (raw cached pkl):  nnz=4,058,285   self-loops=58,423, weight=1.0 each
self.aff_raw   (what training uses): nnz=3,999,862   self-loops=0
```
**So the sampler concern is a non-issue** -- self-loops are stripped before `EdgeDataset`
is ever built, `WeightedRandomSampler` never sees them, and every training edge is a real
cell-cell pair.

**Consequence for section 5(b).** The "edges_inside inflated by self-loop contamination"
half of that finding was measured against the wrong object -- in real training,
`self_term` (the self-loop correction term in the `nassoc_exact` code) is always exactly
0, because there ARE no self-loops in `aff_raw` to find. The **other** half of 5(b) (each
real edge counted twice, once as `(i,j)` and once as `(j,i)`) is unaffected by this and
is still the real, necessary correction -- verified algebraically: even with
`self_term=0`, `edges_in_b = 0.25*(A_na.diagonal() - 2*self_term)` reduces to
`0.25*(4*true_edges - 0) = true_edges`, exactly. **The `nassoc_exact` code fix needed no
change** -- it was written to handle self-loops IF present and correctly degrades to "no
self-loops" when they aren't, which is what real training actually hits. Only the
*narrative* in section 5(b) (and the earlier explanation given to the user in this same
conversation) overstated the self-loop's role; both are corrected here.

**Real numbers, one deterministic minibatch (seed fixed, reproducible):** a real 292-cell
neighborhood around cell 30000 (ground truth: 135 cells after de-duplication, true
`edges_inside=760.6`, true `possible_pairs=9045`, true `density=0.084`), with exactly 5 of
its real members forced into an otherwise-random 1024-cell batch at soft weight 0.9:
```
size_b (this batch)  = 4.50   sumsq_b = 4.05
possible_pairs  BUGGY (c^2, section 7's bug) = 26,366   density=0.362
possible_pairs  FIXED (c,   section 7's fix) = 32,843   density=0.291
possible_pairs  GROUND TRUTH (full membership, no extrapolation) = 9,045   density=0.084
```
Both badly overshoot the true value here -- that overshoot is separate sampling noise
(`edge_scale`, which rescales the few edges visible in one minibatch up to full-dataset
scale, is itself noisy for a single draw) and would average out over many minibatches
across an epoch; it is not the c/c^2 bug. The c/c^2 gap itself (26k vs 33k, ~25% here) is
real but modest for a prototype that gets several members per batch -- it only becomes
catastrophic (collapse to exactly 0, per section 7's donor-cell example) for prototypes
so rare that a batch sees a small FRACTION of one member's soft weight, not whole members.

---

## Untested ideas, parked

- **VQ-VAE style restart of dead prototypes** (`proto_reinit_every`). Implemented. Showed
  nothing — but only because there were no dead prototypes left to revive.
  *Refs: van den Oord et al. NeurIPS 2017; Dhariwal et al. 2020 (Jukebox).*
- **DMoN collapse regularization** (`lambda_collapse`). Implemented. A soft "keep sizes
  balanced" prior that, unlike SwAV, does not force equal sizes.
  *Ref: Tsitsulin et al., JMLR 24 (2023).*
- **min() off-diagonal normalizer** (`nassoc_offdiag_norm='min'`). Implemented, no effect
  yet. Makes a rare prototype leaking into a big one visible.
  *Ref: Kannan, Vempala & Vetta, JACM 2004.*
- **Fix the off-diagonal's real problem.** Using `max` or a top-k mean instead of `mean`
  would target the actual overlapping pairs rather than averaging over 39,800 zeros.
- **Learn gamma instead of setting it.** PMN (arXiv:2308.09644) makes gamma a trained
  parameter, specifically to avoid having to know K in advance.

---

## Known limitations of everything above

1. **Single seed.** Every comparison here is one run. Standard deviations are 0.08-0.25, so
   any difference under ~0.1 is not distinguishable from noise.
2. **One dataset, one slide.** `crcx` (3-patient, real batch effects) has never been re-run
   with the fixed loss — its only comparison predates all of this.
3. **`Tumor cells x Tumor surface` fails for every method** (0.03-0.31). Diagnosed: its
   ground-truth signal is genuinely weak — largest fold-change across all 960 genes is
   0.584, where every other state reaches at least 1.3. Not a method failure. It drags
   every average down and arguably should be excluded.
4. **Gamma cannot act as a resolution parameter while K is fixed.** This is structural, not
   a tuning issue, and it should not be described as a resolution knob in a paper as things
   stand.

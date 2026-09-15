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

**What we found.** The diagonal CPM term, in full, always looks like this:

```
diag_term  =  - ( sum over every prototype p of  edges_inside[p] - gamma * possible_pairs[p] )  /  normalizer

  edges_inside[p]     real weighted connections between two members of prototype p
  possible_pairs[p]   ALL member-pairs of p, connected or not -- the "capacity"
  gamma               price charged per possible pair that is NOT a real edge
                       (equivalently: the density a prototype must clear to be "worth it")
  normalizer          what the summed score gets divided by. pair / graf / gful / dpar
                       differ ONLY in this.
```

**Why `normalizer` has to exist at all.** Sum `edges_inside[p] - gamma*possible_pairs[p]` over
200 prototypes and the raw number's size depends on the dataset -- how many cells, how
many edges, how big the minibatch is -- none of which has anything to do with how much
you actually want this loss term to matter relative to the others. Two separate jobs had
been tangled into that one raw number:

```
normalizer      puts the term on a sensible, dataset-size-independent SCALE. This has a
                right answer: whichever choice keeps the numerator and normalizer
                measured at the SAME scale (batch vs full-dataset -- see section 7) and
                keeps gamma's gradient alive (see `pair` below, where it doesn't). Not
                something you tune by feel.
lambda_nassoc   says how much you CARE about this term relative to every other loss
                (UMAP, usage, ...). This IS a free tuning dial -- there is no "correct"
                value, only what balances well against the other terms.
```

Step 2 (an earlier fix, below) had rescaled the *numerator* (edges, sizes) to
full-dataset units but left `normalizer` at minibatch units -- a ~3900x mismatch. That is not
a tuning problem; no value of `lambda_nassoc` fixes a term whose scale silently drifts
with batch size underneath it.

### The four choices for `normalizer`, in the order they were invented

Code: `scproto.py:1603-1627` and `1830-1836`.

---

**`pair` — the original. BROKEN. Never use.**
```
normalizer = sum_over_prototypes( possible_pairs[p] )        <- attached to the graph
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
normalizer = summed affinity weight of THIS minibatch's sampled edges   (detached)
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
normalizer = total affinity weight of the WHOLE graph   (constant, computed once)
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
normalizer = sum_over_prototypes( possible_pairs[p] )   but DETACHED (frozen, no gradient)
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

## 6. Sweeping gamma

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

Remaining suspects — **section 9 has since settled these, read it instead**. Kept here
as the record of what was believed before the separability experiment:
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

**Does `nassoc_size_scale='btch'` + `nassoc_cpm_norm='dpar'` need this fix too? No --
the rescale step never runs for it.** The bug lives entirely inside the
`if nassoc_size_scale == 'full':` branch that multiplies `sumsq` by `_cell_scale` (or, in
the bug, `_cell_scale**2`). Under `'btch'` that branch is skipped completely:
`size_eff = size_b` (the batch's own raw count, no rescaling at all), and for
`nassoc_exact`, `_sumsq` is used exactly as computed on the batch, no `_cell_scale`
anywhere. So `possible_pairs[p] = (size_b^2 - sumsq_b)/2` runs directly on the batch's own
numbers -- nothing to get wrong, and no rerun is needed for any run that used `'btch'`.
`'btch'`+`'dpar'` trades this bug for a DIFFERENT, older problem instead: a rare
prototype's raw BATCH size can itself land below 1 purely from having few members in one
minibatch (not from any rescaling math), so `possible_pairs` can still be tiny/fragile
there -- just for the reason section 5 originally described, not this one.

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

## 9. Where the niche signal actually dies  (newest)

**What looked wrong.** After steps 1-6, DGE was tied with SEACells (0.839 vs 0.849,
p >= 0.19) but two gaps refused to move no matter what we did to the loss:
coverage 14 vs 19, joint purity 0.431 vs 0.504.

### 9a. First: decompose the joint-purity gap

Joint purity = for each metacell, the fraction of its cells sharing the most common
`celltype|niche` label. Split into its two halves:

```
                  ct_purity   niche_purity   joint_purity
SEACells bk0.2      0.823        0.591          0.504
scProto  bk0.2      0.849        0.488          0.431
                    +0.026       -0.103         -0.073
                    WE WIN       WE LOSE
```

**The whole gap is niche purity.** Our metacells are CLEANER than SEACells' on cell
type and MESSIER on spatial niche. Empirically `joint ~= 1.04 * ct * niche` for both
methods (within 0.4%), so matching SEACells' niche purity while keeping our own
cell-type purity would give `1.04 * 0.849 * 0.591 = 0.522 > 0.504` -- we would WIN.
Target: niche purity 0.488 -> 0.571.

### 9b. Hypothesis tested: is SEACells over-splitting on position?

The idea: SEACells partitions a spatial embedding directly, so maybe it carves out
"states" that are spatially distinct but have NO real gene-expression signal -- and
its coverage and niche-purity advantages are partly fake.

Test: for each state a method found, check whether its metacell DGE matches the real
single-cell DGE. States found but not reproducible = spurious.

```
                                       mean pearson_r
  14 states BOTH methods found             0.855
   5 states ONLY SEACells found            0.850     difference -0.005
```

**REJECTED. SEACells' extra states are real.** Their tpr is 0.95-1.00 (the highest in
the table) and each has 25-87 genes significantly DE in the REAL SINGLE CELLS
(Vascular endothelium|Alveolar spaces 87, Macrophages|Airways 74, Cytotoxic T
cells|Tumor surface 65, Fibroblasts|Alveolar spaces 39, Regulatory T cells|
Desmoplastic stroma 25). scProto found ZERO states SEACells missed -- strict subset.

Side observation from the same check: the coverage ceiling is far higher than either
method reaches. Of 146 (cell type, niche) pairs in the ground truth, **64 have enough
signal to be testable** (>=5 significantly DE genes; MIN_GENES=5 is the gate). SEACells
resolves 19, scProto 14. And three of the STRONGEST states are missed by BOTH:
Macrophages|Tumor core (229 DE genes), Tumor cells|Airways (226), Tumor cells|Vascular
stroma (206). "Match SEACells at 19" is a much lower bar than the data supports.

### 9c. The decisive experiment: where is the niche signal lost?

Protocol: for each cell type separately, predict the NICHE label from a representation.
Logistic regression, class_weight='balanced', 5-fold CV, balanced accuracy, 15 cell
types with >=2 niches of >=30 cells. Run locally, CPU only, no training.
Scripts: `notebooks/s28nsc_dge_lite/scripts/{niche_sep,get_z,sep2,sep3}.py`
Tables:  `notebooks/s28nsc_dge_lite/tables/s28nsc_niche_sep_*.csv`

`z` was recovered by rebuilding the encoder in plain torch from the checkpoint state
dict (960 -> 31 -> ReLU -> 8, single condition so the cond term is a constant bias) and
VALIDATED by reproducing the saved assignments: `adata.X` + ReLU scored 0.669 agreement
versus 0.12-0.32 for every other input/activation combination. Not 1.0 because the saved
labels come from the training-time balanced assignment, not a plain argmax over 200
prototypes; logistic regression is invariant to invertible linear maps, so `z` is right.

```
representation            balanced acc   lift    % of expression signal retained
  chance                      0.162      --
  expression PCA50            0.316     +0.155        100%   (what is available)
  scProto z (8-d latent)      0.303     +0.141         91%   <- ENCODER IS FINE
  soft assignment (200-d)     0.246     +0.084         54%   <- 40% LOST HERE
  hard metacell (one-hot)     0.249     +0.088         57%
  expression PCA8 (control)   0.295     +0.134         86%
  spatial xy coords           0.299     +0.138         89%
```

**THREE FINDINGS, and they settle the argument:**

1. **The signal IS in gene expression, at the CELL level, not just as a group-mean
   effect.** 0.316 vs 0.162 chance, about 2x. Expression is even slightly MORE
   informative about niche than raw spatial coordinates (0.316 vs 0.299). So "the
   information isn't there" is NOT available as an excuse for the loss.

2. **The encoder is exonerated.** z retains 91% of the niche signal in 50 PCs, and
   BEATS a dimension-matched PCA8 (106%). The 8-d bottleneck costs almost nothing.
   => **Raising latent_dim 8 -> 16 could gain at most +0.014 lift (~9%).** Not the fix.
   The earlier "capacity spent on cell type" hypothesis was WRONG.

3. **The prototype assignment is where 40% of the niche signal dies** (91% -> 54%).
   The 200 prototypes are placed such that dot-product assignment discards niche
   information that z demonstrably carries. Hard vs soft makes no difference (0.246 vs
   0.249) because at epsilon=0.060 the softmax is already essentially one-hot: mean
   assignment entropy is 0.113 nats out of a possible 5.30.

**VERDICT: the loss is at fault, and the fault is localized to prototype placement.**
Not the encoder, not the latent dimension, not missing spatial input.

Note this REVERSES an earlier claim in this file and retires the argument that spatial
info must be fed to the encoder as an input. It also retires the evidence previously
offered for it -- "niche purity does not respond to 4x more spatial weight in the graph
(0.491 -> 0.496)" -- which was CIRCULAR: a loss that ignores niche would equally fail to
respond to a better graph. That observation never discriminated between the two
explanations.

### 9d. What this points at next (untested)

The off-diagonal term is the prime suspect for prototype placement: it averages over
39,800 prototype pairs that are almost all zero, contributes ~0.005 to the total loss,
and therefore exerts essentially no pressure to keep prototypes distinct. Candidate
fixes: `max` or top-k instead of `mean` (so the term targets the actually-overlapping
pairs), or the `min` normalizer that is already implemented but has never shown an
effect because nothing else was in a state to reveal it.

---

## 10. [D7] fix rerun on GPU: confirms correctness, moves no metric (2026-09-14)

**What we did.** Reran `H4_g015` (`nassoc_exact=1`, `nassoc_size_scale='full'`,
`nassoc_cpm_norm='gful'`, `gamma=0.15`, `lambda_nassoc=1000`, everything else identical to
"Best config to use") on Modal/T4, same config as the buggy baseline, but with the [D7]
fix from section 7 in the code (verified synced and present before launching). New run
saved as `lf_h4_g015_d7fix` so the buggy baseline stays on disk for comparison. cvae_e50,
train_epochs=20/eval_freq=2/patience=4 (current CLAUDE.md standard -- the buggy run's
folder name doesn't encode epoch count, so this is the best-available match, not a
confirmed exact reproduction).

```
                         pearson_r  kendall_tau   tpr    n_pairs  median_mc_size  joint_purity
H4_g015 (buggy, c^2)       0.800       0.542      0.975    15         297            -- (never computed, see section 5's GAP note)
H4_g015_d7fix (fixed, c)   0.758       0.530      0.974    14         294            0.423
```
`joint_purity` for the fixed run: 0.423 (`purity_micro` in `collapse_row`) -- close to but
slightly under the old NON-exact scProto-best figure quoted in section 9 (0.431), and well
under SEACells (0.504). `n_pairs=14` matches that same old non-exact baseline exactly
(section 9's "14 vs 19" coverage number), not the buggy exact run's 15.

**Verdict: the fix is correct (section 7's derivation and real-data check both hold), but
it does not move DGE, coverage, or joint purity.** If anything, on this single seed, DGE
is very slightly lower with the fix (0.800 -> 0.758, well inside both runs' large stds
[~0.13-0.20] -- not a significant regression, but definitely not an improvement either).

**This is not a surprise in hindsight -- section 9 already explains why.** Section 9c
localized the ENTIRE coverage/joint-purity gap to prototype PLACEMENT (dot-product
assignment discards 40% of the niche signal the encoder demonstrably carries), not to
the diagonal CPM counting math. [D7] only touches counting math. A fix to counting math
was never going to move a problem section 9 already pinned on assignment/placement --
this rerun is a confirming data point for section 9's conclusion, not a contradiction of
it. Same pattern as section 6: gamma being uninterpretable was real and worth fixing, and
fixing it "moved no metric" either, for the identical underlying reason.

**Practical takeaway:** `nassoc_exact=1` should still be considered CORRECT and worth
using for its own sake (the counting is now right, full stop), but it is not a candidate
fix for the coverage/joint-purity gap. Section 9d's off-diagonal-term suspect remains the
live lead.

Artifacts: `notebooks/s28nsc_dge_lite/tables/H4_g015_d7fix_{summary,per_pair}.csv`,
`models/s28nsc/lf_h4_g015_d7fix_.../metrics.json` (Modal volume).

---

## 11. `btch`+`dpar` gamma sweep -- a real n_pairs gain, at a real purity cost (2026-09-14)

**Why this ran.** Section 7 showed `[D7]` only affects `nassoc_size_scale='full'`; under
`'btch'` (`dpar`'s partner normalizer, no full-dataset rescaling) there is nothing to fix,
but `gamma=0.15` was calibrated for `'full'`-scale density and has no reason to be the
right threshold for `'btch'`-scale (raw, unrescaled, noisier) density. Existing `dpar`
runs already on the volume (`lf_g1_bn_g01/g03`, WITHOUT `nassoc_exact`) used gamma in
0.01-0.03, a real prior. Swept `[0.01, 0.03, 0.05]` WITH `nassoc_exact=1`, everything else
matching "Best config to use" except `nassoc_size_scale='btch'`, `nassoc_cpm_norm='dpar'`.

```
                              median_size  n_pairs  pearson_r  ct_purity  joint_purity  coverage  modularity
gamma=0.01 (lf_i_dpar_g01)      291.5        11       0.782      0.850       0.432        0.944     0.279
gamma=0.03 (lf_i_dpar_g03)      158.5        17       0.718      0.634       0.355        0.778     0.208
gamma=0.05 (lf_i_dpar_g05)      197.5        14       0.715      0.658       0.359        0.778     0.204
  for reference --
H4_g015_d7fix (full+gful, 0.15) 294.0        14       0.758      0.827       0.423        0.944     0.261
SEACells bk0.2 (section 9)       --          19        --         0.823      0.504         --        --
```

**Gamma clearly does something here** (unlike section 6's `full`+`gful` sweep, where
gamma only moved size, never coverage): `0.01` barely splits anything (median size stays
near the untouched ~292 average), `0.03` roughly halves it and finds MORE niche states
than any other `nassoc_exact` run so far -- 17, the closest to SEACells' 19 of anything
tested. **But it is a real tradeoff, not a free win**: cell-type purity crashes from
~0.83-0.85 (every other healthy run) to 0.634, joint purity drops to 0.355 (below
`H4_g015_d7fix`'s 0.423), and cell-TYPE coverage (fraction of the 18 cell types
represented at all, not niche-state count) drops 0.944 -> 0.778 -- some cell types are
losing representation even as niche granularity increases for the ones that remain.

**Interpretation, tentative.** This is the first lever in this whole investigation that
moves `n_pairs` at all (sections 6, 7, 10 all "moved no metric" on coverage) -- consistent
with section 9's conclusion that the bottleneck is prototype placement, since `gamma`
here is directly controlling how aggressively prototypes fragment, which IS a placement
lever, unlike the diagonal counting fixes. Worth reading alongside section 9d's
off-diagonal-term suspect: an off-diagonal term with real teeth might let gamma fragment
prototypes for niche resolution WITHOUT paying the cell-type-purity cost seen here, since
right now nothing is stopping a split from cutting across cell-type boundaries instead of
along niche ones.

**Not run yet, open questions:** a finer gamma grid between 0.01 and 0.03 (where the
purity cliff happens), and whether pairing `nassoc_offdiag_norm='min'` (already
implemented, per section 9d) with `dpar` gamma=0.03 recovers cell-type purity while
keeping the n_pairs gain.

Artifacts: `notebooks/s28nsc_dge_lite/tables/lf_i_dpar_exact_g{01,03,05}_{summary,per_pair}.csv`,
`models/s28nsc/lf_i_dpar_exact_g{01,03,05}_.../metrics.json` (Modal volume).

---

## 12. Turn the usage loss OFF — and all 200 prototypes still survive (2026-09-15)

**Decision: run with `lambda_proto_usage = 0.0` at gamma <= 0.15.** Numbers in
`EXPERIMENTS.md` §4; this section is the *why*.

**What changed.** At matched gamma=0.15, `latent_dims=16`: usage ON gives DGE 0.819 /
13 states, usage OFF gives **0.834 / 16 states**. A second matched pair at 8-d
(gamma 0.025 ON -> 0.781/12 vs 0.024 OFF -> 0.821/13) points the same way. Worth roughly
+0.015 to +0.040 DGE and +1 to +3 states.

**Why nothing dies without it.** This was the real surprise — the usage loss exists to
stop prototypes collapsing, and removing it killed nothing. **All 200 prototypes stayed
alive in every usage-OFF run, at every gamma from 0.01 to 0.20.** Two reasons, the first
measured and the second mechanistic:

1. **The collapse it was guarding against was a BUG, and the bug is fixed.** §2 and §7
   document the two counting errors that made the loss *pay* the model to kill
   prototypes — `possible_pairs` clamping to 0 for sub-1-cell soft prototypes, so an
   empty prototype cost 1 while a merely small one cost ~1e16. With `nassoc_exact=1` and
   the [D7] `cell_scale` fix in place, the incentive to empty a prototype is gone, so the
   guard has nothing left to guard.
2. **Softmax makes emptying a prototype self-defeating.** Every cell's memberships sum to
   1, so cells vacating prototype `p` must land in other prototypes, inflating those.
   Under CPM the penalty is quadratic in size, so inflating a neighbour costs more than
   the vacating prototype saves. The equilibrium keeps all 200 populated on its own.

**Why it was HURTING.** The usage loss is
`-log(EMA(max over cells of assign[cell, p])).mean()`. It has no headcount term, but it
*is* a separation pressure: for `max assign[.,p]` to approach 1, prototype `p` needs a
cell much closer to it than to any other prototype, so prototypes are pushed apart. That
pressure competes with nassoc for the same prototype positions, and nassoc is the term
that actually knows about the graph. Removing it frees placement.

**Where it is NOT safe: gamma = 0.20.** Usage OFF there produced a degenerate partition —
density halved (0.02577 -> 0.01011), `max/median` 9.3, joint purity 0.363, DGE 0.493 —
even though it hit 20 states, beating SEACells' 19. Coverage bought that way is
worthless. The usage loss was evidently suppressing that failure mode, so **turning it
off is a decision coupled to keeping gamma low**, not an independent one.

**What it does NOT do: equalize sizes.** At matched gamma=0.15, `max/median` is 1.2 with
usage ON and 1.3 with it OFF — no change. The size-equalizing force is CPM's
`gamma * possible_pairs`, which is quadratic in size and therefore always pushes cells
from big prototypes to small ones. An earlier note in `main-idea.txt` asserted this
without a controlled run; it has now been removed from there and the measurement lives in
`EXPERIMENTS.md` §5.

---

## 13. The gamma sweeps did not solve it — and now we know why (2026-09-15)

**Decision: stop sweeping gamma upward. Confine gamma to below ~0.026.** Full sweep
results in `EXPERIMENTS.md` §3; this is the reasoning.

Gamma was swept across `{0.01, 0.024, 0.025, 0.03, 0.046, 0.05, 0.15, 0.2, 0.5, 0.7}`,
with the usage loss both ON and OFF, at 8 and 16 latent dims. **No setting beat SEACells
on both DGE and coverage.** Higher gamma reliably buys states and loses quality
(gamma 0.7 -> 18 states at DGE 0.647; gamma 0.20 usage-OFF -> 20 states at DGE 0.493).
Two independent measurements explain why raising it can never work:

**(a) Above gamma ~= 0.0265 the request is physically unsatisfiable.** Gamma implicitly
names a metacell size, but size is pinned by the architecture:

```
requested_size = mean_weighted_degree * internal_fraction / gamma + 1
actual_size    = N / K = 58,423 / 200 = 292          (every cell is assigned)

gamma_max = mean_weighted_degree * internal_fraction / (N/K - 1)
          = 26.374 * 0.2926 / 291  =  0.0265
```

Above it every prototype scores negative, the gradient points the same way everywhere and
the term stops discriminating between prototypes. Measured consequence: an 8x change in
gamma (0.024 -> 0.20) moved density by 5%. **Gamma is not a density knob.** Density is
fixed by `mean_weighted_degree * internal_fraction / (size - 1)` — a formula that predicts
every uniform-size run to under 1% — and gamma appears nowhere in it.

**(b) Above gamma = 0.0386 the loss RANKS SEACELLS' BETTER PARTITION LOWER.** Evaluating
our own objective on both label sets:

```
cpm_total = edges_inside - gamma * total_possible_pairs

                        edges_inside   total_possible_pairs
SEACells                  343,971          11,650,407
scProto (best run)        225,400           8,577,888

gamma = 0.024 : scProto  +19,530   SEACells  +64,361   -> SEACells wins 3.3x
gamma = 0.0386: BREAK-EVEN
gamma = 0.15  : scProto -1,061,283 SEACells -1,403,590 -> scProto "wins" by 342,307
```

SEACells captures more edges but is charged for more pairs, because uneven sizes inflate
the sum of squares. **Our long-standing default gamma=0.15 sits deep in the regime where
the objective prefers the worse partition** — so the DGE those runs achieved came from the
UMAP/reconstruction terms, not from nassoc. This splits the problem cleanly:

```
gamma > 0.0386  ->  MISSPECIFIED objective. Hopeless by construction, no optimizer helps.
gamma < 0.0386  ->  objective is correct; the optimizer never found the better partition.
                    A different failure, needing a different fix.
```

**What this retires.** "Higher gamma will fix coverage" — tested at 0.15/0.2/0.5/0.7 and
refuted, with a mechanism for why. Also retires the framing that K=200 is too few:
**SEACells uses the same K=200 on the same 58,423 cells and reaches DGE 0.849 / 19 states
/ internal_fraction 0.4465**, so K=200 is provably sufficient and the gap is ours.

**Still open.** Why `internal_fraction` is 0.29 vs SEACells' 0.4465 — which is the same
question as why our density is lower, since the graph and the sizes match and the formula
leaves no third term. §9 localizes it to prototype placement; `EXPERIMENTS.md` §8-9 record
the first two attempts to pin it down (an expressivity probe whose strong reading was
withdrawn, and an untested lead that scProto's encoder never receives spatial context
while SEACells partitions a BANKSY embedding that does).

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

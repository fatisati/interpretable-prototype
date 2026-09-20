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

## 14. The gap is INFORMATION, not the loss: the encoder never sees spatial context (2026-09-15)

**This supersedes the working assumption behind sections 3-13.** All of that work treated
the gap to SEACells as a loss-design problem. It is not.

**The measurement** (`probe_expressivity_ladder.py`, numbers in `EXPERIMENTS.md` §8b). Fit a
linear, **no-bias** head — matching our real head exactly — to SEACells' own 200 labels,
in-sample, from different feature spaces, and measure the internal_fraction of the
partition each produces:

```
features            dim   int_frac   gap closed    accuracy
pca50                50    0.2790       -9%         0.436
z16 (our encoder)    16    0.3016        6%         0.410
BANKSY X_bklam02     50    0.4262       87%         0.714
SEACells (target)    --    0.4465      100%          --
```

`pca50` and `BANKSY` are the SAME dimension, the SAME head and the SAME target. The only
difference is that BANKSY mixes each cell's expression with its **spatial neighbours'**.
That alone moves internal_fraction from 0.2790 to 0.4262.

**scProto's encoder cannot see that.** `encode_adata` (trainer.py:257) feeds a scPoli
dataloader over `adata`: gene expression plus batch key. Spatial information reaches
scProto *only* through the affinity graph inside the nassoc loss — never through the
encoder input. SEACells, by contrast, partitions `X_bklam02` directly (its run is stored
as `seacell_X_bklam02_K200`).

**Two hypotheses this kills:**
- *The encoder is destroying information.* No — z16 (0.3016) beats pca16 (0.2624) at
  matched dimension. The encoder concentrates signal as intended. PCA-16 was the right
  control because it needs no trained encoder, so it answers "is 16-d enough?" without
  depending on any particular trained z.
- *We need more dimensions.* No — pca8 -> pca50 buys only 0.2402 -> 0.2790, and pca50
  still loses to our 16-d z. Latent 32/64 would not have helped.

**Corroborated independently by §13's follow-up.** Running gamma inside the provably
correct window (0.020-0.032, §13(a)+(b)) fixed the loss's internal behaviour — density
finally tracks gamma (asked/achieved ratio 1.02 at gamma=0.026, confirming the satisfiable
bound exactly), internal_fraction rose to 0.3338, sizes became SEACells-like (median 259
vs 258, max/median 2.7 vs 3.6) — and moved **neither DGE nor coverage** outside noise. A
correct loss on blind features still cannot find the partition.

**The circularity caveat, stated because it is significant.** SEACells was fit on
`X_bklam02`, and the `bk02` affinity graph on which internal_fraction is measured was also
built from `X_bklam02`. So BANKSY features predict a BANKSY-built partition scored on a
BANKSY-built graph, and part of the 87% is circular. **But that circularity is the
finding**: our nassoc loss optimizes containment on that same graph, so scProto is asked
to maximize a spatially-defined objective with an encoder that is structurally blind to
space. Also note the probe imitates SEACells' *label identities*, so each number is a
lower bound on what a feature space can express — evidence, not proof.

**Decision: stop tuning the loss for this gap.** The remaining loss-side levers (gamma,
normalizer, usage, latent dims) are exhausted — see `EXPERIMENTS.md` §10 for the full
ruled-out list. The next change is to what the ENCODER CONSUMES: feed `X_bklam02` (already
present in `adata.obsm`, no computation needed) instead of raw expression. That is also the
fairness fix — SEACells currently gets spatially-aware features and scProto does not, so
the published comparison is not like-for-like as things stand.

**Not yet implemented** — it is a trainer change, pending sign-off.

---

---

## 15. gamma is DERIVED from the graph now, not swept (2026-09-17)

**Decision: gamma = 0.026-0.030 for `s28nsc`/`bk02`, and the derivation method replaces
sweeping on any new graph.** Numbers in `EXPERIMENTS.md` 11a; this is the reasoning.

CLAUDE.md 9b demanded gamma be derived from measured graph properties, and the
2026-09-17 edit marked the derivation UNSETTLED. It is now settled, and the derivation
is the CPM move rule itself:

```
delta(move cell i into p) = (affinity from i to p's members) - gamma * size[p]
                          = size[p] * ( MEAN affinity from i to p's members - gamma )
=> cell i joins p  iff  mean_affinity(i -> members of p) > gamma
```

So gamma is **the minimum AVERAGE affinity a cell needs to a group to be worth joining
it** -- a property of the graph, measurable with no training. The mean is over ALL
members, including the ones i has no edge to, which is why gamma is far smaller than a
typical edge weight (0.34 on `bk02`).

**What this retires.** Blind gamma sweeping. Any new graph gets measured instead --
`modal_scripts/measure_gamma_from_graph.py`. Pancreas/`arbf`, for contrast, measures
0.11 rather than 0.028: gamma is not portable across graphs.

**What it does NOT do.** The derived value lands inside the window already swept in
`EXPERIMENTS.md` 3d, which moved neither DGE nor coverage. Deriving gamma made it
principled; it did not make it a lever.

---

## 16. Three loss bugs fixed, and the honest verdict on them (2026-09-18)

**Decision: keep all three fixes ON for spatial work (`nassoc_exact=1, nassoc_ht=1,
nassoc_cpm_norm='btsz', nassoc_gamma_scale='soft'`), but do NOT report any of them as
closing the gap to SEACells, because none of them moved it.** Results in
`EXPERIMENTS.md` 11d-11f.

**Why they were made.** All three are genuine defects, found by simulating the real
sampler rather than by algebra -- two earlier algebraic derivations in this session were
both wrong, because a batch is 1024 EDGES drawn proportional to `w` and its cells are
their deduplicated endpoints, so nothing in a batch is a uniform sample.

```
[D9]  w applied twice: edges sampled proportional to w, then multiplied by w again, so
      the objective was effectively weighted by w^2 -- while the UMAP term reading the
      SAME batch was not. Bias 1.046 overall, 1.347 under 25 cells (size-dependent, so
      no gamma retune can absorb it). Fixed: sum the membership overlap with NO w.
[D9]  possible_pairs estimated from degree-biased cells with a uniform N/n_b rescale,
      which cannot correct a non-uniform sample. Fixed with a Horvitz-Thompson survey
      weight q_i = p_i/mean(p_i), p_i = 1-(1-2 d_i/W)^B. Bias 1.124 -> 0.999, and flat
      at every prototype size, with LOWER spread than before (p90 1.281 vs 4.157).
[D10] gamma specified on the HARD density scale but compared against a SOFT one.
      Measured ratio 0.68, i.e. gamma was ~1.47x too strict. Fixed by measuring the
      ratio once from the pretrained encoder and converting.
```

**Why they did not help.** The estimator bias is ~1.0x in the 200-900 cell band where
every prototype actually sits (it is 2.4x only below 25 cells), and the gamma rescale is
1.47x, inside a window already swept flat. Both were predicted in advance not to move
the metrics, and both did not. That prediction holding is itself evidence the model of
the system is now correct.

**A hypothesis this KILLED.** "The estimator over-charges small prototypes, which is why
our sizes are rigid, which is why we miss rare states." Correcting it made sizes MORE
uniform (max/median 2.7 -> 1.7, nothing under 100 cells). Size uniformity is CPM's
intrinsic quadratic charge, exactly as section 5 said -- not the estimator, not the
usage loss.

**Still unfixed, deliberately.** Epsilon calibration FAILS on both datasets tried and
silently falls back to `umap_proto_effk=5.0`, a hard-coded default nothing chose; effk
then drifts 5.0 -> 2.7 because calibration runs once. That is the root cause behind
[D10] and it is a real repair, not bundled here.

---

## 17. The CPM rewrite is a SPATIAL-specific fix, not a general improvement (2026-09-18)

**Decision: do NOT port the CPM diagonal to the non-spatial main-table datasets. The
published `volume` + `(M_kk-1)^2` objective stays for those.** Numbers in
`EXPERIMENTS.md` 11g.

**How this came up.** The published objective in `neurips_manuscript/sections/method.tex`
is `M[k,j] = A[k,j]/sqrt(vol[k] vol[j])` with `L = (1/K) sum_k (M_kk - 1)^2 + alpha *
mean_offdiag(M^2)`, at `lambda_nassoc=1` with usage on. **It has no gamma at all.** So
every pathology chased in sections 3-16 -- the density scale, the saturation, the
estimator bias, gamma itself -- belongs to the CPM rewrite, which exists to fix the
spatial collapse of section 2 (15/200 survivors on `s28nsc`) that pancreas never had.

**Evidence.** Reproducing the published pancreas config with today's code gives
modularity 0.617/0.611 against the paper's 0.61 and purity 0.981/0.972 against 0.98 --
an essentially exact match, with 0 unused prototypes and coverage up to 1.0. Porting the
CPM rewrite to pancreas (`gamma=0.11`, `lambda_nassoc=200`, usage off) collapsed
modularity to 0.3127, early-stopped at epoch 4, `min_size=1`, `max/median=17.2`.

That comparison changed four things at once (lambda 1 -> 200, usage 0.1 -> 0, latent
20 -> 8, and the whole diagonal), so it does not isolate the cause -- `lambda_nassoc`
at 200 on a saturated term is the prime suspect. But the direction is clear enough to
stop porting by default.

**Bonus: the reproduction is a REGRESSION TEST and it passed.** Every option added in
this session ([D9], [D9b], [D10]) claims to be behaviour-preserving at its default.
Reproducing the published numbers with the new code proves that, rather than asserting it.

**Process note, worth keeping.** `TASK1_METRICS` in `result_tables.py` uses
`weighted_mean_batch_entropy` / `weighted_mean_cell_type_purity`, NOT the `mean_*` keys.
Reading the unweighted ones in-session produced a spurious "the paper does not
reproduce, 2.5x off" alarm that cost real time. Check the metric keys against
`result_tables.py` before ever claiming a reproduction failure.

---

## 18. Adopting DMoN modularity as an alternative to CPM (2026-09-18)

**Decision: add `nassoc_diag_loss='modu'` -- a soft modularity diagonal in DMoN's exact
form -- as an alternative to the CPM diagonal. CPM stays available; nothing is removed.**
Numbers in `EXPERIMENTS.md` 11i and the Leiden control; this section is the reasoning.

### Why: CPM's own optimum is a bad partition

`leiden_cpm_control.py` ran three optimizers on the same graph at the same K=200:

```
                     DGE   pairs  jointP  int_frac   med   min
Leiden-modularity   0.856    18    0.534   0.5147    282    15
Louvain             0.859    18    0.542   0.5062    291    37
SEACells            0.849    19    0.504   0.4465    258    --
Leiden-CPM          0.630    23    0.530   0.5509    309     1
scProto best (CPM)  0.842    13    0.437   0.3338    259    --
scProto at CPM-opt  0.605    15    0.425   0.5501     42     1
```

**Modularity beats CPM by +0.23 DGE with the same optimizer, same graph, same K.** And
scProto at the K-matched gamma reproduced Leiden-CPM's DGE almost exactly (0.605 vs
0.630) -- so our implementation is faithful and CPM's optimum is simply a poor partition.

Note what is NOT different: **joint purity is a tie** (0.530 vs 0.534) and CPM resolves
MORE states (23 vs 18). CPM is not broadly worse; it is specifically bad at DGE.

### Two mechanisms, both measured

**(a) Degree-awareness.** `bk02` weighted degree spans 17.8x (min 9.95, mean 26.37, max
177.26). CPM charges a flat gamma per pair; modularity charges
`resolution * degree_i * degree_j / total_degree`. Tested directly -- community purity
against the mean degree of its members:

```
purity by mean-degree quartile   sparsest -> densest     spearman(degree, purity)
Leiden-modularity   0.499  0.526  0.522  0.484                +0.025
Leiden-CPM          0.438  0.534  0.537  0.520                +0.118
scProto-CPM         0.302  0.244  0.269  0.414                +0.304
```

Under modularity, being in a sparse region costs nothing. Under CPM the sparsest
communities are the worst, and in our own run the coupling is strongest of all.

**(b) CPM permits singletons; modularity does not.** Minimum community size: CPM **1**,
modularity **15**, Louvain **37**. CPM's resolution-limit-freeness cuts both ways --
nothing stops a 1-cell community. DGE is computed from pseudobulk, and tiny communities
give noisy profiles. This is a plausible route from "more states resolved" to "worse
correlation"; it is consistent with the numbers but was not isolated experimentally.

### Practical reasons, which weigh as much as the metric

```
resolution TRANSFERS from Leiden (19.43 -> K=200 on bk02). gamma never transferred:
  derived from SEACells density -> 0.028, which gives Leiden K=1043 not 200
  K-matched -> 0.00686; after the [D10] soft conversion -> 0.00444; three different
  numbers in one session, none externally checkable.

CORRECTNESS IS VERIFIABLE. Feed Leiden's hard partition into our modularity formula;
  it must equal igraph's modularity at the same resolution. CPM had no reference
  implementation -- which is how two counting bugs survived for weeks (9b).

THE SAMPLER SUITS IT. prototype_volume is a DEGREE-WEIGHTED sum over cells, and our
  cells arrive with probability roughly proportional to degree, so the bias and the
  weighting cancel. CPM's possible_pairs is an UNWEIGHTED pair count, which the same
  sampling gets wrong and needed Horvitz-Thompson to repair ([D9]).

PUBLISHED PRECEDENT. DMoN (Tsitsulin et al., JMLR 24 2023) and DGCluster (arXiv
  2312.12697) both optimize soft modularity in a deep model. A search for a soft or
  differentiable CPM found nothing published -- ours is an unvalidated adaptation.

IT RETURNS TO THE PUBLISHED FAMILY. The NeurIPS loss (volume normalization +
  (M_kk - 1)^2) is normalized association -- degree-aware by DIVIDING by volume.
  CPM moved out of the degree-aware family entirely, and that is when the calibration
  problems started. Modularity is a better-behaved member of the original family.
```

### Why the degrees must be the FULL-GRAPH ones, not batch-local

A tempting simplification -- treat the batch as its own graph and use batch degrees --
**destroys the objective**, and the arithmetic is decisive:

```
1024 sampled edges -> 2048 endpoint slots -> ~2006 unique cells
average appearances per cell = 1.02
```

Almost every cell appears exactly ONCE regardless of its true degree, so the batch
degree distribution is nearly flat while the real one spans 17.8x. A flat null model is
a constant charge -- i.e. **batch-local modularity degenerates back into CPM.** It would
also make `resolution` batch-size-dependent and non-transferable, which is exactly the
problem that consumed the CPM work.

So: `weighted_degree` and `total_degree` are used EXACTLY (precomputed, free). Only the
two sums are estimated from the batch. Unlike CPM, where `possible_pairs` had to be
estimated in full, here the null model itself carries no sampling error.

### Are small / rare states safe now? Yes, down to ~6 cells -- and it took TWO things

This is the question the whole coverage problem reduces to, so it is worth separating
the two independent failure modes. BOTH had to be fixed; neither alone is enough.

**(1) The ESTIMATOR must not over-charge small prototypes.** This was the binding
constraint under CPM. Measured by simulating the real sampler 400 times against known
partitions (`modal_scripts/measure_modularity_estimator.py`):

```
                              overall   25-50 cells   under 25 cells
affinity_inside                1.001       1.011            --         unbiased, flat
volume (Horvitz-Thompson)      0.998       0.993          0.987        unbiased, flat
volume^2 BY SQUARING           1.095       1.923          3.245        BROKEN
volume^2 by the EMA product    0.999       0.988          0.989        unbiased, flat
```

Squaring an estimate inflates it by its own variance (`E[X^2] = (E X)^2 + Var`), and
small prototypes have the most variance -- so they would be charged up to **3.2x too
much**, i.e. the loss would systematically punish exactly the rare states we want.
Multiplying the live estimate by a DETACHED running average removes it completely.
**FIXED.** For comparison, CPM's `possible_pairs` had the same defect at 2.4x under 25
cells, and it was never fully fixed -- `[D9]`'s Horvitz-Thompson repaired the linear
part, not the square.

**(2) The OBJECTIVE ITSELF must be able to resolve them.** This is modularity's
resolution limit and it is NOT an estimator issue -- it would apply to a full-batch
implementation too. Derived from modularity's exact merge rule (two communities merge
iff `affinity_between > resolution * vol_X * vol_Y / total_degree`):

```
min_resolvable_size = n_cells * sqrt(mean_edge_weight) / sqrt(total_degree * resolution)

s28nsc / bk02   at resolution 19.431 ->  6.2 cells
pancreas / arbf at resolution 36.517 ->  2.5 cells
```

Against the data: `s28nsc` has **175 non-empty celltype x niche states**, of which
**149 have at least 10 cells** and only **5 have 3 or fewer** (the smallest are
`Plasma cells | Tumor core` and `Dendritic cells | Smooth muscle structures`, 1 cell
each). So the limit sits below everything we could plausibly resolve at K=200.

**Conclusion: yes, small states are safe down to ~6 cells**, with the floor set by the
objective rather than by our sampling. The 5 states below that are out of reach for any
method at K=200.

**Two caveats, recorded so they are not forgotten.**

- "Can resolve 6 cells" is not "will produce 6-cell communities". Leiden-modularity's
  smallest community on `bk02` was **15 cells** (CPM's was 1). Modularity may simply
  decline to make very small communities even where the limit permits them. Untested
  for our soft version.
- The user's original form of the size formula, `n / sqrt(2m * resolution)`, is correct
  up to a factor: it implicitly assumes the connecting edge has weight 1. With
  `mean_edge_weight = 0.3357` on bk02 it overestimates by `1/sqrt(0.3357) = 1.73x`.
  The derived form above is the one to use.

**Why this is a real improvement over CPM, not just a different failure.** Under CPM,
small prototypes were hit twice: the estimator over-charged them (2.4x under 25 cells),
AND the flat `gamma` meant sparse-region communities were lower quality regardless of
size (purity-vs-degree spearman +0.304 in our run, +0.118 for Leiden-CPM, against
+0.025 for modularity). Modularity removes both: the estimator is flat across size, and
the degree-proportional charge removes the density dependence.


### Known costs, recorded so they are not rediscovered

1. **Resolution limit** (Fortunato & Barthelemy, PNAS 2007). Modularity cannot resolve
   communities below a scale `sqrt(2m/resolution)`. Measured on bk02: at resolution 1 the
   threshold is 1,241 internal edge weight and **39% of SEACells' metacells fall below
   it**; at resolution 19.43 the threshold is 282 and only **7.5%** do. Not binding in the
   regime we use, but it WOULD be at scanpy's default resolution=1.
2. **The squared term is still biased.** `volume^2` is estimated by squaring an estimate,
   so `E[X^2] = (E X)^2 + Var(X)` inflates it -- the same defect as CPM's possible_pairs.
   Fix with one live factor times one detached running average, not by squaring the
   estimate.
3. **DMoN reports needing a collapse regularizer.** Modularity does penalize the one-blob
   solution (it scores `1 - resolution` = -18.4 at our resolution), but the gradient near
   a uniform assignment is weak. Three guards already exist: the off-diagonal term,
   `lambda_proto_usage`, and `lambda_collapse` (DMoN's own, implemented and parked).
4. **All the DGE evidence is from Leiden, a hard optimizer.** One data point says
   objectives transfer to scProto (Leiden-CPM 0.630 -> scProto-CPM 0.605). That is
   encouraging, not proof.
5. **Single seed throughout.**

### What it would buy

```
vs our current best:  +5 resolved states (13 -> 18) at equal DGE, +0.10 joint purity
vs SEACells:          parity on DGE (0.856 vs 0.849) and coverage (18 vs 19), BETTER on
                      joint purity (0.534 vs 0.504) and internal fraction (0.515 vs 0.447)
```

i.e. matching SEACells on its own terms while beating it on joint purity -- the stated
goal, which no scProto run has reached.

### The loss, exactly

**Step 1 -- DMoN's form, on the whole graph.**

```
affinity_inside[k] = sum over ALL ordered pairs (i,j) of
                       affinity[i,j] * assign[i,k] * assign[j,k]
volume[k]          = sum over ALL cells i of  weighted_degree[i] * assign[i,k]
total_degree       = sum over ALL cells i of  weighted_degree[i]        # = 2m = 1,540,839

modularity = (  sum_k affinity_inside[k]
              - resolution * sum_k volume[k]^2 / total_degree  ) / total_degree

loss_diag_modularity = -modularity
```

Each prototype earns the affinity it actually contains and is charged the affinity it
would contain BY CHANCE given how connected its members are. That charge is the only
structural difference from CPM, which charges a flat `gamma` per pair.

**Step 2 -- what one training step actually sees.**

```
n_sampled_edges = 1024        edges drawn WITH REPLACEMENT, probability proportional
                              to affinity, from the DIRECTED list (each undirected edge
                              appears twice, so the list's total weight IS total_degree)
batch_cells     = ~2000       the deduplicated endpoints of those edges
assign[i,k]     = softmax(logits[i] / epsilon)[k]
```

Both sums above run over the whole graph, so both must be estimated. **The degrees are
NOT estimated** -- `weighted_degree` and `total_degree` are precomputed constants
(`self._cell_degree`, `self._full_graph_weight`). Only the two sums carry sampling error.

**Step 3 -- the edge sum.** A draw picks edge e with probability
`affinity[i,j] / total_degree`, so for a single draw

```
E[ assign[head,k] * assign[tail,k] ]
    = sum over all edges of (affinity[i,j]/total_degree) * assign[i,k]*assign[j,k]
    = affinity_inside[k] / total_degree
```

One draw returns, on average, `affinity_inside[k] / total_degree`. Scaling back up:

```
affinity_inside_est[k] = (total_degree / n_sampled_edges)
                       * sum over sampled edges of assign[head,k] * assign[tail,k]
```

**There is deliberately NO `affinity` factor inside the sum** -- the sampler already
applied it. Multiplying again weights the objective by `affinity^2`; that was the [D9]
bug, measured at 1.046 overall and 1.347 for prototypes under 25 cells.

**Step 4 -- the cell sum (Horvitz-Thompson).** Cells appear unevenly: a cell is in the
batch only if one of ITS edges was drawn, so high-degree cells appear more often.
Horvitz-Thompson (1952) is the standard repair -- count each sampled item as
`1 / its probability of being sampled`:

```
prob_in_batch[i] = 1 - ( 1 - 2*weighted_degree[i]/total_degree ) ^ n_sampled_edges
                   # "1 minus the chance no draw touched it"; already computed as
                   # self._nassoc_p_in for [D9]

volume_est[k] = sum over batch_cells of
                  weighted_degree[i] * assign[i,k] / prob_in_batch[i]
```

Exactly unbiased, because
`E[ sum over sampled of f(i)/p_i ] = sum over all i of p_i * f(i)/p_i = sum over all i of f(i)`.

**Why this is cheap here.** For most cells `prob_in_batch[i]` is close to
`n_sampled_edges * 2*weighted_degree[i]/total_degree`, so

```
weighted_degree[i] / prob_in_batch[i]  ~  total_degree / (2*n_sampled_edges)   -- a CONSTANT
```

The degree weighting and the sampling bias cancel: it reduces to "count assignments and
scale". The exact form is still used because `prob_in_batch` saturates for hub cells
(0.21 against a 0.034 mean), so the constant is not quite constant at the top end. This is
the opposite of CPM's `possible_pairs`, an UNWEIGHTED pair count where the same sampling
bias had nothing to cancel against and Horvitz-Thompson did real work.

**Step 5 -- the squared term, which is still biased.** `volume[k]^2` must NOT be computed
by squaring an estimate: `E[X^2] = (E X)^2 + Var(X)`, so noise inflates it. Concretely, a
true volume of 100 estimated as 90 or 110 with equal probability averages to 100
correctly, but squaring gives `(8100+12100)/2 = 10,100` against a true 10,000. Small
prototypes wobble most, so they are over-charged most -- exactly the 2.4x measured on
CPM's possible_pairs below 25 cells. The fix is to multiply the live estimate by a STEADY
one rather than by itself:

```
volume_running[k] <- 0.9 * volume_running[k] + 0.1 * volume_est[k].detach()
volume_squared_est[k] = volume_est[k] * volume_running[k].detach()
```

`90*100 = 9,000` and `110*100 = 11,000` average to exactly 10,000. The gradient still
flows at full strength through the live factor.

**Step 6 -- the objective.**

```
modularity_est = (  sum_k affinity_inside_est[k]
                  - resolution * sum_k volume_squared_est[k] / total_degree
                 ) / total_degree

loss_total = loss_community
           + 0.01 * loss_proto_recon
           + lambda_nassoc * ( -modularity_est + alpha * loss_offdiag )
```

### The normalizer, and why `total_degree`

Divide the whole thing by `total_degree`, which is DMoN's own convention. Two reasons,
the second worth more than the first:

1. **Scale.** The raw numerator is of order 1.5 million; dividing puts modularity in
   `[-1, 1]`, so `lambda_nassoc` stays interpretable against the UMAP term.
2. **It makes our number THE modularity number** -- bit-for-bit what igraph and Leiden
   report for a partition. That buys two things CPM never had:
   - `resolution` transfers in from Leiden unchanged (19.43 -> K=200 on bk02);
   - the implementation is verifiable -- feed Leiden's hard one-hot partition into the
     formula and it must equal igraph's modularity at the same resolution, before any
     GPU time is spent.

**Alternatives considered and rejected.**

```
divide by the batch edge count ('btsz', what CPM uses)
   -> the term becomes batch-native, so `resolution` stops being a graph property,
      drifts with batch size, and no longer transfers from Leiden. This is precisely
      the calibration trap that consumed the CPM work (sections 15-17).

use batch-local degrees instead of the precomputed ones
   -> FATAL. 1024 edges give 2048 endpoints over ~2006 unique cells, i.e. 1.02
      appearances per cell: almost every cell appears exactly once regardless of its
      true degree. The batch degree distribution is nearly FLAT while the real one
      spans 17.8x, so the null model becomes a constant charge and modularity
      DEGENERATES BACK INTO CPM. The exact degrees are precomputed and free; there is
      no reason to do this.

no normalizer
   -> the term is ~1e6 against a UMAP term of ~3.
```

### Checks before trusting any number (CLAUDE.md 9b)

```
range         modularity_est must lie in [-1, 1]. Outside that is a bug -- stop.
at init       near-uniform assignments give (1 - resolution)/n_prototypes
              = (1 - 19.43)/200 = -0.092
one big blob  gives 1 - resolution = -18.43, so collapse is never rewarded
balance       lambda_nassoc * loss_diag should sit in the same range as loss_community (~2.5)
EXACT TEST    our formula on Leiden's partition == igraph's modularity at that resolution
```

### Choosing `resolution`

Ask Leiden: bisect its resolution parameter until it returns the K we use. Already
measured on `bk02`: **resolution = 19.43 gives K = 200**. Because of the normalization
above, that value drops straight into the loss -- no derivation, no sweep, no soft/hard
conversion. Pancreas needs its own from the same one-line search on `arbf`.

Caveat specific to us: in Leiden, K is an OUTPUT that resolution determines; in scProto K
is fixed at 200 architecturally. So resolution here sets the PRESSURE, not the count.
Choosing the value at which the graph naturally supports 200 communities means asking for
something the graph already agrees with -- an argument, not a guarantee. Verify after the
first run by counting non-empty prototypes.


**Status: NOT implemented.** Config shape when it is: `nassoc_diag_loss='modu'` (a new
value of an existing key, 4-char distinct from mse/hing/cpm/nll/nll2, so no
`constants.py` change) plus a new numeric key `nassoc_modu_res` (needs BOTH
`defaults.py` and `interpretable_ssl/constants.py`, per section 9).

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

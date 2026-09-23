# scProto loss terms — exact formulas

Reference for **modularity**, **CPM**, **nassoc off-diagonal**, **cohesion**, and the
**assignment memory**. Every formula here was read off the code, not remembered —
file:line is given for each so you can check it. Plain text, no LaTeX.

Written 2026-09-23.

---

## 0. Shared notation

Used by every formula below.

```
n_cells          number of cells in the dataset
K                number of prototypes (metacells)
assign[i][k]     SOFT assignment of cell i to prototype k. Rows sum to 1:
                 sum over k of assign[i][k] = 1.  Written S[i][k] in the code.
affinity[i][j]   the input graph. Symmetric, non-negative, ZERO diagonal
                 (setdiag(0) -- every graph-cut consumer needs no self-loops).
degree[i]        sum over j of affinity[i][j]        (weighted degree)
total_degree     sum over i of degree[i]             (= 2m, the "2m" in modularity)
size[k]          sum over cells i of assign[i][k]    (soft cell COUNT of prototype k)
volume[k]        sum over cells i of assign[i][k] * degree[i]  (soft DEGREE mass)
bio(i)           cell i's biological batch (pancreas: obs['tech'], 9 values)
```

**size vs volume is the single most load-bearing distinction in this file.**
`size` counts cells; `volume` counts edge endpoints. They behave differently under
merging, and that difference decides whether a loss will split or merge communities.

A minibatch sees only a fraction of the graph:

```
p_batch = n_cells_in_minibatch / n_cells       (0.0625 on pancreas at batch 1024)
```

Anything that is *quadratic* in cells (pair counts) survives subsampling at rate
`p_batch^2`; anything *linear* (cell counts) survives at `p_batch`. Comparing the two
without rescaling is the single most common bug in this codebase — see §7.

---

## 1. Modularity (soft / DMoN)

`scproto.py:2298-2365`, enabled by `lambda_modularity > 0`. A **complete objective on
its own** — one scalar, no diagonal/off-diagonal split. Set `lambda_nassoc = 0` when
using it.

### Formula

```
affinity_inside[k] = sum over cell pairs (i,j) of assign[i][k]*assign[j][k]*affinity[i][j]

modularity = (   sum over k of affinity_inside[k]
               - resolution * sum over k of volume[k]^2 / total_degree
             ) / total_degree

loss = -modularity
```

In words: **the affinity actually found inside prototypes, minus the affinity you would
expect inside them if the same cells were wired up at random, keeping every cell's
degree.** `volume[k]^2 / total_degree` is that null expectation.

`resolution` (`modularity_res`, default 1.0) scales the null term — higher resolution
punishes large prototypes harder, giving more, smaller communities.

### How the code actually computes it

It never forms the full graph. Edges are sampled, and two corrections are applied:

```
affinity_inside[k] = (total_degree / n_sampled_edges)
                     * sum over sampled edges (h,t) of assign[h][k]*assign[t][k]

volume[k]          = sum over minibatch cells i of assign[i][k] * degree[i] / p_in[i]
```

- the `total_degree / n_sampled_edges` factor rescales the sampled edge sum back to
  full-graph scale;
- `p_in[i]` is cell i's **inclusion probability**. The minibatch is built from
  weight-sampled *edges*, so its cells are degree-biased — high-degree cells appear far
  more often. Dividing by `p_in[i]` is a survey weight that undoes that bias.

One subtlety: the squared volume term is computed as `volume[k] * volume_ema[k]`, where
`volume_ema` is a **detached** exponential moving average (`modularity_ema_alpha`,
default 0.9). Using `volume[k]^2` directly on a minibatch would be biased, because
`E[X^2] != E[X]^2` — the EMA supplies a low-variance estimate of the second factor and
carries no gradient.

**Validated**: on a hard one-hot partition this formula matches `igraph`'s modularity to
**1e-16** at four resolutions (K = 9 to 573). It is not an approximation of modularity;
it is modularity.

### Its gradient — and why that is the problem

```
d(modularity charge) / d assign[i][k] = resolution * degree[i] * volume[k] / total_degree
```

Read that carefully: it contains `degree[i]` and `volume[k]`, and **nothing that relates
cell i to the members of prototype k**. The reward half is similarity-driven, but the
charge half can only move a cell for *size* reasons. Since `sum_k volume[k]^2` is
minimised at equal volumes, the charge pushes toward **uniform-size prototypes** —
while the partition we actually want is uneven (alpha cells are 5493 cells, t_cell is 7).

That is the defect cohesion (§4) exists to fix.

### `modularity_agg`

- `'glob'` (default) — one modularity over the whole graph, as above.
- `'pbch'` — compute it **per biological batch**, then for each prototype take the
  **max over batches**, then sum over prototypes (`scproto.py:2307-2346`). Each batch
  keeps its own `total_degree`, its own degrees and its own volume EMA — with 9
  technologies a batch sees only ~1024/9 = 114 sampled edges, so a shared EMA would
  reinstate the variance bias (measured 1.9x at 25-50 cells). With one batch, `'pbch'`
  is exactly `'glob'`.

---

## 2. CPM — Constant Potts Model (the nassoc diagonal)

`scproto.py:2091-2112`, selected by `nassoc_diag_loss='cpm'`.

### Formula

```
edges_inside[k]   = sum over cell pairs (i,j) of assign[i][k]*assign[j][k]*affinity[i][j]

possible_pairs[k] = 0.5 * ( size[k]^2 - sum over i of assign[i][k]^2 )
                    ("how many pairs COULD exist among k's members".
                     This is the SOFT form; the hard form 0.5*size*(size-1) is
                     also in the code. They agree when assignments are one-hot,
                     because then sum_i assign^2 == size.)

density[k]        = edges_inside[k] / ( possible_pairs[k] + 1 )

score[k]          = ( density[k] - gamma ) * possible_pairs[k]
                  = edges_inside[k] - gamma * possible_pairs[k]

loss = - ( sum over k of score[k] ) / normalizer
```

In words: **reward every link present inside a prototype, charge `gamma` for every pair
that could have been linked but isn't.**

**The `+ 1` is a Laplace pseudo-count, not an epsilon — and the difference is the whole
bug fix.** A prototype with fewer than 2 effective members has no pairs, so its density is
undefined. Dividing by `1e-8` yields ~1e8, and under the old `(density-1)^2` loss that
dwarfs every real prototype — which is precisely what made an *empty* prototype cheaper
than a *small* one. `+ 1` instead says "assume one possible pair exists", so a degenerate
prototype gets `density = edges/1 ~= 0`, an ordinary bad value. It is also preferred over a
hard `size >= 2` mask because it is **smooth**: no threshold, no gradient discontinuity,
and a genuinely rare-but-real prototype is nudged rather than dropped.

### What gamma means — this is the part worth internalising

CPM's move rule is

```
delta(move cell i into prototype p) = size[p] * ( mean_affinity(i -> ALL members of p) - gamma )

  =>  cell i joins p  if and only if  mean_affinity(i -> members of p) > gamma
```

So **gamma is the minimum AVERAGE affinity a cell needs to a group to be worth joining
it.** It is a property of the *graph*, measurable with no training at all
(`modal_scripts/measure_gamma_from_graph.py`).

Two things people get wrong about it:

1. The mean is over **all** members, including the ones cell i is not connected to at
   all. So gamma sits far BELOW a typical edge weight — on bk02 a typical edge is 0.22
   while gamma is 0.03.
2. gamma is **not portable across graphs**. Measured: `s28nsc`/`bk02` → 0.026-0.030,
   `pancreas`/`arbf` → 0.11. Re-derive it whenever the graph changes.

### Its gradient

```
d(CPM charge) / d assign[i][k] = gamma * ( size[k] - assign[i][k] )
```

Again — `size[k]` only. Like modularity, CPM cannot move a cell for similarity reasons;
it is a size/capacity charge. Same family, same limitation.

### Why `possible_pairs` and not `volume` (`nassoc_diag_norm='pair'`)

`scproto.py:2060-2074`. This swap is what removed the merge bias:

```
volume[k]         = 2*edges_inside + edges_leaving    -> "containment"
possible_pairs[k] = size*(size-1)/2                   -> "capacity"
```

- `volume` is **additive** under merging: `volume[X and Y together] = volume[X] + volume[Y]`.
  So when you merge two prototypes the denominator doesn't notice, while the numerator
  *gains* all the between-edges. Containment therefore always rises on a merge →
  hierarchical splits are vetoed and **no threshold can flip that**.
- `possible_pairs` is **superadditive**: merging adds `size[X]*size[Y]` mostly-empty
  cross pairs. So density *falls* on a weak merge.

That monotonicity flip is the entire point of the swap.

> **The bug this fixed, and it is the cautionary tale of this project.** `density` is an
> edge density, so it *must* lie in `[0,1]`. The logged value was **371-510**.
> `possible_pairs = size*(size-1)/2` clamps to exactly 0 for any prototype whose soft
> minibatch size is below 1, so `density = edges/1e-8` exploded. Under the old
> `(density-1)^2` loss an EMPTY prototype cost 1 while a merely SMALL one cost ~1e16 —
> **the loss was paying the model to kill prototypes.** That single bug caused the
> 15/200-survivor collapse, and fixing it moved DGE from 0.654 to 0.780. Hence the rule
> in §7: print the value, compare it against its theoretical range, every time.

---

## 3. nassoc off-diagonal — keeping prototypes separate

`scproto.py:1874-1891`. The diagonal (§2) says "be internally dense"; the off-diagonal
says "don't overlap with anyone else".

```
leak[p][q] = sum over edges (h,t) of assign[h][p]*affinity[h][t]*assign[t][q]
             + the same with p and q swapped        (symmetrised)

norm[p][q] = min( volume[p], volume[q] )        <- nassoc_offdiag_norm='min'   CORRECT
           = sqrt( volume[p] * volume[q] )      <- 'sqrt' (the older default)

M[p][q] = leak[p][q] / norm[p][q]

off_diagonal_loss = aggregate over all p != q of  M[p][q]^2
```

### Why the denominator must be `min(volume[p], volume[q])`

`sqrt(vol_p * vol_q)` is **dominated by the larger volume**. So when a rare prototype
(small volume) bleeds into a common one (huge volume), the geometric mean is huge and the
leak registers as almost nothing — exactly the case you most need to catch.

`min(vol_p, vol_q)` bills the leakage against the **smaller prototype's own budget**.
This is conductance-of-the-smaller-side (Kannan, Vempala & Vetta 2004). And because
`min(v,v) = v = sqrt(v*v)`, it never touches the diagonal — the swap is off-diagonal-only
by construction.

Aggregation (`nassoc_offdiag_agg`): prefer `'pclu'` (sum the squared leakages out of each
prototype, then mean over the K prototypes) over `'pair'` (divide by `K*K - K`). Nonzero
partners grow roughly **linearly** in K — measured on bk02: 14 nonzero partners per
prototype at K=16, rising to only 168 at K=1774 — while `K^2` grows quadratically. So
`'pair'` divides by something growing far faster than the signal, and the `nassoc_alpha`
needed for diagonal/off-diagonal parity drifts 77 → 87,009 across that K range (1130x),
i.e. it does not transfer across K at all. `'pclu'` drifts 5.2 → 49 (9.5x).

---

## 4. Cohesion — the term that CAN move a cell for similarity reasons

`scproto.py:2369-2910`, enabled by `lambda_cohesion` (gram) and/or `lambda_cohesion_aff`
(aff). This is the one that fixes the §1/§2 defect.

### The similarity

Two choices, deliberately non-redundant:

```
'aff'   similarity(i,j) = affinity[i][j]
        1-HOP: "are i and j directly connected?"          -> PROXIMITY

'gram'  similarity(i,j) = cos( closed_row[i], closed_row[j] )
        where closed_row[c] = affinity_row[c] + cohesion_diag * e_c
        2-HOP: "do i and j connect to the SAME cells?"    -> STRUCTURAL EQUIVALENCE
```

Neither sees the other's failure case, which is why running both is a real conjunction:

- two cells hanging off the same hub are structurally equivalent but *not* near each
  other → pure-gram wrongly merges them;
- two adjacent cells straddling a type boundary are proximate but embedded differently →
  pure-aff wrongly merges them.

`cohesion_diag` closes the neighbourhood **before** the row-normalisation. That order is
load-bearing: normalising first and setting the diagonal after would not be a cosine at
all. At `cohesion_diag = 0`, two directly adjacent cells in a locally triangle-free region
get `Gram = 0` exactly — the open-neighbourhood Gram calls two neighbours completely
dissimilar. `cohesion_diag` is therefore also the **1-hop/2-hop mixture knob**: on the
un-normalised scale it adds exactly `2*cohesion_diag*affinity[c][j]` to the off-diagonal.

### The objective

```
within[k] = sum over pairs (i,j) of assign[i][k]*assign[j][k]*similarity(i,j)

cohesion  = ( sum over k of within[k] / size[k] ) / n_cells

loss      = 1 - cohesion
```

### Its gradient — the whole reason it exists

```
d(loss) / d assign[i][k] = -( mean_similarity(i -> k's members) - cohesion[k] )
```

**No size. No volume. No degree.** Cell i joins prototype k exactly when i is *more*
similar to k's members than they already are to each other — i.e. when adding i would
IMPROVE k.

Because `cohesion[k]` is a ratio of two quadratics it is **intensive / scale-free**: a
20-cell prototype is as viable as a 2000-cell one. That is what lets it keep rare states
alive where modularity and CPM squeeze them toward the average size.

This is weighted kernel k-means. Dhillon, Guan & Kulis (KDD 2004) prove that family is
equivalent to normalized cut / normalized association — the same family as the published
nassoc loss, which is why both are scale-free while CPM and modularity (grafted on from
outside that family) are not. It also inherits the **k-means splitting property**:
splitting any non-degenerate group strictly lowers the objective, so its optimum uses
every prototype with no anti-collapse term needed.

### `cohesion_form` — four ways to aggregate

```
'perc'  (default)  sum over k of within[k] / size[k]
'pcex'             cell c excluded from its OWN numerator and divisor:
                   for each c: sum_k assign[c][k]
                                     * ( sum over j != c of similarity(c,j)*assign[j][k] )
                                     / ( size[k] - assign[c][k] )
'qwcw'             sum over k of ( size[k]^2 - within[k] ) / n_cells^2
                   = the MISSING internal connections
'flor'             one-sided hinge: mean over k of max(0, cohesion_floor - within[k]/size[k])^2
```

- `'perc'` divides by `size[k]` **including** c — a 0.3% error for a 290-cell prototype
  but a **50% error for a 2-cell one**, so it systematically under-scores exactly the
  small/rare prototypes the term exists to protect. `'pcex'` fixes that.
- `'perc'`/`'pcex'` give a **dead** prototype a contribution of exactly 0 — so they never
  *penalise* collapse, they merely fail to reward it. There is no anti-collapse signal.
  `'flor'` supplies one (a dead slot pays the full `floor^2`) with a target that is
  actually reachable, unlike the published `(containment - 1)^2` whose target of 1 is
  unreachable by ~11x on bk02.
- `'perc'`'s size control is ~`1/m`, which **vanishes** for large prototypes. Measured
  consequence: median metacell size 12-136 against a mean of 292. `'qwcw'` instead has
  gradient `2*(size[k] - similarity_of_i_to_k)`, i.e. O(m) size control that is waived
  exactly when the cell genuinely belongs.

### `cohesion_self` — the weight on the i == j self-pairs

```
within[k] = cross_pairs[k] + cohesion_self * self_pairs[k]
self_pairs[k] = sum over i of assign[i][k]^2 * similarity(i,i)
```

The self-pair's value **is the founder gate**. A singleton prototype scores exactly
`cohesion_self`, so a cell joining it gains `2*sim*a` against a dilution of
`cohesion_self*a`:

```
a cell joins a singleton  if and only if  similarity > cohesion_self / 2
```

At `cohesion_self = 1` the gate is 0.5, and the median cell's best-neighbour gram cosine
on bk02 is 0.472 — **most founders can never recruit their first follower**, which is why
earlier runs kept prototypes alive but tiny (median 69-84).

**With multi-batch cohesion (§5) `cohesion_self` MUST be 0** — see the warning there.

---

## 5. [D41] Multi-batch cohesion — score each cell in its OWN batch graph

`scproto.py`, `coh_batch='locl'`. Added 2026-09-23.

**Premise**: an affinity between two cells measured in *different* biological batches is
a batch effect, not a similarity. This matters because the affinity graph is built on PCA
of the **raw expression matrix** (`arbf` → `SEACellGraph(ad, "X_pca")`), never on the
model's corrected latent. So within-batch affinity is trustworthy biological signal,
and cross-batch affinity in that same raw space *is* the batch effect.

### Formula

```
similarity(c,j) = 0                          whenever bio(c) != bio(j)

numer[c][p]         = sum over j in bio(c) of  assign[j][p] * w[j] * similarity(c,j)
size_in_batch[b][p] = sum over j in b       of  assign[j][p] * w[j]

contribution(c) = sum over p of assign[c][p] * numer[c][p]
                  / max( size_in_batch[bio(c)][p] , 1 )

cohesion = ( sum over c in the minibatch of contribution(c) ) / n_cells_in_minibatch
loss     = 1 - cohesion
```

Both sums run over **every** cell of c's batch — neighbours and non-neighbours alike.
Non-neighbours contribute 0 to the numerator, which is exactly what keeps the value a
*mean similarity* rather than a mean over edges.

### Why restricting the DENOMINATOR is the load-bearing half

Masking the numerator alone would be a bug: the cell then sees only the share of its
prototype living in its own batch, while the divisor still counts all nine. Measured on
pancreas/arbf, that scores a batch-MIXED prototype at **0.169x** an otherwise identical
batch-PURE one — a standing ~5.9x reward for batch purity, created purely by a
numerator/denominator scale mismatch. Restricting both sides to the same population makes
the term **batch-neutral**.

Merge test on pancreas/arbf (% change in the objective when two groups are merged):

```
merge type                          coh_batch='off'   coh_batch='locl'    want
same celltype, different tech            -1.42%            0.00%          ~0
diff celltype, SAME tech (bad)           -1.71%           -1.71%          very negative
```

And on real reference partitions, `celltype` (spread over 9 techs) now scores **exactly**
what `celltype x tech` (the same biology, batch-pure) scores — 0.073253 aff / 0.186007
gram — where the global denominator penalised the mixed one 6.8x.

### WARNING: cohesion_self MUST be 0 here

The self-pair contributes `1/size_in_batch` per cell. Globally that is `1/size`, identical
for batch-pure and batch-spread prototypes, so it carries no batch preference. **Per
batch**, a spread prototype has `size_in_batch = size/n_bio`, so its self part is `n_bio`
times LARGER — the term then pays the model to smear every prototype thinly across
batches. Decomposed on pancreas at K=220:

```
partition     median size_in_batch    TOTAL     self part   PAIR part
leiden              191.0            0.13034    0.01343     0.11691
random-220           10.0            0.12969    0.12007     0.00962
```

The pair part discriminates **12x**; the self bonus hands random 0.107 back, collapsing
the span from 0.107 to 0.0006 and implying an absurd lambda of 2396. The trainer prints a
loud warning if `cohesion_self != 0` with more than one bio batch.

### Accepted consequence

A per-batch denominator makes the objective **separable across batches**, so relabelling
cells in *other* batches is a zero-gradient direction: the term cannot penalise a bad
*cross-batch* merge either (measured 0.00%). That is the price of the premise, not a
defect. Cross-batch alignment comes from **Stage 1** (the scPoli CVAE conditions its
decoder on batch, so the encoder discards batch-specific variation), and must therefore
be **measured** — batch entropy, per-prototype cross-batch cell-type consistency — never
assumed.

With **one** bio batch this is bit-identical to the global path (asserted to <1e-12 in
`tests/test_cohesion_memory.py`).

---

## 6. The assignment memory ([D29] / [D31] / [D32] / [D37])

`interpretable_ssl/cohesion_memory.py` + `scproto.py`, enabled by `cohesion_global=2`.

### The problem it solves

A minibatch sees only `p_batch` of the cells (0.0625 on pancreas). That breaks the
cohesion estimate in two separate ways:

1. **Invisible prototypes.** A prototype with S cells is entirely absent from a batch with
   probability `(1 - p_batch)^S` — 29% at S=10, 54% at S=5. The loss cannot tell
   *invisible* from *empty*, so a cell joining it is scored as if **founding** one:
   `within = a*a` over `size = a` gives gradient 1.0, the maximum, where the truth is its
   mean similarity to members it cannot see (~0.08). A 12x overpayment aimed precisely at
   small prototypes. Measured on a real run: 11.1 of 200 prototypes invisible per batch,
   35 invisible in >10% of batches.
2. **Mixed scales.** `within[k]` mixes a PAIR count (cross terms, needing BOTH cells,
   surviving at `p_batch^2`) with a CELL count (self terms, one per cell, surviving at
   `p_batch`). Dividing both by `size[k]` leaves the cross part unbiased but inflates the
   self part by `1/p_batch` = 8.6x.

### The mechanism

Keep the last-known assignment of **every** cell (an `n_cells x K` matrix, ~47MB on
s28nsc). Each step, swap this batch's rows for the live gradient-carrying ones and leave
the rest as detached context:

```
S_eff[j] = live softmax row       if j is in the current minibatch   (carries gradient)
         = remembered row         otherwise                          (DETACHED)
```

Then both similarities are exact full-graph quantities, and a prototype whose members are
all absent still has a real profile to compare against:

```
aff   numer[c][k] = sum over ALL j of affinity[c][j] * S_eff[j][k]
gram  numer[c][k] = dot( R[c] , sum over ALL j of S_eff[j][k] * R[j] )
                    where R = the L2-normalised closed-neighbourhood rows

size[k] = sum over ALL j of S_eff[j][k]          <- EXACT, not an EMA
```

The inner `sum over j of S_eff[j][k] * R[j]` **is** the per-prototype affinity profile —
one length-n_cells vector per prototype, summed over its members.

Non-batch rows are detached on purpose: they are **context**, so a cell moves under its
own gradient while its prototype-mates hold still. That is the classic Lloyd / EM
assignment step.

### [D31] Age weighting — because a remembered row goes stale

A non-batch row is as old as the last batch that contained that cell (~8.6 steps on
average). So discount it by age:

```
age[j] = current_step - last_seen[j]            (last_seen = -1 means never visited)
w[j]   = 0.5 ** ( age[j] / half_life )
w[j]   = 1                                       for cells in the current minibatch
```

**The both-sides rule**: `w[j]` multiplies `assign[j]` *wherever it appears* — in the
similarity numerators AND in `size`. That keeps every downstream quantity a proper
weighted mean and preserves the `[0,1]` bound. Discounting only the numerator would be
exactly the scale bug of §7.

`coh_mem_hlife = 0` (default) disables it and is bit-identical to plain [D29]. Anchor
value: one epoch (~500 steps at `umap_steps_per_epoch=500`) = "information a full epoch
old counts half". Derive it from the `coh_mem_drift` diagnostic logged every step, not by
sweeping — drift ~0 means leave it off.

Memory-bank semantics: **decay is for the unobserved, replacement for the observed.** A
revisit overwrites the row outright rather than blending.

### [D32] How the memory starts

```
'unif' (default)  every row = 1/K. NEUTRAL (identical contribution to every prototype,
                  measured spread ~1e-16, so no false preference) but DILUTED -- real
                  signal only accumulates over the first epoch. Rows carry
                  last_seen = -1 so under [D31] this prior fades by age.
'full'            ONE no-grad forward pass over all cells fills the memory with real
                  assignments softmax(z @ prototypes / eps). The encoder is already
                  pretrained at finetune start, so step-0 profiles and sizes are honest
                  immediately. Costs one forward-only pass (seconds on GPU).
```

### [D37] The divisor — subtle and it was wrong for a while

```
cohesion = ( sum over c in the MINIBATCH of contribution(c) ) / n_cells_in_minibatch
```

Divide by the number of cells actually **summed over**, not by the full dataset size. The
INNER sum spans every cell (that is what the memory provides, and it is correct), but the
OUTER sum is still a subsample. The original [D29] comment claimed "GS already spans every
cell, so nothing was subsampled away" — true of the inner sum, **false of the outer one**,
and that is what hid the bug. Measured on s28nsc/bk02: batch metric 0.00086 vs true
full-graph 0.0076, ratio 8.8 — matching `n_cells/n_minibatch = 58423/6758 = 8.6`.

Validated against a triple-loop brute force, finite-difference gradients, and the
full-batch exact limit in `interpretable_ssl/tests/test_cohesion_memory.py`.

---

## 7. The rule that catches all of these

Every bug in this file was the same bug wearing a different hat: **a quantity computed at
one scale compared against a quantity computed at another.**

Before trusting any new term:

1. **Print its value and compare against its theoretical range.** A density, probability,
   purity or correlation has a known range. `nassoc` density had to be in `[0,1]` and was
   logged at **371-510** for weeks.
2. **Print every term of the total loss side by side and check the ratios.** One term at
   1000x another is a bug until proven otherwise. `nassoc=2874` vs `umap=2.5` — the
   regularizer was 99.5% of the loss, because the numerator was at full-dataset scale and
   the denominator at minibatch scale (~3900x too large).
3. **Check every denominator that can vanish, and every minibatch estimate for a scale
   mismatch against its numerator.** Anything quadratic in a soft count is biased
   differently under subsampling than anything linear — so a ratio of the two is **not**
   an unbiased estimate of the full-graph ratio. Rescale both sides, or don't compare them.

Corollary: a threshold parameter (`nassoc_gamma`, `cohesion_floor`) is only interpretable
once the quantity it is compared against is on a known scale. Derive it from measured
properties of the actual graph, and re-derive it whenever that scaling changes.

---

## 8. Summary — which term can do what

| | moves a cell for SIMILARITY? | size control | scale-free | anti-collapse |
|---|---|---|---|---|
| **modularity** | no — charge is `degree*volume` | strong, toward UNIFORM sizes | no | yes |
| **CPM** | no — charge is `gamma*size` | strong, via `gamma` | no | yes |
| **nassoc off-diag** | no — it separates, doesn't attract | none | partly | no |
| **cohesion `perc`/`pcex`** | **yes** | ~`1/m`, vanishes when large | yes | **no** (a dead slot scores 0) |
| **cohesion `qwcw`** | **yes** | O(m), waived when the cell fits | no | yes |
| **cohesion `flor`** | **yes** | via the floor | yes | **yes** |

The practical reading: modularity and CPM decide **how big** prototypes are; cohesion
decides **who belongs in which**. They are complementary, not alternatives — which is why
`perc` cohesion needs a size-control partner, and why cohesion-only runs must be read with
that in mind.

# What the loss-term ablation taught us (scProto, pbmc-immune)

*Prepared for the DAII conversation — feedback wanted on the open questions at the end.*
*Sources: [D58] ablation (`files/tables/d58_ablation/`), [D59] epoch-0 measurement
(`files/tables/d59_ctb/`). Dataset: pbmc-immune, K=300 prototypes, one seed per arm
(two seeds for the base config, used as the noise yardstick).*

---

## 1. Setup

The model learns metacells (prototypes) in two stages: **stage 1** pretrains a CVAE
encoder (batch-conditional), **stage 2** fine-tunes the encoder + prototype vectors
with four loss terms:

| term | what it optimizes |
|---|---|
| **cohesion-gram** | mean latent-space (gram) similarity of each prototype's members to the prototype |
| **cohesion-aff** | same, but similarity = the precomputed affinity-graph edge weight (1-hop) |
| **community** | UMAP-style attraction/repulsion on graph edges, routed to the encoder only |
| **recon** | CVAE reconstruction, kept on during stage 2 as an anchor |

### The cohesion loss, precisely (from `scproto.py`, the deployed path)

Soft assignment: `assign[i][k]` = probability that cell `i` belongs to prototype `k`
(rows sum to 1). The core objective — one formula, two instantiations that differ
only in the similarity used:

```
sim(i, j)  = similarity between cells i and j   (two choices, below)

within[k]  = sum over cell pairs (i, j) of  assign[i][k] * assign[j][k] * sim(i, j)
             # how similar prototype k's members are to each other,
             # soft-weighted; the i == j self-pair IS included (sim(i,i) = 1)

size[k]    = sum over cells i of  assign[i][k]          # soft member count

cohesion   = ( sum over prototypes k of  within[k] / size[k] )  /  n_cells
loss       = 1 - cohesion                                # bounded in [0, 1]
```

The two instantiations:

```
cohesion-aff   (1-hop):  sim(i, j) = affinity[i][j]      the graph edge weight itself;
                                                          sim(i, i) set to 1
cohesion-gram  (2-hop):  sim(i, j) = cosine( profile[i], profile[j] )
                         profile[c] = cell c's ROW of the affinity matrix,
                                      with profile[c][c] = 1  (closed neighbourhood)
                         # "do i and j connect to the SAME cells?" -- two cells can
                         # share an edge yet sit across a boundary; identical
                         # neighbour profiles cannot.
```

Why this shape: the gradient (derived in the code header, `scproto.py:2464`) is

```
d(loss) / d assign[i][k]  =  -( mean_sim(i -> members of k)  -  cohesion[k] )
```

i.e. cell `i` is pulled toward prototype `k` exactly when it is MORE similar to k's
members than they are to each other — when adding it would improve k. No size,
volume, or degree term appears, which is the defect this term was built to fix:
CPM / modularity / DMoN gradients depend on assignment column sums only, so they can
move a cell for size reasons but never for similarity reasons. `within[k]/size[k]`
is a ratio of quadratic over linear, so the score is intensive (scale-free): a
20-cell prototype is as viable as a 2000-cell one. This family is weighted kernel
k-means (Dhillon, Guan & Kulis, KDD 2004), equivalent to normalized association.

Two deployment details (what actually runs, `scproto.py:3044-3056`):

```
# (a) full-graph memory: the sums over j run over ALL cells, using a remembered
#     soft assignment refreshed as cells are revisited -- not just the minibatch.
#     The outer average is over the minibatch's cells (an unbiased subsample).
# (b) multi-batch denominator: cell c is scored against V[b(c)], the node set of
#     its own technical batch's graph (all cells with an edge touching batch b).
#     Cross-batch similarities keep their real values -- nothing is masked; what
#     is excluded from the DENOMINATOR are pairs whose zeros carry no information
#     (a cross-batch zero is mostly batch effect, not measured dissimilarity).
#     With one batch, V[b] = all cells and this reduces exactly to the formula
#     above (asserted to < 1e-12 in tests).

numer[c][k] = sum over j in V[b(c)] of  mem_weight[j] * assign[j][k] * sim(c, j)
denom[c][k] = sum over j in V[b(c)] of  mem_weight[j] * assign[j][k]
cohesion    = mean over minibatch cells c of
                  sum over k of  assign[c][k] * numer[c][k] / denom[c][k]
```

**Ablation design ([D58]):** train the full 4-term config (`base`, two seeds s31/s32),
then four arms each removing exactly one term (`gram0`, `aff0`, `comm0`, `recon0`).
Metrics are scored on **held-out cells** (a cell split; held-out cells are assigned to
prototypes by a frozen-encoder forward pass) unless marked "full".

**Metrics** (all size-weighted over metacells unless noted):

```
test cohesion (aff / gram) = mean similarity of a prototype's held-out members
                             to each other / to the prototype
cell type purity           = fraction of a metacell's members that belong to
                             its majority cell type
batch entropy              = entropy of the batch distribution inside a
                             metacell (high = batches are well mixed)
within-batch purity        = purity computed separately per (metacell, batch)
pooled purity              = purity over all members of a metacell (the usual one)
cross-batch agreement      = per metacell: majority cell type per batch, then
                             "do the batches agree on the majority?"
                             -> are we matching like with like ACROSS batches?
rare homogeneity           = how concentrated each rare cell type is in few
                             metacells
rare f1 / coverage         = detection of rare types, per batch
```

**Noise yardstick:** the only replicate we have is base_s31 vs base_s32. Seed spread:
purity 0.0037, batch entropy 0.0133, coh_aff 0.0009, coh_gram 0.0006,
rare homogeneity 0.059 (rare metrics are by far the noisiest — [D58] recorded this
and demoted rare homogeneity as the primary metric for exactly this reason).

---

## 2. Finding 1 — the cohesion losses do not generalize to held-out cells

**Claim: training improves train cohesion but not test cohesion. CONFIRMED — and for
cohesion-aff it is stronger than that: training makes test cohesion *worse* than not
training at all.**

**Direct evidence ([D59], epoch-0 vs trained — same pretrain, same waypoint init;
`epoch0` = argmax assignment with zero stage-2 training):**

| arm | coh_aff TRAIN | coh_aff TEST |
|---|---|---|
| epoch0 (no stage-2) | 0.0898 | **0.0880** |
| scproto_base, 10 ep | **0.1349** | 0.0816 |
| scproto + proto-recon | **0.1365** | 0.0760 |
| scPoli + Leiden | 0.0591 | 0.0535 |

Ten epochs of stage 2 raise TRAIN cohesion-aff by +50% (0.090 → 0.135) while TEST
cohesion drops *below the untrained init* (0.0880 → 0.0816). The entire test-cohesion
advantage over the scPoli/Leiden control (0.082 vs 0.054) is already present at
epoch 0 — it comes from the topology-aware waypoint init + CVAE encoder, not from the
cohesion loss.

**Gram ([D59]):** TRAIN 0.2356 vs TEST 0.1592 after training — a large gap — but no
epoch-0 gram value was saved, so the "worse than untrained" comparison is only
established for aff. Indirect evidence from the ablation closes the loop:

**Ablation evidence ([D58], test cohesion when the term is removed):**

| | base (s31 / s32) | term removed | delta |
|---|---|---|---|
| coh_aff_test | 0.0788 / 0.0796 | aff0: 0.0824 | **+0.003 (removing the aff loss *raises* test aff cohesion)** |
| coh_gram_test | 0.1541 / 0.1547 | gram0: 0.1609 | **+0.007 (same for gram)** |

Both deltas are several times the seed spread (0.0009 / 0.0006), though with only two
seeds that yardstick is soft. On held-out cells, each cohesion loss fails to improve
even its own metric. Meanwhile removing **recon** or **community** hurts test cohesion
more than removing the cohesion terms themselves (recon0: coh_aff 0.0720, coh_gram
0.1395; comm0: 0.0745 / 0.1427) — test cohesion is carried by the encoder-shaping
terms, not by the cohesion objectives.

---

## 3. Finding 2 — cross-batch purity gets worse during training

**Claim: cross-batch purity degrades. CONFIRMED — but the mechanism is not "the
cross-batch force is too weak". [D59] showed cross-batch alignment is already at
Leiden level at epoch 0 and is actively *destroyed* by stage-2 training. The missing
piece is a preservation mechanism, not a stronger attractive force.**

**[D59], cross-batch agreement over training (train split):**

| arm | cross TRAIN | cross TEST |
|---|---|---|
| epoch0 (no stage-2) | **0.963** | **0.954** |
| scproto_base, 10 ep | 0.856 | 0.885 |
| scPoli + Leiden | 0.967 | 0.983 |

Epoch 0 matches Leiden (0.963 vs 0.967); ten epochs of training lose ~0.11. The
convergence check showed the cohesion loss was still improving at epoch 10, i.e. more
training would move *further* in the harmful direction.

**Per-term effect on cross-batch agreement ([D58], `cross_agree_test`; base mean
0.8710, seed spread 0.0033):**

| term removed | cross_agree_test | delta vs base | reading |
|---|---|---|---|
| aff0 | 0.9168 | **+0.046 (14x spread)** | **cohesion-aff is the main term damaging cross-batch purity** |
| comm0 | 0.8530 | **−0.018 (5x spread)** | community is the main term *protecting* it |
| gram0 | 0.8748 | +0.004 | within noise |
| recon0 | 0.8672 | −0.004 | within noise |

**The aff paradox, stated precisely:** cohesion-aff is what buys batch *mixing* —
removing it collapses batch entropy 0.8699 → 0.6501 (−0.22, the largest effect in the
whole ablation) — yet it *hurts* cross-batch *agreement*. These are different things:
entropy asks "do batches co-occur in a metacell?", cross-agreement asks "are the
co-occurring members the same cell type?". So aff forces batches together but partly
matches the wrong cells with each other. (Caveat: aff0 had 1 dead prototype, so it is
gate-flagged under the pre-registered rule; its direction is consistent with [D59].)

---

## 4. Finding 3 — rare metrics: gram and recon carry them

([D58]; remember rare metrics have the largest seed spread, so only the biggest
effects are trustworthy. Base s31 / s32 shown as the pair.)

| metric | base (s31 / s32) | worst arm | delta vs s31 | reading |
|---|---|---|---|---|
| rare F1 | 0.7046 / 0.7226 | gram0: 0.5803 | **−0.124** | **gram buys rare-type recovery** (largest F1 effect in the set) |
| rare coverage | 0.9333 / 0.9333 | gram0 & aff0: 0.8667 | −0.067 | same direction |
| rare homogeneity | 0.5849 / 0.6437 | recon0: 0.4581 | **−0.127 (3.0x spread)** | **recon is the only unambiguous effect on this metric** |

Overall [D58] verdict on the four terms: **recon** is the most load-bearing term
(purity −0.034 = 13x spread when removed, the only term with a real held-out
generalization gap, 1.92x); **aff** buys batch mixing at a purity cost; **gram** buys
rare-type recovery; **community** earns its place but is the weakest single effect
(purity −0.014, 5.3x spread).

---

## 5. One surprise worth flagging ([D58], pre-registered prediction falsified)

We predicted removing recon would *drop* batch entropy (mechanism: recon keeps batch
info out of the latent via the batch embedding). The opposite happened: recon0
*raised* batch entropy (+0.034, 3.6x spread) and cross-batch homogeneity (+0.008).
That mechanism story is not supported by the data.

---

## 6. Open questions for feedback

1. The cohesion losses optimize an assignment-dependent objective on the training
   graph and demonstrably overfit it (Finding 1). Is there a standard way to make such
   a graph-coupled objective generalize — e.g. edge dropout / graph subsampling during
   training, or optimizing on a held-in/held-out edge split?
2. Finding 2 says the init already has the structure we want and training erodes it.
   What preservation mechanisms are sensible here — trust-region / proximal penalty to
   the init assignment, distillation from the epoch-0 soft assignment, or simply much
   earlier stopping on a cross-batch metric?
3. The aff paradox (mixing without matching): is there a principled objective that
   rewards cross-batch *agreement* directly rather than batch entropy as a proxy?

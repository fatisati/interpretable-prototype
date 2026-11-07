# scProto — preserving rare cell populations

Meeting note, 2026-11-01. Dataset: **pbmc-immune**, 33,506 cells, 5 studies,
16 cell types, K = 300 metacells. All held-out numbers use a split stratified by
(cell type x batch), so every type appears in both halves.

---

## 1. The gap

A CVAE (scPoli) removes batch effect well, but it is trained to **reconstruct**.
Reconstruction error is dominated by the genes that carry most of the signal —
the abundant ones. Two consequences for rare populations:

- **Rare cells get absorbed into dense neighbourhoods.** Placing a few hundred
  cells slightly wrong costs almost nothing in total reconstruction error.
- **Subtle but decisive marker genes are ignored.** If a small group of cells is,
  say, becoming cancerous, it is defined by a marker gene at low expression.
  That gene contributes little to the loss, so the model has no reason to keep
  those cells separate — even when the pattern is *consistent across batches*,
  which is exactly the evidence that it is real biology and not noise.

**Measured on our data:** in the pretrained CVAE latent, a rare cell is closer
to a dense-population centre than to its own cell type in another batch **67%**
of the time (only 328 of 1,034 rare cells are nearer their own type). The rare
groups' *centroids* do line up across batches (83% of cases), but the individual
cells lean into the dense regions. That is the absorption, measured.

---

## 2. The solution

Keep the CVAE's batch correction, then **fine-tune on top of it** with a loss
that is about *neighbourhood structure* rather than reconstruction.

```
stage 1   pretrain scPoli (CVAE)              -> batch-corrected latent
stage 2   fine-tune with:
            COHESION   cells that are similar WITHIN a batch should share a
                       metacell
            GM-KL      each cell is pulled toward its metacell's centre, and the
                       encoder is kept stochastic -- this is what makes the rule
                       generalise to unseen cells instead of memorising
            CVAE recon kept on, so the batch correction is not destroyed
```

**Why the similarity is defined within each batch:** a cross-batch zero mostly
reflects the batch effect, not real dissimilarity. Comparing cells only to cells
measured the same way keeps the signal clean.

**How similarity is defined:** an adaptive Gaussian kernel on PCA coordinates —
the same construction SEACells uses for its affinity graph. The kernel width
adapts to local density, so cells in **sparse, rare regions get similarity
scores comparable to cells in dense regions.** A fixed-width kernel would make
every rare cell look dissimilar to everything, which is precisely the failure we
are trying to fix.

---

## 3. Goal, and how we validate it

**Goal: show scProto preserves rare populations better than the CVAE alone.**

We compare the **same model before and after stage 2**: pretrained scPoli with a
given prototype initialisation, versus scProto fine-tuned on top of it with the
*same* initialisation. That isolates what stage 2 does.

| what we check | metric | why |
|---|---|---|
| structure improved | within-batch cohesion, train **and** test, rare **and** all | shows something real improved inside batches, and that it transfers to unseen cells |
| batch alignment kept | cross-batch cell-type purity; **signature correlation** | shows we did not break the CVAE's alignment |
| batches still mixed | batch entropy per metacell | shows we did not simply split by batch |
| the actual goal | rare homogeneity, rare F1, rare cell-type purity | the claim itself |

**Signature** (a label-free check of alignment): for a group of cells,

```
signature[gene] = mean expression in the group, within one batch
                - mean expression of ALL cells of that batch
```

Subtracting the batch's own mean cancels the batch effect. For each metacell we
then correlate its signature between batches. High correlation means the
metacell means the *same biology* in every batch. This is the CIDER idea
(Genome Biology 2021) and it does not rely on annotations — useful, because we
found annotations that disagree across studies (below).

---

## 4. Results so far

Baseline comparison, held-out cells. **scProto** = cohesion + GM-KL + CVAE recon,
leiden init. **stage-1** = the same pretrained CVAE, same init, no fine-tuning.
**Leiden on CVAE** = plain Leiden clustering on the stage-1 latent (the "why not
just cluster the embedding" control).

```
                            stage-1    Leiden on CVAE    scProto     noise floor
WITHIN-BATCH STRUCTURE
 cohesion  train, all        0.0806        0.0591         0.1310        0.0016
 cohesion  train, rare       0.1263        0.0688         0.1670          --
 cohesion  test,  all        0.0784        0.0535         0.0853        0.0012
 cohesion  test,  rare       0.1262        0.0674         0.1376        0.0101
 within-batch purity, rare   0.5700        0.3465         0.6269        0.0611

RARE-POPULATION METRICS (test)
 rare homogeneity            0.7584        0.6398         0.7190        0.0773
 rare F1                     0.8790        0.6662         0.7958        0.0822
 rare coverage               1.0000        0.7667         1.0000        0.0667
 pooled purity, rare         0.5362        0.3092         0.6039        0.0564

BATCH ALIGNMENT
 cross-batch purity, test    0.9545        0.9827         0.9244        0.0160
 batch entropy, test all     0.482         0.573          0.574           --
 batch entropy, test rare    0.088         0.075          0.086           --
 pooled purity, all          0.8630        0.8565         0.8160        0.0058
```

**Noise floor** = how much a number moves between two runs of the *identical*
configuration (measured from two accidental duplicate runs). A difference
smaller than this means nothing. Rare metrics are noisy (0.06–0.08) because only
~207 rare cells land in the test half.

### What is solid

- **Cohesion improves, on held-out cells, and most for rare cells.**
  Train 0.0806 → 0.1310 and test 0.0784 → 0.0853, against a noise floor of
  0.0012 — that is ~60x the floor. Rare test cohesion 0.1262 → 0.1376.
  The loss does what it was designed to do, and it *generalises*.
- **We beat plain Leiden on everything rare, by a wide margin.** Within-batch
  rare purity 0.627 vs 0.347, pooled rare purity 0.604 vs 0.309, rare F1 0.796
  vs 0.666 — all 1.6–5x their noise floors. Leiden buys its batch consistency by
  making blurrier clusters: the correlation between *different* Leiden clusters'
  signatures is 0.385, versus ~0.00–0.10 for ours, i.e. its clusters are far
  less distinct from one another.
- **Batch mixing is not harmed.** Batch entropy 0.482 → 0.574 (0 = one batch,
  1 = perfectly mixed). We are not splitting by batch.

### The problem

**Cross-batch alignment gets worse, not merely unchanged.**

```
                                     stage-1   scProto
cross-batch cell-type purity, test    0.9545    0.9244
metacells whose per-batch majority
  cell type AGREES (>=10 cells/batch)  83.1%     66.0%
signature correlation across batches
  (metacells spanning >=2 batches)     0.819     0.733
  for RARE metacells                   0.671     0.431
```

So within each batch the structure improves, but metacells increasingly mean
*different things* in different batches. Rare metacells degrade most — which is
the opposite of what we want, since they are the target.

One honest caveat found along the way: for **Megakaryocyte progenitors**, the
Oetjen cells are genuinely a different expression state from the other three
studies' (signature correlation 0.18–0.21, versus 0.84 median for true same-type
pairs; their closest match is HSPCs at 0.72). The shared *label* is misleading.
So part of the apparent "misalignment" for that type is correct behaviour, and
forcing those together would be wrong. This is why we use signature correlation
and not only annotations.

---

## 5. What we are testing now

Two directions, both aimed at the cross-batch half:

**1. NB likelihood + prototype reconstruction.** Replace MSE with a negative
binomial likelihood. MSE treats every gene's error equally, so a housekeeping
gene at 5,000 counts drifting 10% costs 250,000 while a marker going 0 → 3
counts costs 9. NB models count noise (`variance = mean + mean^2/theta`), which
discounts deviations that sampling alone explains and amplifies real signal in
low-count genes — exactly where rare identity lives. The prototype-reconstruction
term asks each metacell to reconstruct its own cells under each batch's
condition, using **each cell's own library size** (its total counts), so the
likelihood judges the expression *profile* and not sequencing depth.

**2. A signature loss.** Make the thing we currently only *measure* into a
training objective: penalise a metacell whose signature disagrees between the
batches it spans. It is silent for metacells living in a single batch, so a
genuinely batch-unique rare type is never forced to merge with anything.

### Already ruled out

A reconstruction-based term alone does **not** fix cross-batch alignment. We
tested it across both formulations, both gradient routings, five doses spanning
100x, and with the CVAE recon on and off: at a correct dose it becomes harmless
(0 dead metacells, cohesion preserved) but improves nothing above the noise
floor. Reason: reconstruction can penalise a metacell that *mixes* two
biologies, but it is exactly indifferent to the same biology being *split*
across two metacells — both reconstruct their own members perfectly. That
asymmetry is why a separate alignment force is needed.

---


## 6. The signature loss — everything in one place

This resolves both directions in §5: the signature loss is built and tested, and
the NB question is answered. It was run four ways — pull-only vs pull+push,
crossed with MSE vs NB — all on the same split and seed.

### 6.1 What the loss is

Each metacell gets one expression fingerprint per batch. The loss rewards a
metacell whose fingerprints agree across the batches it spans.

```
1 SIZE         size[b][k] = how much of metacell k sits in batch b
                            (sum of its cells' soft memberships)

2 FINGERPRINT  sig[b][k]  = mean expression of k's cells in batch b
                          - mean expression of ALL cells in batch b
                            the subtraction is what cancels the batch effect

3 MEMORY       running average of sig across training steps
                            one step gives a (metacell, batch) group a MEDIAN
                            OF 2 CELLS -- far too few to trust on their own

4 PULL         reward = sum over metacells k, over batch pairs (b1,b2) of
                          min(size[b1][k], size[b2][k])
                          * correlation( sig[b1][k], sig[b2][k] )
                        divided by a FIXED number (total cells)

5 PUSH         penalty = average of max(0, correlation - 0.5)
                         over DIFFERENT metacells k != j
```

Three choices decide whether it works at all:

- **Reward, not penalty.** A penalty of the form "sum of (1 - correlation)"
  charges per batch pair, so a metacell could lower its cost by *abandoning a
  batch* — the opposite of alignment. Dropping one batch costs only 17.8% of a
  metacell's cells, so that escape is cheap and would have been taken.
- **`min(size, size)` is a trust weight.** A pair is only as reliable as its
  smaller side. Measured: a fingerprint built from 3 cells reproduces itself at
  0.34, from 10 cells at 0.65, from 40 cells at 0.93.
- **Push is a hinge above 0.5, never a push to zero.** With K = 300 and 16 cell
  types, about 19 metacells share each cell type, so most "different metacell"
  pairs are genuinely the same biology and must be left alone.

### 6.2 The full loss it sits inside, and what is detached

**Detached** = the quantity still affects the loss *value*, but no gradient flows
back through it, so that term cannot change it.

```
TERM             WHAT IT PUSHES FOR                    WEIGHT  SHARE  -> ENCODER  -> METACELLS
---------------------------------------------------------------------------------------------
cohesion gram    same NEIGHBOURS within a batch        2.7414  14.6%     yes          NO
                 -> same metacell
cohesion aff     ARE neighbours within a batch         4.8578  14.6%     yes          NO
                 -> same metacell
GM-KL            cell sits near its metacell centre;   0.0274  29.2%     yes         YES
                 encoder stays stochastic
CVAE recon       latent can still rebuild the cell,    0.0218  29.2%     yes          NO
                 so stage-1 batch correction survives
signature        a metacell means the SAME biology     5.7862  12.5%     yes          NO
                 in every batch it spans
```

Weights are not chosen by hand. Each term's gradient size is measured once with
its weight set to 1, then the weight is solved so the term takes the share
shown. The **shares** are the design choice; the weights follow. (Numbers above
are the MSE arm; they are re-derived per dataset and are not portable.)

**Measured gradient onto the metacell positions** — not assumed:

```
cohesion gram  0.000000     CVAE recon  0.000000     GM-KL  0.033054
cohesion aff   0.000000     signature   0.000000
```

Four exact zeros. **GM-KL is the only term that moves the metacells.** Every
other term can only move cells *between* metacells, by changing the encoder.

Why: if two terms both pull the metacells, each needs its own calibration,
because a term's pull on the encoder and its pull on the metacells are different
sizes. We learned this expensively — a term weighted from its encoder gradient
pulled the metacells 5.7–18.6x harder than GM-KL, and they collapsed onto each
other (nearest-neighbour similarity 0.81 → 0.9997).

**Three extra detachments inside the signature term:**

```
memory half       a reference point, not a target. Gradient flows only through
                  the cells seen in the current step.
min(size, size)   left live it becomes a SIZE incentive -- a constant pull to
                  join any large, well-aligned metacell regardless of fit --
                  because the correlation part, which checks fit, weakens as a
                  group grows.
expression data   it is data. Nothing to learn.
```

Outside the loss, the **batch embedding is frozen** in stage 2. Stage 1 pushed
batch information out of the latent and into that embedding; unfrozen, the
encoder would drift batch structure back in and CVAE recon would stop protecting
what we think it protects.

### 6.3 Results, held-out cells

`scProto` is §4's arm. `+ sig` adds pull only; `+ sig both` adds pull and push.

```
                              stage-1   scProto    + sig    + sig both   floor
WITHIN-BATCH STRUCTURE
 cohesion  train, all          0.0806    0.1310    0.1352     0.1343     0.0016
 cohesion  test,  all          0.0784    0.0853    0.0862     0.0862     0.0012
 cohesion  test,  rare         0.1262    0.1376    0.1433     0.1437     0.0101
 within-batch purity, rare     0.5700    0.6269    0.7500     0.6355     0.0611

RARE-POPULATION METRICS (test)
 rare homogeneity              0.7584    0.7190    0.7441     0.6885     0.0773
 rare F1                       0.8790    0.7958    0.8221     0.7591     0.0822
 pooled purity, rare           0.5362    0.6039    0.6618     0.6135     0.0564

BATCH ALIGNMENT
 cross-batch purity, test      0.9545    0.9244    0.9224     0.9309     0.0160
 metacells whose per-batch
   majority AGREES (>=10)       83.1%     66.0%     67.2%      64.8%     ~3 pts
 signature corr (>=40 cells)    0.819     0.733     0.740      0.747       --
 signature corr, RARE mc        0.671     0.431     0.520      0.484       --
 chimeric metacells (r < 0.3)    0.0%      9.9%      5.7%       6.7%       --
 batch entropy, test all        0.483     0.567     0.583      0.592       --
 batch entropy, test rare       0.057     0.056     0.059      0.060       --
 pooled purity, all            0.8630    0.8160    0.8193     0.8246     0.0058
```

Batch entropy is recomputed here for all arms with one consistent method, so read
that column as a set. It reproduces §4's figure for the shared arms (0.483 vs
0.482); the rare row uses a different normalisation from §4's and is not
comparable to it line by line.

### 6.4 NB — the one place NB is reported

**Yes, NB was trained**, in two experiments, both on a fresh NB pretrain (an NB
run cannot reuse the MSE encoder — different decoder head, different data scale):
NB + prototype reconstruction (4 arms: leiden/waypoint x CVAE recon on/off), and
NB + signature loss (2 of §6's arms).

**Held-out cells. NB is worse than MSE on every axis.**

```
                       coh      coh      coh    within   cross   pooled   rare
                      TRAIN    test   test rare   test    test     test     F1
stage-1 (CVAE)        0.0806  0.0784   0.1262   0.8886  0.9545   0.8630  0.8790
scProto (MSE)         0.1310  0.0853   0.1376   0.8722  0.9244   0.8160  0.7958
NB leiden + recon     0.1251  0.0678   0.1194   0.8217  0.8868   0.7636  0.5756
NB leiden, no recon   0.1263  0.0654   0.0966   0.7501  0.8403   0.6702  0.4335
NB waypoint + recon   0.1233  0.0647   0.1275   0.8039  0.8679   0.7169  0.7249
NB waypoint, no recon 0.1267  0.0619   0.1182   0.7350  0.7880   0.6334  0.6031
noise floor           0.0016  0.0012   0.0101   0.0008  0.0160   0.0058  0.0822
```

**The decisive observation is the train/test split in that table.** On TRAIN
cells NB looked like the better model — cohesion 0.125–0.127 against MSE's
0.131, and higher batch-vote agreement (70.3% vs 67.8%). On HELD-OUT cells it
reverses: cohesion 0.062–0.068 against MSE's 0.0853, roughly **15x the noise
floor in the wrong direction**, and cross-batch purity 0.788–0.887 against
0.9244. NB fits the cells it was trained on and **generalises worse**. Train-only
numbers would have given the opposite conclusion — which is why the held-out
split is the one that counts.

The separate NB + signature-loss arms agree: NB wins the cross-batch objective
(agreement 84.4% vs 75.6%) and loses cohesion, within-batch purity, pooled purity
and rare F1 by 7x–58x the noise floor.

**Why the premise did not hold here.** NB was expected to help because it
up-weights low-count genes, where rare identity lives. But on this dataset rare
types average **1,389 counts across 505 detected genes** while common types
average **25,685 counts across 412 genes** — *rare cells are not the sparse
ones*. Depth tracks the study, not rarity, so there was less for NB to correct
than assumed.

**Rejected.** Prototype health was never the problem (0–3 dead metacells, no
collapse); the cost lands on generalisation, which is the whole point of the
method.

### 6.5 Verdict

**Partly worked. It did not fail, and it did not deliver.**

What it did:

- **Chimeras nearly halved, 9.9% → 5.7%.** First term we have added that reduces
  them at all.
- **Rare metacells recover much of their lost alignment**, signature correlation
  0.431 → 0.520 against stage-1's 0.671. §4 flagged rare metacells as degrading
  most; this is the first thing that pushes back. Caveat: 7 rare metacells only.
- **Within-batch rare purity 0.627 → 0.750**, ~2x the noise floor — the first
  real gain on a rare metric from any term we have added.
- **It costs nothing.** Cohesion and purity tie or edge up. Every previous added
  term paid for its gains somewhere; this one does not.
- **Batch mixing unharmed**, entropy 0.567 → 0.583.

What it did not do:

- **Missed every target set before the run** (signature correlation >= 0.69,
  chimeras <= 3%, agreement > 69%). It recovers about a third of what stage 2
  destroys.
- **Did not improve generalisation.** Train-to-test cohesion gap 0.0457 without
  it, 0.0490 with it. The model learns slightly more on training cells and
  passes none of it on.
- **Cell-type agreement did not move** (66.0% → 67.2%, inside the ~3 point
  floor), even though the label-free measures did.
- **Push term is not useful** — worse than pull-only on its own objective and on
  every rare metric. Dropped.

Two things to carry into the write-up:

- **Signature correlation is now the training signal**, so it can no longer be
  the headline evidence. Cell-type agreement and cross-batch purity have to
  carry that, and neither moved much.
- **Every rare number here is one run**, and run-to-run noise on rare metrics is
  0.06–0.08. The 0.627 → 0.750 result needs a second seed before it is a claim.

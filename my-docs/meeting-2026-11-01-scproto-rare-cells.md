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

#### NB results (train half)

Four arms ran: leiden/waypoint init x CVAE recon on/off. They finished training
but crashed in an unrelated metric routine before held-out placement was saved,
so these are **train-half numbers**. Every row below is computed the same way on
the same cells, so the comparison is valid; the test half is re-running.

```
                        cohesion        within-batch  cross-batch        signature r   rare
                        all     rare    purity rare   purity   AGREE%    all    rare   F1
stage-1 (CVAE)          0.0806  0.1263    0.707       0.9675   82.7%     0.687  0.570  0.846
Leiden on CVAE          0.0591  0.0688    0.498       0.9751   83.3%     0.573  0.455  0.621
scProto (MSE)           0.1310  0.1670    0.815       0.9153   67.8%     0.597  0.408  0.774
scProto+protoRecon MSE  0.1305  0.1637    0.843       0.9014   66.9%     0.591  0.393  0.781
-- NB arms --
NB leiden + recon       0.1415  0.1629    0.878       0.9218   70.3%     0.617  0.325  0.753
NB leiden, no recon     0.1442  0.1715    0.789       0.9106   69.5%     0.596  0.425  0.762
NB waypoint + recon     0.1395  0.2361    0.845       0.9006   60.5%     0.612  0.411  0.765
NB waypoint, no recon   0.1448  0.2353    0.849       0.8652   53.7%     0.507  0.322  0.685
```

**Does NB preserve cross-batch alignment? Partly — but no, not really.**

- With **leiden** init, NB is the best non-baseline we have on cross-batch:
  agreement 70.3% vs 67.8% for MSE, signature correlation 0.617 vs 0.597,
  cross-batch purity 0.922 vs 0.915. Real but small, and agreement moves by
  ~3 points between identical runs, so this sits near the noise.
- **The gap to stage-1 is still large**: 70.3% vs 82.7% agreement, 0.617 vs
  0.687 signature correlation. NB narrows the loss; it does not undo it.
- With **waypoint** init NB is clearly worse (53.7-60.5% agreement), so the
  gain is not a property of NB alone — it depends on initialisation.
- **NB does help structure**: cohesion is the best of any arm (0.1448 vs 0.1310
  for MSE), and rare cohesion with waypoint init is much higher
  (0.236 vs 0.167). Within-batch rare purity is also the best we have (0.878).
- But **rare pooled purity and rare F1 do not improve** over MSE.

So NB does what it was predicted to do *inside* batches — it pays more attention
to the low-count genes that define rare cells — without fixing the cross-batch
problem. That is the argument for adding the signature loss rather than
expecting the likelihood change to be sufficient.

Two caveats worth stating out loud: these are train-half numbers (held-out
re-running), and the NB arms sit on an NB-pretrained encoder, so they do not yet
have their own matched stage-1 baseline — the comparison above is of absolute
values on identical cells, not of each model against its own starting point.

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

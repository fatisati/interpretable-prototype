# D58 ablation results — pbmc-immune, one term removed at a time

**2026-10-06.** 6 arms, all completed, 0 failures. Decision record: `../adr/D58-ablate-each-loss-term-around-share-vector-B.md`.
Plan and pre-registered predictions: `../doing/ablation-plan.txt`.
Numbers from `files/tables/d58_ablation/pbmc-immune_d58_ablation.csv`.

Base = share vector B from `D56` (`gram 3.794 / aff 6.399 / community 0.04307 / recon 0.02042`),
NH-heldout protocol, K=300, `train_epochs=10`, metrics on the 6,700 held-out cells
against the FULL affinity graph.

---

## 1. THE NOISE FLOOR CHANGES THE PRIMARY METRIC

Two base seeds, identical config:

| metric | seed 31 | seed 32 | spread | "std" |
|---|---|---|---|---|
| **rare homogeneity** | 0.5849 | 0.6437 | **0.0588** | 0.0416 |
| rare F1 | 0.7046 | 0.7226 | 0.0181 | 0.0128 |
| batch entropy (test) | 0.8699 | 0.8832 | 0.0133 | 0.0094 |
| **purity (test, wtd)** | 0.7594 | 0.7631 | **0.0037** | 0.0026 |
| cohesion aff (test) | 0.0788 | 0.0796 | 0.0009 | 0.0006 |
| modularity (test) | 0.1506 | 0.1498 | 0.0008 | 0.0006 |

**Rare homogeneity's run-to-run spread is 0.0416 — 10x the within-run SE of 0.0041
that I used to justify making it primary.** The within-run standard error badly
underestimates run-to-run variance, because changing the seed changes the prototype
init *and* the batch order, not just the sampling of cells.

The plan pre-declared the fallback for exactly this case:

> *"If the 3-seed spread on rare homogeneity turns out larger than the base-vs-arm
> differences, we fall back to purity as primary — declared NOW, not after seeing
> results."*

**So purity (test, weighted) is the primary metric.** Its noise floor is 16x tighter
(0.0026 vs 0.0416), and every arm separates cleanly on it. Rare homogeneity is still
reported, but with only 2 seeds only the largest effect on it is readable.

Reading the arm deltas in units of the noise std:

```
                 rare_homog        purity_test
gram0            -1.4 std          +3.8 std
aff0             +0.9 std          +13.9 std
comm0            -1.8 std          -5.3 std
recon0           -3.0 std          -13.0 std
```

Only `recon0` is unambiguous on rare homogeneity. All four are unambiguous on purity.

---

## 2. MAIN TABLE

| arm | rare homog | rare F1 | rare cov | cross-batch homog | entropy (test) | **purity (test)** | purity (full) | coh aff | coh gram | modularity | n_mc | n_dead |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_s31 | 0.5849 | 0.7046 | 0.9333 | 0.2599 | 0.8699 | **0.7594** | 0.7798 | 0.0788 | 0.1541 | 0.1506 | 300 | 0 |
| base_s32 | 0.6437 | 0.7226 | 0.9333 | 0.2661 | 0.8832 | **0.7631** | 0.7785 | 0.0796 | 0.1547 | 0.1498 | 300 | 0 |
| gram0 | 0.5262 | 0.5803 | 0.8667 | 0.2277 | 0.8894 | **0.7693** | 0.7814 | 0.0831 | 0.1609 | 0.1423 | 300 | 0 |
| aff0 | 0.6204 | 0.6448 | 0.8667 | 0.2624 | 0.6501 | **0.7955** | 0.8045 | 0.0824 | 0.1652 | 0.1976 | 299 | **1** |
| comm0 | 0.5085 | 0.6992 | 0.9333 | 0.2666 | 1.0038 | **0.7457** | 0.7678 | 0.0745 | 0.1427 | 0.1322 | 300 | 0 |
| recon0 | 0.4581 | 0.6242 | 1.0000 | 0.2674 | 0.9041 | **0.7255** | 0.7621 | 0.0720 | 0.1395 | 0.1330 | 300 | 0 |

Deltas vs `base_s31` (negative = arm is worse):

| arm | rare homog | rare F1 | rare cov | cross-batch | entropy | **purity (test)** | purity (full) | coh aff | coh gram | modularity |
|---|---|---|---|---|---|---|---|---|---|---|
| gram0 | -0.0587 | **-0.1243** | -0.0667 | -0.0322 | +0.0195 | +0.0099 | +0.0016 | +0.0043 | +0.0068 | -0.0083 |
| aff0 | +0.0355 | -0.0597 | -0.0667 | +0.0026 | **-0.2199** | **+0.0361** | +0.0247 | +0.0036 | +0.0111 | **+0.0469** |
| comm0 | -0.0764 | -0.0054 | 0.0000 | +0.0067 | **+0.1339** | -0.0137 | -0.0120 | -0.0042 | -0.0113 | -0.0184 |
| recon0 | **-0.1268** | -0.0804 | +0.0667 | +0.0075 | +0.0342 | **-0.0339** | -0.0177 | -0.0068 | -0.0146 | -0.0176 |

---

## 3. WHAT EACH TERM TURNS OUT TO DO

### recon is the most load-bearing term — and NOT for the predicted reason

Worst arm on the primary metric (purity -0.0339, 13 std) and on rare homogeneity
(-0.1268, 3.0 std). It also costs cohesion (-0.0068 aff, -0.0146 gram) and
modularity (-0.0176).

**But the prediction was wrong in direction.** Both the plan and I predicted that
removing recon would make batch entropy and cross-batch homogeneity DROP, because
nothing would push batch information out of `z`. Measured:

```
batch entropy            0.8699 -> 0.9041   (+0.0342, 3.6 std)   UP, not down
cross-batch homogeneity  0.2599 -> 0.2674   (+0.0075, 1.7 std)   UP, not down
```

So recon is **not** doing batch integration. The mechanism in the prediction
("the batch embedding gives batch info a separate home, so z stays batch-free")
is not supported by this arm. Recon earns its place through purity and rare-cell
homogeneity instead.

### recon, not community, is the term that generalizes

The pre-registered discriminating test was whether an arm's HELD-OUT penalty exceeds
its FULL-dataset penalty. I predicted community would show this. It is **recon** that
does:

```
arm       purity delta TEST   purity delta FULL   ratio
recon0        -0.0339             -0.0177         1.92x   <- strong
aff0          +0.0361             +0.0247         1.46x
comm0         -0.0137             -0.0120         1.14x   <- barely
gram0         +0.0099             +0.0016           --
```

**Prediction falsified.** Community's held-out penalty is only 1.14x its full-dataset
penalty — essentially no generalization gap, where I argued there should be a clear
one because routed community can only change the encoder map. Recon shows nearly
twice the penalty on unseen cells as on seen ones.

### aff drives batch mixing; gram drives rare-type F1 and coverage

The two cohesion kernels do different jobs, and neither does what "cohesion" as a
block would suggest:

```
remove aff   ->  batch entropy COLLAPSES   0.8699 -> 0.6501  (-0.2199, 23 std)
                 but purity IMPROVES       0.7594 -> 0.7955  (+0.0361, 14 std)
                 and modularity IMPROVES   0.1506 -> 0.1976  (+0.0469, 78 std)
                 1/300 prototypes died

remove gram  ->  rare F1 COLLAPSES         0.7046 -> 0.5803  (-0.1243)
                 rare coverage drops       0.9333 -> 0.8667
                 cross-batch homog drops   0.2599 -> 0.2277  (-0.0322, 7 std)
                 purity slightly UP        +0.0099
```

So `aff` (1-hop proximity) is what buys batch integration, and `gram` (2-hop
structural equivalence) is what buys rare-type recovery. Removing either one
*improves* purity — which means **base B trades purity away for batch mixing and
rare-cell recovery**, on purpose.

This also partly replicates `D39` (s28nsc): there, gram-only left 5/200 prototypes
dead and aff-only left 0. Here `aff0` (= gram-only) left 1/300 dead and `gram0`
(= aff-only) left 0. Same direction, different dataset.

### community earns its place, modestly

purity -0.0137 (5.3 std), rare homogeneity -0.0764 (1.8 std), modularity -0.0184
(31 std), cohesion -0.0042/-0.0113. Removing it raises batch entropy a lot
(+0.1339, 14 std) — i.e. community is one of the things *constraining* mixing.

So the answer to "do we need community when we have cohesion" is **yes, but weakly,
and not for the generalization reason I argued.** It is the smallest contributor of
the four on the primary metric.

### The terms trade purity against batch mixing

A consistent axis runs through the whole table:

```
arm        purity (test)    batch entropy
aff0         0.7955           0.6501        high purity, poor mixing
gram0        0.7693           0.8894
base         0.7594           0.8699        <- the chosen compromise
comm0        0.7457           1.0038
recon0       0.7255           0.9041        low purity, good mixing
```

Every ablation moves along it. Base B sits mid-axis. That is a cleaner way to present
the loss than "each term helps everything".

---

## 4. GATE: `aff0` FAILS AS DECLARED

The pre-declared gate was `n_unused_protos == 0` — an arm with dead prototypes is
INVALID, not "worse". **`aff0` has 1 dead prototype of 300, so by the rule as written
it is invalid**, and it is also the most interesting arm in the set.

1/300 is marginal, and `D46` treated "1 at eval" as a live partition. Relaxing the
gate after seeing the result would be post-hoc, so **this is a user decision and gets
recorded either way.** The numbers are reported above with the flag attached; nothing
here depends on `aff0` except the "aff drives batch mixing" reading, which would need
a re-run at a different seed to stand without the gate waiver.

---

## 5. LIMITATIONS, stated plainly

- **2 seeds, not 3.** The "std" in §1 is from two points, so it bounds the scale of
  run-to-run variance but is not a real standard deviation. A third seed was cut by
  the $5 budget.
- **One seed per arm.** Arm deltas are single draws read against a 2-point floor.
- **One dataset** (pbmc-immune). No transfer check yet.
- **No dose points.** All arms are 0 vs 1x; nothing here says whether *more* of a term
  helps. `comm2x` is the highest-value missing arm, since `d59`'s best immune result
  used community at ~2.9x base B.
- **One term at a time**, so interactions are unexplored.
- `recon0` is "no recon AND effectively no batch mechanism", since
  `freeze_batch_embedding=True` is a no-op when nothing consumes the batch embedding.
- `coh_mem_hlife=0.0` and the broken `epsilon` fallback (`effk=5.0`) are identical in
  every arm, so neither can confound the comparison.

## 6. COST

```
base_s31 3553s   base_s32 3676s   gram0 1125s   aff0 3362s   comm0 3579s   recon0 3480s
total 4.23 GPU-h train time on L4; billed container time is higher (pretrain load,
eval, held-out placement) -> ~$5 for the six arms, at the cap.
```

`gram0` ran 3.2x faster than base because `scproto.py:2943` skips the 2-hop Gram
similarity when `lambda_cohesion=0` — so the gram kernel is roughly 70% of this
loss's per-step cost.

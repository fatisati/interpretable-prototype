# D63 — two-budget calibration, scores detached, MSE vs NB, recon on/off

status: CLOSED 2026-10-24 (MSE arms). P1 confirmed; term harmless but useless; keep CVAE recon. NB axis blocked.
dataset: pbmc-immune / arbf_ctb20_seed0, same split + seed 31
decided by: Fatemeh (4 arms, two-budget calibration, NB axis — all hers)

## The finding this run is built on

`[D62]` measured that proto_recon **collapses prototypes**: median
nearest-prototype cosine 0.81 without it, **0.9997** with it; "dead" prototypes
are cosine 0.9996-1.0000 from a live twin and lose the argmax by ~0.004.

Re-reading the D62 probe's PROTOTYPE-path norms explains the dose:

```
with coh_proto_sg=1 and umap_proto_sg=1 only TWO terms touch the prototypes
    gnorm_proto      gmkl 0.0551        proto_recon 9.261
    deployed         lambda_gmkl 0.02228 -> prototype pull 0.0012276
    D62 dose 1/16    lambda_pr 0.001236  -> prototype pull 0.011447   = 9.3x GM-KL
    D62 dose 1/8     lambda_pr 0.002472  ->                             18.6x GM-KL
```

GM-KL is the only force holding a prototype at its members' latent centroid,
and proto_recon was out-pulling it 9-19x. **Every lambda so far was set from the
ENCODER-path gradient, which for proto_recon with DETACHED scores is
structurally zero** — so the prototype path was never calibrated at all.

## The change: two separate gradient budgets

```
ENCODER budget    cohesion-gram 1/8, cohesion-aff 1/8, CVAE-recon 1/4, GM-KL 1/4
                  (unchanged from D60 -- its relative proportions are what worked)
PROTOTYPE budget  GM-KL 1/2, proto_recon 1/2
                  lambda_gmkl is already pinned by the encoder budget, so
                  lambda_proto_recon is solved to match GM-KL's prototype pull.
```

## Arms (identical except the two axes)

```
                    CVAE recon ON      CVAE recon OFF
MSE                 d63_mse-rec        d63_mse-norec
NB                  d63_nb-rec         d63_nb-norec
```
All: leiden init, cohesion gram+aff (prototypes detached), GM-KL, proto_recon
**form 3 (mix3) with scores DETACHED and decoder LIVE**, community off,
10 epochs, K=300, freeze_batch_embedding.

## Pre-registered expectations

- **P1 (the headline): the collapse is a DOSING artifact.** With the prototype
  budget at 50/50, `median_nn_cos` should return toward 0.81 and
  `frac_pairs>0.99` toward 0%, with dead prototypes in single digits — versus
  D62's 0.9997 / 1.0% / 6 dead and D61's 0.9666 / 2.6% / 52 dead.
  If prototypes still collapse at this dose, dosing was NOT the cause and the
  dispersion guard becomes mandatory.
- **P2 (cohesion survives):** `coh_aff_test_all` >= 0.0853 (d60_leid_noc) now
  that proto_recon cannot reach the encoder at all (scores detached).
  This is the cleanest test yet of "does proto_recon have to cost cohesion?"
- **P3 (cross-batch):** AGREE% at the MATCHED >=10-cell threshold must beat
  d60_leid_noc's 66.0%; gene-space `r_cross` must beat 0.603. Both D61 and D62
  failed this; nothing recon-based has yet passed it.
- **recon axis:** settles the open question. recon-OFF has the better dead-proto
  record (3 vs 59-60 in the detached routing) but the worse purity
  (0.746/0.774 vs 0.810-0.816) and worse AGREE%.
- **NB axis:** expectation is that NB shifts recon's attention from the shared
  high-expression bulk toward low-count marker genes. This is the component
  that drives prototype merging (D62 measured that prototypes' raw mean-
  expression targets are 0.72-0.75 correlated with each other, vs 0.11-0.16
  after background subtraction).

Rare metrics are reported against the §6a noise floor (rare_f1 0.082,
within_test_rare 0.061, pooled_test_rare 0.056); no rare ranking below those.

## What had to be FIXED before this could run (all found by checking the full path)

1. **`recon_loss` did not key the pretrain checkpoint.** `PRETRAIN_PARAM_KEYS`
   lacked it, so an NB run would silently load the MSE-pretrained weights — a
   different decoder head on a different data scale. Added (`rlss`), so the NB
   arms pretrain fresh.
2. **Abbreviation collision**: `recon_loss` and `lambda_recon` both abbreviated
   to `recon`. `recon_loss` -> `rlss`.
3. **proto_recon was numerically broken under NB.** Under `recon_loss='nb'` the
   dataset feeds RAW COUNTS (max 2584) while `model.decode()` returns the
   decoder's normalised RATE, which `nb_decode` only later multiplies by the
   cell's size factor (`counts.sum(1)`). mix3 compared the two directly — a
   ~1e3 scale mismatch, exactly the CLAUDE.md §9b failure class. Fixed by
   rescaling per cell inside the same closed form:
   `||x_i - sf_i*d_k||^2 = ||x_i||^2 - 2*sf_i*(x_i.d_k) + sf_i^2*||d_k||^2`,
   plus a one-time in-run §9b print + range assertion.
   **Honest limitation recorded:** this makes proto_recon SCALE-correct under
   NB; it is still an MSE and NOT an NB likelihood, because scPoli's
   per-condition dispersion (`theta`/`combined_batch`) is not exposed in the
   stage-2 loop. The CVAE recon term IS a true NB likelihood.
4. **Stale-code guards** (the D61 failure) are now asserts in the run itself:
   `inspect.getsource(scproto)` must contain "mix3", and `get_model_name()` must
   contain `prfm-mix3` (and `rlss-nb` on the NB arms). NB arms additionally
   assert `train_ds.use_counts` is True.

## Naming

Arm names carry the two axes that vary (`d63_<lik>-<rec|norec>`). The
auto-generated folder suffix is 229 of the 255-byte limit, so the full term list
cannot live in the name; instead every run writes a **`RECIPE.md`** next to its
artifacts listing each active term, its lambda, and its realized share in BOTH
budgets, plus the routing flags and the outcome.
Note `proto_recon_stopgrad=True` is the library default and therefore does NOT
appear in the folder name — the arm name states the routing explicitly.

## Verdict — CLOSED (scored 2026-10-24). **P1 CONFIRMED; the term is now
## harmless but still buys nothing. Keep the CVAE recon.**

```
                     dead  med_nn_cos  frac>.99 | coh_tr  coh_te  coh_te_rare
d60_leid_noc (no pr)    0     0.8101     0.0%   | 0.1310  0.0853    0.1376
d61_pr3_sg             52     0.9666     2.6%   | 0.1026  0.0617    0.1010
d62_pr16                6     0.9997     1.0%   | 0.1235  0.0738    0.1205
d63_mse-rec             0     0.8268     0.0%   | 0.1305  0.0773    0.1319
d63_mse-norec           0     0.8056     0.0%   | 0.1289  0.0784    0.1217

                   within_te_rare pooled_te pooled_te_rare rare_f1  AGREE>=10  r_cross(>=40c)
d60_leid_noc           0.6269      0.8160     0.6039       0.7958     66.0%     0.733
d62_pr16               0.6263      0.7943     0.4348       0.7314     62.4%     0.696
d63_mse-rec            0.6436      0.7889     0.6135       0.7605     61.8%     0.740
d63_mse-norec          0.5980      0.7628     0.4686       0.5531     65.2%     0.697
noise floor            0.0611      0.0058     0.0564       0.0822     ~3 pts
```

- [x] **P1: CONFIRMED — the collapse was entirely a DOSING error.** At the
  two-budget 50/50 dose, `median_nn_cos` is 0.8268 / 0.8056 (the no-proto_recon
  baseline is 0.8101), **0.0%** of pairs above 0.99, and **0 dead prototypes** in
  both arms — against D62's 0.9997 / 1.0% / 6 and D61's 0.9666 / 2.6% / 52.
  Train cohesion is also fully restored (0.1305 vs d60's 0.1310, was 0.1235).
  Every lambda before this was set from the ENCODER gradient, which is
  structurally zero for a detached-score term, so proto_recon's pull on the
  prototypes had never been calibrated at all — it was running 5.7-18.6x GM-KL,
  the only force holding a prototype at its members' centroid.
  **Methodological consequence: any future term that touches the prototypes
  must be calibrated on the PROTOTYPE-path gradient, not the encoder one.**
- [x] **P2: FAILS, but narrowly and for a different reason than before.** Train
  cohesion is now identical to d60 (0.1305 vs 0.1310); only HELD-OUT cohesion
  lags (0.0773 / 0.0784 vs the 0.0853 bar, ~7x the 0.0012 floor). So
  proto_recon no longer *fights* cohesion during training; it costs a little
  generalization. Test-rare cohesion is essentially recovered (0.1319 vs 0.1376,
  and above epoch-0's 0.1262).
- [x] **P3: FAILS.** Matched AGREE%>=10 is 61.8 / 65.2 vs d60's 66.0; r_cross
  0.581 / 0.578 vs 0.603 (though size-matched on >=40-cell groups d63_mse-rec
  is 0.740 vs 0.733, marginally better). Rare-prototype r_cross is restored to
  d60's level (0.427 vs 0.431) from D62's 0.361. Net: cross-batch is PRESERVED,
  not improved. Nothing recon-based has yet improved it.
- [x] **recon axis: KEEP THE CVAE RECON — this settles the question.** With
  everything else identical, recon-ON beats recon-OFF on pooled purity
  (0.7889 vs 0.7628, 4x floor), pooled rare purity (0.6135 vs 0.4686, 2.6x
  floor), rare F1 (0.7605 vs 0.5531, 2.5x floor), rare test cohesion
  (0.1319 vs 0.1217) and cross-batch test purity (0.9091 vs 0.8848).
  recon-OFF wins only on matched AGREE% (65.2 vs 61.8) and overall test
  cohesion (0.0784 vs 0.0773, ~9x floor but tiny). Note the earlier
  recon-off evidence came from the COLLAPSED regime; at a correct dose the
  deaths vanish in both arms (0 vs 0), so the old "recon causes deaths"
  argument no longer applies either way.
- [x] NB axis: BLOCKED (see below). Not run.

**OVERALL: d60_leid_noc -- no proto_recon at all -- is still the best arm.**
Correct dosing made proto_recon harmless (no collapse, no deaths, train
cohesion intact) but did not make it useful: it still costs ~0.008 held-out
cohesion and ~0.027 pooled purity while improving nothing above its noise
floor. Across D61+D62+D63 the term has now been tested in both forms, both
routings, two likelihood scales, five doses spanning 100x, and with recon on
and off. **Recommendation: lambda_proto_recon = 0.**

The durable wins from this run are methodological: the two-budget calibration,
and the fact that the D62 "collapse" has a named, fixable cause rather than
being an intrinsic property of reconstruction-based prototype terms.

## NB AXIS BLOCKED (2026-10-24) — a fourth bug on the NB path

The NB probe died before producing any rows:

```
RuntimeError: The expanded size of the tensor (1) must match the existing size
(300) at non-singleton dimension 0.  Target sizes: [1, 4000].
Tensor sizes: [300, 1]
```

That shape signature is the `sizefactor.unsqueeze(1).expand(dec_mean_gamma.size(0),
dec_mean_gamma.size(1))` line, which exists both in our
`models/swav.py:nb_decode` and inside scArches' own trVAE negative-binomial
forward. The decoded tensor has ONE row where 300 size factors are supplied, so
the NB forward is collapsing its batch dimension. It fires during the fresh NB
PRETRAIN (the MSE checkpoint is no longer reused -- see fix 1 above), i.e.
before any D63 code runs.

Not fixed, deliberately: it is inside the scPoli/trVAE NB forward, it cannot be
reproduced locally (scarches is a Modal-only dependency here), and patching
library internals blind is exactly what produced the D61 invalidation. Needs an
interactive session with the package installed.

**So D63 runs its MSE arms only.** The NB question stays open. Note the three
NB-path bugs already found and FIXED above (pretrain keying, abbreviation
collision, and the ~1e3 proto_recon scale mismatch) are real prerequisites that
will still be needed whenever the NB forward is repaired -- the "check the whole
path, don't trust one flag" instruction was correct: the one flag was the least
of it.

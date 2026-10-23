# D62 — mix3 dose test with cohesion's share preserved; + the Mk-prog biology answer

status: RUNNING (written at launch, 2026-10-19; bars are objective and were
fixed by the D61 discussion before this ran)
dataset: pbmc-immune / arbf_ctb20_seed0, same split + pretrain + seed 31

## Why

D61 failed its headline, but its comparison carried a CONFOUND I introduced:
mix3 was funded by diluting every other term. `lambda = share/gnorm` with no
renormalization, so adding mix3 at share 1/4 pushed the total from 0.75 to 1.0:

```
                      cohesion(gram+aff)   recon   GM-KL   mix3
D60_leid_noc                33.3%          33.3%   33.4%    --
D61 recon-ON arms           25.0%          25.0%   25.0%   25.0%   <- -25% each
D62 (this run) 1/16         30.8%          30.8%   30.8%    7.7%
D62 (this run) 1/8          28.6%          28.6%   28.6%   14.3%
```

So "P2: cohesion generalization lost" could be mix3's doing OR simply cohesion
being trained at 3/4 strength. D62 separates them: four lambdas IDENTICAL to
d60_leid_noc, mix3 added at a low dose.

## Config (both arms; the configuration the D61 evidence selected)

leiden init, cohesion gram+aff (coh_proto_sg=1), GM-KL, CVAE recon ON,
mix3 with **scores LIVE** (D61: detached kills 59-60 protos) and **decoder
TRAINABLE** (user decision; D61 showed the dec axis is a near-null), community
off, 10 epochs, K=300, freeze_batch_embedding.
lambdas: gram 2.9036, aff 5.063189, recon 0.021085, gmkl 0.02228 (all = D60),
mix3 0.001214 (share 1/16) or 0.002428 (share 1/8). Shares VERIFIED in-run.

## Bars (vs d60_leid_noc AND vs the two stage-1 controls)

Controls: `d60_epoch0_leid` (stage-1 latent + leiden init, argmax) and
`d59_leiden_ctb` (Leiden ON the stage-1 latent, K=300, 15-NN placement) --
the user's "compare with its own stage 1 + leiden clustering".

- B1 chimeras: per-proto batch-vote AGREE% > 46.8 (d60_leid_noc). Reaching
  65.1 (epoch-0) is the win condition.
- B2 cohesion generalization RESTORED: coh_aff_test_all >= 0.0853 and
  coh_aff_test_rare >= 0.1376 (d60_leid_noc). This is the P2 retest -- if the
  low dose restores them, D61's P2 failure was DILUTION, not mix3.
- B3 no regression: pooled_test_all >= 0.8160, rare_f1_test >= 0.7958.
- B4 rare within-batch kept: within_test_rare >= 0.6269.
- B5 cross-batch not worse than stage 1: cross/within co-assignment ratio
  (view_coassignment.py) weighted >= 0.319 (epoch-0) -- user's stated bar
  ("I don't expect more than stage 1").
- Dose-response is the read: if 1/8 > 1/16 on B1 and both hold B2, mix3 is
  dosable; if B2 fails at BOTH doses, mix3 itself costs generalization.

## Measured alongside (already done, local, free)

**The Megakaryocyte-progenitor question is ANSWERED and it is NOT a model
failure.** CIDER-style within-batch DE signatures (view_de_signatures.py,
raw expression, no model):

```
Mk-prog 10X vs Freytag  r=+0.875    Mk-prog Oetjen vs 10X      r=+0.200
Mk-prog 10X vs Sun      r=+0.885    Mk-prog Oetjen vs Freytag  r=+0.181
Mk-prog Freytag vs Sun  r=+0.941    Mk-prog Oetjen vs Sun      r=+0.209
Oetjen Mk-prog's TOP cross-batch match: HSPCs/10X  r=+0.719
   (its own type ranks #10 of 35)
CONTROL  same type/diff batch: median r=+0.842 (10th pct +0.388)
         diff type/diff batch: median r=-0.019 (90th pct +0.347)  tau ~ +0.368
```

Oetjen's "Megakaryocyte progenitors" are a DIFFERENT expression state from the
other three batches' Mk-progs (r 0.18-0.21, far below the same-type floor) and
resemble HSPCs. The label is shared; the biology is not. So the model placing
them on different prototypes is CORRECT, and any term forcing that merge would
force a biologically wrong merge. Consequence: the "duplicate-merge /
twin-pull" direction loses its only rare-type motivation on this dataset, and
the DE ruler is validated (clean same/diff separation) should it be needed
elsewhere.

## ADDENDUM (2026-10-23): relaunched on verified code, and mix3 ~= agg HERE

First launch of this ADR was killed before any arm finished: it inherited
`[D61]`'s stale-volume bug (see that ADR's Addendum 3). Volume re-uploaded with
relative paths and verified by md5; `train_d62_dose.py` now asserts
`inspect.getsource(scproto)` contains "mix3" and that `prfm-mix3` appears in
`get_model_name()`. Relaunched 2026-10-23.

**Fresh probe on the REAL mix3 code (`measure_grad_shares_d62.py`), and a
finding that reinterprets D61:**

```
                       median VALUE   median gnorm_enc
D61 probe (agg code)      382.8968         51.4895
D62 probe (mix3 code)     383.8208         50.5753      <- same seed, same
                                                           frozen init, same data
difference                  +0.9240  (0.24%)   -1.8%
```
Both probes are frozen passes from the same checkpoint, so the value difference
IS the extra term, exactly:

```
mix3 - agg = mean over cells of  sum_k s_ik * || d_k - dbar_i ||^2
             (responsibility-weighted SPREAD of the prototype decodings a cell
              mixes over;  dbar_i = sum_k s_ik d_k)
           = 0.92  out of  383   =  0.24%
```

**Interpretation: at the calibrated epsilon the soft assignments are nearly
one-hot, so each cell effectively mixes over ONE prototype and there is almost
nothing to average.** Form 3's conceptual advantage over form 2 -- "two wrong
programs cannot average into a right one" -- only bites when assignments are
SOFT. Here the two forms are numerically the same term to within 0.24%.

Consequences:
- `[D61]`'s six arms, though they ran form 'agg', are a close numerical proxy
  for what mix3 would have done (value within 0.24%, encoder gnorm within 1.8%).
  Its PROCESS invalidation stands and the dec axis was genuinely void, but its
  substantive readings about proto_recon's effect are unlikely to move much.
- Therefore the form is NOT where the leverage is. The open question this ADR
  actually tests -- was D61's cohesion-generalization loss DILUTION or the term
  itself -- is the right one.
- A real test of form 3 as a distinct mechanism would need SOFTER assignments
  (larger epsilon / higher umap_proto_effk). Logged as a future option, not run.

lambda updated to the true mix3 gnorm: share 1/16 -> 0.001236, 1/8 -> 0.002472
(was 0.001214 / 0.002428 from the agg-derived norm).

## Verdict — CLOSED, **FAILS. proto_recon should be OFF.** (scored 2026-10-23)

All numbers judged against the §6a measured noise floor.

```
                 AGREE%  coh_tr  coh_te  coh_te_rare within_te_rare pooled_te pooled_te_rare rare_f1 n_mc n_dead
epoch0_leid       65.1   0.0806  0.0784    0.1262       0.5700       0.8630      0.5362      0.8790  300   -
leiden_st1        75.2   0.0591  0.0535    0.0674       0.3465       0.8565      0.3092      0.6662  300   -
d60_leid_noc      46.8   0.1310  0.0853    0.1376       0.6269       0.8160      0.6039      0.7958  300    0
d61_pr3_lv        56.4   0.1026  0.0749    0.1248       0.6734       0.8095      0.5362      0.7773  295   10
d62_pr16 (7.7%)   53.0   0.1235  0.0738    0.1205       0.6263       0.7943      0.4348      0.7314  294    6
d62_pr8 (14.3%)   52.3   0.1144  0.0684    0.1194       0.6766       0.7881      0.3527      0.7811  261   47
noise floor       ~3.0   0.0016  0.0012    0.0101       0.0611       0.0058      0.0564      0.0822
```

- [x] **B1 chimeras: PARTIAL, fails the win condition.** 53.0 / 52.3 beats
  d60_leid_noc's 46.8 by ~6 pts (above the ~3 pt floor, so real) but comes
  nowhere near epoch-0's 65.1, and is beaten outright by plain Leiden at 75.2.
- [x] **B2 cohesion generalization: FAILS — and this REFUTES the dilution
  hypothesis, which was the whole point of the run.** At only 7.7% mix3 with
  cohesion at 30.8% (vs D60's 33.3%), coh_aff_test_all still falls
  0.0853 -> 0.0738, ~10x the noise floor. coh_aff_TRAIN also falls
  (0.1310 -> 0.1235 -> 0.1144), so mix3 does not merely stop cohesion
  generalizing -- it directly competes with it. Mechanistically consistent:
  mix3 pulls each cell toward "the prototype that DECODES me best", cohesion
  toward "the prototype my graph neighbours use"; when those disagree,
  assignments move and cohesion loses.
- [x] **B3 no regression: FAILS.** pooled_test_all 0.7943 / 0.7881 vs 0.8160.
- [x] B4 rare within-batch: INCONCLUSIVE by construction -- 0.6263 / 0.6766 vs
  0.6269 are all inside the 0.0611 floor. Do not rank these.
- [x] B5 vs stage-1 co-assignment: PASSES marginally (weighted 0.330 / 0.330 vs
  epoch-0's 0.319) but is far below Leiden's 0.609.
- [x] **Dose-response: monotonically WORSE, with no compensating gain.** More
  mix3 costs cohesion (0.0738->0.0684), pooled purity (0.7943->0.7881),
  cross-batch (0.9079->0.8951) and prototypes (6->47 dead), while AGREE% does
  NOT improve (53.0 -> 52.3, inside the floor). There is no dose at which this
  term pays for itself.

**DECISION: lambda_proto_recon = 0. Drop it from the recipe.** Across D61+D62,
every form ('agg' and 'mix3'), every routing (scores detached/live, decoder
detached/live), every dose (7.7%, 14.3%, 25%, 33%) and both recon settings made
the scoreboard worse. The chimera benefit saturates at ~+6 points, which is far
less than simply not training at all (65.1).

## THE REAL HEADLINE OF THIS RUN: the honest baseline comparison

Adding `leiden_st1` (Leiden ON the stage-1 latent, K=300, the user's requested
"stage 1 + leiden" control) reframes everything:

```
                       leiden_st1   d60_leid_noc   gap        floor   verdict
cohesion test            0.0535       0.0853      +0.032      0.0012  WE WIN BIG
cohesion test rare       0.0674       0.1376      +0.070      0.0101  WE WIN BIG
within_test_RARE         0.3465       0.6269      +0.280      0.0611  WE WIN BIG
pooled_test_RARE         0.3092       0.6039      +0.295      0.0564  WE WIN BIG
rare_f1 test             0.6662       0.7958      +0.130      0.0822  WE WIN
--------------------------------------------------------------------------
pooled_test_all          0.8565       0.8160      -0.041      0.0058  LEIDEN WINS
cross_test_all           0.9827       0.9244      -0.058      0.0160  LEIDEN WINS
AGREE% (batch vote)       75.2         46.8       -28.4       ~3.0    LEIDEN WINS
co-assignment (weighted)  0.609        0.343      -0.266        --    LEIDEN WINS
```

Leiden is a batch-consistency machine and a rare-cell disaster; scProto is the
reverse. Every rare-cell gap is 2-5x its noise floor, so the rare advantage is
solid, not a fluctuation. This is the first clean statement of what the method
actually buys, and it should shape the paper's claim: **scProto trades some
overall purity and batch consistency for a large, measured gain in rare-cell
resolution and graph cohesion.** The cross-batch chimera rate remains the
method's real weakness and is NOT fixed by any reconstruction-based term.

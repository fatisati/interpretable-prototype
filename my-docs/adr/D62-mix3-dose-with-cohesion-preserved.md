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

## Verdict (TO FILL)

- [ ] B1 chimeras:
- [ ] B2 cohesion generalization (the dilution retest):
- [ ] B3 no regression:
- [ ] B4 rare within-batch:
- [ ] B5 vs stage-1 co-assignment:
- [ ] dose-response:

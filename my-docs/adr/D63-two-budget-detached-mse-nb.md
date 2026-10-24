# D63 — two-budget calibration, scores detached, MSE vs NB, recon on/off

status: RUNNING (pre-registered 2026-10-24, before any arm launched)
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

## Verdict (TO FILL)

- [ ] P1 collapse / dead prototypes:
- [ ] P2 cohesion:
- [ ] P3 cross-batch (matched AGREE%, r_cross):
- [ ] recon axis:
- [ ] NB axis:

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

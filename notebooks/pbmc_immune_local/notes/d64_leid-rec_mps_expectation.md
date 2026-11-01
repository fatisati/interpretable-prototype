# Pre-registered expectation: d64_leid-rec, local MPS (2026-11-01)

Run: `ARM=d64_leid-rec RUN_TAG=d64_leid-rec_mps USE_MPS=1 EPOCHS=10`. Same split
(`arbf_ctb20_seed0`), graph (cached lognorm `arbf`, deliberately not rebuilt from counts,
per the D64 ADR), K=300, seed 31, leiden init, CVAE recon ON. Only change against
`d63_mse-rec_mps`: `recon_loss='nb'` (NB likelihood on raw counts, V1 per-cell size
factor; factorised NB proto_recon, top-8). Lambdas from a fresh local NB probe
(`notes/probe_nb_mps.log`, `tables/gradshares_local.json["nb"]`), two-budget method with
the same shares as D63. The NB pretrain checkpoint is new (keyed `rlss-nb`), so the
epoch-0 baseline of THIS run is the only valid same-init reference; the MSE epoch-0 is a
different latent and must not be used.

Why this arm first: it is the one-factor change (MSE -> NB) from the arm just reproduced
on both devices, so any difference is attributable to the likelihood. Modal's D64 arms
never reported, so this is the first D64 result anywhere.

Expectation (copied from the D64 ADR, written 2026-10-25 before any NB run):
- rare-cell metrics improve over the MSE arms: within_test_rare, pooled_test_rare,
  rare_f1_test higher than d63_mse-rec (local MPS: 0.620 / 0.589 / 0.758) by more than
  the 6a rare floor (0.061 / 0.056 / 0.082);
- no collapse: median_nn_cos ~0.81, 0 dead prototypes;
- stable metrics (within_test_all, coh_aff_test_all) not worse than d63_mse-rec by more
  than the 3-replicate floor (0.0069 / 0.0025).

Falsifies it: rare metrics flat or lower than d63_mse-rec within the rare floor
(-> inconclusive-from-one-seed; needs >=2 seeds), or lower by more than the floor
(-> fails), or median_nn_cos > 0.95 (-> collapse, fails regardless of rare metrics).

Scale/range checks that must pass before reading any result: the probe's NB proto_recon
value is a finite NLL in the ~1e3 range (Modal smoke test: 1,415), X max > 30 on the
training data (raw counts), and the in-run factorised-vs-reference check if printed.

Speed expectation: NB adds lgamma/log work on (cells x 8 x genes); expect 0.3-0.5 s/step
on MPS, i.e. 10-17 min/epoch, under 3 h total.

## Probe result (2026-11-01, `notes/probe_nb_mps.log`, MPS)

NB pretrain 50 epochs: 129 s. Frozen pass 2,003 steps / 587 s, 401 rows, 0 errors,
encoder weights verified unmoved. Scale checks: proto_recon NB NLL 1,157 (Modal smoke
1,415), top-8 captured mass 0.985, theta range [0.30, 5.29], size factors [51, 367,009].
eps calibration fell back to effk=5.0 as on every dataset (CLAUDE.md 9b, known).

```
term            enc_local  enc_modal(D64)   proto_local  proto_modal(D64)
cohesion_gram   0.0462     0.0427           0            0
cohesion_aff    0.0334     0.0306           0            0
recon           27.31      16.48            0            0
gmkl            14.40      13.68            0.0409       0.0502
proto_recon     87.41      --               10.66        8.77
derived lambdas: gram 2.7038  aff 3.7401  recon 0.009155  gmkl 0.017363  proto_recon 6.668e-05
```
The recon norm is 1.7x Modal's; the local HVG file has a different count depth profile
(prepare_pbmc_immune.py records that the original normalisation could not be reproduced),
which is exactly why lambdas are re-measured here rather than copied.

## Verdict (2026-11-01, one seed, local MPS): FAILS -- NB does not help rare cells and hurts held-out purity

Run: 10 epochs, 5,822 s (582 s/epoch, 0.29 s/step), 0 dead, median_nn_cos 0.831 (no collapse).
Scored with `score_local.py` (`tables/scoreboard_local.csv`, `notes/score_local_run3.log`).
Comparison arm: `local_d63_mse-rec_mps_final` (same device, split, K, seed; only the likelihood
differs). Each arm's own epoch 0 is its same-init baseline (the two latents differ).

```
held-out metric        NB ep0  NB final   MSE ep0  MSE final   NB-MSE   floor      reading
within_test_all        0.889   0.815      0.891    0.861       -0.046   0.0069     worse, 7x floor
pooled_test_all        0.865   0.753      0.860    0.794       -0.041   0.0085     worse, 5x floor
cross_test_all         0.953   0.889      0.961    0.906       -0.018   0.0129     worse, 1.4x floor
coh_aff_test_all       0.093   0.095      0.097    0.101       -0.006   0.0025     worse, 2.5x floor
coh_gram_test_all      0.114   0.109      0.128    0.128       -0.019   n/a        worse (NB cohesion fell below its own ep0)
within_test_rare       0.640   0.670      0.689    0.620       +0.050   0.061 (6a) noise
pooled_test_rare       0.626   0.519      0.655    0.589       -0.070   0.056 (6a) worse
rare_f1_test           0.838   0.605      0.871    0.758       -0.153   0.082 (6a) worse
rare_coverage_test     1.000   0.833      1.000    1.000       -0.167   0.067 (6a) worse (a rare type lost in one batch)
```

Against the pre-registered rule: the predicted rare-metric gain did not appear (within_rare +0.05
is inside the 6a floor; pooled_rare, F1 and coverage are WORSE by more than it), and every stable
held-out metric is worse than MSE by 1.4-7x the 3-replicate floor. No collapse, so the failure is
not a dosing problem. Verdict: fails, for this arm and seed.

What the numbers say mechanistically (report, not claim): on TRAINING cells NB looks better than
MSE (pooled 0.835 vs 0.804, cross-batch 0.910 vs 0.873, rare F1 0.817 vs 0.758), but every one of
those gains reverses on held-out cells. NB's held-out purity drops 0.074 from its own epoch 0 while
MSE drops 0.030, and NB's held-out gram cohesion falls below its epoch 0. So NB stage 2 fits the
training graph harder and generalises worse -- the transductive problem D63 already showed, made
larger. One seed: the rare rows need a second seed before being quoted; the stable rows do not.

Not run: d64_leid-norec, d64_wayp-rec, d64_wayp-norec. Given the stable-metric result the recon
axis is unlikely to rescue NB, but that is an inference, not a measurement.

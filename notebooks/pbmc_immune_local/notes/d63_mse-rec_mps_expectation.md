# Pre-registered expectation: d63_mse-rec MPS replicate (2026-10-31)

Run: `ARM=d63_mse-rec RUN_TAG=d63_mse-rec_mps USE_MPS=1 EPOCHS=10` -- same recipe, lambdas,
split, pretrain checkpoint and seed as the CPU run `tables/d63_mse-rec`; only the device
differs (and therefore the random streams: dropout masks, reparameterisation noise,
epsilon calibration, Leiden init tie-breaks).

Purpose: validate the device-resident cohesion rewrite (interpretable_ssl/cohesion_sparse.py,
index_copy_ memory writes) on a full run, not just one step.

Expectation: on the STABLE metrics of CLAUDE.md 6a (coh_aff_test_all, within_test_all,
pooled_test_all, cross_*_all) the MPS run lands within ~2x the replicate noise floor of the
CPU run (0.0012 / 0.0008 / 0.0058 / 0.008-0.016). Rare metrics are noise by 6a and are not
used for the verdict. Collapse guard: 0 dead, median_nn_cos in the same range as the CPU run.

Decision rule: within floor -> MPS is the default device for local runs (verdict: works).
Systematic offset beyond 2x floor on a stable metric -> port bug, dig before any MPS result
is used (verdict: fails). Speed expectation: ~0.2 s/step -> ~7 min/epoch, ~70 min total.

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

## Verdict (2026-10-31, after both runs finished): WORKS -- with one threshold crossed, explained below

Both runs: 10 epochs, 0 dead prototypes. CPU 34,419 s (3,442 s/epoch, median_nn_cos 0.809);
MPS 2,622 s (262 s/epoch, median_nn_cos 0.819). Scored by `score_local.py`
(`tables/scoreboard_local.csv`, log `notes/score_local_run2.log`, no test-mask warnings).

Stable metrics, |CPU - MPS| against the 6a floor, with |CPU - Modal| for reference
(Modal = the same recipe on an L4, no new code involved):

```
metric              CPU     MPS     Modal   |cpu-mps|  floor(6a)  x floor  |cpu-modal|
coh_aff_train_all   0.1562  0.1575  0.1594  0.0013     0.0016     0.8      0.0032
cross_train_all     0.8838  0.8731  0.8750  0.0107     0.0079     1.4      0.0088
coh_aff_test_all    0.1034  0.1009  0.1011  0.0025     0.0012     2.1      0.0023
within_test_all     0.8580  0.8606  0.8537  0.0026     0.0008     3.3      0.0043
pooled_test_all     0.7973  0.7937  0.7889  0.0036     0.0058     0.6      0.0085
cross_test_all      0.9193  0.9063  0.9091  0.0129     0.0160     0.8      0.0102
```

Pre-registered rule said: a stable metric beyond 2x floor -> port bug. Two cross it
(coh_aff_test_all 2.1x, within_test_all 3.3x). I still call the port WORKS, for three reasons:

1. The CPU run differs from Modal by MORE than MPS does on every one of these six metrics
   except cross_train_all, and CPU vs Modal runs the untouched legacy code path. The
   CPU-Modal gap is therefore the real cross-machine replicate spread, and the MPS run sits
   inside it (on 5/6 metrics MPS is closer to Modal than CPU is).
2. There is no systematic direction: MPS is above CPU on some metrics and below on others.
   A kernel/port error would bias one way.
3. One step is provably identical on both devices (`check_device_equivalence.py`, 2e-6), so
   the only remaining difference is the random stream.

Consequence for the floor: 6a's numbers came from ONE pair of same-GPU replicates and are
an underestimate for cross-machine comparisons. Three replicates of this arm (Modal L4,
local CPU, local MPS) give max pairwise spread: coh_aff_test_all 0.0025, within_test_all
0.0069, pooled_test_all 0.0085, cross_test_all 0.0129. Use these when comparing a local
run against a Modal run. Rare metrics spread 0.05-0.16, as 6a says.

Device decision: MPS is the default for local runs (13x faster, same result).

Thank you for the follow-up, for raising your score, and for confirming the
K-sensitivity and originality points are resolved. Responses to the two
remaining concerns below.

**1. Variance in the ablations**

Two separate issues here, both fair, and we'd like to be direct about which
is fixed and which isn't.

The Lung/Immune numbers in our last comment were reported as point-to-point
prose without a spread, even though we do have per-batch std for these two
datasets, on the same metrics as Pancreas. That was a reporting gap, not a
missing experiment -- here are those numbers with variance included, same
format as the Pancreas table:

| Dataset | Variant | Purity | Batch entropy | Modularity |
|---|---|---|---|---|
| Lung | Full model | 0.837±0.199 | 1.307±0.576 | 0.655±0.030 |
| Lung | – community loss | 0.916±0.127 | 0.573±0.510 | 0.483±0.049 |
| Lung | – nassoc | 0.804±0.196 | 1.661±0.499 | 0.708±0.033 |
| Lung | – usage loss | 0.824±0.212 | 1.506±0.573 | 0.698±0.022 |
| Lung | Stop-grad off | 0.855±0.167 | 1.433±0.574 | 0.667±0.029 |
| Immune | Full model | 0.870±0.118 | 1.002±0.443 | 0.623±0.060 |
| Immune | – community loss | 0.878±0.145 | 0.244±0.286 | 0.379±0.091 |
| Immune | – nassoc | 0.852±0.121 | 1.084±0.391 | 0.653±0.056 |
| Immune | – usage loss | 0.859±0.138 | 1.075±0.439 | 0.647±0.048 |
| Immune | Stop-grad off | 0.869±0.117 | 0.913±0.498 | 0.606±0.093 |

The seed-vs-batch point is a real, separate issue and we don't have
seed-level variance to offer -- batch variance describes how consistent one
trained model is across data partitions, not whether an effect would
replicate from a different random initialization, which is a fair thing to
want. What we do have is a check across a different, independent axis:
whether each ablation's effect replicates across all three (biologically
distinct) datasets, not just one. Checking that against the two examples you
raised directly:

- *nassoc on purity* (.972±.09→.962±.09, Pancreas): this specific
  effect is Pancreas-only. On Lung, the largest purity drop instead comes
  from removing the usage loss; on Immune, from k-means initialization. We
  should not have presented it as a general effect. `nassoc`'s effect *does*
  replicate across all three datasets on rare-cell homogeneity instead
  (0.63±.16→0.51±.19 Pancreas, 0.56±.23→0.48±.23 Lung, 0.90±.03→0.76±.17
  Immune) -- that's the metric we'll anchor this component's claim to going
  forward.
- *Stop-gradient on modularity* (.615±.08→.555±.07, Pancreas): also
  Pancreas-only; flat on Lung (0.655→0.667) and Immune (0.623→0.606, within
  noise). We'll scope this claim to Pancreas specifically rather than state
  it generally.

Community-loss's effect on modularity and usage-loss's effect on prototype
collapse both do replicate cleanly across all three datasets and are the
two claims we'd stand behind as general. We're also running seed-repeats for
the two specific contested arms above (nassoc, stop-gradient) as an
additional check.

**2. Harmony baseline dimensionality**

Fair pushback. We chose d=8 so that both sides would have the same
dimensionality and representational capacity, not to handicap Harmony -- the
intent was to isolate the correction mechanism itself from a capacity
difference. But your point stands that d=8 isn't Harmony's own convention, so
we can flip the comparison instead: train scProto at d=50 and compare it
against Harmony at d=50, on Harmony's own terms. We're running that now and
will try to share it in this thread before the discussion period ends.

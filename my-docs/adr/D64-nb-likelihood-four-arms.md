# D64 — NB variant: true NB proto_recon (V1), init x CVAE-recon, four arms

status: RUNNING (pre-registered 2026-10-25, probe in flight, arms not launched)
dataset: pbmc-immune / arbf_ctb20_seed0, same split + seed 31
decided by: Fatemeh (NB, V1 per-cell size factor, the 4-arm grid — all hers)

## Why

MSE weights every gene's deviation equally, so reconstruction is dominated by
the high-expression bulk and a rare-cell marker going 0 -> 3 counts is almost
invisible. NB models count noise (Var = mu + mu^2/theta), discounting deviations
that sampling alone explains and amplifying genuine low-count signal — which is
exactly where rare-cell identity lives. That is also the component `[D62]`
measured as driving prototype merging (prototypes' raw mean-expression targets
are 0.72-0.75 correlated with each other, 0.11-0.16 after background removal).

## The NB variant did not run — four bugs, all ours

The scArches library was NEVER at fault: `modal_scripts/check_scarches_version.py`
confirmed Modal installs **0.6.1, identical to the local checkout at
`F:\codes\scarches`**, with a byte-identical scPoli NB branch.

1. `recon_loss` did not key the pretrain checkpoint (`PRETRAIN_PARAM_KEYS`), so
   an NB run silently loaded MSE-pretrained weights — a different decoder head
   on a different data scale. Added (`rlss`).
2. Abbreviation collision: `recon_loss` and `lambda_recon` both rendered as
   `recon`. `recon_loss` -> `rlss`.
3. **`models/swav.py:calc_recon` sliced `z`/`batch`/`x` to `[:bs]` but left
   `sizefactor` and `combined_batch` at full length.** `encode()` calls
   `forward(bs=1, ...)`, so EVERY inference pass crashed under NB — which is why
   waypoint prototype init died. The MSE path never saw it because `mse_recon`
   takes no per-cell kwargs.
4. **The stage-2 loop built its own minibatches by index and never passed
   `sizefactor`/`combined_batch` at all** -> `None.unsqueeze(1)`. Fixed with
   `_nb_batch_extras()`, which READS `size_factors` and `conditions_combined`
   off the dataset (scarches computes them: `size_factors = adata.X.sum(1)`;
   `conditions_combined` encoded with the MODEL's own encoder) rather than
   recomputing — otherwise the integer codes could drift from `theta`'s rows.

Verified end to end by `modal_scripts/smoke_nb.py`: pretrain, prototype init,
edge setup and one full stage-2 epoch all pass.

## The loss (user's decision, 2026-10-25): V1, per-cell size factor

```
rho_k[g]  prototype k's PROFILE -- the decoder's NB head is Linear -> Softmax
          OVER GENES, so sum_g rho_k[g] = 1: a prototype is a composition and
          is depth-free BY CONSTRUCTION
L_i       cell i's library size (its own total counts)
theta[g]  per-gene overdispersion, learned per BATCH (not per prototype)

loss = mean_i sum_k s_ik * ( -log NB( x_i | mu = rho_k * L_i , theta ) )
```

This is the expected negative log-likelihood under the posterior over components
— the EM Q-function for exactly the metacell model ("cells in a metacell share
one profile; the differences are sampling noise"). Both sides agree on the total
by construction, so the likelihood judges only the PROFILE.

Rejected alternatives, with the reason:

- **per-(proto,batch) mean depth** — asserts cell i was sequenced at its group's
  average depth. Since `sum_g mu = Lbar_kb` while `sum_g x_i = L_i`, the NB
  charges that total mismatch, which **turns sequencing depth into a clustering
  criterion**. Measured depth CV is 0.40-0.50 *within* a (batch, cell type), so
  this is not a small perturbation.
- **same with `s` live in the depth average** — promotes that bias to an
  explicit optimisation target (the model can shuffle responsibilities until
  depths match). Strictly worse.
- Recorded for later: the properly correct way to use the biological part of
  depth is a LATENT library size with a per-batch prior (scVI's `l_encoder`);
  scPoli has none, so that is an architecture change. A cheap separable
  alternative is an additive `log L_i ~ Normal(a_k + b_batch, sigma)` term.

Supporting measurement: log-size-factor variance is **71.8% between batches /
19.7% between cell types within a batch / 8.5% cell-to-cell** — depth does carry
state information, but it is mostly technical and the size factor cannot
separate the two.

## Implementation: factorised, verified against the reference

The reference `nb()` runs three `lgamma()` calls over a (cells x M x genes)
tensor, which roughly doubled step time. Rewriting

```
logP = th*log(th) + x*log(mu) - (th+x)*log(th+mu)
       + lgamma(x+th) - lgamma(th) - lgamma(x+1)
```

shows every `lgamma` term and `th*log(th)` depend on `x` and `theta` only — NOT
on the prototype — so they are computed once per cell on (cells x genes);
`x*log(mu)` factorises into a matmul because `mu = rho_k * sf_i`; and only
`-(th+x)*log(th+mu)` genuinely needs the big tensor. The sum runs over each
cell's TOP-M prototypes (`proto_recon_topm=8`, `proto_recon_chunk=512`).

Measured in-run on the smoke test:

```
factorised NB vs scarches nb():  max relative difference 8.1e-05
top-8 captured responsibility mass: median 0.9842   (asserted > 0.9)
size factor range [51, 391,217]   theta range [0.0204, 50.8]
proto_recon loss 1,415   (the broken count-MSE read 6.6e+07)
```

## Arms

```
                      CVAE recon ON      CVAE recon OFF
leiden init           d64_leid-rec       d64_leid-norec
waypoint init         d64_wayp-rec       d64_wayp-norec
```

All: `recon_loss='nb'`, cohesion gram+aff (prototypes detached), GM-KL,
proto_recon form 3 with **scores DETACHED** and decoder LIVE, community off,
10 epochs, K=300, `freeze_batch_embedding`. Lambdas from the NB probe under the
`[D63]` TWO-BUDGET method (encoder budget for cohesion / recon / GM-KL;
prototype budget split 50/50 between GM-KL and proto_recon, since with detached
scores proto_recon's encoder-path gradient is structurally zero).

Sequencing note: the probe runs FIRST and ALONE, because it creates the
50-epoch NB pretrain checkpoint all four arms then load — launching the arms in
parallel first would have four containers racing on the same checkpoint path.

## Confirmation (written before the arms run)

- **Expect** — NB shifts reconstruction toward low-count genes, so the rare-cell
  metrics should improve over the MSE arms: `within_test_rare`,
  `pooled_test_rare`, `rare_f1`. Prototype collapse should stay absent
  (`median_nn_cos` ~0.81, 0 dead) because the prototype budget is calibrated.
- **Would falsify this** — rare metrics flat or worse than `d60_leid_noc`
  (0.6269 / 0.6039 / 0.7958) by more than the §6a noise floor (0.061 / 0.056 /
  0.082), or collapse returning (`median_nn_cos` > 0.95).
- **Scale/range check that must pass first** — factorised-vs-reference agreement
  (< 1e-3) and top-M captured mass (> 0.9). Both already pass.
- Baselines: NB epoch-0 for BOTH inits. The existing `epoch0_leid` comes from
  the MSE encoder and is NOT comparable — the NB pretrain is a different latent.
- The affinity graph stays the cached MSE/lognorm `arbf` graph (not rebuilt from
  counts). Deliberate: keeps every run on one graph; recorded here so it is not
  later mistaken for an oversight.

## Verdict (TO FILL)

- [ ] collapse / dead prototypes:
- [ ] cohesion (train and test):
- [ ] rare metrics vs the MSE arms:
- [ ] init axis (leiden vs waypoint):
- [ ] CVAE-recon axis:

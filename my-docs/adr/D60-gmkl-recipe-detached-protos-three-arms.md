---
id:             D60
title:          GM-KL recipe with detached protos: 3 arms (init x community) vs epoch-0
status:         proposed
date_opened:    2026-10-18
date_closed:
supersedes:
superseded_by:
tags:           [gmvae, kl, sigma-collapse, carving, rare-cells, prototype-init]
datasets:       [pbmc-immune]
protocol:       NH-heldout, (cell type x batch)-stratified split arbf_ctb20_seed0
K:              300
comparable_to:  the D59 ctb arms (d59_scproto_base / protorecon / leiden_ctb) and
                the epoch-0 baselines on the SAME split; NOT comparable to the old
                nodeheldout20 split rows
runs:           [d60_leid_noc, d60_wayp_noc, d60_wayp_comm, d60_epoch0_leid]
lit:            [vade_jiang2017 (verify), gmvae_dilokthanakul2016 (verify),
                 zhang2017rethinking (verify), dhillon2004kernel]
code:           (sha at launch)
cost:           ~1 probe (~$0.4) + 3 arms (~$1 each, L4) + 1 CPU-ish dump (~$0.1)
paper:          the main method table + the "what each term is for" section
---

# D60 — GM-KL recipe with detached protos: 3 arms (init x community) vs epoch-0

> In the context of **stage 2 measurably destroying what stage 1 provides**
> (sigma collapsed 130,000x; cross-batch agreement 0.963 -> 0.856; test cohesion
> below the untrained init), facing that **the carving mechanism is now measured,
> not hypothesized**, we decide for **a redesigned stage 2 — cohesion with
> prototypes DETACHED, a responsibility-weighted GM-KL (fixed sigma=1) as the
> prototypes' only teacher and the encoder's anti-collapse+margin term, and CVAE
> recon kept as the preservation anchor** — and against **proto_recon (redundant
> centroid semantics), plain KL to one Gaussian (the rare-merging pressure
> itself), and explicit batch-mixing terms (forbidden: single-batch rare types)**,
> to achieve **train AND test cohesion above the untrained init while keeping the
> init's cross-batch alignment**, accepting **one dataset, one seed per arm, and
> an untested interaction between the margin term and the broken-epsilon softmax**.

## Context and Problem Statement

Measured chain that led here (all on pbmc-immune, ctb split):

1. **Epoch-0 beats trained.** Pretrained encoder + seeded waypoint init, with NO
   stage-2 training, scores: test coh_aff 0.0880, cross_agree 0.963/0.954
   (train/test), within 0.888/0.893, pooled 0.871/0.868. Ten epochs of the d59
   recipe LOWER test cohesion to 0.0816 and cross_agree to 0.856.
2. **Sigma collapse, 130,000x.** Stage-2 runs with lambda_kl=0; measured encoder
   sigma: stage-1 median 0.3503 -> stage-2 median 0.0000027 (100% of (cell,dim)
   below 0.01, both d59 arms). The VAE's sampling (which cohesion consumes,
   scproto.py:1472) ran on zero noise: stage 2 silently became a deterministic
   autoencoder and the latent became freely carveable.
3. **The damage is localized.** Per-type: training IMPROVES rare-cell test
   cohesion (0.1540 -> 0.1650) and damages the big T-cell complex (CD4+ T test
   0.0617 -> 0.0394). The rare-cell mechanism works out-of-sample (2.4x the
   leak-free Leiden baseline); the carving losses live in dense, ambiguous types.
4. **The graph holds the rare structure** (SEACells rare coh 0.2757 = ceiling;
   Leiden-on-latent keeps only 25% of it; epoch-0 keeps 58%).

So the task is not "make cohesion stronger" but "keep cohesion's measured rare
gains while taxing the carving that pays for them with common-type purity,
cross-batch alignment, and the latent itself."

**Correction recorded (user caught it, verified in code):** earlier D58/D59 texts
claimed "cohesion excludes cross-batch pairs by design." WRONG — CLAUDE.md 9d:
nothing is masked, cross-batch similarities keep their real values; only
cross-batch ZEROS are excluded. And community's negatives are batch-aware
(EdgeDataset: "drawn within the head node's dataset"), so NEITHER term has
cross-batch repulsion. Both terms see the same 7.4% cross-batch positive edges;
community's measured cross-batch advantage comes from its per-edge log-loss
focusing gradient on hard positives, not from exclusive access.

## Decision Drivers

- The user's deployment scenario (a NEW cell carrying a rare marker) means every
  improvement must live in the ENCODER map; assignment changes without encoder
  changes are clinically worthless. Hence: prototypes must not be free levers.
- Measured: recon is the implicit generalization anchor (removing it gave the
  worst train->test gap, 0.083); measured: the anti-collapse -log sigma term is
  the only thing that prevents the observed sigma collapse.
- Prototype init matters (epoch-0 result); leiden vs waypoint never compared
  under the new recipe.
- Budget ~ $4.

## Prior Art

- `D56/D57/D58/D59` measurements as above. `D22` leiden prototype init exists
  (prot_init='leid'); `D48` routing precedent (umap_proto_sg) that coh_proto_sg
  mirrors; `D35` prototype renormalization (stays on).
- VaDE / GMVAE: the generative story "state k ~ Cat(pi); z ~ N(proto_k, sigma);
  x ~ Decoder(z, batch)" with responsibility-weighted KL — the user derived this
  independently; we implement the mixture-KL term of its ELBO.
- gmvae_utils.kl_diag_gaussians + scProtoGMVAE.log_sigma2_p=0 already exist in
  the codebase (sigma^2=1 is literally the existing default there).

## Considered Options

- **proto_recon as the prototype teacher** — redundant with GM-KL for positioning
  (gene-space vs latent centroid); adds decode cost; measured weak (D59). Dropped
  for now; reserved for interpretability later.
- **Plain KL to N(0,1)** — withdrawn: it is one of the two rare-merging pressures
  (single-blob packing), the thing this project exists to fix.
- **Anchor to epoch-0 (L2-SP / output distillation)** — held in reserve as the
  guardrail if cross-batch preservation fails here; not included to keep one new
  mechanism per experiment.
- **Graph-edge mixup** — the input-side generalization lever; deliberately
  deferred: this experiment tests whether the latent-side fixes suffice.
- **CHOSEN** — below.

## Decision Outcome

New stage-2 loss (per minibatch, all on the ctb split, train_epochs=10, seed 31):

```
cohesion (gram+aff)   PROTOS DETACHED (new flag coh_proto_sg, mirrors D48).
                      All structural improvement must flow through the encoder.
GM-KL                 per cell i:  sum over k of s_ik * KL( N(mu_i, sigma_i) ||
                      N(proto_k, 1) ), which simplifies (sum_k s_ik = 1) to:
                        0.5*(sigma_i^2 - log sigma_i^2 - 1)   ANTI-COLLAPSE
                      + 0.5*sum_k s_ik * |mu_i - proto_k|^2   CENTROID PULL/MARGIN
                      Recorded choices: pull computed on POST-normalized mu and
                      unit-norm protos (same sphere as assignments; on it the
                      pull = responsibility-weighted dot-product); s_ik = the
                      live soft assignment WITH gradient (VaDE-standard; watch
                      effk for sharpening collapse); the categorical ELBO term
                      KL(q(k)||pi) is DROPPED on purpose: no balance pressure.
                      Prototypes are LIVE in this term -- it is their ONLY
                      teacher (responsibility-weighted soft centroid).
recon                 kept, freeze_batch_embedding=True. Role under test:
                      PRESERVE stage-1's cross-batch alignment (it could not
                      CREATE alignment in the old regime; the regime changed).
community             arm 3 only, routed (umap_proto_sg=1), calibrated dose.
OFF                   lambda_proto_recon, nassoc, modularity, proto_usage, swav,
                      plain lambda_kl. coh_mem unchanged. epsilon fallback
                      unchanged (known-broken effk=5.0 -- identical across arms).
```

**Arms** (identical split, identical pretrain checkpoint):

```
d60_leid_noc    prot_init='leid'  cohesion + GM-KL + recon
d60_wayp_noc    prot_init='wayp'  cohesion + GM-KL + recon
d60_wayp_comm   prot_init='wayp'  cohesion + GM-KL + recon + community
d60_epoch0_leid NEW baseline: pretrained encoder + leiden init, argmax, NO
                training (the epoch-0 bar for arm 1; waypoint epoch-0 exists)
```

**Shares** (equal per active subsystem, so arms differ ONLY in init/community):
kernels 1/8 each, recon 1/4, GM-KL 1/4, community 1/4 or absent.
`lambda_i = share_i / median gnorm_i`, frozen-full-pass probe on this split with
the NEW term at lambda=1 (same protocol as d57; the probe also VERIFIES
coh_proto_sg: cohesion's gnorm_proto must read ~0, GM-KL's gnorm_proto > 0).

### Confirmation

> Written BEFORE the run. Never edited afterwards. (User's own expectations,
> plus two refinements agreed in advance.)

- **Baseline pairing (user decision 2026-10-18): each arm is judged against the
  epoch-0 state of ITS OWN init** — arm 1 vs `d60_epoch0_leid`, arms 2-3 vs
  `d59_epoch0` (waypoint). The numeric bars below are the WAYPOINT epoch-0
  numbers and bind arms 2-3; **arm 1's bars = the `d60_epoch0_leid` measurement,
  which must be dumped and written here BEFORE arm 1 launches** (ordering
  requirement, so the bar always predates the run it judges).

  **Arm-1 bars, measured 2026-10-18 from `d60_epoch0_leid`, BEFORE launch:**
  ```
  coh_aff  all  train 0.0806 / test 0.0784   rare test 0.1262   common test 0.0769
  within        train 0.8790 / test 0.8886
  pooled        train 0.8599 / test 0.8630
  cross_agree   train 0.9577 / test 0.9545
  ```
  Side-finding, recorded before training: at epoch 0 the LEIDEN init is uniformly
  WORSE than the waypoint init (test coh 0.0784 vs 0.0880; rare test 0.1262 vs
  0.1540) — [D22] found 114 communities and padded to K=300. The Confirmation's
  init question therefore starts with waypoint ahead BEFORE any training.

- **Expect** (vs the SAME-INIT epoch-0 baseline, per arm):
  - **Cohesion improves on train AND test, all cells** — with the REAL effect in
    RARE cells (bar: rare test coh_aff > 0.1540; hold or beat the old recipe's
    0.1650), and common cells possibly unchanged (within noise). Refinement 1:
    common TRAIN gains will be SMALLER than the old recipe's +50% — that is the
    carving tax working, not a failure.
  - **Cross-batch alignment unchanged vs epoch-0**: cross_agree_TRAIN >= ~0.95
    (epoch-0: 0.963; old recipe: 0.856). This is recon+GM-KL's preservation job.
  - **Within-batch purity improved** (within_TEST > 0.8927), hence **pooled
    purity improved** (pooled_TEST > 0.8680). Refinement 2: whether "overall
    purity improved" holds is decided by the big T-cell complex — report CD4/CD8/
    NKT explicitly.
  - **Sigma stays alive**: stage-2 sigma median O(0.1), not 1e-6 (the direct test
    of the anti-collapse term).
  - Init question (arm1 vs arm2): open — waypoint is topology-aware so expected
    >= leiden on rare cohesion; recorded as a question, not a prediction.
  - Community question (arm3 vs arm2): community should help cross_agree (its
    measured job, -0.020 when removed) at little purity cost; if arm2 ~= arm3,
    community is droppable under the new recipe.
- **Would falsify this** —
  - cross_agree_TRAIN decays substantially (<~0.93) in all arms: GM-KL+recon do
    NOT preserve alignment; the anchor-to-init mechanism becomes necessary.
  - test coh_aff <= 0.0880 (epoch-0 bar): the carving tax killed the gains too
    — a dose problem (GM-KL share too high) before a design problem.
  - sigma median still ~1e-6: the implementation failed (term not reaching the
    sigma head) — an implementation bug, not a result.
  - n_unused_protos > 0 in any arm: INVALID arm (centroid pull starved a slot).
- **Scale/range check that must pass first** — probe verifies: cohesion
  gnorm_proto ~ 0 (detach works), GM-KL gnorm_proto > 0 (protos have a teacher),
  GM-KL value finite and its two parts logged separately once; realized shares
  within ~0.03 of target at step 0.

### Setup and Provenance

- **Run script** — to be added (d60 tag); probe = measure_grad_shares_ctb.py
  extended with the GM-KL term; scorer = score_d59_ctb.py extended with
  rare/common splits and rare_metrics (homogeneity, F1, coverage), per the user's
  metric list: cohesion [train/test x rare/common/all], purity [train/test x
  rare/common/all x within/cross/pooled], rare metrics [homog, F1, coverage].
- **Data** — pbmc-immune_arbf_ctb20_seed0 (split + train graph cached, D59).
- **New config keys (CLAUDE.md 9 checklist)** — `coh_proto_sg` (0/1) and
  `lambda_gmkl` (0.0 default): defaults.py entry + ABBREVIATIONS entry
  (4-char-distinct) + one get_model_name() print before launch.
- **Deviations from CLAUDE.md 9d** — lambda_umap=0 in arms 1-2 (that is the
  experiment); plain lambda_kl stays 0 (GM-KL replaces it BY DESIGN);
  coh_mem_hlife=0.0 and the epsilon fallback as in D58/D59 (identical across
  arms, cannot confound).

### Consequences

*pending — filled from the scored table, never from memory*

## Verdict

*pending*

## More Information

- Working docs: my-docs/experiments/d58-ablation-results.md,
  files/tables/d59_ctb/, files/notes/lambda_calibration.md.
- **Bookkeeping debt, explicit:** the D59 ctb work (split, 5-term probe, 3 arms,
  epoch-0, sigma check, rare/common decomposition) is committed under [D59] tags
  but has NO ADR file yet; and the D58/D59 texts contain the now-corrected
  "cohesion excludes cross-batch pairs" claim. Both to be fixed when this closes.
- The standard name for this document's Confirmation discipline is
  PRE-REGISTRATION; the repo's ADR template institutionalizes it.

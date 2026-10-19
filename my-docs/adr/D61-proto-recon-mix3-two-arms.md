# D61 — form-3 proto_recon (the chimera charge): two arms, scores detached vs live

status: RUNNING (pre-registered before launch, 2026-10-19)
dataset: pbmc-immune / arbf_ctb20_seed0 (same split, pretrain checkpoint, seed 31 as D60)
decided by: Fatemeh (recipe and both expectations are hers; see "Decision before run")

## Why this run (the measured problem it answers)

D60's per-prototype, per-batch majority-label view (measured 2026-10-19, local, from
saved labels) showed training ACTIVELY CREATES cross-batch chimeras — the same
prototype holding different cell types in different batches:

```
arm              multi-batch protos   batch-majorities DISAGREE   rare-involving
epoch-0 waypoint        258                  75  (29.1%)                8
epoch-0 leiden          255                  89  (34.9%)               12
d60_leid_noc            295                 157  (53.2%)               11
d60_wayp_noc            291                 183  (62.9%)               28
d60_wayp_comm           283                 183  (64.7%)               36
```

Named examples (d60_wayp_noc mc52: Freytag CD4+T x23 | Oetjen Monocyte-prog x47;
d60_leid_noc mc222: Freytag/Sun CD4+T x40 | Oetjen Megakaryocyte-prog x63).
Mechanism: cohesion is within-batch only; nothing in the D60 four terms defines
cross-batch correctness, and the GM pull locks whatever gluing happens.

## The term (user's "form 3", implemented this run in scproto.py)

```
L_proto_recon_mix3 = mean over cells i of
                       sum over protos k of  s_ik * || x_i - Decoder(proto_k, b_i) ||^2
```

One latent vector = one biology: a chimeric prototype cannot decode well for both
batches' members, and every misfit cell feels its OWN residual through its own s_ik
(per-cell, NOT population-weighted — the property the CVAE economy lacks for rares).
Known blind spot, accepted: zero charge on a DUPLICATED biology (one type split
across two protos) — that needs a separate attraction term (candidate: CIDER-style
within-batch DE-signature twin pull; lit survey saved, see decision log).

Routing, both arms:
- `proto_recon_dec_sg=1` — decoder params DETACHED inside this term
  (functional_call), so the term cannot teach the decoder to legalize a chimera
  via e_b-conditional biology switching. Gradient still flows through the decoder
  into the prototype input. The live decoder under lambda_recon is untouched.
- `proto_recon_form='mix3'` new key (defaults 'agg' preserves old behavior);
  `proto_recon_stopgrad` existed in code but was NEVER in ABBREVIATIONS — now
  registered ('prsg'), plus 'prfm', 'prds'. (§9 both-files rule followed.)

## Arms (identical except ONE flag; same lambdas, so the flag is the only variable)

```
d61_pr3_sg   proto_recon_stopgrad=True    scores DETACHED: gradient -> PROTOTYPES
                                          only (through detached-param decoder)
d61_pr3_lv   proto_recon_stopgrad=False   scores LIVE: gradient also through s_ik
                                          -> ENCODER (reassign/move misfit cells)
```

Base = d60_leid_noc exactly (leiden init, no community, coh_proto_sg=1, GM-KL,
recon, freeze_batch_embedding, 10 epochs, K=300): its lambdas unchanged
(gram 2.904 / aff 5.063 / recon 0.02108 / gmkl 0.02228). lambda_proto_recon =
0.25 / median gnorm_enc from the D61 frozen full-pass probe
(measure_grad_shares_d61.py: mix3, dec_sg=1, scores LIVE at lambda=1; same lambda
deployed in BOTH arms by design — recorded choice: for arm _sg the encoder-path
gnorm is structurally 0, so share-matching per-arm would confound lambda with
routing; one shared lambda keeps the contrast clean).

## Decision before run (user's words, 2026-10-19)

"i expect that for each proto cells of different batches vote for same cell type -
and check if we will have batch specific prototypes or not. and also i expect the
cohesion thing still valid - i think we should go with leiden init."

Pre-registered bars, all vs d60_leid_noc and epoch-0 leiden on the SAME labels
pipeline (clusters_full.npz; scorer + proto_batch_labels view):

- P1 (chimera reduction — the headline): per-proto batch-majority AGREE fraction
  (protos with >=2 batches at >=3 cells) must beat d60_leid_noc's 46.8% and at
  least reach epoch-0 leiden's 65.1%. Cell-level cross indicators (cross_train_all,
  cross_test_all) must beat d60_leid_noc.
- P2 (cohesion still valid): coh_aff_test_all and coh_aff_test_rare still improve
  over epoch-0 leiden (the D60 result that must survive adding the new term).
- P3 (rare within-batch gains survive): within_test_rare >= epoch-0 leiden
  (D60-leid gained +0.057; losing it means the new term fights cohesion).
- M (batch-specific-prototypes check, a MEASUREMENT not a bar): count of
  multi-batch protos. If chimeras resolve by SPLITTING into batch-specific protos,
  multi-batch count drops hard (<< 255) with AGREE% up — a different outcome than
  resolving by reassignment (count stays ~255-295, AGREE% up). Which one happens
  decides whether the duplicate-merge term is the next mandatory step (split =
  mandatory) or optional cleanup.
- Arm contrast: if only _lv clears P1, the encoder/assignment path is the active
  ingredient (user's reassignment hypothesis); if _sg also clears it, moving
  prototypes alone suffices. If NEITHER clears P1, form-3 as dosed does not fix
  chimeras -> verdict fails, do not re-dose blindly; re-derive from the probe and
  the §9b value/ratio printout first.

Also recorded in-run: post-training sigma median (anti-collapse guard, D60 was
~0.90), n_dead, §9b-style side-by-side term values from grad_probe=25.

## Cost

Probe 1x L4 frozen pass (~$1) + 2 arms L4 x ~1.5h (~$2). Within budget.

## Verdict (TO FILL AFTER SCORING — works / fails / inconclusive-because)

- [ ] P1:
- [ ] P2:
- [ ] P3:
- [ ] M (split vs reassign):
- [ ] arm contrast:

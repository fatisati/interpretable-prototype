# D61 — form-3 proto_recon (the chimera charge): two arms, scores detached vs live

status: CLOSED, FAILS its headline bar (P1) — scored 2026-10-19, all six arms
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

## ADDENDUM 2026-10-19 (before any arm finished): user decision -> 2x2

User: "i want trainbale decode - trainble prototypes - scores trainbale or node."
Decision: decoder TRAINABLE inside the term too (her reasoning: the decoder must
be able to learn to reconstruct the rare states stage-2 is forming — the stage-1
decoder was population-weighted and may not express rare programs, making the
charge misfire for rares). Prototypes were already live in the term. The two
dec-detached arms were mid-run and are kept; two dec-live arms added:

```
                     scores DETACHED        scores LIVE
decoder DETACHED     d61_pr3_sg (running)   d61_pr3_lv (running)
decoder LIVE         d61_pr3_sgdec (new)    d61_pr3_lvdec (new)
```

Same lambda_proto_recon=0.004855 in all four (valid: decoder-param detachment
does not change the ENCODER-path gradient the probe measured). The dec axis now
tests, empirically, her hypothesis (live decoder helps rares: rare-proto
residuals drop AND cross-batch agreement rises) against the registered cheat
risk (decoder legalizes chimeras via e_b-conditional switching: mix3 loss falls
while cross-batch agreement stays low/falls). The per-proto batch-vote view
plus per-(proto,batch) residuals distinguish the two outcomes.

## ADDENDUM 2 (2026-10-19, still before any arm finished): the recon axis

User question: "do we need cvae loss any more at all? i feel now the decoder
shoud only be trained via the proto recon loss." The dec-live launch was STOPPED
before training to settle this first. During the discussion the user corrected
two of my claims, both verified in code:
- cohesion-aff: similarities are fixed graph weights — z-collapse buys nothing.
- cohesion-GRAM too: scproto.py:2717-2771 builds the gram from the AFFINITY
  rows (cos of closed-neighbourhood affinity vectors, normalized A@A^T 2-hop).
  NO live z in either kernel's similarity; z enters cohesion only via the
  assignment matrix. So: the only collapse-REWARDING term is the GM-KL pull,
  the only collapse-RESISTING term is recon; everything scored uses fixed
  data-side targets, so within-metacell z-contraction corrupts no metric.
  The one scored quantity that depends on the encoder staying a sane function
  of expression is HELD-OUT placement.
Precedent recorded: DEC (Xie, Girshick & Farhadi, ICML 2016) trains encoder
with clustering loss only; IDEC (Guo et al., IJCAI 2017) added recon back for
structure preservation. Unlike DEC we keep two fixed-target cohesion anchors.

DEPLOYED MATRIX (user: "run the 1*1 - make sure decoder is trainable for proto
recon loss") — all dec-live (proto_recon_dec_sg=0), protos live, leiden init:

```
                        scores DETACHED        scores LIVE
cvae recon ON           d61_pr3_sgdec          d61_pr3_lvdec
cvae recon OFF          d61_pr3_sgdec_nr       d61_pr3_lvdec_nr
```

recon-off = lambda_recon=0, all other lambdas unchanged (D58 ablation
convention). Predictions, pre-registered: USER — recon-off works; the decoder,
taught only by mix3, learns the rare-state programs; chimeras drop. CLAUDE —
recon-off shows within-metacell z-contraction (measure: within-cluster z
variance vs epoch-0, from latent_z_full.npy) and risks a wider train->test
purity gap (encoder carving without the per-cell information anchor). Both
readable from saved artifacts; no new instrumentation needed.
The two dec-DETACHED arms (d61_pr3_sg / d61_pr3_lv) completed/complete as
decoder-axis reference only; the headline bars (P1-P3, M) now bind the 2x2.

## Verdict (scored 2026-10-19; all six arms; scorer = score_d61_pr3.py,
## batch-vote = view_proto_batch_labels.py, dispersion = normalized-z
## within-metacell variance)

```
AGREE% (bar: >46.8 AND >=65.1)   sg 49.0  lv 56.4  sgdec 51.9  lvdec 55.2
                                 sgdec_nr 41.0  lvdec_nr 46.5     ALL FAIL
cross_train_all (d60_leid 0.8956): only lv 0.9083 / lvdec 0.9004 beat it
coh_aff_test_all (e0 0.0784, d60_leid 0.0853): every d61 arm <= 0.0759
within_test_rare (bar >= 0.5700): all pass; best lvdec_nr 0.6904
pooled_test_all  (d60_leid 0.8160): every d61 arm below (0.746-0.811)
n_dead: sg/sgdec 59/60, lv/lvdec 10/11, *_nr 3/0;  sigma 0.73-0.94 (healthy)
```

- [x] P1: **FAILS in every arm.** Best (lv 56.4%) beats d60_leid_noc's 46.8%
      but stays well under the epoch-0 bar of 65.1%. The chimera charge as
      dosed reduces but does not fix cross-batch misalignment.
- [x] P2: **FAILS.** Test cohesion drops below even epoch-0 in all six arms
      while TRAIN cohesion rises (+0.021..+0.047) — a train/test carving
      signature that appears exactly when mix3 is added. mix3's per-cell
      residual pressure pushes the encoder toward memorized boundaries.
- [x] P3: passes everywhere; rare within-batch purity best in lvdec_nr
      (+0.120 vs epoch-0, beats d60_leid_noc's +0.057).
- [x] M: recon-ON arms resolve chimeras partly by SPLITTING/killing
      (multi-batch protos 200-259 vs 295; n_dead up to 60). recon-OFF arms
      keep all 300 alive and multi-batch (295-297) but fix nothing. Per the
      pre-registration, split-dominant resolution makes the duplicate-MERGE
      term the next mandatory step.
- [x] scores axis: **LIVE wins decisively.** Scores-detached (proto-only
      teaching) kills 59-60 protos and craters rare metrics (rare_f1 0.61 vs
      0.78, pooled_test_rare 0.32-0.37 vs 0.49-0.54). The encoder/assignment
      path is the active ingredient — user's reassignment hypothesis
      confirmed on this axis.
- [x] recon axis: **collapse prediction REFUTED** — within-metacell z
      dispersion 0.100-0.122 vs epoch-0's 0.0866 (no contraction at 10
      epochs; recon-off slightly lower than recon-on but nowhere near
      collapse). User right on no-collapse. BUT recon-off arms are worst on
      AGREE% (41.0/46.5) and pooled_test_all (0.746/0.774), with one bright
      spot: lvdec_nr pooled_test_rare 0.5942 (+0.058 vs epoch-0, the only
      positive pooled-rare delta in D61). CVAE recon still pays for
      common/overall quality; question not closed but evidence says keep it.
- [x] dec axis: near-null. sg~sgdec and lv~lvdec on every headline number —
      decoder live vs detached barely matters at this dose/duration.

**OVERALL: fails.** d60_leid_noc still dominates every d61 arm on pooled/
cross/F1. Per the pre-registered rule: do NOT re-dose mix3 blindly; the
measured next step is the duplicate-merge (signature twin-pull) direction,
plus understanding P2's carving cost before any mix3 redeployment.

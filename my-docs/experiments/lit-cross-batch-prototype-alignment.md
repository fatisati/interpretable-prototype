# Lit survey: making prototypes show the SAME biology across batches

Collected 2026-10-19 (web-verified by a search agent; URLs are published versions)
to answer: is there a STANDARD solution for cross-batch cluster/prototype-level
alignment, especially for rare types? Context: the D60/D61 chimera + duplication
failure (see D61 ADR). Items below are the usable summary; verify a paper before
citing it in the manuscript.

## Cluster-level mechanisms (the relevant family)

- **Harmony** (Korsunsky et al., Nature Methods 2019,
  https://www.nature.com/articles/s41592-019-0619-0): soft k-means + an ADDITIVE
  diversity penalty (dependence between batch and cluster assignment), then
  per-cluster linear correction toward the global centroid. The diversity term is
  separable (theta=0 = plain soft k-means) and differentiable in the soft
  assignments. Fights DUPLICATION; overweighting is the canonical way to CREATE
  chimeras. Latent geometry only, no biology check. (We rejected batch-mixing
  pressure for rare types on principle — rare types may be batch-unique.)
- **scDML** (Yu et al., Nature Communications 2023,
  https://www.nature.com/articles/s41467-023-36635-5): over-cluster so rare types
  get own clusters -> cluster-pair similarity = count of inter-batch MNN pairs ->
  hierarchical MERGE by that vote -> triplet loss on merged labels. The published
  template for "merge duplicated prototypes by cross-batch evidence", validated on
  rare pancreas types. Merge decision is periodic/discrete, not differentiable.
- **CIDER** (Hu et al., Genome Biology 2021,
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02561-2):
  per-group WITHIN-BATCH differential-expression signature (limma, confounders
  regressed out), cross-batch similarity = consistency of signatures. The ONLY
  mechanism in this list using EXPRESSION-space evidence — the only one that can
  catch a chimera the latent itself cannot see. Post-hoc as published.
- **scPoli** (De Donno et al., Nature Methods 2023,
  https://www.nature.com/articles/s41592-023-02035-2): prototypes are SUPERVISED
  (labeled cell-type prototypes); batch robustness comes from CVAE conditioning,
  chimera/duplication prevented by the labels, not by any unsupervised term.
  Exactly the gap we must fill without labels.

## Cell-level (for contrast; already ruled out for rares at cell level)

MNN, Seurat anchors, SCALEX (batch-free encoder/batch-specific decoder), iMAP
(GAN on MNN pairs, explicitly tries to DETECT batch-specific cells rather than
force-mix), Concerto (argues against over-mixing), scDREAMER (adversarial batch
classifier; validated under skewed type composition), INSCT (batch-aware
triplets). All latent/graph geometry.

## Optimal transport

- **uniPort** (Cao et al., Nature Communications 2022): minibatch UNBALANCED OT
  loss between datasets' latents, trained into the encoder — proof that OT works
  as a differentiable loss. Unbalanced relaxation is the standard OT answer to
  "rare type present in one batch only" (also SCOTv2). moscot (Nature 2025) is
  post-hoc.

## The gap (usable for the paper's positioning)

Nothing published combines (a) UNSUPERVISED prototype learning, (b) an explicit
cross-batch prototype-identity term, (c) expression-space (DE) validation of
prototype identity, in one training objective. Harmony has (b) without persistent
prototypes; scDML has (b) as a discrete pre-step; scPoli has (a)+(b) only via
supervision; CIDER has (c) post-hoc only.

## How it maps onto our two failure modes

```
CHIMERA   (one proto, wrong biologies glued)  -> no published in-training charge;
          closest is CIDER's signature check (post-hoc). Our form-3 proto_recon
          (D61) is the in-training version: one latent vector = one biology,
          decoder as ruler.
DUPLICATE (one biology split across protos)   -> scDML's MNN-vote merge (discrete,
          periodic) and Harmony's diversity pressure (rejected: forced mixing).
          Our candidate: CIDER-style within-batch DE-signature twin-pull
          (periodic recompute, differentiable latent attraction), which stays
          silent for batch-unique rares by construction.
```

# Lit survey 2 (2026-10-19): duplicate-merge mechanisms, dead mixture components,
# held-out evaluation of clustering losses

Follow-up to lit-cross-batch-prototype-alignment.md, targeted at the three
measured D61 failures: (1) duplicates uncharged, (2) mixture-recon term causes
carving + dead protos, (3) model prefers splitting into batch-specific protos.
Web-verified by a search agent; flags preserved. Verify before citing.

## The headline pattern

**Every published method that merges duplicated clusters across batches while
protecting batch-unique populations separates the "should these merge" decision
from the gradient**: a DISCRETE, evidence-thresholded, cluster-level matching
step layered on a differentiable cell-level loss. None charges duplicates
inside a recon/ELBO term. The differentiable cross-domain prototype losses
(TPN CVPR 2019, ProDA CVPR 2021, SwAV NeurIPS 2020) all assume a fully shared
label space or force usage/mixing — none protects domain-unique clusters.

## Evidence-gated merge templates

- **scDML** (Yu et al., Nat Commun 14:960, 2023): over-cluster ON PURPOSE
  (Louvain res 3.0), then sim(i,j) = n_NN_pairs(i,j)/min(n_i,n_j) using
  intra-batch KNN + inter-batch MNN pairs; hierarchical merge to target K (or
  auto-K from eigenvalues of the sim matrix); merged labels then supervise a
  triplet loss. Batch-unique: implicit protection (no MNN evidence -> never
  merged). https://www.nature.com/articles/s41467-023-36635-5
- **BERMUDA** (Wang et al., Genome Biology 20:165, 2019): cluster each batch
  SEPARATELY, score cross-batch cluster pairs with MetaNeighbor AUROC, apply
  MMD transfer loss ONLY between pairs above threshold, weighted by sim.
  EXPLICIT headline claim: preserves cell types not shared by both batches.
  The closest template to an in-training gated twin-pull.
  https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1764-6
- **CIDER** (Genome Biology 22:337, 2021): within-batch Louvain -> limma DE
  signature per group (covariates incl. batch) -> sim = Pearson r of DE
  signatures -> merge by hierarchical agglomerative clustering, COMPLETE
  linkage (not dbscan). Fully post-hoc. Its EVALUATION mode splits integrated
  clusters into batch-specific subclusters and scores their cross-batch
  signature similarity -- i.e., the chimera detector.
- **Conos** (Nat Methods 16:695, 2019): joint cell graph from pairwise
  inter-sample MNN edges, community detection; unique populations form their
  own communities. Cell-level, no merge step.

## Dead mixture components (published remedies, if mix3 is ever revived)

- GMVAE (arXiv:1611.02648): "cluster degeneracy"; remedy = free-bits clamp on
  the categorical KL (differentiable minimum-usage floor).
- DEM-VAE (ICML 2020): mixture-ELBO mode collapse proven; dispersion term.
- DeepDPM (CVPR 2022): discrete split/merge moves during training, dynamic K.
- VQ codebook resets: Jukebox (arXiv:2005.00341) usage-triggered random
  restart; CVQ-VAE (ICCV 2023) dead codes pulled to running anchors.
  Our proto_reinit_every is the Jukebox-style move.
- VaDE itself relies on initialization only. FLAG: no VaDE/GMVAE paper found
  doing explicit dead-component reinit.

## Held-out (inductive) evaluation of clustering losses

- SCAN (ECCV 2020): the standard citation for inductive deep-clustering
  evaluation (test-set-only), noting prior work was transductive.
- DEC reported dropping ~84->75 train->held-out vs geometry-preserving methods
  (via GCML, arXiv:2009.09590). FLAG: numbers NOT verified against the PDF.
- **NO single-cell paper found reporting that adding a clustering loss hurt
  held-out generalization — our D61 P2 measurement appears to be an
  unreported, citable finding.**
- CarDEC (Genome Research 2021): batch-blind DEC loss; third-party benchmarks
  report it leaving batch-specific clusters and losing rare types (matches our
  failure 3). FLAG: full text inaccessible, inferred from abstract+secondary.

## Mapping to our failures

```
(1) duplicates        -> discrete evidence merge (scDML/CIDER/BERMUDA pattern)
(2) carving + deaths  -> known GMVAE degeneracy family; remedies are usage
                         floors/resets/split-merge -- NOT better recon weighting
(3) split-preference  -> not a failure under the over-cluster-then-merge
                         design; scDML over-clusters DELIBERATELY
```

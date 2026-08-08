"""
Helper functions for the "batch-correct-then-cluster" rebuttal baselines
(Rebuttal experiment E1: scPoli(Stage-1) / Harmony / scVI / BBKNN latents ->
{SEACells, Leiden}), moved out of batch_correct_then_cluster_baselines.ipynb
so the notebook only holds config + the per-dataset run/results cells.

Import this AFTER nb_setup.py has run (it needs the project root on sys.path
and expects `seacell_train` / `interpretable_ssl.*` to be importable).
"""

import os
import json
import pickle
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import anndata
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

import SEACells
import SEACells.core
import SEACells.build_graph

from interpretable_ssl.experiments.tasks import get_trainer
from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import (
    get_dataset_model_dir, get_seacell_model_dir, get_affinity_path,
)
from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
    compute_seacells_own_affinity, agg_obs, save_seacell,
)
from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import (
    add_scvi_emb, load_seacell,
)
from interpretable_ssl.evaluation.mc_metric_utils import (
    compute_task1_metrics, calc_task2_metrics, compute_modularity, calc_modularity_per_batch,
)
from seacell_train import eval_seacell_task1, eval_seacell_task2


AFF_K_NEIGHBORS = 50   # kNN for the adaptive-RBF affinity graph (matches SEACells' own default)
KNN_PURITY_K = 15      # neighbors used for the rare-cell embedding diagnostic


# --- environment patches (idempotent -- safe to import this module more than once) ---

def _apply_seacells_dtype_patch():
    """SEACells' own code (installed unpinned from GitHub main, --no-deps) calls
    `AnnData(..., dtype=...)` internally in a few places -- `dtype` was a valid
    AnnData.__init__ kwarg in old anndata, removed in modern anndata. Without this,
    `SEACells.core.summarize_by_SEACell` (called from both
    compute_seacells_own_affinity and run_leiden_on_latent's Task 2 aggregation)
    raises `TypeError: AnnData.__init__() got an unexpected keyword argument 'dtype'`.

    Patch: make AnnData.__init__ silently drop a stray dtype= kwarg instead of
    raising -- harmless, since dropping it just means anndata infers dtype from the
    data (normal behavior). Guarded so importing/reloading this module never stacks
    the wrapper more than once (stacking enough times blows Python's recursion limit).
    """
    if getattr(anndata.AnnData.__init__, '_dtype_patch_applied', False):
        return
    _orig_anndata_init = anndata.AnnData.__init__

    def _patched_anndata_init(self, *args, **kwargs):
        kwargs.pop('dtype', None)
        _orig_anndata_init(self, *args, **kwargs)

    _patched_anndata_init._dtype_patch_applied = True
    anndata.AnnData.__init__ = _patched_anndata_init


def _apply_cuda_fallback_patch():
    """SCProtoTrainer.build_model() (scproto.py) unconditionally calls
    self.model.cuda(), with no CPU fallback -- raises
    'AssertionError: Torch not compiled with CUDA enabled' on any CPU-only runtime.
    get_stage1_latent only needs this model for a cheap forward pass (encoding an
    existing checkpoint, not training), so CPU is fine here -- it just needs to not
    crash. Makes nn.Module.cuda() fall back to .to('cpu') when no GPU is available,
    instead of raising. Only changes behavior when torch.cuda.is_available() is
    False, where the original call would have crashed anyway -- never affects a real
    GPU runtime. Guarded so importing/reloading this module never stacks the wrapper.
    """
    if getattr(torch.nn.Module.cuda, '_cpu_fallback_patch_applied', False):
        return
    _orig_cuda = torch.nn.Module.cuda

    def _patched_cuda(self, device=None):
        if not torch.cuda.is_available():
            return self.to('cpu')
        return _orig_cuda(self, device)

    _patched_cuda._cpu_fallback_patch_applied = True
    torch.nn.Module.cuda = _patched_cuda


def _apply_pandas_string_patch():
    """Some pandas/anndata version combinations auto-infer a newer Arrow-backed
    'str' extension dtype even from plain object-dtype input -- disable that
    inference where possible so pd.Index(..., dtype=object) in _fix_arrow_strings
    below actually sticks instead of being silently re-upgraded.
    """
    for opt in ('future.infer_string',):
        try:
            pd.set_option(opt, False)
        except Exception:
            pass
    try:
        pd.options.mode.string_storage = 'python'
    except Exception:
        pass


_apply_seacells_dtype_patch()
_apply_pandas_string_patch()
_apply_cuda_fallback_patch()


def get_stage1_latent(ds_id, cvae_epochs=50, batch_size=1024):
    """Load the existing Stage-1 (scPoli pretrain) checkpoint for ds_id and encode
    every cell with it -- no Stage-2 (prototype/community) training happens here.

    Returns:
        t:  the SCProtoTrainer, with t.model holding ONLY Stage-1 weights and
            t.train_ds.adata the preprocessed AnnData used for pretraining.
        ad: t.train_ds.adata (kept as a separate name for clarity below).
        z1: (N, d) numpy array -- the Stage-1-only latent for every cell in ad,
            in the same row order as ad.
    """
    t = get_trainer(
        experiment_name='stage1_latent_extract',
        cvae_epochs=cvae_epochs,
        dataset_id=ds_id,
        l2norm=1,
        assignment_metric='dotp',
        batch_size=batch_size,
        affinity_type='arbf',
    )
    t.load_pretrain_checkpoint()  # raises FileNotFoundError if not pretrained yet

    ad = t.train_ds.adata
    with torch.no_grad():
        z1 = t.encode_adata(ad, t.model, z_idx=1).cpu().numpy()

    print(f"[{ds_id}] Stage-1 latent: {z1.shape[0]} cells x {z1.shape[1]} dims")
    return t, ad, z1


def get_preprocessed_adata(ds_id):
    """Load ds_id's AnnData exactly as scProto's own pipeline does (read_adata():
    HVG subsetting, batch key, lognorm layer) WITHOUT building or running any model.

    get_stage1_latent goes through a full SCProtoTrainer (checkpoint + model +
    a .cuda() call) purely to reach `t.train_ds.adata` -- needed only by the
    'stage1z' correction method, which actually uses the Stage-1 z1 latent. Every
    other method here (combat, bbknn) never touches z1 at all, so that whole
    trainer/model load was pure overhead riding along for no reason.

    use_counts=False matches Trainer.get_dataset's own resolution
    (use_counts=self.recon_loss=="nb"; recon_loss defaults to 'mse' for every
    dataset used in these baselines) -- ad.X stays log-normalized, same as what
    get_stage1_latent's ad.X already was.
    """
    from interpretable_ssl.datasets.dataset import SingleCellDataset
    return SingleCellDataset(name=ds_id, use_counts=False, **DATASETS[ds_id]).adata


def get_harmony_embedding(ad, bk):
    """PCA + Harmony batch correction. Calls harmonypy directly (pinned to 0.0.9,
    the version scanpy's own harmony_integrate wrapper assumes) instead of going
    through embedding_metrics.add_pca_harmoney / scanpy.external.pp.harmony_integrate --
    those hardcode `harmony_out.Z_corr.T`, which silently breaks (wrong shape, not an
    error) if a newer harmonypy changes what .Z_corr holds. Checking + fixing
    orientation explicitly here fails loudly instead of cascading into an opaque
    anndata ValueError three layers down.
    """
    import harmonypy

    if 'X_pca' not in ad.obsm:
        sc.pp.pca(ad, n_comps=50)

    ho = harmonypy.run_harmony(ad.obsm['X_pca'], ad.obs, bk)
    Z = np.asarray(ho.Z_corr)
    if Z.ndim != 2:
        raise ValueError(f"harmonypy returned Z_corr with ndim={Z.ndim}, expected 2 -- "
                          f"check the installed harmonypy version's API (pinned to 0.0.9).")
    if Z.shape[0] != ad.n_obs and Z.shape[1] == ad.n_obs:
        Z = Z.T
    if Z.shape[0] != ad.n_obs:
        raise ValueError(f"harmony output shape {Z.shape} doesn't match n_obs={ad.n_obs} "
                          f"on either axis.")
    return Z


def get_harmony_embedding_matched_dim(ad, bk, n_comps, ds_id=None, use_cache=True, nclust=None):
    """Same correction as get_harmony_embedding, but PCA'd to `n_comps` first instead
    of the usual 50 -- for a dimensionality-MATCHED comparison against a lower-dim
    embedding (e.g. scProto's 8-dim latent). Rationale: rare_type_affinity_ratio_per_batch
    showed Harmony's (50-dim) embedding beating scProto's (8-dim) latent on same-type
    affinity mass for rare cells on 2/3 datasets -- before concluding that's a real
    mechanism difference, need to rule out that it's just "8 dims of cell-type variation
    forced to share a smaller space is inherently more crowded than 50 dims," independent
    of whether the correction/prototype mechanism itself is doing anything differently.

    nclust: optional override for harmonypy's OWN internal soft-clustering resolution
    (its `nclust` kwarg -- distinct from the downstream SEACells/Leiden target K, which
    is already matched to scProto elsewhere). Left unset, harmonypy defaults to
    min(N/30, 100), which caps at 100 for all three RNA-seq datasets here -- 2-3x
    coarser than scProto's own prototype count (220-300). Pass nclust=K (the dataset's
    num_prototypes) to test whether that default cap was itself handicapping (or
    helping -- direction isn't obvious a priori, see module docstring discussion)
    Harmony's rare-cell handling, independent of the embedding-dimension question above.

    Computes its own PCA at n_comps (via sklearn directly on ad.X, not sc.pp.pca --
    avoids touching/overwriting ad.obsm['X_pca'], which everything else in this module
    assumes is the 50-comp version) and caches it under a dedicated obsm key so repeat
    calls at the same n_comps don't recompute within one session.

    If ds_id is given (and use_cache, the default), also persists the corrected
    embedding to disk via _cache_paths(ds_id, tag) where tag encodes BOTH n_comps and
    the RESOLVED nclust -- always, not just when explicitly overridden -- e.g.
    'harmony_d8_k100' (left at harmonypy's own default) vs. 'harmony_d8_k220'
    (explicitly matched). Without this, a default-nclust run and a matched-nclust run
    would look identical from the filename alone (only 'harmony_d8', no way to tell
    which cap was actually used) -- resolving and naming it explicitly either way
    means nothing here is ever ambiguous on disk, without re-reading this code or any
    notebook. Pass use_cache=False to force a fresh recompute regardless of what's
    already cached (e.g. while still validating this code path).

    Note: when nclust is left None, the resolved value used for the FILENAME is
    computed here with the same formula harmonypy uses internally
    (min(round(N/30), 100), per harmonypy.harmony.run_harmony) rather than read back
    off harmonypy's own result -- if a future harmonypy version changes that default
    formula, the filename could drift out of sync with what was actually run. Low
    risk, but worth knowing if a cached '_k100' file's contents ever look surprising.
    """
    import harmonypy
    from sklearn.decomposition import PCA as _SKPCA

    resolved_nclust = nclust if nclust is not None else int(min(round(ad.n_obs / 30.0), 100))
    tag = f'harmony_d{n_comps}_k{resolved_nclust}'

    if ds_id is not None and use_cache:
        emb_path, _ = _cache_paths(ds_id, tag)
        if os.path.exists(emb_path):
            print(f"[{ds_id}] {tag}: reusing cached embedding at {emb_path}")
            return np.load(emb_path)

    pca_key = f'_X_pca_d{n_comps}'
    if pca_key not in ad.obsm:
        X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
        ad.obsm[pca_key] = _SKPCA(n_components=n_comps, random_state=0).fit_transform(X)

    harmony_kwargs = {} if nclust is None else {'nclust': nclust}
    ho = harmonypy.run_harmony(ad.obsm[pca_key], ad.obs, bk, **harmony_kwargs)
    Z = np.asarray(ho.Z_corr)
    if Z.ndim != 2:
        raise ValueError(f"harmonypy returned Z_corr with ndim={Z.ndim}, expected 2.")
    if Z.shape[0] != ad.n_obs and Z.shape[1] == ad.n_obs:
        Z = Z.T
    if Z.shape[0] != ad.n_obs:
        raise ValueError(f"harmony output shape {Z.shape} doesn't match n_obs={ad.n_obs} "
                          f"on either axis.")

    if ds_id is not None and use_cache:
        emb_path, _ = _cache_paths(ds_id, tag)
        np.save(emb_path, Z)
        print(f"[{ds_id}] {tag}: cached embedding to {emb_path}")

    return Z


def get_combat_corrected_pca(ad, bk, n_comps=50, ds_id=None, use_cache=True):
    """Original ComBat (Johnson, Li & Rabinovic 2007) batch correction -- via scanpy's
    built-in sc.pp.combat, no extra package needed -- run on log-normalized expression
    (ad.layers['lognorm']), followed by PCA on the corrected matrix to get an embedding.

    Unlike Harmony (which corrects an existing PCA embedding) or scVI/scPoli (which
    learn a corrected latent directly), ComBat corrects the full gene-expression matrix
    itself; PCA here happens AFTER correction, on the corrected matrix, not before.

    n_comps matters a lot: comparing scProto's low-dim latent (e.g. 8) against a
    ComBat embedding PCA'd to the default 50 isn't apples-to-apples -- lower
    dimensionality forces cell-type variation into a more crowded space, independent
    of whatever correction mechanism is actually being compared (the same reasoning
    that motivated get_harmony_embedding_matched_dim). Callers should pass scProto's
    own latent dimension here for a fair comparison -- see run_correction_method's
    matched_n_comps.

    Runs on a fresh AnnData built from ad.layers['lognorm'] (never ad.X, which holds raw
    counts for scProto/scVI training) so the shared `ad` object other baselines use is
    never mutated by this call.

    Cached under a dimension-qualified tag ('combat_d{n_comps}', via _cache_paths) --
    e.g. 'combat_d8' vs 'combat_d50' never collide or silently reuse each other's
    cache, the same reasoning get_harmony_embedding_matched_dim's tag naming uses.
    """
    tag = f'combat_d{n_comps}'

    if ds_id is not None and use_cache:
        emb_path, _ = _cache_paths(ds_id, tag)
        if os.path.exists(emb_path):
            print(f"[{ds_id}] {tag}: reusing cached embedding at {emb_path}")
            return np.load(emb_path)

    if 'lognorm' not in ad.layers:
        raise ValueError(
            "ad.layers['lognorm'] not found -- ComBat needs log-normalized expression, "
            "not raw counts. Check the dataset's preprocessing (dataset.py's read_adata "
            "always populates this layer, so its absence here means `ad` came from "
            "somewhere unexpected)."
        )

    ad_combat = anndata.AnnData(
        X=ad.layers['lognorm'].copy(),
        obs=ad.obs.copy(),
        var=ad.var.copy(),
    )
    sc.pp.combat(ad_combat, key=bk)
    sc.pp.pca(ad_combat, n_comps=n_comps)
    Z = np.asarray(ad_combat.obsm['X_pca'])

    if ds_id is not None and use_cache:
        emb_path, _ = _cache_paths(ds_id, tag)
        np.save(emb_path, Z)
        print(f"[{ds_id}] {tag}: cached embedding to {emb_path}")

    return Z


def get_scvi_embedding(ad, ds_id, bk):
    """Trains a fresh scVI model (via embedding_metrics.add_scvi_emb: reference/query
    split using DATASETS[ds_id]['test_studies'] if present, else trains on everything).
    NOTE: this is real model training (slower than the other methods, not a lookup).
    Runs on ad.copy(): add_scvi_emb overwrites adata.X with raw counts internally, and
    we don't want that touching the shared `ad` other baselines on this dataset use.

    Uses scvi-tools' own default gene_likelihood='zinb' -- NOT loss-matched to
    scProto's Stage-1 pretrain, which uses recon_loss='mse' on log-normalized
    expression (see interpretable_ssl/configs/defaults.py). See
    get_scvi_gauss_embedding below for the loss-matched variant.
    """
    query_studies = DATASETS[ds_id].get('test_studies', [])
    z_scvi, _key = add_scvi_emb(ad.copy(), query_studies, bk)
    return z_scvi


def get_scvi_gauss_embedding(ad, ds_id, bk):
    """Same training recipe as get_scvi_embedding (same reference/query split, same
    epoch budget, same n_latent=8) but with gene_likelihood='normal' instead of
    scvi-tools' ZINB default -- a Gaussian reconstruction likelihood (Normal NLL
    with learned per-gene variance) in place of the zero-inflated negative-binomial
    count likelihood.

    Why this exists: scProto's own Stage-1 (scPoli) pretrain uses recon_loss='mse'
    on log-normalized expression (defaults.py), while stock scVI reconstructs raw
    counts under a ZINB likelihood -- two different noise-model families, not just
    two different architectures, which confounds "is scVI's mechanism better" with
    "is ZINB just a better-suited loss for scRNA-seq counts than MSE." Gaussian NLL
    reduces to (learned-variance-weighted) MSE, and removes the biggest piece of
    that mismatch -- the NB/zero-inflation shape -- while keeping everything else
    about scVI's real, unmodified architecture (encoder, batch-conditioned decoder,
    n_latent=8) identical to the ZINB run above.

    NOT a perfect match to scProto's plain MSE-on-lognorm even so: the decoder
    mean is still `library_size * softmax(...)` -- library-scaled and constrained
    to a per-cell simplex -- rather than an unconstrained prediction directly
    against log-normalized targets (stock scVI has no supported way to bypass its
    library-size machinery; see add_scvi_emb's docstring). Report this as "scVI
    (Gaussian)", not "scVI (MSE)" -- it is a meaningfully closer, not identical,
    comparison point. `stage1z` (scProto's own actual MSE-on-lognorm Stage-1
    encoder, already computed elsewhere in this module) remains the more literal
    MSE control.

    Runs on ad.copy() for the same reason as get_scvi_embedding: add_scvi_emb
    overwrites adata.X with raw counts internally.
    """
    query_studies = DATASETS[ds_id].get('test_studies', [])
    z_scvi_gauss, _key = add_scvi_emb(ad.copy(), query_studies, bk, gene_likelihood='normal')
    return z_scvi_gauss


def get_bbknn_graph(ad, bk, n_comps=None, ds_id=None, use_cache=True):
    """BBKNN produces a batch-balanced kNN graph directly -- unlike the embedding-based
    methods there's no separate 'build an RBF kernel on top of an embedding' step: this
    graph itself is the affinity fed to SEACells + Leiden below. Runs on a copy so
    bbknn's neighbor-graph side effects don't touch the shared `ad`.

    n_comps: if given (e.g. scProto's own latent dimension), computes a DEDICATED PCA
    at this dimension instead of the shared 50-dim ad.obsm['X_pca'] -- same
    dimension-matching reasoning as get_harmony_embedding_matched_dim /
    get_combat_corrected_pca: comparing scProto's low-dim latent against a graph
    built on a 50-dim PCA isn't apples-to-apples. None falls back to the plain
    50-dim path (the original default).

    Cached under a dimension-qualified tag ('bbknn_d{n_comps}') -- unlike the
    embedding-based methods, BBKNN has no separate embedding to cache, so this
    caches the resulting GRAPH directly.
    """
    resolved_n_comps = n_comps if n_comps is not None else 50
    tag = f'bbknn_d{resolved_n_comps}'

    if ds_id is not None and use_cache:
        _, aff_path = _cache_paths(ds_id, tag)
        if os.path.exists(aff_path):
            print(f"[{ds_id}] {tag}: reusing cached graph at {aff_path}")
            return sp.load_npz(aff_path)

    pca_key = f'_X_pca_d{resolved_n_comps}' if n_comps is not None else 'X_pca'
    if pca_key not in ad.obsm:
        if n_comps is not None:
            from sklearn.decomposition import PCA as _SKPCA
            X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
            ad.obsm[pca_key] = _SKPCA(n_components=resolved_n_comps, random_state=0).fit_transform(X)
        else:
            sc.pp.pca(ad, n_comps=50)

    ad_bb = ad.copy()
    sce.pp.bbknn(ad_bb, batch_key=bk, use_rep=pca_key)
    aff = sp.csr_matrix(ad_bb.obsp['connectivities'])
    aff = (aff + aff.T) / 2

    if ds_id is not None and use_cache:
        _, aff_path = _cache_paths(ds_id, tag)
        sp.save_npz(aff_path, sp.csr_matrix(aff))
        print(f"[{ds_id}] {tag}: cached graph to {aff_path}")

    return aff


def build_latent_affinity(ad, latent_key, k=AFF_K_NEIGHBORS):
    """Adaptive-RBF affinity graph on ad.obsm[latent_key], same construction SEACells
    itself uses on X_pca (interpretable_ssl.augmenters.graph_generator's 'arbf' affinity_type
    calls exactly this: SEACellGraph(ad, "X_pca", ...).rbf(k, graph_construction="union")) --
    here pointed at any corrected embedding instead of PCA, and built ONCE per method so
    both baselines below (SEACells + Leiden) see the identical graph.
    """
    kernel_model = SEACells.build_graph.SEACellGraph(ad, latent_key, verbose=True)
    aff = kernel_model.rbf(k, graph_construction='union')
    aff = sp.csr_matrix(aff)
    return (aff + aff.T) / 2


def rare_type_knn_purity(X, labels, k=KNN_PURITY_K, rare_quantile=0.25):
    """For each RARE-type cell (bottom-quartile global frequency, matching
    mc_metric_utils.get_rare's default), the fraction of its k nearest neighbors in
    embedding X that share its label -- averaged. High = rare-type cells still form
    locally-coherent neighborhoods in X; low = they've been absorbed into other types'
    neighborhoods.

    Embedding-only diagnostic -- no clustering involved -- isolates whether the
    correction method ALREADY destroys rare-type separability, independent of whichever
    clustering algorithm runs downstream.
    """
    freq = pd.Series(labels).value_counts(normalize=True)
    thresh = freq.quantile(rare_quantile)
    rare_types = freq[freq < thresh].index.tolist()
    if not rare_types:
        return {'mean_purity': None, 'n_rare_cells': 0}

    labels = np.asarray(labels)
    rare_mask = np.isin(labels, rare_types)

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X[rare_mask])
    idx = idx[:, 1:]  # drop self

    neighbor_labels = labels[idx]
    own_labels = labels[rare_mask][:, None]
    purity_per_cell = (neighbor_labels == own_labels).mean(axis=1)

    return {'mean_purity': float(purity_per_cell.mean()), 'n_rare_cells': int(rare_mask.sum())}


def rare_type_affinity_ratio_per_batch(ad, latent_key, label_key, batch_key,
                                        k=AFF_K_NEIGHBORS, rare_quantile=0.25):
    """Embedding-only, clustering-free counterpart to the F1/homogeneity rare-cell
    table: builds one ARBF affinity graph on ad.obsm[latent_key] (same construction as
    build_latent_affinity / SEACells' own kernel), then for each locally-rare-type
    cell scores what FRACTION OF ITS TOTAL AFFINITY MASS (not just top-k kNN
    membership, unlike rare_type_knn_purity above) goes to cells of its own type.
    No SEACells/Leiden step involved -- isolates the embedding from whichever
    downstream clustering algorithm would otherwise run on top of it.

    Locally-rare definition (per-batch outlier rule, Q25 fallback) matches
    paper_figures.py's rare_celltype_purity_table exactly, so results here are
    directly comparable/pairable against the F1/homogeneity significance test.

    Returns: list of per-batch scores (one float per batch with >=1 locally-rare
    type) -- caller aggregates/tests as needed (e.g. paired Wilcoxon vs scProto).
    """
    aff = sp.csr_matrix(build_latent_affinity(ad, latent_key=latent_key, k=k))
    n = aff.shape[0]
    row_sum = np.asarray(aff.sum(axis=1)).ravel()

    labels = ad.obs[label_key].astype(str).values
    label_cat = pd.Categorical(labels)
    onehot = sp.csr_matrix(
        (np.ones(n), (np.arange(n), label_cat.codes)),
        shape=(n, len(label_cat.categories)),
    )
    mass_by_type = aff @ onehot  # (n, n_types): each cell's total affinity mass per type
    same_type_mass = np.asarray(mass_by_type[np.arange(n), label_cat.codes]).ravel()
    ratio = np.divide(same_type_mass, row_sum, out=np.zeros(n), where=row_sum > 0)

    batches = ad.obs[batch_key].values
    per_batch_scores = []
    for b in pd.unique(batches):
        bmask = batches == b
        batch_labels = labels[bmask]
        freq = pd.Series(batch_labels).value_counts(normalize=True)
        thresh = float(freq.mean() - freq.std())
        rare_types = set(freq[freq < thresh].index)
        if not rare_types:
            rare_types = set(freq[freq < freq.quantile(rare_quantile)].index)
        if not rare_types:
            continue
        rare_mask = bmask & np.isin(labels, list(rare_types))
        if rare_mask.sum() == 0:
            continue
        df = pd.DataFrame({'label': labels[rare_mask], 'ratio': ratio[rare_mask]})
        per_type_mean = df.groupby('label')['ratio'].mean()
        per_batch_scores.append(float(per_type_mean.mean()))

    return per_batch_scores


def leiden_resolution_search(adjacency, target_k, seed=42, start_res=1.0, max_res=512.0):
    """Get EXACTLY target_k Leiden clusters via over-segment + greedy modularity-gain
    merge-down. Leiden has no native "give me K clusters" parameter -- resolution is
    an indirect knob, and plain bisection on it can land far from target_k if the
    graph's cluster-count-vs-resolution curve is coarse near that value (this is what
    caused an earlier run to land at ~70 clusters when the target was 220).

    Step 1 (over-segment): push resolution up until the partition has >= target_k
    communities -- always achievable, since higher resolution never produces fewer
    clusters in practice.
    Step 2 (merge-down): repeatedly merge whichever pair of CURRENT clusters loses the
    LEAST modularity when combined (the standard greedy agglomerative rule -- the same
    principle Louvain/Leiden's own multilevel algorithm uses internally to build its
    hierarchy), until exactly target_k clusters remain. This guarantees exact K while
    staying graph-native, not an arbitrary cut.

    Returns:
        (labels, resolution_used_for_oversegmentation, n_clusters) -- n_clusters ==
        target_k unless over-segmentation itself couldn't reach target_k even at
        max_res, in which case it's capped there (printed as a warning).
    """
    N = adjacency.shape[0]
    A = sp.csr_matrix(adjacency)
    A = (A + A.T) / 2
    degrees = np.asarray(A.sum(axis=1)).ravel()
    two_m = float(degrees.sum())

    # --- Step 1: over-segment ---
    tmp = anndata.AnnData(np.zeros((N, 1), dtype=np.float32))
    res = start_res
    labels = None
    while True:
        sc.tl.leiden(tmp, adjacency=A, resolution=res, key_added='leiden',
                     random_state=seed, flavor='leidenalg')
        labels = tmp.obs['leiden'].astype(int).values
        n_k = len(np.unique(labels))
        print(f"  [leiden oversegment] resolution={res:.4f} -> {n_k} clusters (need >= {target_k})")
        if n_k >= target_k or res >= max_res:
            break
        res *= 2

    uniq = np.unique(labels)
    remap = {old: i for i, old in enumerate(uniq)}
    labels = np.array([remap[l] for l in labels])
    n_current = len(uniq)

    if n_current < target_k:
        print(f"  [leiden oversegment] WARNING: capped at {n_current} clusters (resolution "
              f"limit {max_res}) -- below target {target_k}; returning as-is, no merge-down.")
        return labels, res, n_current

    # --- Step 2: greedy modularity-gain merge-down to exactly target_k ---
    while n_current > target_k:
        rows = np.arange(N)
        M = sp.csr_matrix((np.ones(N), (rows, labels)), shape=(N, n_current))
        C = (M.T @ A @ M).toarray()
        np.fill_diagonal(C, 0.0)
        cluster_deg = np.asarray(M.T @ degrees).ravel()

        i_idx, j_idx = np.where(np.triu(C, k=1) > 0)
        if len(i_idx) == 0:
            order = np.argsort(cluster_deg)
            bi, bj = order[0], order[1]
        else:
            dQ = 2 * (C[i_idx, j_idx] / two_m
                      - (cluster_deg[i_idx] / two_m) * (cluster_deg[j_idx] / two_m))
            best = np.argmax(dQ)
            bi, bj = i_idx[best], j_idx[best]

        labels[labels == bj] = bi
        uniq = np.unique(labels)
        remap = {old: k for k, old in enumerate(uniq)}
        labels = np.array([remap[l] for l in labels])
        n_current = len(uniq)
        if n_current % 10 == 0 or n_current <= target_k + 5:
            print(f"  [leiden merge] -> {n_current} clusters (target {target_k})")

    print(f"[leiden merge-down] final: {n_current} clusters (target {target_k})")
    return labels, res, n_current


def save_baseline_umap_data(ds_id, ad, mc_key, save_path, lk, bk, nk=None):
    """Write umap_cells.csv / umap_protos.csv / cell_assignments.csv in the same
    format seacell_train.py's _save_seacell_umap_data() produces for SEACells, so
    rare_celltype_purity_table() / paper_figures helpers can find this run too.
    Reuses ad.obsm['X_pca'] (computed once per dataset) for the joint cell+metacell
    UMAP, exactly like _save_seacell_umap_data does.
    """
    if 'X_pca' not in ad.obsm:
        sc.pp.pca(ad, n_comps=50)

    mc_idx = ad.obs[mc_key].astype(int).values
    unique_mcs = np.unique(mc_idx)
    n_mc = len(unique_mcs)
    mc_pos = {m: i for i, m in enumerate(unique_mcs)}
    cell_pca = ad.obsm['X_pca']

    mc_pca = np.zeros((n_mc, cell_pca.shape[1]))
    for m in unique_mcs:
        mask = mc_idx == m
        mc_pca[mc_pos[m]] = cell_pca[mask].mean(axis=0)

    combined = np.vstack([cell_pca, mc_pca])
    tmp_ad = anndata.AnnData(combined)
    sc.pp.neighbors(tmp_ad, use_rep='X', n_neighbors=15, metric='cosine', random_state=42)
    sc.tl.umap(tmp_ad, random_state=42)
    z_umap = tmp_ad.obsm['X_umap'][:len(ad)]
    proto_umap = tmp_ad.obsm['X_umap'][len(ad):]

    cells_df = pd.DataFrame({
        'cell_id': ad.obs_names, 'umap_1': z_umap[:, 0], 'umap_2': z_umap[:, 1],
        'metacell_id': mc_idx,
    })
    for col in [lk, bk, nk]:
        if col and col in ad.obs.columns:
            cells_df[col] = ad.obs[col].values
    cells_df.to_csv(os.path.join(save_path, 'umap_cells.csv'), index=False)
    cells_df[['cell_id', 'metacell_id'] + [c for c in [lk, bk, nk] if c]].to_csv(
        os.path.join(save_path, 'cell_assignments.csv'), index=False)

    lk_vals = ad.obs[lk].values if lk in ad.obs.columns else None
    proto_rows = []
    for m in unique_mcs:
        mask = mc_idx == m
        n = int(mask.sum())
        row = {'proto_id': int(m), 'umap_1': proto_umap[mc_pos[m], 0],
               'umap_2': proto_umap[mc_pos[m], 1], 'n_cells': n}
        if lk_vals is not None and n > 0:
            row[f'majority_{lk}'] = Counter(lk_vals[mask]).most_common(1)[0][0]
        proto_rows.append(row)
    pd.DataFrame(proto_rows).to_csv(os.path.join(save_path, 'umap_protos.csv'), index=False)

    print(f"[{mc_key}] umap_cells.csv / umap_protos.csv saved to {save_path}")


def _fix_arrow_strings(ad):
    """Force any Arrow-backed string columns/index to plain numpy object-dtype
    strings, so anndata==0.10.6's h5ad writer (predates Arrow-string support --
    IORegistryError otherwise) can serialize them.

    Passing dtype=object EXPLICITLY to the pd.Index/pd.array constructor (not just
    handing it an already-object-dtype numpy array) is required -- some pandas
    versions auto-infer a newer 'str' extension dtype (still Arrow-backed internally)
    even from object-dtype input unless dtype=object is forced.
    """
    for df in (ad.obs, ad.var):
        idx_vals = np.array([str(x) for x in df.index], dtype=object)
        df.index = pd.Index(idx_vals, dtype=object, name=df.index.name)
        for col in df.columns:
            if 'Arrow' in type(df[col].array).__name__ or 'string' in str(df[col].dtype).lower() or str(df[col].dtype) == 'str':
                col_vals = np.array([str(x) for x in df[col]], dtype=object)
                df[col] = pd.array(col_vals, dtype=object)
    return ad


def verify_arrow_string_fix():
    """Sanity check that _fix_arrow_strings actually solves the h5ad write bug --
    tiny dummy AnnData, no real data/models touched. Returns True if the fix works,
    raises AssertionError otherwise.
    """
    test_ad = anndata.AnnData(
        X=np.random.rand(10, 5).astype('float32'),
        obs=pd.DataFrame(index=[f'cell_{i}' for i in range(10)]),
    )
    test_ad.obs.index = pd.Index(test_ad.obs.index.astype('string[pyarrow]'))
    test_ad.obs['dummy_col'] = pd.array([f'val_{i}' for i in range(10)], dtype='string[pyarrow]')

    _fix_arrow_strings(test_ad)

    test_path = os.path.join(tempfile.gettempdir(), 'arrow_string_fix_test.h5ad')
    try:
        test_ad.write(test_path)
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)
    print("verify_arrow_string_fix: OK -- _fix_arrow_strings resolves the Arrow-string h5ad write bug.")
    return True


def _load_metrics(save_path):
    p = os.path.join(save_path, 'metrics.json')
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _find_existing_leiden_dir(ds_id, tag, n_target_clusters):
    """Look up the exact 'leiden_{tag}_K{n_target_clusters}' folder -- NOT a
    'leiden_{tag}_K*' prefix scan. A prefix scan would happily reuse a stale run left
    over from an earlier num_prototypes value (e.g. 'leiden_X_harmony_K119' sitting
    next to what should now be a K300 target) as if it were a valid skip_if_exists
    cache hit, silently scoring the wrong K against scProto. Returns None (a cache
    miss, triggering a fresh run at the correct K) if no folder at exactly this K
    exists yet, even if a different-K folder for the same (ds_id, tag) does.
    """
    base_dir = get_dataset_model_dir(ds_id)
    run_dir = os.path.join(base_dir, f'leiden_{tag}_K{n_target_clusters}')
    if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, 'metrics.json')):
        return run_dir
    return None


def get_realized_seacell_count(ds_id, tag):
    """Realized metacell count for the on-disk 'seacell_{tag}' run for ds_id, read
    directly from its saved cell_assignments.csv (metacell_id column) -- a read-only
    check, no archetypal analysis re-run. Does NOT read metrics.json's K_target field:
    recompute_modularity_canonical unconditionally overwrites that field to whatever
    the CURRENT caller believes the target is, on every skip_if_exists cache hit, even
    for a stale run computed under an old num_prototypes value -- so it can't be
    trusted to reflect what K the run actually used.

    Returns the realized count (int), or None if the run or that CSV don't exist yet.
    """
    save_path = get_seacell_model_dir(ds_id, tag)
    assign_path = os.path.join(save_path, 'cell_assignments.csv')
    if not os.path.exists(assign_path):
        return None
    try:
        return int(pd.read_csv(assign_path)['metacell_id'].nunique())
    except Exception:
        return None


def _cached_seacell_run_matches_k(save_path, expected_k, tolerance=0.05):
    """SEACells folders don't encode K in their name (unlike 'leiden_{tag}_K{n}') --
    so a skip_if_exists check on 'does metrics.json exist' alone can't distinguish a
    fresh run at the CURRENT target K from a stale one computed under an OLD
    num_prototypes value (this is the exact bug found 2026-07-30: pbmc-immune's
    leiden_X_harmony was silently reusing a K=119 run left over from before
    num_prototypes was changed to 300 -- the SEACells path has the identical
    vulnerability, just without a folder-name tell).

    Cross-checks against the actual realized metacell count in cell_assignments.csv
    (see get_realized_seacell_count). Returns True (safe to skip) only if that count
    is within `tolerance` of expected_k; False (treat as cache miss, recompute fresh)
    otherwise, or if the CSV is missing and the run can't be verified.
    """
    assign_path = os.path.join(save_path, 'cell_assignments.csv')
    if not os.path.exists(assign_path):
        return False
    try:
        n_actual = pd.read_csv(assign_path)['metacell_id'].nunique()
    except Exception:
        return False
    return abs(n_actual - expected_k) <= tolerance * expected_k


def _topk_sparsify_rows(M, k=20):
    """Truncate a dense or sparse (n_cells x n_protos) matrix to each row's top-k
    entries, returned as a CSR sparse matrix. Used to bound the on-disk size of
    soft-assignment matrices regardless of how sparse/dense the true distribution
    turns out to be (a very soft epsilon could leave most of the 800 columns
    non-negligible for every row) -- keeps only the k entries that matter most for
    reconstructing a soft-weighted pseudobulk, dropping the long tail whose
    contribution would be negligible anyway.
    """
    if sp.issparse(M):
        M = np.asarray(M.todense())
    else:
        M = np.asarray(M)
    n, p = M.shape
    k = min(k, p)
    # argpartition finds the top-k per row without a full sort
    top_idx = np.argpartition(-M, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = top_idx.ravel()
    vals = M[rows, cols]
    keep = vals > 0
    return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, p))


def save_soft_assignments(save_path, soft_assign_cells_x_protos, cell_ids):
    """Save a (n_cells x n_protos) soft-assignment matrix + the cell_id order it's
    aligned to, as soft_assignments.npz / soft_assignments_cell_ids.npy in
    save_path. Common format shared by both the scProto and SEACells save paths
    (SEACells' native model.A_ is n_protos x n_cells -- transpose before calling
    this) so downstream soft-weighted labeling/pseudobulk code is written once."""
    sp.save_npz(os.path.join(save_path, 'soft_assignments.npz'), soft_assign_cells_x_protos.tocsr())
    np.save(os.path.join(save_path, 'soft_assignments_cell_ids.npy'), np.asarray(cell_ids))
    print(f"  saved soft_assignments.npz {soft_assign_cells_x_protos.shape} to {save_path}")


def run_seacells_on_latent(ds_id, ad, n_seacells, aff, tag, skip_if_exists=True):
    """SEACells archetypal analysis directly on `aff`, with waypoint archetype seeding
    ALSO computed from `aff` (matches scProto's own waypoint-init philosophy -- see
    compute_seacells_own_affinity docstring). Saves + evaluates via the exact same
    save_seacell / eval_seacell_task1/2 pipeline used for the paper's existing
    SEACells(PCA) numbers, tagged '{tag}' -> folder 'seacell_{tag}'.

    Skips entirely (no archetypal analysis re-run) if metrics.json already exists
    for this (ds_id, tag), skip_if_exists is True, AND the cached run's realized
    metacell count matches n_seacells (see _cached_seacell_run_matches_k) -- a stale
    run at the wrong K is treated as a cache miss and recomputed fresh, not silently
    reused. Task 2 DGE consistency is turned off (compute_dge=False) -- not needed
    for this rebuttal's hypothesis test (Task 1 + rare-cell coverage/homogeneity
    carry the argument) and it was failing on non-lognormalized data anyway.
    Coverage + scGraph are still computed.

    Modularity is always recomputed against the canonical ARBF-on-PCA graph (see
    recompute_modularity_canonical below), on both the fresh-run and skip_if_exists
    cache-hit paths -- eval_seacell_task1 scores modularity off whatever
    ad.obsp['connectivities'] its freshly-reloaded adata happens to carry, which is
    not guaranteed to be this canonical graph. The cell->metacell assignment needed
    for this is read from this run's saved cell_assignments.csv (written by
    eval_seacell_task1's _save_seacell_umap_data) rather than by reloading the full
    seacell_sc.h5ad -- that CSV is small and far more likely to still be on disk than
    the big h5ad, which can be pruned separately. If even that CSV is missing, the
    recompute is skipped with a warning and whatever modularity was already in
    metrics.json is kept as-is, rather than crashing the whole dataset run.
    """
    lk = DATASETS[ds_id]['label_key']
    bk = DATASETS[ds_id].get('batch_key')
    save_path = get_seacell_model_dir(ds_id, tag)

    def _try_recompute(existing_metrics):
        assign_path = os.path.join(save_path, 'cell_assignments.csv')
        if not os.path.exists(assign_path):
            print(f"[{ds_id}] seacell_{tag}: no cell_assignments.csv found at {save_path} "
                  f"-- cannot recompute modularity against the canonical graph for this "
                  f"run; keeping its existing (possibly unverified) modularity value.")
            return existing_metrics
        try:
            assign_df = pd.read_csv(assign_path).set_index('cell_id')
            mc_idx_reload = assign_df.loc[ad.obs_names, 'metacell_id'].values.astype(int)
            return recompute_modularity_canonical(ds_id, ad, mc_idx_reload, bk, save_path, K=n_seacells)
        except Exception as e:
            print(f"[{ds_id}] seacell_{tag}: modularity recompute failed ({type(e).__name__}: {e}) "
                  f"-- keeping existing (possibly unverified) modularity value, continuing.")
            return existing_metrics

    if skip_if_exists:
        existing = _load_metrics(save_path)
        if existing is not None:
            if _cached_seacell_run_matches_k(save_path, n_seacells):
                print(f"[{ds_id}] seacell_{tag} already computed -- skipping (metrics.json found)")
                existing = _try_recompute(existing)
                return {'task1': existing, 'task2': existing, 'save_path': save_path, 'skipped': True}
            else:
                print(f"[{ds_id}] seacell_{tag}: found cached metrics.json at {save_path}, "
                      f"but its realized metacell count doesn't match target K={n_seacells} "
                      f"(stale run from an old num_prototypes value) -- treating as a cache "
                      f"miss and recomputing fresh (will overwrite this folder).")

    ad_sc, SEACell_ad, model = compute_seacells_own_affinity(
        ad, n_SEACells=n_seacells, aff=aff, build_kernel_on=tag,
    )
    agg_obs(SEACell_ad, ad_sc, lk)
    if bk is not None:
        agg_obs(SEACell_ad, ad_sc, bk)

    _fix_arrow_strings(ad_sc)
    _fix_arrow_strings(SEACell_ad)
    save_seacell(ad_sc, SEACell_ad, ds_id, build_kernel_on=tag)

    try:
        # model.A_ is SEACells' own archetypal soft-assignment matrix, n_SEACells x
        # n_cells (confirmed via its existing use elsewhere in this codebase, e.g.
        # scproto.py:3080 `model.A_.argmax(axis=0)` for the hard assignment) --
        # transpose to the shared (n_cells x n_protos) convention before saving.
        A_soft = model.A_
        A_soft = A_soft.T if not sp.issparse(A_soft) else sp.csr_matrix(A_soft).T
        A_soft_topk = _topk_sparsify_rows(A_soft, k=20)
        save_soft_assignments(save_path, A_soft_topk, ad_sc.obs_names.to_numpy())
    except Exception as e:
        print(f"WARNING: failed to save SEACells soft assignment matrix "
              f"({type(e).__name__}: {e}) -- continuing without it.")

    t1 = eval_seacell_task1(ds_id, build_kernel_on=tag)
    t1 = _try_recompute(t1)

    try:
        t2 = eval_seacell_task2(ds_id, build_kernel_on=tag, compute_dge=False)
    except Exception as e:
        print(f"WARNING: task2 failed for seacell_{tag}: {type(e).__name__}: {e} "
              f"-- keeping task1 results, continuing.")
        t2 = None

    return {'task1': t1, 'task2': t2, 'save_path': save_path}


def run_leiden_on_latent(ds_id, ad, n_target_clusters, aff, tag, skip_if_exists=True):
    """Leiden on the SAME affinity graph `aff` used for run_seacells_on_latent, so the
    two baselines differ only in clustering algorithm, not in graph construction --
    directly isolating e9Ho's "confounded with clusterer, not embedding" question.

    Computes Task 1 / Task 2 metrics via the exact same functions the rest of the
    paper's numbers come from (compute_task1_metrics / calc_task2_metrics), saved to
    an explicit, isolated folder ('leiden_{tag}_K{n}') -- never touches scProto's own
    trainer dump path.

    Skips entirely (no resolution search re-run) if a leiden_{tag}_K*/metrics.json
    already exists and skip_if_exists is True. Task 2 DGE consistency is turned off
    (compute_dge=False) -- not needed for this rebuttal's hypothesis test and it was
    failing on non-lognormalized data anyway; coverage + scGraph are still computed
    and non-fatal (kept if they fail, Task 1 results are kept regardless).

    Modularity is always recomputed against the canonical ARBF-on-PCA graph (see
    recompute_modularity_canonical below): compute_task1_metrics' own modularity call
    scores against `aff` (this method's own clustering target) via
    ad.obsp['connectivities'], which is close to circular -- Leiden is scored on the
    exact graph it was asked to partition. On the skip_if_exists cache-hit path, the
    cell-to-metacell assignment is reloaded from this run's saved
    cell_assignments.csv so modularity can be recomputed there too.
    """
    lk = DATASETS[ds_id]['label_key']
    bk = DATASETS[ds_id].get('batch_key')
    nk = DATASETS[ds_id].get('niche_key')

    if skip_if_exists:
        existing_dir = _find_existing_leiden_dir(ds_id, tag, n_target_clusters)
        if existing_dir is not None:
            existing = _load_metrics(existing_dir)
            print(f"[{ds_id}] leiden_{tag} already computed -- skipping (found {existing_dir})")
            assign_path = os.path.join(existing_dir, 'cell_assignments.csv')
            if not os.path.exists(assign_path):
                print(f"[{ds_id}] leiden_{tag}: no cell_assignments.csv found at "
                      f"{existing_dir} -- cannot recompute modularity against the "
                      f"canonical graph for this cached run; its reported modularity "
                      f"may still reflect the old, unverified reference graph.")
            else:
                try:
                    assign_df = pd.read_csv(assign_path).set_index('cell_id')
                    mc_idx_reload = assign_df.loc[ad.obs_names, 'metacell_id'].values.astype(int)
                    existing = recompute_modularity_canonical(
                        ds_id, ad, mc_idx_reload, bk, existing_dir,
                        K=existing.get('n_clusters', n_target_clusters),
                    )
                except Exception as e:
                    print(f"[{ds_id}] leiden_{tag}: modularity recompute failed "
                          f"({type(e).__name__}: {e}) -- keeping existing (possibly "
                          f"unverified) modularity value, continuing.")
            return {'task1': existing, 'task2': existing, 'save_path': existing_dir,
                    'resolution': existing.get('resolution'), 'n_clusters': existing.get('n_clusters'),
                    'skipped': True}

    labels, resolution, n_k = leiden_resolution_search(aff, n_target_clusters)
    mc_key = f'leiden_{tag}'
    ad.obs[mc_key] = pd.Categorical(labels.astype(str))

    run_name = f'leiden_{tag}_K{n_k}'
    save_path = os.path.join(get_dataset_model_dir(ds_id), run_name)
    os.makedirs(save_path, exist_ok=True)

    mc_idx = labels.astype(int)
    compute_task1_metrics(
        ad, mc_idx, lk, bk, nk, save_path, ds_id, run_name,
    )
    task1 = recompute_modularity_canonical(ds_id, ad, mc_idx, bk, save_path, K=n_k)

    metrics_path = os.path.join(save_path, 'metrics.json')

    task2 = None
    try:
        if 'counts' not in ad.layers:
            ad.layers['counts'] = ad.X.copy()
        mc_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label=mc_key, summarize_layer='counts')
        agg_obs(mc_ad, ad, lk)
        if bk is not None:
            agg_obs(mc_ad, ad, bk)
        sc.tl.pca(mc_ad)
        obsm_key = f'{run_name}_mc_pca'
        mc_ad.obsm[obsm_key] = mc_ad.obsm['X_pca']

        task2 = calc_task2_metrics(ad, mc_ad, lk, bk, [obsm_key], run_name, save_path, compute_dge=False)
    except Exception as e:
        print(f"WARNING: task2 failed for leiden_{tag}: {type(e).__name__}: {e} "
              f"-- keeping task1 results, continuing.")

    # merge whatever we have into the same metrics.json (already holds the
    # canonical-graph modularity written by recompute_modularity_canonical above)
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    if task2 is not None:
        metrics.update({k: v for k, v in task2.items() if v is not None})
    metrics['resolution'] = resolution
    metrics['n_clusters'] = int(n_k)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    try:
        save_baseline_umap_data(ds_id, ad, mc_key, save_path, lk, bk, nk)
    except Exception as e:
        print(f"WARNING: save_baseline_umap_data failed for leiden_{tag}: {type(e).__name__}: {e} "
              f"-- rare_celltype_purity_table won't find this run, but metrics.json is saved.")

    print(f"[{ds_id}] leiden-on-{tag} saved to {save_path}")
    return {'task1': task1, 'task2': task2, 'save_path': save_path, 'resolution': resolution, 'n_clusters': int(n_k)}


def _cache_paths(ds_id, method):
    base = get_dataset_model_dir(ds_id)
    return (os.path.join(base, f'_cache_X_{method}_emb.npy'),
            os.path.join(base, f'_cache_X_{method}_aff.npz'))


def _load_cached_embedding_affinity(ds_id, method):
    """Reuse a previously-computed embedding/affinity for (ds_id, method) instead of
    recomputing it just because Leiden alone needs a redo (SEACells already being done
    doesn't help Leiden -- run_correction_method's top-level skip_if_exists check only
    short-circuits when BOTH are already done). Two sources, checked in order:

    1. The already-saved SEACells output for this tag, if `seacell_X_{method}` exists.
       save_seacell's delta-vs-base diffing persists any obsm key unique to a tag (the
       corrected embedding itself, e.g. 'X_scvi') into that tag's own seacell_sc.h5ad
       even though nothing there saved it "on purpose" -- so a SEACells run completed
       under an older version of this notebook already has this on disk. Recovers it
       retroactively, with zero prior setup -- critical for scVI (real model training).
    2. Our own emb/aff cache (written by _save_cache below every time this function
       runs from here on), so future reruns never need step 1 or a full recompute.

    'bbknn' has no embedding (aff only, never stored on `ad`), so only the cache (2)
    can help it; a cache miss there just means a cheap graph rebuild, not model retraining.
    """
    tag = f'X_{method}'
    z = None

    if method != 'bbknn':
        try:
            saved_ad, _ = load_seacell(ds_id, build_kernel_on=tag)
            if tag in saved_ad.obsm:
                z = np.asarray(saved_ad.obsm[tag])
        except (FileNotFoundError, OSError):
            pass

    emb_path, aff_path = _cache_paths(ds_id, method)
    if z is None and os.path.exists(emb_path):
        z = np.load(emb_path)
    aff = sp.load_npz(aff_path) if os.path.exists(aff_path) else None

    if z is not None or aff is not None:
        print(f"[{ds_id}] {method}: reusing cached embedding/affinity "
              f"(z={'hit' if z is not None else 'miss'}, aff={'hit' if aff is not None else 'miss'})")
    return z, aff


def _save_cache(ds_id, method, z, aff):
    emb_path, aff_path = _cache_paths(ds_id, method)
    if z is not None:
        np.save(emb_path, z)
    if aff is not None:
        sp.save_npz(aff_path, sp.csr_matrix(aff))


_CANONICAL_AFF_CACHE = {}


def get_canonical_affinity(ds_id, n_cells, n_components=50, k_neighbors=50,
                            affinity_type='arbf', graph_dir='./graphs'):
    if ds_id in _CANONICAL_AFF_CACHE:
        return _CANONICAL_AFF_CACHE[ds_id]
    graph_path = get_affinity_path(ds_id, n_cells, n_components, k_neighbors, affinity_type, graph_dir)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Canonical affinity graph not found at {graph_path} -- cannot score "
            f"modularity consistently against it. (ds_id={ds_id}, n_cells={n_cells})"
        )
    with open(graph_path, 'rb') as f:
        aff = pickle.load(f)
    aff = sp.csr_matrix(aff)
    _CANONICAL_AFF_CACHE[ds_id] = aff
    print(f"[{ds_id}] canonical ARBF-on-PCA affinity graph loaded from {graph_path} "
          f"(nnz={aff.nnz}) -- caching for reuse across all methods for this dataset.")
    return aff


def recompute_modularity_canonical(ds_id, ad, mc_idx, bk, save_path, K=None):
    """Overwrite metrics.json's modularity fields with scores against the canonical
    ARBF-on-PCA graph, so every method/dataset in Table 1 is judged against the same
    reference graph regardless of which embedding it was actually clustered on.

    Needed because `ad` is one shared object reused across all correction methods per
    dataset, and build_latent_affinity() never writes its graph back into
    ad.obsp['connectivities'] -- so a Leiden/SEACells run's reported modularity could
    otherwise silently be scored against whatever graph a DIFFERENT correction method
    (or a plain fallback) last left sitting there.
    """
    aff = get_canonical_affinity(ds_id, len(ad))
    ad.obsp['connectivities'] = aff
    mod_result = compute_modularity(ad, mc_idx)

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    old_mod = metrics.get('mean_modularity_batch')
    metrics['modularity'] = mod_result['modularity']
    if bk is not None and bk in ad.obs.columns:
        batch_mod_s = calc_modularity_per_batch(aff, mc_idx, ad.obs[bk].values)
        metrics['mean_modularity_batch'] = float(batch_mod_s.mean())
        metrics['std_modularity_batch'] = float(batch_mod_s.std())
    if K is not None:
        metrics['K_target'] = int(K)

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[{save_path}] modularity recomputed against canonical graph: "
          f"mean_modularity_batch={metrics.get('mean_modularity_batch')} +/- "
          f"{metrics.get('std_modularity_batch')} (was {old_mod}), K_target={K}")
    return metrics


DIM_MATCHED_METHODS = {'harmony', 'combat', 'bbknn'}


def _downstream_runs_complete(ds_id, tag, K, run_seacells=True, run_leiden=True,
                               skip_if_exists=True):
    """True when every downstream arm requested for this (ds_id, tag) is already saved on
    disk at the right K -- i.e. run_seacells_on_latent / run_leiden_on_latent would both
    take their cache-hit path and never look at the affinity graph.

    Deliberately mirrors those two functions' own cache conditions (metrics.json present +
    realized metacell count matching K for SEACells via _cached_seacell_run_matches_k; an
    existing leiden_{tag}_K* dir via _find_existing_leiden_dir) rather than inventing a
    looser check -- if this returned True while one of them actually recomputed, that one
    would receive aff=None and fail. A stale run at the wrong K correctly reports False
    here, so the graph is rebuilt exactly when it is really needed.
    """
    if not skip_if_exists:
        return False
    if run_seacells:
        save_path = get_seacell_model_dir(ds_id, tag)
        if _load_metrics(save_path) is None or not _cached_seacell_run_matches_k(save_path, K):
            return False
    if run_leiden:
        if _find_existing_leiden_dir(ds_id, tag, K) is None:
            return False
    return run_seacells or run_leiden


def run_correction_method(ds_id, ad, method, K, lk, bk, skip_if_exists=True,
                           run_seacells=True, run_leiden=True, matched_n_comps=None):
    """One full batch-correction -> {SEACells, Leiden} -> metrics pipeline.

    method: one of CORRECTION_METHODS.
    For embedding-based methods: builds one shared adaptive-RBF affinity graph on the
    embedding, fed to both SEACells and Leiden. For 'bbknn': its own batch-balanced kNN
    graph IS the affinity (no separate RBF-on-embedding step).

    Assumes ad.obsm['X_stage1z'] is already set by the caller for method='stage1z'
    (get_stage1_latent was already run once per dataset to build `ad` itself).

    run_seacells / run_leiden let either clustering arm be switched off entirely (e.g.
    to skip SEACells for a quick Leiden-only comparison run) -- the corresponding
    result dict entry is None when its flag is False, rather than the pipeline being
    duplicated or commented out at the call site.

    matched_n_comps: scProto's own latent dimension (e.g. 8). For harmony, combat, and
    bbknn (DIM_MATCHED_METHODS) -- when given, each is corrected/embedded at THIS
    dimension instead of the usual 50, and `tag` becomes 'X_{method}_d{n}' instead of
    the plain 'X_{method}'. This is the ONE variant computed for each of these three
    methods -- not an addition alongside a d=50 version. Comparing scProto's low-dim
    latent against a 50-dim baseline isn't apples-to-apples: lower dimensionality
    forces cell-type variation into a more crowded space, independent of whatever
    correction mechanism is actually being compared (originally identified for
    Harmony via get_harmony_embedding_matched_dim; applies identically to ComBat and
    BBKNN, which also both start from a 50-dim PCA by default).

    Using a dimension-qualified tag is what makes this safe: it can never collide with
    old plain 'X_{method}' / 'seacell_X_{method}' / 'leiden_X_{method}_K*' names left
    on disk from before this fix existed, so skip_if_exists correctly forces a fresh
    compute the first time at a given dimension and correctly reuses it on every run
    after that. Those old, now-orphaned d=50 files/folders are harmless leftovers, not
    read by anything once this is in use. If matched_n_comps is None (e.g. scProto's
    checkpoint wasn't available), each of the three falls back to its own plain d=50
    path as a safety net, not a parallel mode.

    When run, always calls run_seacells_on_latent / run_leiden_on_latent even if both
    are already computed for this (ds_id, method) -- those two functions have their own
    skip_if_exists cache-hit paths (cheap: a metrics.json load + a modularity
    recompute against the canonical graph, no retraining), so results written before
    the canonical-modularity fix still get corrected on a rerun. Embedding/affinity
    construction itself still uses the cache in _load_cached_embedding_affinity below,
    so this doesn't retrain scVI or rebuild embeddings unnecessarily -- only the
    (cheap) affinity-graph build is redone when both baselines are already done.
    """
    tag = f'X_{method}'
    purity = None  # embedding-level rare-cell kNN purity -- None for 'bbknn' (graph only, no embedding)
    dim_matched = (method in DIM_MATCHED_METHODS)
    resolved_n_comps = matched_n_comps if matched_n_comps is not None else 50
    if dim_matched:
        tag = f'X_{method}_d{resolved_n_comps}'

    # dim-matched methods each handle their own dimension-qualified caching internally
    # (get_harmony_embedding_matched_dim / get_combat_corrected_pca / get_bbknn_graph)
    # -- the generic _load_cached_embedding_affinity/_save_cache pair below is only
    # for the non-dimension-qualified methods (stage1z, scvi).
    cached_z, cached_aff = (
        (None, None) if dim_matched else
        (_load_cached_embedding_affinity(ds_id, method) if skip_if_exists else (None, None))
    )

    # If BOTH downstream arms are already complete on disk, neither the affinity graph
    # nor the embedding-level kNN purity is needed: run_seacells_on_latent /
    # run_leiden_on_latent return from their cache-hit paths without ever touching `aff`
    # (their modularity recompute uses the separately-cached canonical graph), and
    # purity is already saved in rare_knn_purity_{ds}.json. Building them anyway costs
    # a full kNN + adaptive-bandwidth RBF pass per rerun -- minutes on a 16k-cell dataset,
    # for output that is thrown away. `aff` stays None in that case; nothing downstream
    # reads it. Not applied to 'bbknn', whose graph IS its embedding (and which has its
    # own on-disk cache).
    downstream_cached = method != 'bbknn' and _downstream_runs_complete(
        ds_id, tag, K, run_seacells=run_seacells, run_leiden=run_leiden,
        skip_if_exists=skip_if_exists,
    )
    if downstream_cached:
        print(f"[{ds_id}] {tag}: SEACells + Leiden already complete -- skipping affinity "
              f"graph construction and kNN purity (nothing downstream needs them).")

    def _aff(cached=None):
        if downstream_cached:
            return None
        return cached if cached is not None else build_latent_affinity(ad, latent_key=tag)

    def _purity(z_arr):
        return None if downstream_cached else rare_type_knn_purity(z_arr, ad.obs[lk].values)

    if method == 'stage1z':
        z = ad.obsm[tag]
        aff = _aff(cached_aff)
        purity = _purity(z)
    elif method == 'harmony':
        z = get_harmony_embedding_matched_dim(ad, bk, resolved_n_comps, ds_id=ds_id)
        ad.obsm[tag] = z
        aff = _aff()
        purity = _purity(z)
    elif method == 'scvi':
        z = cached_z if cached_z is not None else get_scvi_embedding(ad, ds_id, bk)
        ad.obsm[tag] = z
        aff = _aff(cached_aff)
        purity = _purity(z)
    elif method == 'scvi_gauss':
        # loss-matched scVI variant (gene_likelihood='normal') -- see
        # get_scvi_gauss_embedding's docstring. tag='X_scvi_gauss' here, distinct
        # from plain 'scvi' (ZINB)'s 'X_scvi' -- separate cache files
        # (_cache_X_scvi_gauss_*), separate seacell_X_scvi_gauss /
        # leiden_X_scvi_gauss_K{K} folders, never collides with the ZINB run.
        z = cached_z if cached_z is not None else get_scvi_gauss_embedding(ad, ds_id, bk)
        ad.obsm[tag] = z
        aff = _aff(cached_aff)
        purity = _purity(z)
    elif method == 'combat':
        z = get_combat_corrected_pca(ad, bk, n_comps=resolved_n_comps, ds_id=ds_id, use_cache=skip_if_exists)
        ad.obsm[tag] = z
        aff = _aff()
        purity = _purity(z)
    elif method == 'bbknn':
        aff = get_bbknn_graph(ad, bk, n_comps=matched_n_comps, ds_id=ds_id, use_cache=skip_if_exists)
    else:
        raise ValueError(f"unknown method '{method}', expected one of "
                          f"['stage1z', 'harmony', 'scvi', 'scvi_gauss', 'combat', 'bbknn']")

    if not dim_matched:
        _save_cache(ds_id, method, z if method != 'bbknn' else None, aff)

    seacells_res = None
    if run_seacells:
        seacells_res = run_seacells_on_latent(ds_id, ad, n_seacells=K, aff=aff, tag=tag, skip_if_exists=skip_if_exists)

    leiden_res = None
    if run_leiden:
        leiden_res = run_leiden_on_latent(ds_id, ad, n_target_clusters=K, aff=aff, tag=tag, skip_if_exists=skip_if_exists)

    return {'method': method, 'tag': tag, 'purity': purity, 'seacells': seacells_res, 'leiden': leiden_res}


def _merge_dict_json(path, new_data):
    """Write new_data into path, MERGED with whatever's already there, instead of a
    plain overwrite. Fixes a real bug (found 2026-07-31): rare_affinity_purity_{ds}.json
    and rare_knn_purity_{ds}.json are each written to by every sibling batch-correction
    notebook that runs for a given dataset (this notebook computes harmony/scvi/
    scvi_gauss; combat_then_cluster_baselines.ipynb computes combat; etc.) -- but all of
    them share the exact same file path per dataset. A plain `json.dump(data, open(path,
    'w'))` overwrites the WHOLE file with only whatever that one run just computed, so
    whichever notebook happens to run last silently deletes every other notebook's rows.
    That's exactly what was observed: a scVI/Harmony run's rows vanished from
    rare_affinity_purity_pancreas.json after a ComBat notebook run overwrote the file
    with only its own (pca/combat/scproto) rows.

    Fix: read the existing file first, then merge new_data into it one level of dict
    nesting deep (new_data's own values win on any overlapping key), and write the
    merged result back -- so each notebook's contribution accumulates instead of
    replacing. Handles both JSON shapes used in this module: a flat {method: value}
    dict (rare_knn_purity), and a one-level-nested {section: {key: value}} dict
    (rare_affinity_purity's per_batch_scores/embedding_dims/display_names) -- for both,
    any top-level key whose value is itself a dict gets merged key-by-key with the
    existing dict at that path; every other key is simply overwritten.

    Not a substitute for a real file lock -- two processes writing at the exact same
    instant could still race (and Drive-synced files can't be locked across machines
    anyway). This fixes the far more common case: sibling notebooks running
    sequentially or interleaved over the course of a session, which is what was
    actually happening here.
    """
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    merged = dict(existing)
    for key, val in new_data.items():
        if isinstance(val, dict) and isinstance(existing.get(key), dict):
            merged[key] = {**existing[key], **val}
        else:
            merged[key] = val

    with open(path, 'w') as f:
        json.dump(merged, f, indent=2)
    return merged


AFFINITY_PURITY_DISPLAY_NAMES = {
    'X_pca': 'Raw PCA (uncorrected)',
    'X_stage1z': 'scPoli (Stage-1)',
    'X_scvi': 'scVI',
    'X_scvi_gauss': 'scVI (Gaussian)',
    'X_combat': 'ComBat',
    'X_scproto': 'scProto',
}


def _json_has_key(path, key):
    """True if `path` is a readable JSON dict already holding a non-empty `key`. Used to
    decide whether an expensive diagnostic can be reused instead of recomputed; any read
    problem returns False, so a corrupt/missing file falls back to recomputing rather
    than silently skipping.
    """
    if not os.path.exists(path):
        return False
    try:
        return bool(json.load(open(path)).get(key))
    except Exception:
        return False


def _affinity_purity_has_key(ds_id, key):
    """True if this dataset's saved affinity-purity results already contain `key`
    (e.g. 'X_scproto'), i.e. that embedding's row needs no recompute."""
    path = os.path.join(get_dataset_model_dir(ds_id), f'rare_affinity_purity_{ds_id}.json')
    if not os.path.exists(path):
        return False
    try:
        return bool(json.load(open(path)).get('per_batch_scores', {}).get(key))
    except Exception:
        return False


def _load_scproto_latent(ds_id, ad):
    """Loads scProto's own Stage-2 latent (existing trained checkpoint, not retrained
    here -- load_umap=True) and aligns it to ad.obs_names. Returns the (N, d) array,
    or None (with a printed warning) if the checkpoint is missing or obs_names don't
    align. Shared by run_all_baselines_for_dataset (which needs scProto's dimension
    before running the matched-dim Harmony baseline) and
    compute_and_save_embedding_affinity_purity, so the checkpoint only ever loads once
    per dataset, not once per caller.

    skip_eval=True (see find_metacells/run_mc_task in tasks.py): load_umap=True alone
    only skips TRAINING -- eval_metacell_quality/eval_task2/3, aff_dc_compactness, and
    save_metacells()/save_umap_data() all still ran unconditionally on every call
    without this, silently overwriting clusters.npz/metacells.h5ad in the checkpoint's
    own directory (including the canonical scProto run directories whose numbers are
    verified against the published paper) purely to get an encoded latent. Only the
    loaded model is needed here, so skip all of that.
    """
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    try:
        t2, _, _ = run_mc_task(
            ds_id, cvae_epochs=50, train_epochs=50, eval_freq=3, patience=6,
            batch_size=1024, umap_steps_per_epoch=500,
            lambda_config=LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'},
            affinity_type='arbf', load_umap=True, skip_eval=True,
        )
        with torch.no_grad():
            z_scproto = t2.encode_adata(t2.train_ds.adata, t2.model, z_idx=1).cpu().numpy()
        z_scproto_df = pd.DataFrame(z_scproto, index=t2.train_ds.adata.obs_names)
        missing = set(ad.obs_names) - set(z_scproto_df.index)
        if missing:
            print(f"[{ds_id}] WARNING: {len(missing)} cells missing from scProto's own "
                  f"adata -- can't align X_scproto (obs_names mismatch).")
            return None
        return z_scproto_df.reindex(ad.obs_names).values
    except FileNotFoundError as e:
        print(f"[{ds_id}] scProto checkpoint not found -- skipping X_scproto ({e}).")
        return None


def compute_and_save_embedding_affinity_purity(ds_id, ad, lk, bk, K=None,
                                                harmony_tag=None, combat_tag=None,
                                                compute_nclust_control=False,
                                                skip_if_exists=True):
    """Embedding-only, clustering-free rare-cell affinity purity for every embedding
    available for this dataset: scProto's own Stage-2 latent, Harmony/ComBat
    (whatever tag each was actually computed under -- see harmony_tag/combat_tag),
    X_stage1z, X_scvi (whichever are present on `ad`), raw PCA.

    harmony_tag / combat_tag: the exact obsm key each embedding for this dataset
    lives under (e.g. 'X_harmony_d8' / 'X_combat_d8' when run_correction_method used
    the dimension-matched path -- see its docstring). Passed in rather than assumed,
    since which tag is correct depends on whether the matched-dim fix was applied.
    If None, that method is simply left out of this comparison.

    compute_nclust_control (default False): if True AND K is given (the dataset's
    num_prototypes) AND harmony_tag is set, ALSO computes a SEPARATE nclust-matched
    Harmony embedding (same dimension as harmony_tag, PLUS harmonypy's own internal
    `nclust` set to K instead of its min(N/30, 100) default) purely for this
    embedding-only diagnostic -- not run through SEACells/Leiden. Off by default --
    this is a deeper hyperparameter question nobody actually asked about; only worth
    turning on if the main dimension-matched results alone still look unfavorable and
    it's worth ruling this out too.

    Called automatically at the end of run_all_baselines_for_dataset -- runs every
    time the 'Run: {dataset}' cells run, no separate notebook step needed. Saves to
    '{MODEL_DIR}/{ds_id}/rare_affinity_purity_{ds_id}.json'
    (per_batch_scores/embedding_dims/display_names); load_and_compare_affinity_purity
    below just reads that file back -- no computation happens in the notebook itself.
    """
    keys = [k for k in ('X_pca', 'X_stage1z', 'X_scvi', 'X_scvi_gauss') if k in ad.obsm]
    dynamic_tags = {}  # obsm key -> display name, for dimension-qualified tags
    if harmony_tag is not None and harmony_tag in ad.obsm:
        keys.append(harmony_tag)
        dynamic_tags[harmony_tag] = 'Harmony'
    if combat_tag is not None and combat_tag in ad.obsm:
        keys.append(combat_tag)
        dynamic_tags[combat_tag] = 'ComBat'

    # scProto's own Stage-2 latent -- reused from ad.obsm if a caller (e.g.
    # run_all_baselines_for_dataset, which also needs it to size the matched-dim
    # Harmony run) already loaded it, so the checkpoint is never loaded twice in one
    # dataset run. Loads it itself otherwise, so this function still works standalone.
    if 'X_scproto' not in ad.obsm:
        z_scproto = _load_scproto_latent(ds_id, ad)
        if z_scproto is not None:
            ad.obsm['X_scproto'] = z_scproto
    if 'X_scproto' in ad.obsm:
        keys.append('X_scproto')

    if compute_nclust_control and K is not None and harmony_tag is not None and bk is not None:
        n_comps = ad.obsm[harmony_tag].shape[1]
        ad.obsm['X_harmony_nclustK'] = get_harmony_embedding_matched_dim(
            ad, bk, n_comps, ds_id=ds_id, nclust=K,
        )
        keys.append('X_harmony_nclustK')
        AFFINITY_PURITY_DISPLAY_NAMES.setdefault('X_harmony_nclustK', 'Harmony (matched nclust)')

    # Each key here costs a full ARBF graph build (kNN + adaptive-bandwidth RBF) -- with
    # 5-6 embeddings present that dominates the wall-clock of a rerun whose runs are all
    # cache hits. Rows already saved for this dataset are reused as-is: the score depends
    # only on (embedding, labels, batches), all fixed once the embedding is on disk, so
    # recomputing reproduces the same number. Pass skip_if_exists=False to force a fresh
    # recompute (e.g. after changing rare_type_affinity_ratio_per_batch itself).
    save_path = os.path.join(get_dataset_model_dir(ds_id), f'rare_affinity_purity_{ds_id}.json')
    existing = {}
    if skip_if_exists and os.path.exists(save_path):
        try:
            existing = json.load(open(save_path)).get('per_batch_scores', {}) or {}
        except Exception as e:
            print(f"[{ds_id}] could not read cached affinity purity ({type(e).__name__}: {e}) "
                  f"-- recomputing all embeddings.")

    per_ds = {}
    dims_this_ds = {}
    for key in keys:
        dims_this_ds[key] = int(ad.obsm[key].shape[1])
        if key in existing and existing[key]:
            per_ds[key] = existing[key]
            print(f"[{ds_id}] {key}: affinity purity already saved -- reusing "
                  f"(no graph rebuild).")
        else:
            per_ds[key] = rare_type_affinity_ratio_per_batch(ad, key, lk, bk)
        vals = per_ds[key]
        # dynamic_tags covers dimension-qualified keys (e.g. 'X_harmony_d8',
        # 'X_combat_d8') that won't be static keys in AFFINITY_PURITY_DISPLAY_NAMES.
        name = dynamic_tags.get(key, AFFINITY_PURITY_DISPLAY_NAMES.get(key, key))
        if vals:
            print(f"[{ds_id}] {name} (d={dims_this_ds[key]}): "
                  f"{np.mean(vals):.3f} +/- {np.std(vals):.3f} (n={len(vals)} batches)")
        else:
            print(f"[{ds_id}] {name}: no locally-rare batches found")

    _merge_dict_json(save_path, {
        'per_batch_scores': per_ds,
        'embedding_dims': dims_this_ds,
        'display_names': {
            k: dynamic_tags.get(k, AFFINITY_PURITY_DISPLAY_NAMES.get(k, k))
            for k in per_ds
        },
    })
    print(f"[{ds_id}] affinity purity saved to {save_path} "
          f"(merged with any existing entries from other runs/notebooks)")

    return per_ds


def load_and_compare_affinity_purity(ds_ids, ref_key='X_scproto', dataset_display_names=None):
    """Loads rare_affinity_purity_{ds_id}.json (written automatically by
    compute_and_save_embedding_affinity_purity inside run_all_baselines_for_dataset)
    for each ds_id and builds one comparison table: mean/std per embedding, plus
    paired one-sided Wilcoxon signed-rank (ref_key > other) with Bonferroni
    correction per dataset -- same convention as
    paper_figures.rare_metric_significance_paired. Pure load + aggregate/test, no
    embedding computation happens here.
    """
    from scipy.stats import wilcoxon

    def _stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    dataset_display_names = dataset_display_names or {}
    rows = []
    for ds_id in ds_ids:
        path = os.path.join(get_dataset_model_dir(ds_id), f'rare_affinity_purity_{ds_id}.json')
        if not os.path.exists(path):
            print(f"[{ds_id}] {path} not found -- run run_all_baselines_for_dataset('{ds_id}') first.")
            continue
        d = json.load(open(path))
        per_ds = d['per_batch_scores']
        dims = d.get('embedding_dims', {})
        names = d.get('display_names', {})

        ref_vals = per_ds.get(ref_key)
        ref_arr = np.array(ref_vals) if ref_vals else None
        if ref_arr is None:
            print(f"[{ds_id}] '{ref_key}' not in saved results -- skipping significance test for this dataset.")
        n_comparisons = sum(1 for k in per_ds if k != ref_key and per_ds[k])

        for key, vals in per_ds.items():
            arr = np.array(vals)
            row = {
                'dataset': dataset_display_names.get(ds_id, ds_id),
                'method': names.get(key, key),
                'dim': dims.get(key),
                'n': len(arr),
                'mean': round(float(arr.mean()), 3) if len(arr) else float('nan'),
                'std': round(float(arr.std()), 3) if len(arr) else float('nan'),
            }
            if key != ref_key and ref_arr is not None and len(arr) > 0 and len(arr) == len(ref_arr):
                row['n_wins'] = int((ref_arr > arr).sum())
                try:
                    _, p_raw = wilcoxon(ref_arr, arr, alternative='greater')
                    p_adj = min(p_raw * n_comparisons, 1.0)
                    row['p_vs_ref'] = round(p_raw, 4)
                    row['p_adj'] = round(p_adj, 4)
                    row['sig'] = _stars(p_adj)
                except ValueError:
                    pass
            elif key != ref_key and ref_arr is not None and len(arr) != len(ref_arr):
                row['note'] = f'batch count mismatch vs ref ({len(arr)} vs {len(ref_arr)}) -- not paired'
            rows.append(row)

    return pd.DataFrame(rows)


def run_all_baselines_for_dataset(ds_id, correction_methods, skip_if_exists=True,
                                   run_seacells=True, run_leiden=True,
                                   compute_nclust_control=False,
                                   force_matched_n_comps=None):
    """Encapsulates the full run for one dataset: Stage-1 latent extraction, then all
    correction_methods x {SEACells, Leiden}, saving metrics for each along the way.

    run_seacells / run_leiden: see run_correction_method -- pass run_seacells=False to
    skip the SEACells arm entirely (e.g. for a quick Leiden-only baseline check).

    compute_nclust_control: see compute_and_save_embedding_affinity_purity -- off by
    default (a deeper Harmony hyperparameter question nobody asked about); flip to
    True from the notebook call if the dimension-matched affinity-purity results alone
    still look unfavorable and it's worth ruling this out too, without editing this file.

    force_matched_n_comps: run every DIM_MATCHED_METHODS method (harmony/combat/bbknn)
    at THIS dimension instead of the default behavior of deriving it from scProto's own
    latent dimension. The only reason this exists: Reviewer nG29 asked for Harmony
    reported at its conventional ~50-PC setting as well as at scProto's dimension
    (harmony_at_default_dim.ipynb), so 50 has to be passable explicitly rather than
    being whatever scProto's checkpoint happens to be. Safe alongside the existing
    matched runs -- the dimension is encoded in every tag/folder/cache name
    ('X_harmony_d50', 'seacell_X_harmony_d50', 'leiden_X_harmony_d50_K{K}'), so a d=50
    run never reads or overwrites a d=8 one. Left None (the default), behavior is
    exactly as before.
    """
    K = DATASETS[ds_id]['num_prototypes']
    lk = DATASETS[ds_id]['label_key']
    bk = DATASETS[ds_id].get('batch_key')

    if 'stage1z' in correction_methods:
        t, ad, z1 = get_stage1_latent(ds_id)
        ad.obsm['X_stage1z'] = z1
    else:
        # No method here needs the Stage-1 z1 latent -- skip the full
        # SCProtoTrainer/checkpoint/model load entirely (see get_preprocessed_adata).
        ad = get_preprocessed_adata(ds_id)
    if 'X_pca' not in ad.obsm:
        sc.pp.pca(ad, n_comps=50)

    # scProto's own Stage-2 latent, loaded once up front (not retrained -- existing
    # checkpoint only) so its dimension is known BEFORE the correction_methods loop
    # runs -- Harmony, ComBat, and BBKNN all need it to correct/embed at the matched
    # dimension instead of the usual 50 (see run_correction_method's matched_n_comps
    # and DIM_MATCHED_METHODS). Also reused by the embedding-only affinity-purity
    # diagnostic below, so the checkpoint never loads twice in one dataset run.
    # scProto's own Stage-2 latent is needed for two things: sizing the matched-dim
    # correction, and being the reference row of the affinity-purity diagnostic below.
    # When the dimension is given explicitly AND that diagnostic already has scProto's
    # row saved for this dataset, neither applies -- so skip the checkpoint load
    # entirely (it builds the full model and encodes every cell, which is pure overhead
    # on a rerun whose results are all cache hits).
    scproto_row_cached = skip_if_exists and _affinity_purity_has_key(ds_id, 'X_scproto')
    if force_matched_n_comps is not None and scproto_row_cached:
        print(f"[{ds_id}] skipping scProto checkpoint load -- dimension given explicitly "
              f"(force_matched_n_comps={force_matched_n_comps}) and its affinity-purity "
              f"row is already saved.")
        z_scproto = None
    else:
        z_scproto = _load_scproto_latent(ds_id, ad)
    if z_scproto is not None:
        ad.obsm['X_scproto'] = z_scproto
    matched_n_comps = z_scproto.shape[1] if z_scproto is not None else None
    if force_matched_n_comps is not None:
        # Explicit dimension wins over scProto's own (see docstring).
        if matched_n_comps is not None and matched_n_comps != force_matched_n_comps:
            print(f"[{ds_id}] force_matched_n_comps={force_matched_n_comps} overrides "
                  f"scProto's own latent dimension ({matched_n_comps}) for "
                  f"{sorted(DIM_MATCHED_METHODS & set(correction_methods))}.")
        matched_n_comps = force_matched_n_comps

    results = {}
    for method in correction_methods:
        print(f"\n=== [{ds_id}] batch-correction method: {method} ===")
        results[method] = run_correction_method(
            ds_id, ad, method, K, lk, bk, skip_if_exists=skip_if_exists,
            run_seacells=run_seacells, run_leiden=run_leiden,
            matched_n_comps=matched_n_comps if method in DIM_MATCHED_METHODS else None,
        )

    # raw_pca's kNN purity never changes for a dataset (X_pca is deterministic given the
    # preprocessed adata), so recomputing it on every rerun is wasted -- reuse whatever is
    # already in rare_knn_purity_{ds}.json. _merge_dict_json keeps existing keys that
    # aren't in the new dict, so leaving it out preserves the saved value.
    knn_purity_path = os.path.join(get_dataset_model_dir(ds_id), f'rare_knn_purity_{ds_id}.json')
    raw_pca_cached = skip_if_exists and _json_has_key(knn_purity_path, 'raw_pca')
    purity_summary = {m: r['purity'] for m, r in results.items() if r['purity'] is not None}
    if raw_pca_cached:
        print(f"[{ds_id}] raw_pca kNN purity already saved -- reusing.")
    else:
        purity_summary['raw_pca'] = rare_type_knn_purity(ad.obsm['X_pca'], ad.obs[lk].values)

    purity_summary = _merge_dict_json(knn_purity_path, purity_summary)
    print(f"\n[{ds_id}] rare-type kNN purity by method: "
          f"{ {m: round(v['mean_purity'], 3) if v['mean_purity'] is not None else None for m, v in purity_summary.items()} }")

    harmony_tag = results.get('harmony', {}).get('tag')
    combat_tag = results.get('combat', {}).get('tag')
    compute_and_save_embedding_affinity_purity(
        ds_id, ad, lk, bk, K=K, harmony_tag=harmony_tag, combat_tag=combat_tag,
        compute_nclust_control=compute_nclust_control, skip_if_exists=skip_if_exists,
    )

    return results

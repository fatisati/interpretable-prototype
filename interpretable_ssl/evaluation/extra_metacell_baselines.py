"""
Helper functions for the "more metacell baselines" rebuttal ask (Reviewer e9Ho,
Weakness 7 / Question 5: "Only SEACells and MetaQ are true metacell baselines ...
Will you add more baselines like scVI / SuperCell / Metacell-2?"). This module adds
the two still-missing ones: SuperCell and Metacell-2.

Both are run directly on each dataset's own (uncorrected) representation, at
K = DATASETS[ds_id]['num_prototypes'] metacells -- the same within-batch setup as the
paper's existing 'SEACells (PCA)' row in Table 1/2, so they slot into the exact same
comparison as a third and fourth "true metacell baseline", not a batch-correction
baseline.

SuperCell (Bilous et al. 2022, BMC Bioinformatics -- GfellerLab/SuperCell) has no
pip-installable Python port (verified against PyPI directly, not just GitHub/CRAN --
the `supercell`/`pysupercell` PyPI names that DO exist are an unrelated Tornado REST
framework and a crystal-structure tool). The only Python-reachable path is
GfellerLab/MetacellAnalysisToolkit, which wraps the R package via rpy2 + a dedicated
conda environment with R installed -- a much heavier and more fragile install in
Colab than what's already used for the Leiden baselines elsewhere in this notebook
family (see the module-level pipInstall discussion in
`batch_correct_then_cluster_baselines.ipynb` for what "fragile" has meant in practice
for lighter installs than this one).

`run_supercell_baseline` below is instead a direct Python port of SuperCell's actual
R source (`R/SCimplify.R`, `R/build_knn_graph.R`, read from
github.com/GfellerLab/SuperCell directly, not inferred from the paper), matching its
real default recipe:
  1. select the top `n_var_genes` (default 1000) highest-variance genes from
     log-normalized expression;
  2. z-score them (`do.scale=TRUE`) and run PCA, keeping `n_pc` (default 10)
     components (SuperCell's own defaults -- deliberately NOT the shared 50-dim
     `X_pca` the SEACells(PCA)/Leiden baselines elsewhere use, since SCimplify()
     always computes its own bespoke PCA for this step, never accepts a
     precomputed one);
  3. build a kNN graph (`k.knn`, default 5) via nearest neighbors in that PCA space,
     made undirected as a UNION of each direction's neighbor relation (cell A
     connects to B if B is among A's k-NN OR A is among B's k-NN -- exactly
     `igraph::graph_from_adj_list(..., mode='all')`'s semantics), with every kept
     edge given UNIFORM weight 1 (`igraph::E(graph.knn)$weight <- 1` -- SuperCell's
     default kNN graph is unweighted, not an RBF/distance-weighted kernel);
  4. run `igraph::cluster_walktrap` on that graph and cut the dendrogram
     (`igraph::cut_at`) to exactly `k = round(N/gamma)` groups -- here, exactly
     `DATASETS[ds_id]['num_prototypes']`, requested directly rather than via a
     gamma ratio.
Step 2 above only applies to the primary, expression-based comparison
(`latent_key='native'`, matching R's `SCimplify()`). For the secondary "SuperCell on
an existing embedding" comparison (`latent_key='X_stage1z'`, matching R's
`SCimplify_from_embedding()`), R does NOT rescale the given embedding -- it only
subsets to the first `n_pc` columns -- so that path skips step 2's z-scoring too.
Not a call into the official package -- a source-verified port of its mechanism.

Metacell-2 (Ben-Kiki et al. 2022, Genome Biology -- tanaylab/metacells) IS a real,
pip-installable Python package (`pip install metacells`), so `run_metacell2_baseline`
below calls it directly (`metacells.pl.divide_and_conquer_pipeline`). Unlike
SuperCell/SEACells/Leiden, Metacell-2's homogeneity model operates on raw UMI counts,
not an arbitrary embedding -- it cannot be pointed at a Stage-1 latent the way
SuperCell/Leiden/SEACells can, so there is no "Metacell-2 on Stage-1 embedding"
variant here (that would need a batch-corrected count matrix, which none of the
correction methods elsewhere in this codebase produce).

Import this AFTER nb_setup.py has run (needs the project root on sys.path).
"""

import os
import json

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import SEACells
import SEACells.core

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import get_dataset_model_dir
from interpretable_ssl.evaluation.mc_metric_utils import compute_task1_metrics, calc_task2_metrics

# these three already apply the SEACells dtype patch / pandas string patch / cuda
# fallback patch on import (idempotent) -- reused here rather than re-declared.
from interpretable_ssl.evaluation.batch_correct_baselines import (
    get_preprocessed_adata, recompute_modularity_canonical, save_baseline_umap_data,
    _load_metrics,
)


def _apply_metacells_readonly_patch():
    """metacells' own internal rare-gene-module detection
    (metacells/tools/rare.py::_identify_genes) calls `np.fill_diagonal(...)` in
    place on a gene-gene similarity matrix. On some numpy versions that array comes
    back genuinely read-only (numpy has gotten stricter over time about which views/
    computed arrays keep the writeable flag off), which np.fill_diagonal has no
    fallback for -- `ValueError: underlying array is read-only`, raised from deep
    inside divide_and_conquer_pipeline, before this module's own code ever runs.
    metacells' own changelog (HISTORY.rst 0.9.2: "Fix numpy compatibility issue")
    shows this exact class of numpy-version friction has bitten this package before;
    this is that same class of issue recurring against a newer numpy than the
    package was tested against, not something wrong with how this module calls it.

    Patch: wrap np.fill_diagonal so that if it's asked to write into a read-only
    array, it flips that array's own WRITEABLE flag on first (safe here -- the
    array in question is a fresh similarity matrix metacells computed for its own
    internal use, not a shared/frozen buffer this codebase relies on staying
    protected) and then proceeds as normal. If the array can't be made writable at
    all (e.g. a read-only view into someone else's buffer), the original
    ValueError still propagates -- this only removes the specific, needlessly
    strict case metacells' own code doesn't defend against.
    Idempotent: guarded against re-applying if this module is imported/reloaded
    more than once (stacking wrappers would still work, just wastefully).
    """
    if getattr(np.fill_diagonal, '_readonly_patch_applied', False):
        return
    _orig_fill_diagonal = np.fill_diagonal

    def _patched_fill_diagonal(a, val, wrap=False):
        if isinstance(a, np.ndarray) and not a.flags.writeable:
            a.flags.writeable = True
        return _orig_fill_diagonal(a, val, wrap=wrap)

    _patched_fill_diagonal._readonly_patch_applied = True
    np.fill_diagonal = _patched_fill_diagonal


_apply_metacells_readonly_patch()


def _agg_obs_by(mc_ad, ad, mc_key, obs_key):
    """Majority-vote a per-cell obs column onto mc_ad, grouped by ad.obs[mc_key].

    Deliberately NOT metric_helpers.metacell_metrics.agg_obs -- that helper hardcodes
    `adata.obs.groupby("SEACell")` regardless of the obs_key/column it's asked to
    aggregate, so it only works when the assignment column is literally named
    'SEACell' (true for the paper's real SEACells runs, not for the arbitrary
    mc_key names used here/in run_leiden_on_latent). This is the same aggregation,
    parametrized by the actual assignment column.
    """
    mc_ad.obs[obs_key] = (
        ad.obs.groupby(mc_key)[obs_key]
        .agg(lambda x: x.mode()[0])
        .reindex(mc_ad.obs_names)
    )
    return mc_ad


# ---------------------------------------------------------------------------
# SuperCell (source-verified Python port -- bespoke PCA + unweighted union kNN +
# igraph walktrap cut to K -- see module docstring for the exact recipe read out
# of GfellerLab/SuperCell's actual R source)
# ---------------------------------------------------------------------------

def _supercell_native_pca(ad, n_var_genes=1000, n_pc=10, do_scale=True, seed=42):
    """SCimplify()'s own PCA recipe (R/SCimplify.R), computed directly here rather
    than reusing the shared ad.obsm['X_pca'] -- SCimplify never accepts a
    precomputed PCA. It always: (1) picks the `n_var_genes` highest-variance genes
    from log-normalized expression, (2) z-scores them (`do.scale=TRUE`), NaNs (from
    zero-variance genes, same as R's `scale()`) filled with 0 exactly like R's
    `X.for.pca[is.na(X.for.pca)] <- 0`, (3) runs PCA keeping `n_pc` components.
    R defaults reused as this function's own defaults: n.var.genes=min(1000,
    nrow(X)), n.pc=10, do.scale=TRUE.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = ad.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)

    n_var_genes = min(n_var_genes, X.shape[1])
    top_genes = np.argsort(X.var(axis=0))[::-1][:n_var_genes]
    X = X[:, top_genes]

    if do_scale:
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

    n_pc = min(n_pc, X.shape[1], X.shape[0] - 1)
    return PCA(n_components=n_pc, random_state=seed).fit_transform(X)


def _build_union_knn_graph(X, k=5):
    """SuperCell's own kNN graph (`build_knn_graph_nn2`'s default path in
    R/build_knn_graph.R: `use.nn2=TRUE, DoSNN=FALSE, pruning=NULL`): k nearest
    neighbors (Euclidean) per cell, made undirected as a UNION of each direction's
    relation (`igraph::graph_from_adj_list(..., mode='all')` -- cell A connects to
    B if EITHER is in the other's k-NN, not only mutual neighbors), every kept edge
    given uniform weight 1 (`igraph::E(graph.knn)$weight <- 1` -- unweighted, not an
    RBF/distance kernel).

    (Minor, deliberately-not-reproduced R quirk: RANN::nn2's default query=data
    returns each cell as its own first "neighbor" at distance 0, which then gets
    stripped as a self-loop by `igraph::simplify(..., remove.loops=TRUE)`'s default
    -- net effect, R's k.knn=5 yields ~4 real distinct neighbors per cell before the
    union step. sklearn's kneighbors_graph excludes self natively, so this function's
    k=5 means 5 real distinct neighbors -- functionally equivalent, off by
    one recovered neighbor, not worth reproducing the quirk for.)

    Returns a symmetric sparse 0/1 adjacency matrix (no self-loops).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    knn = nn.kneighbors_graph(mode='connectivity')  # X omitted -> self excluded natively
    union = knn.maximum(knn.T)  # union: edge present if EITHER direction has it
    union.setdiag(0)
    union.eliminate_zeros()
    union.data[:] = 1.0
    return sp.csr_matrix(union)


def supercell_walktrap_labels(aff, target_k):
    """igraph walktrap community detection on `aff`, cut to exactly `target_k` groups
    -- `igraph::cluster_walktrap` + `igraph::cut_at(g.s, k)` in R/SCimplify.R.
    `Dendrogram.as_clustering(n=target_k)` cuts at EXACTLY target_k groups (unlike
    Leiden, which needs the over-segment+merge-down search in
    batch_correct_baselines.leiden_resolution_search to hit an exact K) -- so no
    resolution search is needed here.

    aff: symmetric sparse adjacency matrix -- SuperCell's own graph is unweighted
         (see _build_union_knn_graph), but a weighted graph works too.
    """
    import igraph as ig

    A = sp.csr_matrix(aff)
    A = (A + A.T) / 2
    A.setdiag(0)
    A.eliminate_zeros()

    n = A.shape[0]
    rows, cols = A.nonzero()
    upper = rows < cols  # one edge per undirected pair
    edges = list(zip(rows[upper].tolist(), cols[upper].tolist()))
    weights = np.asarray(A[rows[upper], cols[upper]]).ravel().tolist()

    g = ig.Graph(n=n, edges=edges)
    g.es['weight'] = weights
    if not g.is_connected():
        n_comp = len(g.connected_components())
        print(f"  [supercell] warning: kNN graph has {n_comp} connected components "
              f"(walktrap still runs, but a cut this coarse can't separate cells in "
              f"different components until very near the top of the dendrogram)")

    dendrogram = g.community_walktrap(weights='weight')
    clustering = dendrogram.as_clustering(n=target_k)
    return np.array(clustering.membership)


def run_supercell_baseline(ds_id, latent_key='native', tag=None, k_knn=5, n_pc=10,
                            n_var_genes=1000, do_scale=True, skip_if_exists=True,
                            seed=42):
    """SuperCell baseline, matching R/SCimplify.R's exact default recipe (see module
    docstring) -- cut to exactly DATASETS[ds_id]['num_prototypes'] groups.

    latent_key: 'native' (default) recomputes SuperCell's own bespoke PCA directly
        from log-normalized expression (_supercell_native_pca) -- matching R's
        `SCimplify()`, the primary e9Ho-requested comparison; deliberately NOT the
        shared 50-dim ad.obsm['X_pca'] the SEACells(PCA)/Leiden baselines reuse (see
        module docstring for why). Pass an existing obsm key (e.g. 'X_stage1z', after
        get_stage1_latent has set it on `ad`) to instead match R's
        `SCimplify_from_embedding()` -- subset to the first n_pc columns, no
        z-scoring (R doesn't rescale a caller-supplied embedding either) -- for the
        secondary "SuperCell on scProto's own corrected embedding" comparison.

    tag: folder name under MODEL_DIR/{ds_id}/ -- defaults to 'supercell' for
        latent_key='native', else f'supercell_{latent_key}'.

    Saves to MODEL_DIR/{ds_id}/{tag}/ with the same metrics.json / *_per_mc.csv /
    cell_assignments.csv layout every other baseline in this codebase uses, so it
    shows up automatically in load_task1_multi / rare_celltype_purity_table.
    """
    if tag is None:
        tag = 'supercell' if latent_key == 'native' else f'supercell_{latent_key}'

    K = DATASETS[ds_id]['num_prototypes']
    lk = DATASETS[ds_id]['label_key']
    bk = DATASETS[ds_id].get('batch_key')
    nk = DATASETS[ds_id].get('niche_key')
    save_path = os.path.join(get_dataset_model_dir(ds_id), tag)

    if skip_if_exists:
        existing = _load_metrics(save_path)
        if existing is not None and existing.get('K_target') == K:
            print(f"[{ds_id}] {tag} already computed -- skipping (metrics.json found, K matches)")
            return existing
        elif existing is not None:
            print(f"[{ds_id}] {tag} found but K_target={existing.get('K_target')} != {K} "
                  f"-- treating as stale, recomputing.")

    ad = get_preprocessed_adata(ds_id)

    if latent_key == 'native':
        print(f"[{ds_id}] {tag}: computing SuperCell's own PCA (top {n_var_genes} "
              f"var genes, scale={do_scale}, {n_pc} PCs) ...")
        knn_input = _supercell_native_pca(ad, n_var_genes=n_var_genes, n_pc=n_pc,
                                           do_scale=do_scale, seed=seed)
    else:
        if latent_key not in ad.obsm:
            raise KeyError(f"ad.obsm['{latent_key}'] not found -- compute/attach it "
                            f"before calling run_supercell_baseline(latent_key='{latent_key}').")
        knn_input = ad.obsm[latent_key][:, :n_pc]

    print(f"[{ds_id}] {tag}: building union kNN graph (k={k_knn}, unweighted) ...")
    knn_graph = _build_union_knn_graph(knn_input, k=k_knn)

    print(f"[{ds_id}] {tag}: walktrap community detection, cutting to K={K} ...")
    labels = supercell_walktrap_labels(knn_graph, target_k=K)
    n_realized = len(np.unique(labels))
    print(f"[{ds_id}] {tag}: {n_realized} groups realized (target {K})")

    mc_key = tag
    ad.obs[mc_key] = pd.Categorical(labels.astype(str))
    if 'counts' not in ad.layers:
        ad.layers['counts'] = ad.X.copy()

    mc_ad = SEACells.core.summarize_by_SEACell(ad, SEACells_label=mc_key, summarize_layer='counts')
    _agg_obs_by(mc_ad, ad, mc_key, lk)
    if bk is not None:
        _agg_obs_by(mc_ad, ad, mc_key, bk)

    os.makedirs(save_path, exist_ok=True)
    compute_task1_metrics(ad, labels, lk, bk, nk, save_path, ds_id, tag)
    recompute_modularity_canonical(ds_id, ad, labels, bk, save_path, K=K)

    sc.tl.pca(mc_ad)
    obsm_key = f'{tag}_mc_pca'
    mc_ad.obsm[obsm_key] = mc_ad.obsm['X_pca']
    task2 = calc_task2_metrics(ad, mc_ad, lk, bk, [obsm_key], tag, save_path, compute_dge=False)

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics.update({k: v for k, v in task2.items() if v is not None})
    metrics['n_clusters_realized'] = int(n_realized)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    try:
        save_baseline_umap_data(ds_id, ad, mc_key, save_path, lk, bk, nk)
    except Exception as e:
        print(f"WARNING: save_baseline_umap_data failed for {tag}: {type(e).__name__}: {e} "
              f"-- rare_celltype_purity_table won't find this run, metrics.json still saved.")

    print(f"[{ds_id}] {tag} saved to {save_path}")
    return metrics


# ---------------------------------------------------------------------------
# Metacell-2 (real `pip install metacells` package)
# ---------------------------------------------------------------------------

def _get_raw_counts_adata(ds_id):
    """Raw UMI counts, resolved via SingleCellDataset itself (use_counts=True) --
    NOT by re-reading DATASETS[ds_id]['path'] and assuming its on-disk adata.X is raw
    counts (an earlier version of this function did exactly that and broke on
    'pancreas': its on-disk X is ALREADY log-normalized, max 13.0, with real raw
    counts stored separately in adata.layers['counts']).

    SingleCellDataset.read_adata() (interpretable_ssl/datasets/dataset.py) already
    resolves this correctly and is what every other raw-counts consumer in this
    codebase relies on -- e.g. embedding_metrics.add_scvi_emb (used for the scVI
    baseline) does the equivalent `adata.X = adata.layers.get("counts", adata.X)`.
    Passing use_counts=True here reuses that exact logic: if the on-disk X is
    already normalized (max < 30), it swaps X for layers['counts'] when present; if
    the on-disk X genuinely IS raw counts, it's left as-is. Metacell-2's homogeneity
    model (rare-gene-module detection + per-gene fold-change deviant/outlier tests)
    needs the real thing either way, not log-normalized expression.
    """
    from interpretable_ssl.datasets.dataset import SingleCellDataset

    ad_pp = get_preprocessed_adata(ds_id)
    raw = SingleCellDataset(name=ds_id, use_counts=True, **DATASETS[ds_id]).adata
    raw = raw[ad_pp.obs_names].copy()
    if raw.X.max() <= 20:
        raise ValueError(
            f"[{ds_id}] SingleCellDataset(use_counts=True).adata.X still doesn't look "
            f"like raw counts (max value {raw.X.max():.2f} <= 20) even after resolving "
            f"via layers['counts'] -- this dataset may not have real raw UMI counts "
            f"preserved anywhere on disk. Available layers: {list(raw.layers.keys())}."
        )
    return ad_pp, raw


def run_metacell2_baseline(ds_id, skip_if_exists=True, max_size_search_iters=3,
                            size_tol=0.15, random_seed=42):
    """Metacell-2 baseline via the real `tanaylab/metacells` package
    (`pip install metacells`; https://github.com/tanaylab/metacells).

    Runs `metacells.pl.divide_and_conquer_pipeline` directly on raw UMI counts for
    every cell in the dataset (no upfront QC gene/cell exclusion step -- these
    datasets are already HVG-subsetted/curated by the rest of this codebase, and
    skipping Metacell-2's own `exclude_genes`/`exclude_cells` QC step, which expects
    human-gene-symbol patterns like 'MT-.*', avoids a cross-dataset assumption that
    may not hold for every dataset here).

    Metacell-2 has no direct "give me exactly K groups" knob -- group count is an
    emergent property of `target_metacell_umis` (target UMIs per group; NOT the
    similarly-named `target_metacell_size`, which as of package version 0.9.3 is a
    separate, cell-COUNT-based target the pipeline tries to satisfy simultaneously --
    passing a UMI-scale number there trips its own internal
    `assert target_metacell_size < 1000`. This function only ever sets
    `target_metacell_umis`; `target_metacell_size` is left at the package default).
    This does a small bounded search (<= max_size_search_iters attempts): start from
    target_metacell_umis = mean_UMIs_per_cell * n_cells / K (the size that would give
    K equal-UMI groups on average), then rescale target_metacell_umis by
    (realized_K / K) and retry if realized K is off by more than size_tol (15%,
    looser than SEACells' 5% tolerance elsewhere in this codebase -- Metacell-2's
    group count is not a hard target the way SEACells'/Leiden's is, so exact
    convergence isn't guaranteed or expected -- it's also nudged by the
    cell-count-based target_metacell_size running alongside it, which this function
    doesn't control). Whatever the last iteration produces is kept and its realized K
    reported honestly in metrics.json, same as the "K mismatch" pattern already used
    for Harmony/BBKNN elsewhere in this codebase.

    Cells Metacell-2 marks as outliers (obs['metacell'] == -1, its own deviant/rare-
    cell detection declining to group them) are each given their OWN singleton
    metacell rather than being dropped -- compute_task1_metrics/calc_task2_metrics
    assume every cell has a group id, and a method declining to pool a cell is itself
    informative about its within-batch aggregation limits, not something to hide by
    excluding those cells from the denominator.
    """
    K = DATASETS[ds_id]['num_prototypes']
    lk = DATASETS[ds_id]['label_key']
    bk = DATASETS[ds_id].get('batch_key')
    nk = DATASETS[ds_id].get('niche_key')
    tag = 'metacell2'
    save_path = os.path.join(get_dataset_model_dir(ds_id), tag)

    if skip_if_exists:
        existing = _load_metrics(save_path)
        if existing is not None:
            print(f"[{ds_id}] {tag} already computed -- skipping (metrics.json found, "
                  f"realized K={existing.get('n_clusters_realized')}, target K={K})")
            assign_path = os.path.join(save_path, 'cell_assignments.csv')
            if os.path.exists(assign_path):
                ad_pp = get_preprocessed_adata(ds_id)
                assign_df = pd.read_csv(assign_path).set_index('cell_id')
                mc_idx_reload = assign_df.loc[ad_pp.obs_names, 'metacell_id'].values.astype(int)
                existing = recompute_modularity_canonical(ds_id, ad_pp, mc_idx_reload, bk, save_path, K=K)
            return existing

    import metacells as mc

    ad_pp, raw = _get_raw_counts_adata(ds_id)
    total_umis = np.asarray(raw.X.sum(axis=1)).ravel()
    mean_umis = float(total_umis.mean())
    n_cells = raw.n_obs
    target_umis = mean_umis * n_cells / K

    labels = None
    n_realized = None
    for it in range(max_size_search_iters):
        full = raw.copy()
        mc.ut.set_name(full, f'{ds_id}_metacell2_iter{it}')

        # divide_and_conquer_pipeline's internal gene-selection step
        # (metacells/pipeline/select.py::extract_selected_data) hardcodes
        # additional_gene_masks=["&~lateral_gene"] as its DEFAULT -- i.e. it
        # unconditionally requires an adata.var['lateral_gene'] boolean mask to
        # exist, regardless of whether any genes should actually be marked lateral
        # (confirmed by reading that source directly: combine_masks() raises
        # KeyError('unknown mask data: lateral_gene') if it's missing, despite the
        # function's own docstring saying this annotation is only used "if it
        # exists"). mark_lateral_genes() with no name/pattern args creates the mask
        # as all-False (tl.find_named_genes with names=None, patterns=None matches
        # nothing) -- satisfies the requirement without hardcoding any
        # dataset/species-specific gene list, which we deliberately don't do (see
        # this function's docstring on skipping the QC step generically). noisy_gene
        # is NOT required the same way -- every reference to it elsewhere in the
        # package (metacells/tools/deviants.py) is guarded by
        # `if ut.has_data(adata, "noisy_gene")`, so its absence is a genuine no-op --
        # marked anyway (also all-False) purely for parity with the package's own
        # documented one-pass recipe, not because omitting it would crash.
        mc.pl.mark_lateral_genes(full)
        mc.pl.mark_noisy_genes(full)

        try:
            max_piles = mc.pl.guess_max_parallel_piles(full)
            mc.pl.set_max_parallel_piles(max_piles)
        except Exception as e:
            print(f"  [metacell2] guess_max_parallel_piles failed ({type(e).__name__}: {e}) "
                  f"-- continuing with the package default.")

        print(f"[{ds_id}] metacell2 size-search iter {it}: target_metacell_umis={target_umis:.0f} ...")
        # Deliberately NOT wrapped in mc.ut.progress_bar(): that context manager's
        # own start_progress_bar()/end_progress_bar() pair (metacells/utilities/
        # progress.py) has a real bug -- end_progress_bar() only resets its
        # PROGRESS_BAR global, never TQDM_KWARGS, so if divide_and_conquer_pipeline
        # raises before the progress bar's first update (e.g. a crash early in rare-
        # gene-module detection, exactly what happened here before the read-only-
        # array patch above), TQDM_KWARGS is left non-None -- permanently, for the
        # rest of the Python process -- and every later start_progress_bar() call
        # (i.e. every retry in this very loop, or any other metacells call in this
        # notebook) then fails `assert TQDM_KWARGS is None` immediately, regardless
        # of whether THIS run would otherwise succeed. Cosmetic feature, not worth
        # the fragility -- our own per-iteration print() above already reports
        # progress. Defensively clear any such leaked state up front too, so a
        # kernel that already hit this in an earlier attempt self-heals here
        # without needing a restart.
        import metacells.utilities.progress as _mc_progress
        _mc_progress.PROGRESS_BAR = None
        _mc_progress.TQDM_KWARGS = None
        mc.pl.divide_and_conquer_pipeline(full, random_seed=random_seed,
                                           target_metacell_umis=int(round(target_umis)))

        iter_labels = full.obs['metacell'].to_numpy().astype(int)
        n_realized = len(np.unique(iter_labels[iter_labels >= 0]))
        n_outliers = int((iter_labels < 0).sum())
        print(f"  [metacell2] -> {n_realized} metacells + {n_outliers} outliers "
              f"(target {K}, tolerance +/-{size_tol:.0%})")
        labels = iter_labels

        if abs(n_realized - K) <= size_tol * K:
            break
        target_umis = target_umis * (n_realized / K)

    outlier_mask = labels < 0
    n_assigned = int(labels.max()) + 1 if (~outlier_mask).any() else 0
    labels = labels.copy()
    labels[outlier_mask] = n_assigned + np.arange(outlier_mask.sum())

    ad_pp.obs[tag] = pd.Categorical(labels.astype(str))
    if 'counts' not in ad_pp.layers:
        ad_pp.layers['counts'] = ad_pp.X.copy()

    mc_ad = SEACells.core.summarize_by_SEACell(ad_pp, SEACells_label=tag, summarize_layer='counts')
    _agg_obs_by(mc_ad, ad_pp, tag, lk)
    if bk is not None:
        _agg_obs_by(mc_ad, ad_pp, tag, bk)

    os.makedirs(save_path, exist_ok=True)
    compute_task1_metrics(ad_pp, labels, lk, bk, nk, save_path, ds_id, tag)
    recompute_modularity_canonical(ds_id, ad_pp, labels, bk, save_path, K=K)

    sc.tl.pca(mc_ad)
    obsm_key = f'{tag}_mc_pca'
    mc_ad.obsm[obsm_key] = mc_ad.obsm['X_pca']
    task2 = calc_task2_metrics(ad_pp, mc_ad, lk, bk, [obsm_key], tag, save_path, compute_dge=False)

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics.update({k: v for k, v in task2.items() if v is not None})
    metrics['n_clusters_realized'] = int(n_realized)
    metrics['n_outliers'] = int(outlier_mask.sum())
    metrics['target_metacell_umis'] = float(target_umis)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    try:
        save_baseline_umap_data(ds_id, ad_pp, tag, save_path, lk, bk, nk)
    except Exception as e:
        print(f"WARNING: save_baseline_umap_data failed for {tag}: {type(e).__name__}: {e} "
              f"-- rare_celltype_purity_table won't find this run, metrics.json still saved.")

    print(f"[{ds_id}] {tag} saved to {save_path}")
    return metrics


def run_extra_baselines_for_dataset(ds_id, skip_if_exists=True, run_supercell=True,
                                     run_metacell2=True):
    """Convenience wrapper: run SuperCell (PCA) + Metacell-2 for one dataset, matching
    the run_all_baselines_for_dataset pattern in batch_correct_baselines.py.

    Each baseline runs in its own try/except: SuperCell and Metacell-2 are
    independent computations (different packages, different failure modes), so one
    raising (e.g. Metacell-2 hitting a version-specific bug in a third-party
    package -- see extra_metacell_baselines.py's own patches for two already found
    and fixed) must not discard the other's already-computed result, or force a
    full dataset re-run once it's fixed. `results[name]` holds the exception object
    itself (not the metrics dict) for whichever baseline failed, so the return
    value always tells you unambiguously which of the two succeeded.

    For the secondary "SuperCell on scProto's Stage-1 embedding" comparison, call
    run_supercell_baseline(ds_id, latent_key='X_stage1z', ...) directly after
    get_stage1_latent(ds_id) has attached 'X_stage1z' to that same `ad` -- not
    wired through this wrapper since it needs the Stage-1 checkpoint loaded first
    (see the notebook's own cell for this).
    """
    results = {}
    if run_supercell:
        try:
            results['supercell'] = run_supercell_baseline(ds_id, latent_key='native', skip_if_exists=skip_if_exists)
        except Exception as e:
            import traceback
            print(f"[{ds_id}] supercell FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results['supercell'] = e
    if run_metacell2:
        try:
            results['metacell2'] = run_metacell2_baseline(ds_id, skip_if_exists=skip_if_exists)
        except Exception as e:
            import traceback
            print(f"[{ds_id}] metacell2 FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results['metacell2'] = e
    return results

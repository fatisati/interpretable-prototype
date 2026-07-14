"""
niche_composition_comparison.py — compare neighborhood representations against
the cell-type composition vector (as used in Pentimalli et al. NSCL paper).

For each cell, its spatial neighborhood defines a "niche" via the cell-type
composition vector (fraction of each cell type among k spatial neighbors).
We compare four neighborhood embeddings:

  V1 — mean PCA of neighbors            1st moment, continuous
  V2 — COVET of neighbors               2nd moment (covariance), continuous
  V3 — soft-cluster avg of neighbors    1st moment, discretized via learned prototypes
  V4 — concat(V1, V2), var-balanced     1st + 2nd moment combined (theoretically best)

Three comparison metrics (all against the composition vector as ground truth):
  kNN label purity      — cluster composition vectors (k-means), measure fraction of
                          kNN in rep-space sharing the same composition cluster
  composition kNN Jaccard — Jaccard overlap between kNN-in-composition-space and
                            kNN-in-rep-space (neighbor identity recovery)
  composition cosine sim  — cosine similarity between composition profiles of
                            kNN-in-rep-space neighbors (neighborhood profile coherence)
Plus silhouette score in rep-space using composition-cluster labels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import scanpy as sc


# ---------------------------------------------------------------------------
# Step 1: cell-type composition vector (ground-truth niche representation)
# ---------------------------------------------------------------------------

def compute_celltype_composition(ad, k=35, celltype_key='celltypes',
                                  batch_key=None, radius=None):
    """For each cell, fraction of each cell type among its spatial neighbors.

    Two modes (mutually exclusive — radius takes priority if given):
      kNN mode    (radius=None): k nearest spatial neighbors, fixed count per cell.
      radius mode (radius>0):    all neighbors within `radius` coordinate units.
                                 The padded I array still uses -1 for missing slots
                                 so all downstream V1–V4 functions work unchanged.
                                 Prints median / mean / min / max neighbor count.

    Args:
        ad:           AnnData with obsm['spatial'] and obs[celltype_key]
        k:            kNN mode only — number of spatial neighbors (excludes self)
        celltype_key: obs column with cell type labels
        batch_key:    restrict neighbors to same batch/section
        radius:       radius mode — distance threshold in coordinate units
                      (e.g. 12.57 = 50 µm for NSCLC CosMx; set SPATIAL_RADIUS in notebook)

    Returns:
        comp_df: pd.DataFrame (N × C) — fraction of each cell type. Rows sum to 1.
        I:       (N, max_k) int array — neighbor indices, -1 for empty slots.
                 max_k = k in kNN mode; max neighbors across all cells in radius mode.
    """
    cell_types = ad.obs[celltype_key].astype(str).values
    unique_types = sorted(set(cell_types))
    ct_to_idx = {ct: i for i, ct in enumerate(unique_types)}
    ct_codes = np.array([ct_to_idx[ct] for ct in cell_types], dtype=np.int32)
    n_types = len(unique_types)
    n_cells = len(ad)
    spatial = ad.obsm['spatial'][:, :2].astype(np.float32)

    if radius is not None:
        # --- radius mode ---
        nn = NearestNeighbors(radius=radius, algorithm='ball_tree').fit(spatial)
        _, idx_list = nn.radius_neighbors(spatial, return_distance=False)
        # exclude self
        idx_list = [nbrs[nbrs != i].astype(np.int32) for i, nbrs in enumerate(idx_list)]

        counts = np.array([len(nbrs) for nbrs in idx_list])
        print(f"  [composition radius={radius}]  "
              f"median={np.median(counts):.0f}  mean={counts.mean():.1f}  "
              f"min={counts.min()}  max={counts.max()}  "
              f"cells_with_0={(counts == 0).sum()}")

        max_k = int(counts.max()) if counts.max() > 0 else 1
        I = np.full((n_cells, max_k), -1, dtype=np.int32)
        comp = np.zeros((n_cells, n_types), dtype=np.float32)
        for i, nbrs in enumerate(idx_list):
            if len(nbrs) == 0:
                continue
            I[i, :len(nbrs)] = nbrs
            for t in ct_codes[nbrs]:
                comp[i, t] += 1

    elif batch_key is None:
        # --- kNN mode, single batch ---
        import faiss
        index = faiss.IndexFlatL2(2)
        index.add(spatial)
        _, I = index.search(spatial, k + 1)
        I = I[:, 1:].astype(np.int32)  # exclude self

        comp = np.zeros((n_cells, n_types), dtype=np.float32)
        for i in range(n_cells):
            for t in ct_codes[I[i]]:
                comp[i, t] += 1

    else:
        # --- kNN mode, per-batch ---
        import faiss
        batches = ad.obs[batch_key].values
        I = np.full((n_cells, k), -1, dtype=np.int32)
        comp = np.zeros((n_cells, n_types), dtype=np.float32)
        for b in np.unique(batches):
            idx = np.where(batches == b)[0]
            if len(idx) <= 1:
                continue
            sp_b = spatial[idx]
            index = faiss.IndexFlatL2(2)
            index.add(sp_b)
            ki = min(k + 1, len(idx))
            _, I_b = index.search(sp_b, ki)
            I_b = I_b[:, 1:]
            for local_i, global_i in enumerate(idx):
                nbr_local = I_b[local_i]
                nbr_global = idx[nbr_local[nbr_local < len(idx)]]
                n_take = min(k, len(nbr_global))
                I[global_i, :n_take] = nbr_global[:n_take]
                for t in ct_codes[nbr_global[:n_take]]:
                    comp[global_i, t] += 1

    comp /= comp.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(comp, index=ad.obs_names, columns=unique_types), I


# ---------------------------------------------------------------------------
# Step 2: cluster composition vectors → niche labels (replicates paper)
# ---------------------------------------------------------------------------

def cluster_composition_vectors(comp_df, n_clusters=10, random_state=0):
    """K-means on cell-type composition vectors → niche cluster labels.

    Args:
        comp_df:    (N, C) DataFrame from compute_celltype_composition
        n_clusters: k for k-means (paper uses 10)

    Returns:
        labels:    (N,) int cluster assignments
        centroids: (n_clusters, C) cluster centroids
    """
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = km.fit_predict(comp_df.values)
    return labels, km.cluster_centers_


# ---------------------------------------------------------------------------
# Step 3: neighborhood embeddings — V1, V2, V3, V4
# ---------------------------------------------------------------------------

def _mask_avg(X, I):
    """Mean of X[I] along axis 1, masking out -1 padding entries in I."""
    valid = I >= 0
    I_safe = np.where(valid, I, 0)
    X_nbr = X[I_safe]                                           # (N, k, d)
    count = valid.sum(axis=1, keepdims=True).clip(min=1)[:, :, None]  # (N, 1, 1)
    return (X_nbr * valid[:, :, None]).sum(axis=1) / count.squeeze(-1)  # (N, d)


def compute_mean_pca_neighbors(ad, I):
    """V1 — mean PCA of spatial neighbors (continuous 1st moment).

    Args:
        ad: AnnData with obsm['X_pca']
        I:  (N, k) neighbor index array from compute_celltype_composition

    Returns:
        (N, d_pca) float array
    """
    return _mask_avg(ad.obsm['X_pca'], I)


def compute_covet_for_neighbors(ad, I, n_pcs=10):
    """V2 — COVET: flattened upper-triangle covariance of neighbor PCA, PCA-reduced.

    2nd moment of the neighborhood PCA distribution. Reuses the spatial kNN index I
    so we don't repeat the expensive faiss search.

    Args:
        ad:    AnnData with obsm['X_pca']
        I:     (N, k) neighbor index array (same as used for composition)
        n_pcs: PCA dims used as covariance input (require k_spatial >= 3*n_pcs)

    Returns:
        (N, n_pcs*(n_pcs+1)//2 reduced to ≤n_pca_own dims) float array
    """
    import anndata as ann

    X_pca_full = ad.obsm['X_pca'][:, :n_pcs]
    k = I.shape[1]
    valid = I >= 0
    I_safe = np.where(valid, I, 0)

    X_nbr = X_pca_full[I_safe]                                       # (N, k, n_pcs)
    count = valid.sum(axis=1, keepdims=True).clip(min=1)             # (N, 1)
    mu = (X_nbr * valid[:, :, None]).sum(axis=1, keepdims=True) / count[:, :, None]
    X_centered = (X_nbr - mu) * valid[:, :, None]

    denom = (count - 1).clip(min=1)[:, :, None]                               # (N, 1, 1)
    cov = np.einsum('ijk,ijl->ikl', X_centered, X_centered) / denom           # (N, p, p)
    ti, tj = np.triu_indices(n_pcs)
    cov_flat = cov[:, ti, tj]                                        # (N, p*(p+1)/2)

    n_comps = min(ad.obsm['X_pca'].shape[1], cov_flat.shape[1] - 1)
    tmp = ann.AnnData(cov_flat.astype(np.float32))
    sc.tl.pca(tmp, n_comps=n_comps)
    return tmp.obsm['X_pca']                                         # (N, n_comps)


def compute_soft_cluster_avg(ad, I, n_proto=30, tau=None, random_state=0):
    """V3 — soft-cluster average of spatial neighbors (discretized 1st moment).

    Algorithm:
      1. Fit k-means(n_proto) globally on all cells' X_pca → prototype centroids C
      2. Assign each cell a soft membership vector:
             s_i = softmax(-||x_i - C||² / tau)   shape (n_proto,)
         where tau is the mean nearest-center distance (adaptive bandwidth).
      3. For each cell's spatial neighborhood, average its neighbors' soft vectors:
             v3_i = mean_{j ∈ nbrs(i)} s_j         shape (n_proto,)

    This is a *discretized* first moment: instead of averaging raw PCA coordinates
    (V1), we average probability mass over a learned vocabulary of expression states.
    Combined with V2 (COVET, 2nd moment), it gives V4.

    Args:
        ad:          AnnData with obsm['X_pca']
        I:           (N, k) spatial neighbor index array
        n_proto:     number of prototypes / soft clusters
        tau:         softmax temperature (None → adaptive: mean nearest-center dist)
        random_state: for k-means reproducibility

    Returns:
        (N, n_proto) float array — soft-cluster histogram of spatial neighborhood
    """
    X_pca = ad.obsm['X_pca']

    km = KMeans(n_clusters=n_proto, n_init=10, random_state=random_state)
    km.fit(X_pca)
    C = km.cluster_centers_                                 # (n_proto, d)

    diff = X_pca[:, None, :] - C[None, :, :]               # (N, n_proto, d)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)          # (N, n_proto)

    if tau is None:
        tau = float(np.sqrt(dist_sq.min(axis=1)).mean())
        tau = max(tau, 1e-6)
        print(f"  [soft_cluster_avg] adaptive tau={tau:.4f}  n_proto={n_proto}")

    # softmax soft assignments
    log_s = -dist_sq / tau
    log_s -= log_s.max(axis=1, keepdims=True)               # numerical stability
    s = np.exp(log_s)
    s /= s.sum(axis=1, keepdims=True)                       # (N, n_proto)

    # average over spatial neighbors
    return _mask_avg(s, I)                                  # (N, n_proto)



def compute_concat_mean_covet(X_v1, X_v2, alpha=0.5):
    """V4 — variance-balanced concat of V1 (mean PCA) and V2 (COVET).

    1st moment + 2nd moment of the neighborhood distribution.
    Uses BANKSY-style lambda balancing so neither part dominates by variance:

        lambda = sqrt(alpha * V1 / ((1-alpha) * V2 + alpha * V1))
        X_v4  = concat([sqrt(1-lambda²) * X_v1,  lambda * X_v2])

    After scaling, var(X_v4) = V1 * alpha/V1 * ... = exactly alpha * (V1+V2)
    from V1 and (1-alpha)*(V1+V2) from V2, regardless of original scale.

    Args:
        X_v1:  (N, d1) mean-PCA embedding
        X_v2:  (N, d2) COVET embedding
        alpha: target variance fraction from V1 [0, 1]. Default 0.5 = equal weight.

    Returns:
        (N, d1+d2) array — variance-balanced concatenation
    """
    V1 = float(X_v1.var(axis=0).sum())
    V2 = float(X_v2.var(axis=0).sum())

    if V1 < 1e-12 or V2 < 1e-12:
        print("  [concat_mean_covet] WARNING: near-zero variance in one part, returning raw concat")
        return np.concatenate([X_v1, X_v2], axis=1)

    lam = np.sqrt(alpha * V1 / ((1 - alpha) * V2 + alpha * V1))
    X_v4 = np.concatenate([
        np.sqrt(1 - lam ** 2) * X_v1,
        lam * X_v2,
    ], axis=1)
    print(f"  [concat_mean_covet] V_v1={V1:.3f}  V_v2={V2:.3f}  lambda={lam:.3f}  "
          f"alpha={alpha}  out_shape={X_v4.shape}")
    return X_v4


# ---------------------------------------------------------------------------
# Step 4: comparison metrics (generic over any set of representations)
# ---------------------------------------------------------------------------

def knn_label_purity(X, labels, k=15):
    """Fraction of k nearest neighbors in X that share the same label as self.

    Args:
        X:      (N, d) embedding
        labels: (N,) labels
        k:      neighbors (excluding self)

    Returns:
        (N,) float
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idxs = nn.kneighbors(X)
    idxs = idxs[:, 1:]
    labels = np.asarray(labels)
    return (labels[idxs] == labels[:, None]).mean(axis=1)


def composition_knn_recovery(X_rep, comp_vecs, k=15):
    """Jaccard overlap between kNN-in-composition-space and kNN-in-rep-space.

    High = rep recovers the same neighbors as the ground-truth composition vector.

    Returns:
        (N,) float
    """
    nn_c = NearestNeighbors(n_neighbors=k + 1).fit(comp_vecs)
    _, I_c = nn_c.kneighbors(comp_vecs)
    I_c = I_c[:, 1:]

    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(X_rep)
    _, I_r = nn_r.kneighbors(X_rep)
    I_r = I_r[:, 1:]

    return np.array([
        len(set(I_c[i]) & set(I_r[i])) / len(set(I_c[i]) | set(I_r[i]))
        for i in range(len(X_rep))
    ])


def composition_cosine_sim(X_rep, comp_vecs, k=15):
    """Mean cosine similarity between composition profiles of kNN-in-rep-space.

    High = cells grouped together by the rep also have similar composition profiles.

    Returns:
        (N,) float
    """
    from sklearn.preprocessing import normalize
    comp_norm = normalize(comp_vecs, norm='l2')

    nn_r = NearestNeighbors(n_neighbors=k + 1).fit(X_rep)
    _, I_r = nn_r.kneighbors(X_rep)
    I_r = I_r[:, 1:]

    sims = (comp_norm[:, None, :] * comp_norm[I_r]).sum(axis=2)  # (N, k)
    return sims.mean(axis=1)


def _metrics_for_rep(X_rep, comp_labels, comp_vecs, k_eval, random_state):
    """Compute all four metrics for one representation."""
    purity  = knn_label_purity(X_rep, comp_labels, k=k_eval)
    jaccard = composition_knn_recovery(X_rep, comp_vecs, k=k_eval)
    cos_sim = composition_cosine_sim(X_rep, comp_vecs, k=k_eval)
    try:
        sil = silhouette_score(X_rep, comp_labels, metric='euclidean',
                               sample_size=5000, random_state=random_state)
    except Exception:
        sil = np.nan
    return purity, jaccard, cos_sim, sil


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_neighborhood_reps(
    ad,
    k_spatial=35,
    celltype_key='celltypes',
    niche_key=None,
    n_clusters=10,
    n_pcs_covet=10,
    n_proto=30,
    concat_alpha=0.5,
    k_eval=15,
    batch_key=None,
    spatial_radius=None,
    random_state=0,
    verbose=True,
):
    """Compare V1–V4 neighborhood representations against a ground-truth niche label.

    V1 — mean PCA of neighbors           (continuous 1st moment)
    V2 — COVET of neighbors              (continuous 2nd moment)
    V3 — soft-cluster avg of neighbors   (discretized 1st moment, soft k-means)
    V4 — concat(V1, V2), var-balanced    (1st + 2nd moment, theoretically best)

    Args:
        ad:             AnnData with obsm['spatial'], obsm['X_pca'], obs[celltype_key]
        k_spatial:      kNN spatial neighbors (used when spatial_radius is None)
        celltype_key:   obs column with cell type labels (used to build V1–V4)
        niche_key:      if set, use ad.obs[niche_key] as ground-truth labels instead of
                        k-means on composition vectors. Jaccard and cosine metrics will
                        use one-hot niche encoding as the ground-truth vectors.
        n_clusters:     k-means clusters on composition vectors (used only if niche_key=None)
        n_pcs_covet:    PCA dims used as COVET covariance input
        n_proto:        soft prototypes for V3 (default 30)
        concat_alpha:   V4 variance weight for V1 side (0.5 = equal)
        k_eval:         kNN k for purity and Jaccard metrics
        batch_key:      if set, restrict spatial neighbors within batches
        spatial_radius: if set, use radius-based neighbors instead of kNN
                        (e.g. 12.57 coordinate units = 50 µm for NSCLC CosMx)
        random_state:   for k-means

    Returns:
        dict with keys:
            'comp_df'      — (N, C) cell-type composition DataFrame
            'comp_labels'  — (N,) ground-truth labels (niche or composition k-means)
            'centroids'    — (n_clusters, C) composition cluster centroids (None if niche_key set)
            'reps'         — dict: name → (N, d) embedding
            'metrics'      — pd.DataFrame: rows=metrics, cols=rep names
            'per_cell'     — dict: name → {'purity', 'jaccard', 'cos_sim'} arrays
    """
    assert 'spatial' in ad.obsm, "Need ad.obsm['spatial']"
    assert 'X_pca'   in ad.obsm, "Need ad.obsm['X_pca']"

    if spatial_radius is None:
        assert k_spatial >= 3 * n_pcs_covet, (
            f"k_spatial={k_spatial} < 3*n_pcs_covet={3*n_pcs_covet}: COVET covariance unstable"
        )

    nbr_desc = f"radius={spatial_radius}" if spatial_radius is not None else f"k={k_spatial}"
    gt_desc = f"niche={niche_key}" if niche_key else f"n_clusters={n_clusters}"
    if verbose:
        print(f"[compare_neighborhood_reps]  n={len(ad)}  neighbors={nbr_desc}  "
              f"ground_truth={gt_desc}  n_pcs_covet={n_pcs_covet}  "
              f"n_proto={n_proto}  concat_alpha={concat_alpha}  k_eval={k_eval}")

    # 1 — composition vectors + spatial neighbor index I (shared across all reps)
    if verbose:
        print("  step 1/6: cell-type composition vectors ...")
    comp_df, I = compute_celltype_composition(
        ad, k=k_spatial, celltype_key=celltype_key,
        batch_key=batch_key, radius=spatial_radius,
    )

    # 2 — ground-truth labels: known niche labels OR k-means on composition
    centroids = None
    if niche_key is not None:
        if verbose:
            print(f"  step 2/6: using known niche labels from obs['{niche_key}'] ...")
        niche_raw = ad.obs[niche_key].fillna('unknown').astype(str).values
        unique_niches = sorted(set(niche_raw))
        niche_to_idx = {n: i for i, n in enumerate(unique_niches)}
        gt_labels = np.array([niche_to_idx[n] for n in niche_raw], dtype=np.int32)
        # one-hot niche vectors as ground-truth for Jaccard / cosine metrics
        gt_vecs = np.zeros((len(ad), len(unique_niches)), dtype=np.float32)
        gt_vecs[np.arange(len(ad)), gt_labels] = 1.0
        gt_label_name = niche_key
    else:
        if verbose:
            print(f"  step 2/6: k-means(k={n_clusters}) on composition ...")
        gt_labels, centroids = cluster_composition_vectors(comp_df, n_clusters=n_clusters,
                                                           random_state=random_state)
        gt_vecs = comp_df.values
        gt_label_name = "composition clusters"

    # 3 — compute all four embeddings
    if verbose:
        print("  step 3/6: V1 (mean PCA of neighbors) ...")
    X_v1 = compute_mean_pca_neighbors(ad, I)

    if verbose:
        print(f"  step 4/6: V2 (COVET, n_pcs={n_pcs_covet}) ...")
    X_v2 = compute_covet_for_neighbors(ad, I, n_pcs=n_pcs_covet)

    if verbose:
        print(f"  step 5/6: V3 (soft-cluster avg, n_proto={n_proto}) ...")
    X_v3 = compute_soft_cluster_avg(ad, I, n_proto=n_proto, random_state=random_state)

    if verbose:
        print(f"  step 6/6: V4 (concat V1+V2, alpha={concat_alpha}) ...")
    X_v4 = compute_concat_mean_covet(X_v1, X_v2, alpha=concat_alpha)

    reps = {
        'V1 mean-PCA':     X_v1,
        'V2 COVET':        X_v2,
        'V3 soft-cluster': X_v3,
        'V4 mean+COVET':   X_v4,
    }

    # 4 — compute metrics for each rep
    if verbose:
        print("  metrics ...")
    metric_names = [
        f'kNN label purity ({gt_label_name})',
        f'kNN Jaccard ({gt_label_name})',
        f'cosine similarity ({gt_label_name})',
        f'silhouette score ({gt_label_name})',
    ]
    per_cell = {}
    rows = {}
    for name, X_rep in reps.items():
        pur, jac, cos, sil = _metrics_for_rep(X_rep, gt_labels, gt_vecs,
                                               k_eval, random_state)
        rows[name] = [pur.mean(), jac.mean(), cos.mean(), sil]
        per_cell[name] = {'purity': pur, 'jaccard': jac, 'cos_sim': cos}

    metrics = pd.DataFrame(rows, index=metric_names)

    if verbose:
        print("\n--- Results ---")
        print(metrics.round(4).to_string())
        best = metrics.mean(axis=0).idxmax()
        print(f"\nBest overall (mean across metrics): {best}")

    return {
        'comp_df':     comp_df,
        'comp_labels': gt_labels,
        'centroids':   centroids,
        'reps':        reps,
        'metrics':     metrics,
        'per_cell':    per_cell,
    }


# ---------------------------------------------------------------------------
# Per-cell-type breakdown
# ---------------------------------------------------------------------------

def per_celltype_metrics(results, ad, celltype_key='celltypes'):
    """Per-cell-type mean of purity and Jaccard for each representation.

    Returns:
        pd.DataFrame: rows=cell types, columns=V1_purity, V2_purity, ... V1_jaccard, ...
                      plus delta columns (V4 - V1) to highlight cell types that benefit most
    """
    ct = ad.obs[celltype_key].astype(str).values
    rep_names = list(results['reps'].keys())
    rows = []
    for t in sorted(set(ct)):
        mask = ct == t
        row = {'cell_type': t, 'n_cells': int(mask.sum())}
        for name in rep_names:
            row[f'{name}_purity']  = float(results['per_cell'][name]['purity'][mask].mean())
            row[f'{name}_jaccard'] = float(results['per_cell'][name]['jaccard'][mask].mean())
        rows.append(row)

    df = pd.DataFrame(rows).set_index('cell_type')
    # delta: best (V4) minus cheapest (V1)
    if 'V4 mean+COVET_purity' in df.columns and 'V1 mean-PCA_purity' in df.columns:
        df['delta_purity']  = df['V4 mean+COVET_purity']  - df['V1 mean-PCA_purity']
        df['delta_jaccard'] = df['V4 mean+COVET_jaccard'] - df['V1 mean-PCA_jaccard']
    return df.sort_values('delta_purity', ascending=False)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

REP_COLORS = {
    'V1 mean-PCA':     '#4C8BE0',   # blue
    'V2 COVET':        '#E05C4C',   # red
    'V3 soft-cluster': '#27AE60',   # green
    'V4 mean+COVET':   '#8E44AD',   # purple — theoretically best
}


def plot_comparison(results, ad=None, celltype_key='celltypes',
                    figsize_bars=(12, 4), figsize_violin=(12, 4)):
    """Bar chart of summary metrics + violin of per-cell distributions.

    Args:
        results:      output dict from compare_neighborhood_reps
        ad:           if provided, also show per-cell-type breakdown
        celltype_key: obs column for per-cell-type breakdown
    """
    metrics  = results['metrics']
    rep_names = list(results['reps'].keys())
    colors   = [REP_COLORS.get(n, '#888888') for n in rep_names]

    # --- 1. Summary bar chart per metric ---
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize_bars)
    if n_metrics == 1:
        axes = [axes]
    for ax, (metric_name, row) in zip(axes, metrics.iterrows()):
        vals = [row[n] for n in rep_names]
        bars = ax.bar(range(len(rep_names)), vals, color=colors,
                      edgecolor='white', width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(range(len(rep_names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in rep_names], fontsize=8)
        ax.set_title(metric_name, fontsize=8)
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.25 + 0.02)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Neighborhood representation vs cell-type composition', fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- 2. Per-cell violin for the three non-silhouette metrics ---
    violin_pairs = [
        ('purity',  'kNN purity\n(composition clusters)'),
        ('jaccard', 'composition kNN Jaccard'),
        ('cos_sim', 'composition cosine sim'),
    ]
    fig, axes = plt.subplots(1, len(violin_pairs), figsize=figsize_violin)
    for ax, (metric_key, title) in zip(axes, violin_pairs):
        data = [results['per_cell'][n][metric_key] for n in rep_names]
        parts = ax.violinplot(data, positions=range(len(rep_names)),
                              showmedians=True, showextrema=False)
        for pc, col in zip(parts['bodies'], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color('black')
        ax.set_xticks(range(len(rep_names)))
        ax.set_xticklabels([n.replace(' ', '\n') for n in rep_names], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis='y', alpha=0.3)
        for pos, vals in enumerate(data):
            ax.text(pos, np.median(vals) + 0.02, f'{np.median(vals):.3f}',
                    ha='center', fontsize=8, fontweight='bold')
    plt.suptitle('Per-cell distributions', fontsize=11)
    plt.tight_layout()
    plt.show()

    # --- 3. Per-cell-type breakdown ---
    if ad is not None:
        ct_df = per_celltype_metrics(results, ad, celltype_key=celltype_key)
        _plot_per_celltype(ct_df, rep_names, colors)


def _plot_per_celltype(ct_df, rep_names, colors):
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(ct_df) * 0.45 + 1)))
    pairs = [
        ([f'{n}_purity'  for n in rep_names], 'kNN purity by cell type'),
        ([f'{n}_jaccard' for n in rep_names], 'composition Jaccard by cell type'),
    ]
    for ax, (col_names, title) in zip(axes, pairs):
        y = np.arange(len(ct_df))
        w = 0.8 / len(rep_names)
        offsets = np.linspace(-(len(rep_names)-1)/2, (len(rep_names)-1)/2, len(rep_names)) * w
        for offset, col, color, rname in zip(offsets, col_names, colors, rep_names):
            vals = ct_df[col].values
            ax.barh(y + offset, vals, w * 0.9, label=rname,
                    color=color, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(ct_df.index, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3)
    plt.suptitle('Per-cell-type: V1 vs V2 vs V3 vs V4', fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_composition_heatmap(results, ad, celltype_key='celltypes', figsize=(10, 5)):
    """Heatmap of mean cell-type composition per cluster (replicates NSCL paper Fig 2c)."""
    comp_df  = results['comp_df']
    labels   = results['comp_labels']
    n_clusters = len(np.unique(labels))

    cluster_means = pd.DataFrame(
        results['centroids'],
        columns=comp_df.columns,
        index=[f'Cluster {i}' for i in range(n_clusters)],
    )
    order = cluster_means.sort_values(cluster_means.idxmax(axis=1)
                                       .value_counts().index[0],
                                       ascending=False).index

    fig, ax = plt.subplots(figsize=figsize)
    try:
        import seaborn as sns
        sns.heatmap(cluster_means.loc[order], ax=ax, cmap='Blues',
                    annot=True, fmt='.2f', linewidths=0.3)
    except ImportError:
        im = ax.imshow(cluster_means.loc[order].values, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(comp_df.columns)))
        ax.set_xticklabels(comp_df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels(order, fontsize=9)
        plt.colorbar(im, ax=ax)

    for i, cluster_name in enumerate(order):
        cid = int(cluster_name.split()[-1])
        ax.text(len(comp_df.columns) + 0.1, i, f'n={(labels==cid).sum()}',
                va='center', fontsize=7)

    ax.set_title('Cell-type composition per cluster\n(replicating NSCL paper Fig 2c)', fontsize=11)
    ax.set_xlabel('Cell type')
    ax.set_ylabel('Composition cluster')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return cluster_means

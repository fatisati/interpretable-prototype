import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import silhouette_score


def weighted_purity(aff, labels):
    """Affinity-weighted label purity per cell.

    For each cell i, computes sum(w_ij * same_label(i,j)) / sum(w_ij) over all
    nonzero neighbors j. Higher-weight neighbors contribute proportionally more,
    which faithfully reflects how the affinity is used during aggregation (SEACells
    does not binarize — it uses raw weights throughout).

    Args:
        aff:    scipy sparse affinity matrix (n x n), diagonal should be 0.
        labels: array-like of length n.

    Returns:
        np.ndarray (n,) — per-cell weighted purity. NaN for isolated cells.
    """
    aff = sp.csr_matrix(aff)
    labels = np.asarray(labels)
    n = aff.shape[0]
    weighted_same = np.zeros(n)
    total_weight = np.zeros(n)

    for i in range(n):
        s, e = int(aff.indptr[i]), int(aff.indptr[i + 1])
        if s == e:
            continue
        cols = aff.indices[s:e]
        vals = aff.data[s:e]
        mask = cols != i
        cols, vals = cols[mask], vals[mask]
        if len(cols) == 0:
            continue
        same = (labels[cols] == labels[i]).astype(float)
        weighted_same[i] = np.dot(vals, same)
        total_weight[i] = vals.sum()

    with np.errstate(invalid='ignore'):
        return np.where(total_weight > 0, weighted_same / total_weight, np.nan)


def weighted_purity_within_group(aff, group_labels, target_labels):
    """Affinity-weighted niche purity within cell type.

    For each cell i, restricts neighbors to those sharing the same group label
    (cell type), then computes affinity-weighted target label (niche) purity
    over that restricted set.

    Args:
        aff:           scipy sparse affinity matrix (n x n), diagonal should be 0.
        group_labels:  cell-type label per cell — used to restrict neighbors.
        target_labels: niche label per cell — the label purity is measured on.

    Returns:
        np.ndarray (n,) — per-cell within-group weighted purity.
        NaN for cells with no same-group neighbors in the affinity graph.
    """
    aff = sp.csr_matrix(aff)
    group_labels = np.asarray(group_labels)
    target_labels = np.asarray(target_labels)
    n = aff.shape[0]
    weighted_same = np.zeros(n)
    total_weight = np.zeros(n)
    has_group_neighbor = np.zeros(n, dtype=bool)

    for i in range(n):
        s, e = int(aff.indptr[i]), int(aff.indptr[i + 1])
        if s == e:
            continue
        cols = aff.indices[s:e]
        vals = aff.data[s:e]
        mask = (cols != i) & (group_labels[cols] == group_labels[i])
        cols, vals = cols[mask], vals[mask]
        if len(cols) == 0:
            continue
        has_group_neighbor[i] = True
        same = (target_labels[cols] == target_labels[i]).astype(float)
        weighted_same[i] = np.dot(vals, same)
        total_weight[i] = vals.sum()

    with np.errstate(invalid='ignore'):
        result = np.where(total_weight > 0, weighted_same / total_weight, np.nan)
    result[~has_group_neighbor] = np.nan
    return result


def affinity_tradeoff_row(aff, ct_labels, niche_labels, name="method"):
    """Compute (cell-type purity, within-CT niche purity) for one affinity.

    Both metrics are affinity-weighted: neighbors with higher edge weight
    contribute proportionally more, matching how the affinity is used downstream.

    Returns a one-row DataFrame — concat multiple rows to get the comparison table.
    """
    ct_pur    = float(np.nanmean(weighted_purity(aff, ct_labels)))
    niche_pur = float(np.nanmean(weighted_purity_within_group(aff, ct_labels, niche_labels)))
    return pd.DataFrame(
        {'ct_weighted_purity': [ct_pur], 'within_ct_niche_weighted_purity': [niche_pur]},
        index=[name],
    )


def compare_affinities(aff_dict, ct_labels, niche_labels):
    """Run affinity_tradeoff_row for each method and return a combined DataFrame.

    Args:
        aff_dict:     {method_name: sparse_affinity_matrix}
        ct_labels:    cell-type label array (length n)
        niche_labels: niche label array (length n)

    Returns:
        pd.DataFrame with columns [ct_weighted_purity, within_ct_niche_weighted_purity],
        one row per method.
    """
    return pd.concat([
        affinity_tradeoff_row(aff, ct_labels, niche_labels, name)
        for name, aff in aff_dict.items()
    ])


def embedding_ct_silhouette(X, ct_labels, sample_size=5000, random_state=0):
    """Overall cell-type silhouette score computed directly on a raw embedding.

    Unlike the affinity-graph purity metrics above, this measures separation in
    the *feature space itself* (e.g. X_pca, X_mean_pca, X_banksy) rather than in
    a derived kNN graph — this is the "silhouette score ... in each embedding
    space" quantification called for in R2.

    Returns NaN if fewer than 2 cell types are present.
    """
    ct_labels = np.asarray(ct_labels)
    if len(np.unique(ct_labels)) < 2:
        return np.nan
    n = X.shape[0]
    ss = min(sample_size, n) if sample_size else None
    return float(silhouette_score(X, ct_labels, sample_size=ss, random_state=random_state))


def embedding_niche_silhouette_within_ct(X, ct_labels, niche_labels,
                                         sample_size=5000, random_state=0,
                                         min_cells=4):
    """Within-cell-type niche silhouette, averaged across cell types (weighted by size).

    For each cell type with >= min_cells cells and >= 2 niche labels present,
    computes the niche-label silhouette restricted to that cell type's rows of
    X, then averages across cell types weighted by cell count. This is the
    embedding-space analogue of `weighted_purity_within_group`.

    Returns NaN if no cell type has enough cells/niches to score.
    """
    ct_labels = np.asarray(ct_labels)
    niche_labels = np.asarray(niche_labels)
    scores, weights = [], []
    for ct in np.unique(ct_labels):
        mask = ct_labels == ct
        n_ct = int(mask.sum())
        niches_ct = niche_labels[mask]
        if n_ct < min_cells or len(np.unique(niches_ct)) < 2:
            continue
        ss = min(sample_size, n_ct) if sample_size else None
        try:
            s = silhouette_score(X[mask], niches_ct, sample_size=ss, random_state=random_state)
        except ValueError:
            continue
        scores.append(s)
        weights.append(n_ct)
    if not scores:
        return np.nan
    return float(np.average(scores, weights=weights))


def embedding_tradeoff_row(X, ct_labels, niche_labels, name="embedding",
                           sample_size=5000, random_state=0):
    """Compute (cell-type silhouette, within-CT niche silhouette) for one embedding.

    Returns a one-row DataFrame — concat multiple rows to get the comparison table.
    """
    ct_sil = embedding_ct_silhouette(X, ct_labels, sample_size, random_state)
    niche_sil = embedding_niche_silhouette_within_ct(
        X, ct_labels, niche_labels, sample_size, random_state)
    return pd.DataFrame(
        {'ct_silhouette': [ct_sil], 'within_ct_niche_silhouette': [niche_sil]},
        index=[name],
    )


def compare_embeddings(emb_dict, ct_labels, niche_labels, sample_size=5000, random_state=0):
    """Run embedding_tradeoff_row for each raw embedding and return a combined DataFrame.

    Args:
        emb_dict:     {embedding_name: (n, d) array from ad.obsm}
        ct_labels:    cell-type label array (length n)
        niche_labels: niche label array (length n)

    Returns:
        pd.DataFrame with columns [ct_silhouette, within_ct_niche_silhouette],
        one row per embedding.
    """
    return pd.concat([
        embedding_tradeoff_row(X, ct_labels, niche_labels, name, sample_size, random_state)
        for name, X in emb_dict.items()
    ])

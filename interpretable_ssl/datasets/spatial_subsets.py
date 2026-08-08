"""spatial_subsets.py — build single-cell-type spatial subsets with a precomputed
spatial context feature, for scoped scProto-vs-baseline experiments.

Usage:
    from interpretable_ssl.datasets.spatial_subsets import build_celltype_subset_with_context

    build_celltype_subset_with_context(
        's28nsc', ct_key='celltypes', ct_value='Fibroblasts',
        out_path=os.path.join(DATA_DIR, 'spatial', 'fibnsc.h5ad'),
    )
"""
import os

import numpy as np
import scanpy as sc

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.augmenters.graph_generator import build_context, compute_banksy_embedding

DEFAULT_TARGET_MEDIAN_NEIGHBOURS = 32  # Pentimalli et al., median 2D (single-section)
# neighbour count within their 50um radius (their 3D median is 71, but our X_ctx is
# 2D/single-section, so 32 is the comparable figure) -- see nscl.pdf p.9.


def calibrate_radius_for_target_median_neighbours(ad, target=DEFAULT_TARGET_MEDIAN_NEIGHBOURS):
    """Find the 2D spatial radius r such that the median cell has `target` neighbours
    within r, in THIS dataset's own coordinate units.

    We don't know whether our coordinate units match the NSCLC paper's micrometers, so
    picking a literal radius=50 to match their number isn't safe. Instead this calibrates
    to the same DENSITY they report (median 32 cells within their 2D 50um neighbourhood),
    which is coordinate-system-independent.

    Exact single-shot computation, not a search loop: for any cell, the distance to its
    `target`-th nearest 2D neighbour IS, by definition, the radius that gives that cell
    exactly `target` neighbours. So the median of those per-cell distances, across the
    dataset, is exactly the radius giving a median neighbour count of `target` -- no
    trial-and-error needed.
    """
    import faiss

    spatial = ad.obsm["spatial"][:, :2].astype(np.float32)
    index = faiss.IndexFlatL2(spatial.shape[1])
    index.add(spatial)

    D_sq, _ = index.search(spatial, target + 1)  # +1 to exclude self
    dist_to_target_nn = np.sqrt(D_sq[:, target])
    radius = float(np.median(dist_to_target_nn))

    # Sanity check: recompute the actual mean/median neighbour count AT this radius
    # directly (median should land very close to `target` by construction -- a big gap
    # would flag ties/duplicate coordinates rather than a calibration error). Mean is
    # reported alongside as a skew check (dense tumor regions vs. sparse stroma) -- it's
    # not the calibration target, only extra visibility.
    recheck_k = min(target * 4, spatial.shape[0])
    D_sq_full, _ = index.search(spatial, recheck_k)
    actual_counts = (np.sqrt(D_sq_full) <= radius).sum(axis=1) - 1  # exclude self
    actual_median = float(np.median(actual_counts))
    actual_mean = float(np.mean(actual_counts))
    n_capped = int((actual_counts >= recheck_k - 1).sum())

    print(f"[radius calibration] target_median_neighbours={target}  ->  "
          f"radius={radius:.4f} (dataset coordinate units)  "
          f"actual median={actual_median:.1f}  actual mean={actual_mean:.1f}  "
          f"(sanity-checked against up to {recheck_k} nearest neighbours per cell)")
    if n_capped > 0:
        print(f"[radius calibration] WARNING: {n_capped} cell(s) hit the {recheck_k}-neighbour "
              f"recheck cap -- their true count may be higher (very dense region); "
              f"mean may be understated, median is unaffected.")
    return radius


def build_celltype_subset_with_context(
    src_ds_id, ct_key, ct_value,
    target_median_neighbours=DEFAULT_TARGET_MEDIAN_NEIGHBOURS,
    out_path=None,
    banksy_configs=None,
):
    """Load src_ds_id, compute the spatial context feature (and, optionally, one or more
    BANKSY embeddings) on the FULL tissue, then subset to a single cell type and save.

    Order matters: a cell's true spatial context includes its real neighbours of every
    cell type. Subsetting first would silently redefine "neighbour" to mean "same-type
    neighbour", changing what the context feature captures. Computing it before
    subsetting keeps the niche-composition signal in X_ctx intact, while the resulting
    single-cell-type dataset's own affinity graph (built later, on this subset) becomes
    same-cell-type only automatically — no explicit cross-cell-type edge masking needed.
    The identical reasoning applies to BANKSY: its neighbour-mean-expression component
    needs each cell's real (mixed-cell-type) spatial neighbours, not just same-type ones
    left over after filtering — computing it here, before subsetting, is exactly as
    important as it is for X_ctx.

    Uses build_context (radius-based, unweighted mean PCA) -- matching the NSCLC paper's
    own stated method (a fixed physical radius, not a fixed neighbour count) and this
    codebase's own appendix (`app:spatial_affinity`, radius r). The radius itself is
    calibrated automatically every time this runs (see
    calibrate_radius_for_target_median_neighbours) rather than hand-picked, since our
    coordinate units aren't confirmed to match the paper's micrometers.

    Args:
        src_ds_id:                dataset id already in DATASETS (e.g. 's28nsc').
        ct_key:                   obs column with cell type labels.
        ct_value:                 cell type to keep.
        target_median_neighbours: calibration target for the radius (default 32, the
                                   paper's own reported 2D median — see nscl.pdf p.9).
        out_path:                 where to write the subset .h5ad. Required.
        banksy_configs:           optional list of (obsm_key, lambda_param, num_neighbours)
                                   triples -- BANKSY embeddings to precompute on the full
                                   tissue before subsetting (e.g. [('X_bk32', 0.5, 32)] for
                                   AFFINITY_TYPE='bk32'). None/[] skips BANKSY entirely
                                   (the common case -- most affinity types don't need it).

    Returns:
        The saved AnnData subset (also written to out_path).
    """
    if out_path is None:
        raise ValueError("out_path is required")

    conf = DATASETS[src_ds_id]
    ad = sc.read_h5ad(conf['path'])
    print(f"[{src_ds_id}] loaded FULL, unfiltered tissue: {ad.n_obs} cells "
          f"(step 1 of 3: load full -> step 2: compute embeddings on it -> "
          f"step 3: filter to {ct_key}=={ct_value!r})")

    if 'X_pca' not in ad.obsm:
        sc.tl.pca(ad, n_comps=50)

    radius = calibrate_radius_for_target_median_neighbours(ad, target=target_median_neighbours)

    print(f"[{src_ds_id}] computing spatial context (radius={radius:.4f}) on all "
          f"{ad.n_obs} cells, before cell-type filtering ...")
    ad.obsm['X_ctx'] = build_context(ad, radius)

    for obsm_key, lam, k in (banksy_configs or []):
        print(f"[{src_ds_id}] computing BANKSY embedding ({obsm_key}, lambda={lam}, "
              f"num_neighbours={k}) on all {ad.n_obs} cells (FULL tissue), "
              f"before cell-type filtering ...")
        compute_banksy_embedding(ad, lambda_param=lam, num_neighbours=k, obsm_key=obsm_key)
        print(f"[{src_ds_id}] {obsm_key} done: shape={ad.obsm[obsm_key].shape} "
              f"(rows == full {ad.n_obs}-cell tissue, not yet filtered)")

    sub = ad[ad.obs[ct_key] == ct_value].copy()
    print(f"[{src_ds_id}] filtered to {ct_key}=={ct_value!r}: {sub.n_obs} / {ad.n_obs} cells. "
          f"obsm now carried into the subset: {sorted(sub.obsm.keys())}")

    # Provenance, so a stale file from a different src_ds_id/ct_value/target can be
    # detected instead of silently reused just because out_path already exists (this bit
    # us once: a fibnsc.h5ad built from ss28nsc — 7592 Fibroblasts — was silently kept
    # after switching the source to s28nsc, which has 15309).
    sub.uns['spatial_subset_src_ds_id'] = src_ds_id
    sub.uns['spatial_subset_ct_key'] = ct_key
    sub.uns['spatial_subset_ct_value'] = ct_value
    sub.uns['spatial_subset_target_median_neighbours'] = target_median_neighbours
    sub.uns['spatial_subset_calibrated_radius'] = radius

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.write_h5ad(out_path)
    print(f"saved to {out_path}")
    return sub


def subset_matches(out_path, src_ds_id, ct_key, ct_value, target_median_neighbours,
                    banksy_configs=None):
    """Check an existing subset .h5ad's recorded provenance against the requested build
    params, without loading the full file (obs/X untouched — reads uns only).

    banksy_configs (same [(obsm_key, lambda_param, num_neighbours), ...] list passed to
    build_celltype_subset_with_context): also checks that every requested obsm_key is
    actually present in the existing file. Presence-only, not a lambda/num_neighbours
    equality check -- catches the real failure mode (a file built before AFFINITY_TYPE
    was switched to something BANKSY-based, so the embedding was never computed at all)
    without needing to also serialize/compare those params. Loads each obsm_key's array
    to check (a few MB, not the whole file) -- still far cheaper than a full reload.
    """
    import anndata as ad_module

    if not os.path.exists(out_path):
        return False
    existing = ad_module.read_h5ad(out_path, backed='r')
    base_ok = (
        existing.uns.get('spatial_subset_src_ds_id') == src_ds_id
        and existing.uns.get('spatial_subset_ct_key') == ct_key
        and existing.uns.get('spatial_subset_ct_value') == ct_value
        and existing.uns.get('spatial_subset_target_median_neighbours') == target_median_neighbours
    )
    if not base_ok:
        return False
    return all(obsm_key in existing.obsm for obsm_key, _, _ in (banksy_configs or []))

# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="title"
# # Spatial Coordinate Units & Radius Calibration
#
# The NSCLC paper defines neighbourhoods as **50 um center-to-center radius** with a
# median of **32 cells in 2D**.  
# This notebook calibrates the coordinate units of `ad.obsm['spatial']` by finding the
# radius (in coordinate units) that reproduces that median neighbour count.
#
# **Strategy - inverted calibration**:
# 1. Sweep candidate radii; measure median neighbour count at each.
# 2. Interpolate to find the exact radius where median = 32.
# 3. Back-calculate implied um/unit; cross-check against NN cell-spacing.
# 4. Confirm visually and compute the full neighbour-count distribution.
# 5. Compute radius-COVET and compare to fixed-k COVET (k=35).

# %% [markdown] id="setup-md"
# ## Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="setup-drive" executionInfo={"status": "ok", "timestamp": 1782629668655, "user_tz": -210, "elapsed": 14808, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="935ef1e5-75ca-4e55-d491-67950d9a67de"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="setup-install" executionInfo={"status": "ok", "timestamp": 1782629884405, "user_tz": -210, "elapsed": 215753, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="7101fb9f-6a1b-48a0-c053-816a9f48a184"
# !pip install -q scarches SEACells faiss-gpu-cu12 scib-metrics

# %% colab={"base_uri": "https://localhost:8080/"} id="setup-nbsetup" executionInfo={"status": "ok", "timestamp": 1782629972490, "user_tz": -210, "elapsed": 88083, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="08958831-9066-43fe-b3d7-55c4d1858ea4"
# %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py

# %% [markdown] id="config-md"
# ## Config

# %% id="config" executionInfo={"status": "ok", "timestamp": 1782630097137, "user_tz": -210, "elapsed": 11, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}}
import os

DATA_PATH        = os.path.join(os.environ["DATA_DIR"], "spatial/NSCLC_3D_section_28.h5ad")
TARGET_RADIUS_UM = 50.0  # paper: 50 um center-to-center
TARGET_NEIGHBORS = 32    # paper: median 32 cells in 2D within 50 um

# %% [markdown] id="load-md"
# ## Load data

# %% colab={"base_uri": "https://localhost:8080/"} id="load" executionInfo={"status": "ok", "timestamp": 1782630104315, "user_tz": -210, "elapsed": 6163, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="fecab27d-a640-4c90-858b-559cf6e4f844"
import scanpy as sc
import numpy as np

ad = sc.read_h5ad(DATA_PATH)
print(ad)
print('obsm keys:', list(ad.obsm.keys()))

# %% [markdown] id="diag-md"
# ## Step 1 — coordinate range & metadata
#
# Check coordinate scale and whether `ad.uns` already stores pixel size.

# %% colab={"base_uri": "https://localhost:8080/"} id="diag-range" executionInfo={"status": "ok", "timestamp": 1782630104322, "user_tz": -210, "elapsed": 5, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="0d9ffb4f-f698-4820-d4d7-a017a207acc6"
spatial = ad.obsm['spatial'][:, :2].astype(float)

print('Coordinate ranges:')
print(f'  x: {spatial[:,0].min():.2f} – {spatial[:,0].max():.2f}  '
      f'(span {spatial[:,0].max() - spatial[:,0].min():.2f})')
print(f'  y: {spatial[:,1].min():.2f} – {spatial[:,1].max():.2f}  '
      f'(span {spatial[:,1].max() - spatial[:,1].min():.2f})')

# check uns for any pixel size metadata
pixel_size_keys = ['pixel_size', 'microns_per_pixel', 'um_per_pixel',
                   'resolution', 'pixel_size_um', 'scale']
found = {k: ad.uns[k] for k in pixel_size_keys if k in ad.uns}
if found:
    print('\nFound in ad.uns:', found)
else:
    print('\nNo pixel size metadata in ad.uns — will estimate empirically below.')

# %% [markdown] id="nn-md"
# ## Step 2 - nearest-neighbour distance (reference only)
#
# We use the median NN distance as a **cross-check** after calibration, not as the
# primary scale source. The cell-diameter assumption gave the wrong neighbour count.

# %% colab={"base_uri": "https://localhost:8080/"} id="diag-nn" executionInfo={"status": "ok", "timestamp": 1782630104471, "user_tz": -210, "elapsed": 148, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="111a4ae8-c5b6-44c0-f638-0aa193c34f7f"
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=2).fit(spatial)
dists, _ = nn.kneighbors(spatial)
nn_dists = dists[:, 1]
median_nn = float(np.median(nn_dists))
p25, p75  = np.percentile(nn_dists, [25, 75])
print(f"Nearest-neighbour distance (coordinate units):")
print(f"  median={median_nn:.3f}  p25={p25:.3f}  p75={p75:.3f}")
print("  (cross-check reference -- scale set by sweep below)")

# %% [markdown] id="sweep-md"
# ## Step 3 - radius sweep: find radius where median neighbours = 32
#
# Sweep candidate radii on a subsample, measure median neighbour count at each,
# then interpolate to find the radius where median = TARGET_NEIGHBORS.
# No cell-diameter assumption needed.

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="radius-sweep" executionInfo={"status": "ok", "timestamp": 1782630423882, "user_tz": -210, "elapsed": 30174, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="668c4223-cbe3-4a70-cd87-8c7b7d49b8c8"
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
sub_idx = rng.choice(len(spatial), size=min(len(spatial), len(spatial)), replace=False)
spatial_sub = spatial[sub_idx]

candidate_radii = np.linspace(3, 40, 30)
median_counts = []
mean_counts   = []
nn_sweep = NearestNeighbors().fit(spatial_sub)

for r in candidate_radii:
    nbrs = nn_sweep.radius_neighbors(spatial_sub, radius=r, return_distance=False)
    counts = np.array([len(idx) - 1 for idx in nbrs])
    median_counts.append(float(np.median(counts)))
    mean_counts.append(float(counts.mean()))

median_counts = np.array(median_counts)
mean_counts   = np.array(mean_counts)

print(f"  {chr(39)}radius{chr(39):>8}  {chr(39)}median_nbrs{chr(39):>12}  {chr(39)}mean_nbrs{chr(39):>10}")
for r, med, mn in zip(candidate_radii, median_counts, mean_counts):
    marker = " <-- near target" if abs(med - TARGET_NEIGHBORS) < 3 else ""
    print(f"  {r:6.2f}  {med:12.1f}  {mn:10.1f}{marker}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(candidate_radii, median_counts, "o-", color="steelblue", label="median neighbours")
ax.plot(candidate_radii, mean_counts, "s--", color="gray", label="mean neighbours", alpha=0.6)
ax.axhline(TARGET_NEIGHBORS, color="red", linestyle="--", label=f"paper target = {TARGET_NEIGHBORS}")
ax.set_xlabel("Radius (coordinate units)")
ax.set_ylabel("Neighbour count")
ax.set_title("Neighbour count vs radius sweep")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown] id="calibrate-md"
# ## Step 4 - interpolate calibrated radius and back-calculate scale

# %% colab={"base_uri": "https://localhost:8080/"} id="calibrate" executionInfo={"status": "ok", "timestamp": 1782630446969, "user_tz": -210, "elapsed": 24, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="65d02bb9-e851-478c-ed67-2f4753cb0a96"
RADIUS_CALIBRATED = float(np.interp(TARGET_NEIGHBORS, median_counts, candidate_radii))
UM_PER_UNIT_calibrated = TARGET_RADIUS_UM / RADIUS_CALIBRATED
implied_cell_diameter_um = median_nn * UM_PER_UNIT_calibrated

print(f"Calibrated radius (median neighbours = {TARGET_NEIGHBORS}):")
print(f"  RADIUS_CALIBRATED = {RADIUS_CALIBRATED:.2f} coordinate units")
print(f"  implies {TARGET_RADIUS_UM} um = {RADIUS_CALIBRATED:.2f} units")
print(f"  UM_PER_UNIT = {UM_PER_UNIT_calibrated:.4f} um/unit")
print()
print(f"Cross-check:")
print(f"  median NN distance = {median_nn:.3f} units")
print(f"  implied cell diameter = {implied_cell_diameter_um:.1f} um")
ok = 8 < implied_cell_diameter_um < 20
print(f"  CosMx cells ~10-15 um  ->  {'OK' if ok else 'UNEXPECTED'}")
print()
radius_old = TARGET_RADIUS_UM / (12.0 / median_nn)
old_median = float(np.interp(radius_old, candidate_radii, median_counts))
print(f"Old estimate (cell-diameter=12 um assumption):")
print(f"  radius = {radius_old:.2f} units  ->  median neighbours = {old_median:.0f}  (target {TARGET_NEIGHBORS})")

# %% [markdown] id="confirm-md"
# ## Step 5 - visual sanity check
#
# Left: calibrated radius (should enclose ~32 cells).  
# Right: old cell-diameter estimate (too small, only ~18 cells).

# %% colab={"base_uri": "https://localhost:8080/", "height": 937} id="diag-plot" executionInfo={"status": "ok", "timestamp": 1782630450806, "user_tz": -210, "elapsed": 550, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="356d09e9-b273-46af-8733-0757a43d5d0e"
import matplotlib.patches as patches

center_approx = np.median(spatial, axis=0)
anchor_idx = int(np.argmin(np.linalg.norm(spatial - center_approx, axis=1)))
cx, cy = spatial[anchor_idx]
window = RADIUS_CALIBRATED * 4
mask = (
    (spatial[:, 0] > cx - window) & (spatial[:, 0] < cx + window) &
    (spatial[:, 1] > cy - window) & (spatial[:, 1] < cy + window)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, radius, label, color in [
    (axes[0], RADIUS_CALIBRATED, f"Calibrated ({RADIUS_CALIBRATED:.1f} units = {TARGET_RADIUS_UM:.0f} um)", "green"),
    (axes[1], radius_old,        f"Old estimate ({radius_old:.1f} units)", "red"),
]:
    ax.scatter(spatial[mask, 0], spatial[mask, 1], s=4, c="steelblue", alpha=0.5)
    ax.scatter([cx], [cy], s=60, c="red", zorder=5)
    circle = patches.Circle((cx, cy), radius=radius,
                             fill=False, edgecolor=color, linewidth=2, label=label)
    ax.add_patch(circle)
    n_inside = (np.linalg.norm(spatial - spatial[anchor_idx], axis=1) < radius).sum() - 1
    ax.set_aspect("equal")
    ax.set_title(f"{label}\n{n_inside} cells inside", fontsize=10)
    ax.legend(fontsize=8)
plt.suptitle("Calibrated vs old radius", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown] id="result-md"
# ## Step 6 - full neighbour-count distribution at calibrated radius

# %% colab={"base_uri": "https://localhost:8080/", "height": 573} id="final-radius" executionInfo={"status": "ok", "timestamp": 1782630458212, "user_tz": -210, "elapsed": 545, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="9f66075f-0716-47e7-e574-44b73a76ab7c"
RADIUS_UNITS = RADIUS_CALIBRATED
print(f"RADIUS_UNITS = {RADIUS_UNITS:.2f}  (calibrated = {TARGET_RADIUS_UM} um, target median={TARGET_NEIGHBORS})")
print(f"UM_PER_UNIT  = {UM_PER_UNIT_calibrated:.4f}")

nn_r = NearestNeighbors(radius=RADIUS_UNITS).fit(spatial)
neighbor_counts = np.array([
    len(idx) - 1
    for idx in nn_r.radius_neighbors(spatial, return_distance=False)
])

print(f"\nNeighbour counts (full dataset, 2D):")
print(f"  median={np.median(neighbor_counts):.0f}  mean={neighbor_counts.mean():.1f}  "
      f"p10={np.percentile(neighbor_counts, 10):.0f}  p90={np.percentile(neighbor_counts, 90):.0f}")
status = "MATCH" if abs(np.median(neighbor_counts) - TARGET_NEIGHBORS) <= 2 else "CHECK"
print(f"  target median = {TARGET_NEIGHBORS}  -> {status}")
print(f"  cells with <3 neighbours: {(neighbor_counts < 3).sum()} ({100*(neighbor_counts < 3).mean():.1f}%)")

fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(neighbor_counts, bins=40, color="steelblue", edgecolor="white")
ax.axvline(np.median(neighbor_counts), color="red", linestyle="--",
           label=f"median={np.median(neighbor_counts):.0f}")
ax.axvline(TARGET_NEIGHBORS, color="green", linestyle=":",
           label=f"paper target={TARGET_NEIGHBORS}")
ax.set_xlabel("Neighbours within calibrated radius")
ax.set_ylabel("Cells")
ax.set_title(f"Neighbour count distribution at calibrated radius ({RADIUS_UNITS:.1f} units)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown] id="covet-md"
# ## Step 7 - radius-based COVET
#
# Use the calibrated radius for COVET neighbourhood computation.

# %% id="covet-radius-fn" executionInfo={"status": "ok", "timestamp": 1782630466162, "user_tz": -210, "elapsed": 45, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}}
import numpy as np
from sklearn.neighbors import NearestNeighbors

def covet_spatial_knn_radius(ad, radius_units):
    """Spatial neighbors within radius_units. Returns list of index arrays (excl. self)."""
    spatial = ad.obsm['spatial'][:, :2].astype(np.float32)
    nn = NearestNeighbors(radius=radius_units).fit(spatial)
    neighbors = nn.radius_neighbors(spatial, return_distance=False)
    return [nbrs[nbrs != i] for i, nbrs in enumerate(neighbors)]


def covet_cov_flat_radius(ad, neighbors, n_pcs):
    """Flattened upper-triangle covariance of neighbour PCA, variable neighbourhood size.

    Cells with fewer than 2 neighbours get a zero vector (not enough for covariance).
    """
    X_pca = ad.obsm['X_pca'][:, :n_pcs]
    n = len(ad)
    d = n_pcs * (n_pcs + 1) // 2
    cov_flat = np.zeros((n, d), dtype=np.float32)
    ti, tj = np.triu_indices(n_pcs)

    for i, idx in enumerate(neighbors):
        if len(idx) < 2:
            continue
        X_nbr = X_pca[idx]
        X_nbr = X_nbr - X_nbr.mean(0)
        cov = (X_nbr.T @ X_nbr) / (len(idx) - 1)
        cov_flat[i] = cov[ti, tj]

    return cov_flat


def compute_covet_features_radius(ad, radius_units, n_pcs=25,
                                  alpha=1.0, n_comps=None, obsm_key='X_covet_radius'):
    """COVET with 50 µm radius neighbourhood instead of fixed-k kNN.

    Args:
        ad:           AnnData with obsm['spatial'] and obsm['X_pca']
        radius_units: 50 µm converted to coordinate units
        n_pcs:        PCA dims used as input to covariance
        alpha:        variance fraction from covariance side (1.0 = covet-only)
        n_comps:      PCA components from flattened covariance (None = auto)
        obsm_key:     where to store the result
    """
    from interpretable_ssl.augmenters.graph_generator import _covet_apply

    print(f'[covet_radius] radius={radius_units:.1f} units  n_pcs={n_pcs}  alpha={alpha}')
    neighbors = covet_spatial_knn_radius(ad, radius_units)

    counts = np.array([len(n) for n in neighbors])
    print(f'[covet_radius] neighbour counts: median={np.median(counts):.0f}  '
          f'min={counts.min()}  max={counts.max()}  '
          f'cells_with_<2={( counts < 2).sum()}')

    cov_flat = covet_cov_flat_radius(ad, neighbors, n_pcs)
    n_comps_used, n_cov_dims = _covet_apply(ad, cov_flat, n_comps, alpha, obsm_key)
    print(f'[covet_radius] stored in obsm["{obsm_key}"]  shape={ad.obsm[obsm_key].shape}')
    return neighbors


# %% colab={"base_uri": "https://localhost:8080/"} id="covet-radius-run" executionInfo={"status": "ok", "timestamp": 1782630474171, "user_tz": -210, "elapsed": 6852, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="96a61b39-f3b9-48b3-d761-33940d480d9e"
if "X_pca" not in ad.obsm:
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)

neighbors = compute_covet_features_radius(
    ad,
    radius_units=RADIUS_UNITS,
    n_pcs=25,
    alpha=1.0,
    obsm_key="X_covet_radius",
)

# %% [markdown] id="compare-md"
# ## Step 8 - compare radius-COVET vs fixed-k COVET (k=35)
#
# k=35 is the paper-grounded default in `compute_covet_features`.
# High per-dimension correlation means both capture the same ~32-cell neighbourhood.

# %% colab={"base_uri": "https://localhost:8080/", "height": 660} id="compare" executionInfo={"status": "ok", "timestamp": 1782630503626, "user_tz": -210, "elapsed": 23277, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="8b8bbda5-3e64-425c-d67f-7d93e7449367"
from interpretable_ssl.augmenters.graph_generator import compute_covet_features

compute_covet_features(ad, k=35, n_pcs=25, alpha=1.0, obsm_key="X_covet_k35")

X_k35    = ad.obsm["X_covet_k35"]
X_radius = ad.obsm["X_covet_radius"]

n_dims = min(X_k35.shape[1], X_radius.shape[1])
corrs = [
    float(np.corrcoef(X_k35[:, d], X_radius[:, d])[0, 1])
    for d in range(n_dims)
]
print(f"Per-dimension correlation (kNN k=35 vs calibrated radius {RADIUS_UNITS:.1f} units):")
print(f"  mean={np.mean(corrs):.3f}  median={np.median(corrs):.3f}  "
      f"min={np.min(corrs):.3f}  max={np.max(corrs):.3f}")
print("  (high = both capture the same ~32-cell neighbourhood)")

fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(n_dims), corrs, color="steelblue")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("COVET dimension")
ax.set_ylabel("Pearson r")
ax.set_title(f"kNN (k=35) vs calibrated radius ({RADIUS_UNITS:.1f} units) COVET -- per-dim correlation")
plt.tight_layout()
plt.show()

print(f"\n=== Calibration summary ===")
print(f"  Calibrated radius  = {RADIUS_UNITS:.2f} coordinate units")
print(f"  Physical radius    = {TARGET_RADIUS_UM} um")
print(f"  Scale              = {UM_PER_UNIT_calibrated:.4f} um/unit")
print(f"  Implied cell diam  = {implied_cell_diameter_um:.1f} um  (expected 10-15 um)")
print(f"  Median neighbours  = {int(np.median(neighbor_counts))}  (paper target {TARGET_NEIGHBORS})")
print(f"  k=35 vs radius corr: mean={np.mean(corrs):.3f}")

# %% id="9NMUTIRZAjcG"

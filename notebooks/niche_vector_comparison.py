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

# %% [markdown] id="3jSb-jpLlzD9"
# # Niche vector comparison: V1–V4
#
# **Experiment** (inspired by Pentimalli et al. NSCL paper):
#
# For each cell, its spatial neighborhood defines a **niche** via the **cell-type composition vector**  
# (fraction of each cell type among `k` spatial neighbors — exactly what the paper uses to cluster niches).
#
# We compare four neighborhood embeddings against that ground truth:
#
# | Vector | What it encodes | Moment |
# |--------|----------------|--------|
# | **V1 — mean PCA** | Mean of neighbors' `X_pca` | 1st moment, continuous |
# | **V2 — COVET** | Covariance of neighbors' `X_pca`, PCA-reduced | 2nd moment, continuous |
# | **V3 — soft-cluster avg** | Mean of neighbors' soft prototype assignments | 1st moment, discretized |
# | **V4 — concat(V1, V2)** | Variance-balanced concat of V1 and V2 | 1st + 2nd moment (theoretically best) |
#
# **V4 rationale**: V1 captures *where* neighbors are in PCA space (mean = 1st moment).  
# V2 captures *how spread out* they are (covariance = 2nd moment).  
# Together they encode the full distribution up to 2nd order — ideally matching composition better than either alone.
#
# **V3 rationale**: Discretizing via soft prototypes avoids the PCA averaging problem  
# (averaging PCA can cancel out opposing cell types). Instead, each cell votes over a shared vocabulary.
#
# **Three metrics** relative to composition vector as ground truth:
# 1. **kNN label purity** — k-means cluster composition → labels; fraction of kNN in rep-space sharing same label
# 2. **Composition kNN Jaccard** — Jaccard overlap between kNN-in-composition and kNN-in-rep-space
# 3. **Composition cosine similarity** — cosine similarity between composition profiles of kNN-in-rep-space

# %% [markdown] id="fSes_9_mlzEC"
# ## Setup

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1286, "status": "ok", "timestamp": 1782639448166, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="pkYCVoAZlzEC" outputId="a69a45e7-08db-485b-c638-39a4d816a2c5"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 27439, "status": "ok", "timestamp": 1782639476799, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="6f9UBghAlzEE" outputId="e7f5cf89-fd7b-4146-b5ba-7f07401df01d"
# !pip install -q faiss-gpu-cu12 SEACells scarches scib-metrics

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 89462, "status": "ok", "timestamp": 1782639670562, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="iYVHrVjZlzEE" outputId="49d99311-41f9-425c-fee4-591857d17f62"
# %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py

# %% [markdown] id="gXMMtDYBlzEF"
# ## Config

# %% executionInfo={"elapsed": 69, "status": "ok", "timestamp": 1782639670616, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="64IgNOwnlzEF"
import os

DATA_PATH    = os.path.join(os.environ['DATA_DIR'], 'spatial/NSCLC_3D_section_28.h5ad')
CT_KEY       = 'celltypes'   # obs column with cell type labels
NICHE_KEY    = 'niches_3D'   # obs column with known niche labels (for reference comparison)

# ── Spatial neighborhood ────────────────────────────────────────────────────
K_SPATIAL      = 35      # kNN spatial neighbors (used when USE_RADIUS=False)
SPATIAL_RADIUS = 12.57   # coordinate units = 50 µm for NSCLC CosMx (used when USE_RADIUS=True)
USE_RADIUS     = False   # set True to switch to radius-based neighbors
BATCH_KEY      = None    # e.g. 'section' to restrict neighbors within a section

# ── Composition clustering (ground truth) ───────────────────────────────────
N_CLUSTERS   = 10   # k-means on composition vectors; paper uses 10

# ── V2: COVET ────────────────────────────────────────────────────────────────
N_PCS_COVET  = 10   # PCA dims for covariance; K_SPATIAL must be >= 3 * N_PCS_COVET

# ── V3: soft-cluster average ──────────────────────────────────────────────────
N_PROTO      = 30   # number of soft prototypes (learned from all cells' X_pca)
                    # higher = finer vocabulary; lower = more robust on small datasets

# ── V4: concat(V1, V2) ───────────────────────────────────────────────────────
CONCAT_ALPHA = 0.5  # variance weight for V1 side; 0.5 = equal; <0.5 = more COVET

# ── Evaluation ───────────────────────────────────────────────────────────────
K_EVAL       = 15   # kNN k for purity / Jaccard metrics

# %% [markdown] id="Kl9_tlxQlzEG"
# ## Load & preprocess

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 1681, "status": "ok", "timestamp": 1782639672300, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="5A1kZ6G6lzEG" outputId="cf191169-0beb-4b03-ff1f-07e54413f25f"
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ad = sc.read_h5ad(DATA_PATH)
print(ad)
print('obs columns:', ad.obs.columns.tolist())
print('obsm keys:  ', list(ad.obsm.keys()))
print(f'\ncell types ({ad.obs[CT_KEY].nunique()}):', sorted(ad.obs[CT_KEY].unique()))
if NICHE_KEY in ad.obs.columns:
    print(f'niches ({ad.obs[NICHE_KEY].nunique()}):', sorted(ad.obs[NICHE_KEY].dropna().unique()))

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 52, "status": "ok", "timestamp": 1782639672357, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="peDo6OMwlzEH" outputId="30dc5a6a-1818-4e1d-d177-5f2f3d650a08"
if 'X_pca' not in ad.obsm:
    print('Computing PCA ...')
    if ad.X.max() > 30:
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
    print('X_pca computed:', ad.obsm['X_pca'].shape)
else:
    print('X_pca already present:', ad.obsm['X_pca'].shape)

assert 'spatial' in ad.obsm, "Need ad.obsm['spatial'] (2D or 3D coordinates)"

# %% [markdown] id="02nBcbq9lzEH"
# ## Run comparison: V1, V2, V3, V4

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 592558, "status": "ok", "timestamp": 1782640264921, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="IyzCYjNRlzEH" outputId="b716e15a-203d-4e66-d976-9130747c99f4"
from interpretable_ssl.evaluation.niche_composition_comparison import compare_neighborhood_reps

results = compare_neighborhood_reps(
    ad,
    k_spatial      = K_SPATIAL,
    celltype_key   = CT_KEY,
    n_clusters     = N_CLUSTERS,
    n_pcs_covet    = N_PCS_COVET,
    n_proto        = N_PROTO,
    concat_alpha   = CONCAT_ALPHA,
    k_eval         = K_EVAL,
    batch_key      = BATCH_KEY,
    spatial_radius = SPATIAL_RADIUS if USE_RADIUS else None,
    verbose        = True,
)

# %% [markdown] id="QJTo7kiHlzEH"
# ## Summary metrics table

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 22, "status": "ok", "timestamp": 1782640264925, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="gGPQSW8ZlzEI" outputId="b07822d8-ee32-4dc5-e5cc-ba541f4eaf2c"
metrics = results['metrics']
print(metrics.round(4).to_string())
print()
print("Rank per metric (1 = best):")
print(metrics.rank(axis=1, ascending=False).astype(int).to_string())
print()
mean_rank = metrics.rank(axis=1, ascending=False).mean(axis=0)
print("Mean rank (lower = better):")
print(mean_rank.sort_values().round(2))

# %% [markdown] id="FnSjBZVwlzEI"
# ## Plots: bars + per-cell violins + per-cell-type breakdown

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"elapsed": 5600, "status": "ok", "timestamp": 1782640270530, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="9thW3dIllzEI" outputId="3767a737-3e6a-47d2-e281-e80a30149394"
from interpretable_ssl.evaluation.niche_composition_comparison import plot_comparison

plot_comparison(results, ad=ad, celltype_key=CT_KEY)

# %% [markdown] id="JRoEq-79lzEI"
# ## Cell-type composition clusters (replicating NSCL paper Fig 2c)

# %% colab={"base_uri": "https://localhost:8080/", "height": 751} executionInfo={"elapsed": 2831, "status": "ok", "timestamp": 1782640273367, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="cM1U8HdKlzEI" outputId="18d90171-b794-42e3-e5ab-6a1b17a82898"
from interpretable_ssl.evaluation.niche_composition_comparison import plot_composition_heatmap

cluster_means = plot_composition_heatmap(results, ad, celltype_key=CT_KEY)

# %% [markdown] id="B5WIqvMVlzEJ"
# ## Compare composition clusters to known niche labels

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 24, "status": "ok", "timestamp": 1782640273394, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="pE4JaidulzEJ" outputId="92b5faec-532a-4fd3-9ff4-aec838949c1c"
ad.obs['comp_cluster'] = results['comp_labels'].astype(str)

if NICHE_KEY in ad.obs.columns:
    ct_tab = pd.crosstab(
        ad.obs['comp_cluster'],
        ad.obs[NICHE_KEY],
        normalize='index',
    ).round(2)
    print("Composition cluster vs known niches (row-normalized):")
    print(ct_tab.to_string())

# %% [markdown] id="Ay7cKFCilzEJ"
# ## UMAP: own PCA vs V1 vs V2 vs V3 vs V4
#
# Each UMAP colored by cell type, composition cluster, and (if available) known niches.
#
# **Key questions to read from these plots:**
# - Does V4 (purple) cluster niches cleanly within cell-type clouds, better than V1/V2 alone?
# - Does V3 (green, discretized) change the structure compared to V1 (blue, continuous)?

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 610687, "status": "ok", "timestamp": 1782640884088, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="EwzSvDIGlzEJ" outputId="6f9106e0-8fc1-4127-8f83-5228ce6d4415"
# Store embeddings so scanpy can run UMAP on them
for name, X_rep in results['reps'].items():
    key = name.replace(' ', '_').replace('+', 'plus')
    ad.obsm[f'X_{key}'] = X_rep

emb_map = {
    'PCA (own)': 'X_pca',
    **{name: f'X_{name.replace(" ","_").replace("+","plus")}'
       for name in results['reps']}
}

for label, key in emb_map.items():
    sc.pp.neighbors(ad, use_rep=key, n_neighbors=30, key_added=f'nbrs_{label}')
    sc.tl.umap(ad, neighbors_key=f'nbrs_{label}', min_dist=0.3)
    ad.obsm[f'X_umap_{label}'] = ad.obsm['X_umap'].copy()
    print(f'UMAP: {label}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "output_embedded_package_id": "1X0nquBJ8AayC7x1Rj17dXB-qyrL_yvm1"} executionInfo={"elapsed": 21689, "status": "ok", "timestamp": 1782640905769, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="ToeAtP-slzEK" outputId="336a48c8-9c05-41ce-940f-d68ca7a97080"
from matplotlib.lines import Line2D

color_keys = [CT_KEY, 'comp_cluster'] + ([NICHE_KEY] if NICHE_KEY in ad.obs.columns else [])
emb_labels = list(emb_map.keys())
palette    = plt.cm.tab20.colors

n_rows = len(color_keys)
n_cols = len(emb_labels)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.8 * n_rows))
if n_rows == 1:
    axes = axes[None, :]

for row_i, ck in enumerate(color_keys):
    cats = ad.obs[ck].astype(str).astype('category').cat.categories
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
    cell_colors = [cmap[str(c)] for c in ad.obs[ck]]

    for col_i, emb_label in enumerate(emb_labels):
        ax = axes[row_i, col_i]
        X2 = ad.obsm[f'X_umap_{emb_label}']
        ax.scatter(X2[:, 0], X2[:, 1], c=cell_colors, s=1, alpha=0.4, rasterized=True)
        ax.set_title(f'{emb_label}\nby {ck}', fontsize=8)
        ax.axis('off')

    handles = [Line2D([0],[0], marker='o', color='w',
                      markerfacecolor=cmap[str(c)], markersize=5, label=str(c))
               for c in cats]
    axes[row_i, -1].legend(handles=handles, fontsize=6,
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            ncol=max(1, len(cats) // 18))

plt.suptitle('UMAP: PCA vs V1 vs V2 vs V3 vs V4', fontsize=12, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown] id="iFYM_ddslzEK"
# ## Ablation: effect of n_proto on V3

# %% colab={"base_uri": "https://localhost:8080/", "height": 776} executionInfo={"elapsed": 774795, "status": "ok", "timestamp": 1782641680684, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="dxWTN6aOlzEK" outputId="22b8d52f-bb38-45d1-a200-7c9bbf32b4f1"
from interpretable_ssl.evaluation.niche_composition_comparison import (
    compute_soft_cluster_avg, _metrics_for_rep
)

# Reuse comp_df and I from results (already computed)
comp_df    = results['comp_df']
comp_labels = results['comp_labels']
comp_vecs   = comp_df.values

# Recompute I with the same neighbor settings used above
from interpretable_ssl.evaluation.niche_composition_comparison import compute_celltype_composition
_, I = compute_celltype_composition(
    ad, k=K_SPATIAL, celltype_key=CT_KEY,
    batch_key=BATCH_KEY, radius=SPATIAL_RADIUS if USE_RADIUS else None,
)

proto_sweep = [10, 20, 30, 50, 100]
proto_rows = []
for n_p in proto_sweep:
    X_v3 = compute_soft_cluster_avg(ad, I, n_proto=n_p)
    pur, jac, cos, sil = _metrics_for_rep(X_v3, comp_labels, comp_vecs, K_EVAL, 0)
    proto_rows.append({'n_proto': n_p, 'purity': pur.mean(), 'jaccard': jac.mean(),
                       'cos_sim': cos.mean(), 'silhouette': sil})
    print(f'n_proto={n_p:3d}  purity={pur.mean():.4f}  jaccard={jac.mean():.4f}')

proto_df = pd.DataFrame(proto_rows).set_index('n_proto')
proto_df.plot(marker='o', figsize=(8, 4), title='V3 sensitivity to n_proto')
plt.ylabel('metric value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown] id="6gEujw_tlzEL"
# ## Ablation: effect of concat_alpha on V4

# %% colab={"base_uri": "https://localhost:8080/", "height": 776} executionInfo={"elapsed": 856182, "status": "ok", "timestamp": 1782642536873, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}, "user_tz": -210} id="nqbUtFKZlzEL" outputId="9946df5a-c4ca-4095-b92d-f1578f4ee395"
from interpretable_ssl.evaluation.niche_composition_comparison import compute_concat_mean_covet

X_v1 = results['reps']['V1 mean-PCA']
X_v2 = results['reps']['V2 COVET']

alpha_sweep = [0.1, 0.25, 0.5, 0.75, 0.9]
alpha_rows = []
for alpha in alpha_sweep:
    X_v4 = compute_concat_mean_covet(X_v1, X_v2, alpha=alpha)
    pur, jac, cos, sil = _metrics_for_rep(X_v4, comp_labels, comp_vecs, K_EVAL, 0)
    alpha_rows.append({'alpha': alpha, 'purity': pur.mean(), 'jaccard': jac.mean(),
                       'cos_sim': cos.mean(), 'silhouette': sil})
    print(f'alpha={alpha:.2f}  purity={pur.mean():.4f}  jaccard={jac.mean():.4f}')

alpha_df = pd.DataFrame(alpha_rows).set_index('alpha')
alpha_df.plot(marker='o', figsize=(8, 4),
              title='V4 sensitivity to concat_alpha (0=all-COVET, 1=all-meanPCA)')
plt.xlabel('alpha (fraction of variance from V1)')
plt.ylabel('metric value')
plt.axvline(0.5, color='grey', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown] id="UCtZFo55lzEL"
# ## Sensitivity: vary k_spatial

# %% colab={"background_save": true, "base_uri": "https://localhost:8080/"} id="IZong2NklzEL"
k_sweep = [30, 50, 70, 100]
sweep_rows = []

for k in k_sweep:
    print(f"\n=== k_spatial={k} ===")
    res = compare_neighborhood_reps(
        ad, k_spatial=k, celltype_key=CT_KEY,
        n_clusters=N_CLUSTERS, n_pcs_covet=N_PCS_COVET,
        n_proto=N_PROTO, concat_alpha=CONCAT_ALPHA,
        k_eval=K_EVAL, batch_key=BATCH_KEY, verbose=False,
    )
    for metric_name, row in res['metrics'].iterrows():
        for rep_name, val in row.items():
            sweep_rows.append({'k': k, 'metric': metric_name, 'rep': rep_name, 'value': val})

sweep_df = pd.DataFrame(sweep_rows)
for metric_name in sweep_df['metric'].unique():
    sub = sweep_df[sweep_df['metric'] == metric_name]
    pivot = sub.pivot(index='k', columns='rep', values='value')
    pivot.plot(marker='o', figsize=(8, 3), title=metric_name)
    plt.ylabel('value')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% id="7hZ8V7ATmj3I"

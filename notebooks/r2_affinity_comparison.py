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

# %% [markdown] id="23ea3ce0"
# # Affinity Comparison: PCA vs. Spatial-Context vs. Product (R2)
#
# Compares three affinity constructions for capturing within-cell-type niche structure,
# **without COVET** — second-order (covariance) neighbourhood features are excluded here.
# kNN-based context uses **k=35** (~50 µm) for the spatial mean, **k=50** for the affinity graph.
#
# | Method | Context | What it encodes |
# |---|---|---|
# | `arbf_opt` (RBF PCA) | — | Transcriptomics only — baseline |
# | `ctx_mean` (RBF mean-PCA) | mean k=35 | RBF directly on the neighbourhood-averaged PCA (1st moment only) |
# | `mean_product` (sigma union) | mean k=35 | rbf(own PCA) x rbf(mean-neighbour PCA), bandwidth from the union kNN graph |
# | `mean_product` (sigma per-space) | mean k=35 | same product, but each factor's bandwidth comes from its own kNN — stricter AND logic |
#
# The two `mean_product` variants isolate the effect of the `per_space_sigma` flag in
# `rbf_product`: sigma-from-union gives a looser product (edges from either space can
# survive with a moderate weight), sigma-per-space enforces a tighter AND (an edge needs
# to be close in *both* spaces under their own bandwidths to keep a high weight).
#
# **Metrics** — both are affinity-weighted (edge weight, not just kNN membership):
# - `ct_weighted_purity` — weighted fraction of affinity neighbors sharing the same **cell type**
# - `within_ct_niche_weighted_purity` — same-cell-type neighbors only, weighted fraction sharing the same **niche**
#

# %% [markdown] id="bffec824"
# ## Setup

# %% colab={"base_uri": "https://localhost:8080/"} id="06580a24" executionInfo={"status": "ok", "timestamp": 1783156795915, "user_tz": -210, "elapsed": 19397, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="a0cbbc33-32bc-4879-d98b-f17e7b393005"
from google.colab import drive
drive.mount('/content/drive')

# %% colab={"base_uri": "https://localhost:8080/"} id="d581b895" executionInfo={"status": "ok", "timestamp": 1783156993780, "user_tz": -210, "elapsed": 197872, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="7327eeb7-ab16-4a66-8e7f-cb42926f3355"
# !pip install -q scarches SEACells faiss-gpu-cu12 scib-metrics

# %% colab={"base_uri": "https://localhost:8080/"} id="5c3f8446" executionInfo={"status": "ok", "timestamp": 1783157097229, "user_tz": -210, "elapsed": 103442, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="4fff100e-4b26-4e60-8d34-1d8152e2a220"
# %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py

# %% [markdown] id="7fb3271d"
# ## Config — set these for your dataset

# %% id="8b035983" executionInfo={"status": "ok", "timestamp": 1783157097236, "user_tz": -210, "elapsed": 3, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}}
import os

DATA_PATH = os.path.join(os.environ['DATA_DIR'], 'spatial/NSCLC_3D_section_28.h5ad')
GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
DS_NAME   = 'NSCLC_3D_section_28'   # used in affinity filenames
CT_KEY    = 'celltypes'
NICHE_KEY = 'niches_2D'
K_AFF     = 50                       # affinity graph kNN
K_CTX     = 35                       # spatial neighbours for the mean-PCA context (~50 um)


# %% [markdown] id="5f9e199d"
# ## Load & preprocess

# %% colab={"base_uri": "https://localhost:8080/"} id="d568fa0f" executionInfo={"status": "ok", "timestamp": 1783157111856, "user_tz": -210, "elapsed": 12423, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="4ab15aff-7714-436b-9981-538fc9f1c3a0"
import scanpy as sc
import numpy as np

ad = sc.read_h5ad(DATA_PATH)
print(ad)
print('obs columns:', ad.obs.columns.tolist())
print('obsm keys:  ', list(ad.obsm.keys()))
print(f'\ncell types ({ad.obs[CT_KEY].nunique()}):', sorted(ad.obs[CT_KEY].unique()))
print(f'niches ({ad.obs[NICHE_KEY].nunique()}):', sorted(ad.obs[NICHE_KEY].dropna().unique()))

# %% colab={"base_uri": "https://localhost:8080/"} id="6d5451fd" executionInfo={"status": "ok", "timestamp": 1783157111859, "user_tz": -210, "elapsed": 18, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="6ba2493a-f261-47ff-923a-0f66cfa570b9"
if 'X_pca' not in ad.obsm:
    sc.pp.highly_variable_genes(ad, n_top_genes=2000)
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
    print('PCA computed:', ad.obsm['X_pca'].shape)
else:
    print('X_pca already present:', ad.obsm['X_pca'].shape)

# %% [markdown] id="f7d47d80"
# ## Generate and save affinity graphs
#
# Each cell runs independently — skip any you've already computed.

# %% colab={"base_uri": "https://localhost:8080/", "height": 191} id="52109877" executionInfo={"status": "ok", "timestamp": 1783157208727, "user_tz": -210, "elapsed": 96867, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="1ca319de-7e15-4fe2-f781-101a0afaa138"
from interpretable_ssl.augmenters.graph_generator import generate_affinity, save_affinity

# RBF PCA -- transcriptomics-only baseline
aff_pca = generate_affinity(ad, k=K_AFF, bk=None, affinity_type='arbf_opt')
aff_pca.setdiag(0)
aff_pca.eliminate_zeros()
save_affinity(aff_pca, DS_NAME, len(ad), affinity_type='arbf_opt',
              k_neighbors=K_AFF, graph_dir=GRAPH_DIR)

# %% colab={"base_uri": "https://localhost:8080/", "height": 191} id="bd91eea9" executionInfo={"status": "ok", "timestamp": 1783157263641, "user_tz": -210, "elapsed": 54910, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="9d232136-b2be-47e7-8495-a56af3c3cf4f"
# RBF mean-PCA -- RBF applied directly to the spatially-averaged PCA (1st moment only, no product)
aff_ctx = generate_affinity(ad, k=K_AFF, bk=None, affinity_type='ctx_mean')
aff_ctx.setdiag(0)
aff_ctx.eliminate_zeros()
save_affinity(aff_ctx, DS_NAME, len(ad), affinity_type='ctx_mean',
              k_neighbors=K_AFF, graph_dir=GRAPH_DIR)

# %% colab={"base_uri": "https://localhost:8080/"} id="88c36105" executionInfo={"status": "ok", "timestamp": 1783157453330, "user_tz": -210, "elapsed": 189686, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="a5fda043-9d58-45a4-e8bb-769d3b7f1581"
# Product of RBF(PCA) x RBF(mean-PCA) -- both sigma modes
for pss in [False, True]:
    aff = generate_affinity(ad, k=K_AFF, bk=None, affinity_type='mean_product',
                            per_space_sigma=pss)
    aff.setdiag(0)
    aff.eliminate_zeros()
    save_affinity(aff, DS_NAME, len(ad),
                  affinity_type='mean_product' + ('_pss' if pss else ''),
                  k_neighbors=K_AFF, graph_dir=GRAPH_DIR)

# %% [markdown] id="d7e5d5bf"
# ## Compute affinity metrics
#
# Load saved affinities (or reuse in-memory ones if they're still in scope).

# %% id="35310f3a" executionInfo={"status": "ok", "timestamp": 1783157454738, "user_tz": -210, "elapsed": 1362, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}}
import pickle, os
from interpretable_ssl.configs.paths import get_affinity_path

def load_aff(affinity_type):
    path = get_affinity_path(DS_NAME, len(ad), k_neighbors=K_AFF,
                             affinity_type=affinity_type, graph_dir=GRAPH_DIR)
    with open(path, 'rb') as f:
        return pickle.load(f)

affinities = {
    'RBF PCA'                   : load_aff('arbf_opt'),
    'RBF mean-PCA'              : load_aff('ctx_mean'),
    'Product (sigma union)'     : load_aff('mean_product'),
    'Product (sigma per-space)' : load_aff('mean_product_pss'),
}

# %% colab={"base_uri": "https://localhost:8080/"} id="9dc3ac56" executionInfo={"status": "ok", "timestamp": 1783157476927, "user_tz": -210, "elapsed": 22172, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="c2c23848-6aa4-4e3b-ff91-c3c42a1d97d1"
from interpretable_ssl.evaluation.metric_helpers.affinity_metrics import compare_affinities

ct_labels    = ad.obs[CT_KEY].astype(str).values
niche_labels = ad.obs[NICHE_KEY].astype(str).replace('nan', 'Unknown').values

results = compare_affinities(affinities, ct_labels, niche_labels)
print(results.round(4))

# %% [markdown] id="5f8149d8"
# ## Visualize
#
# ### Trade-off scatter (hero figure)
#
# x = cell-type weighted purity, y = within-CT niche weighted purity.
# Top-right corner = best of both worlds.

# %% colab={"base_uri": "https://localhost:8080/", "height": 752} id="8a099efd" executionInfo={"status": "ok", "timestamp": 1783157479185, "user_tz": -210, "elapsed": 2262, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="e804dcbe-e9d2-4fd2-afbc-805577ed8704"
import matplotlib.pyplot as plt

colors = {
    'RBF PCA'                   : '#888888',
    'RBF mean-PCA'              : '#4C8BE0',
    'Product (sigma union)'     : '#27AE60',
    'Product (sigma per-space)' : '#E05C4C',
}

fig, ax = plt.subplots(figsize=(6, 5))
for method, row in results.iterrows():
    ax.scatter(row['ct_weighted_purity'], row['within_ct_niche_weighted_purity'],
               color=colors.get(method, 'black'), s=120, zorder=3, label=method)
    ax.annotate(method, (row['ct_weighted_purity'], row['within_ct_niche_weighted_purity']),
                textcoords='offset points', xytext=(6, 4), fontsize=8)

ax.set_xlabel('Cell-type weighted purity', fontsize=11)
ax.set_ylabel('Within-CT niche weighted purity', fontsize=11)
ax.set_title('Affinity trade-off', fontsize=12)
ax.legend(fontsize=8, loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'r2_affinity_tradeoff.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown] id="76126d93"
# ### Bar charts

# %% colab={"base_uri": "https://localhost:8080/", "height": 602} id="6aa2fdeb" executionInfo={"status": "ok", "timestamp": 1783157479803, "user_tz": -210, "elapsed": 617, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="d9db7d4f-76ab-4307-e383-04644193364b"
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, col, title in zip(
    axes,
    ['ct_weighted_purity', 'within_ct_niche_weighted_purity'],
    ['Cell-type weighted purity', 'Within-CT niche weighted purity'],
):
    bar_colors = [colors.get(m, 'steelblue') for m in results.index]
    results[col].plot(kind='bar', ax=ax, color=bar_colors, edgecolor='white', width=0.6)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(axis='y', alpha=0.3)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'r2_affinity_bars.pdf'), bbox_inches='tight')
plt.show()

# %% [markdown] id="f31ef52c"
# ### UMAP of each embedding
#
# Colored by cell type and by niche — shows what structure each *feature space* sees.
# Only `RBF PCA` and `RBF mean-PCA` have a standalone embedding to visualize (`X_pca`,
# `X_mean_pca`); the product affinities combine both graphs and don't correspond to a
# single feature space, so they're omitted from this panel but still compared quantitatively above.

# %% colab={"base_uri": "https://localhost:8080/"} id="3bfefa92" executionInfo={"status": "ok", "timestamp": 1783157658225, "user_tz": -210, "elapsed": 178420, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="23fd5457-f863-4a8a-e0cb-884e55c509d3"
from interpretable_ssl.augmenters.graph_generator import compute_mean_pca_context

if 'X_mean_pca' not in ad.obsm:
    compute_mean_pca_context(ad, k=K_CTX, obsm_key='X_mean_pca')

embeddings = {
    'PCA'      : 'X_pca',
    'mean_pca' : 'X_mean_pca',
}

for label, key in embeddings.items():
    sc.pp.neighbors(ad, use_rep=key, n_neighbors=30, key_added=f'neighbors_{label}')
    sc.tl.umap(ad, neighbors_key=f'neighbors_{label}', min_dist=0.3)
    ad.obsm[f'X_umap_{label}'] = ad.obsm['X_umap'].copy()
    print(f'UMAP done: {label}')

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="6a9ff3ed" executionInfo={"status": "ok", "timestamp": 1783157677522, "user_tz": -210, "elapsed": 19301, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="e6eb7337-7a01-48a2-e4a9-c9eef359fc30"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

color_keys = [CT_KEY, NICHE_KEY]
emb_labels = list(embeddings.keys())
n_rows = len(color_keys)
n_cols = len(emb_labels)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

for row_i, ck in enumerate(color_keys):
    categories = ad.obs[ck].astype('category').cat.categories
    palette = plt.cm.tab20.colors
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    cell_colors = [color_map[c] for c in ad.obs[ck].astype(str)]

    for col_i, emb_label in enumerate(emb_labels):
        ax = axes[row_i][col_i]
        X_umap = ad.obsm[f'X_umap_{emb_label}']
        ax.scatter(X_umap[:, 0], X_umap[:, 1],
                   c=cell_colors, s=1, alpha=0.4, rasterized=True)
        ax.set_title(f'{emb_label} — colored by {ck}', fontsize=10)
        ax.axis('off')

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color_map[c], markersize=6, label=c)
               for c in categories]
    axes[row_i][-1].legend(handles=handles, fontsize=6,
                            loc='center left', bbox_to_anchor=(1, 0.5),
                            ncol=max(1, len(categories) // 20))

plt.suptitle('UMAP of embeddings: does niche cluster within cell type?', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'r2_affinity_umap_comparison.pdf'),
            bbox_inches='tight', dpi=150)
plt.show()

# %% [markdown] id="8e195ddd"
# ### Per-cell-type breakdown
#
# Which cell types benefit most from niche-aware (context-informed) affinity?

# %% colab={"base_uri": "https://localhost:8080/"} id="daafb78c" executionInfo={"status": "ok", "timestamp": 1783157682721, "user_tz": -210, "elapsed": 5194, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="c43b4afa-2356-49b2-9942-b5990b4a7ce0"
from interpretable_ssl.evaluation.metric_helpers.affinity_metrics import weighted_purity_within_group
import pandas as pd

REF_METHOD = 'Product (sigma per-space)'   # column used to sort the heatmap/table

per_ct = {}
for method, aff in affinities.items():
    per_cell = weighted_purity_within_group(aff, ct_labels, niche_labels)
    df = ad.obs[[CT_KEY]].copy()
    df['niche_purity'] = per_cell
    per_ct[method] = df.groupby(CT_KEY)['niche_purity'].mean()

per_ct_df = pd.DataFrame(per_ct)
print(per_ct_df.round(3).sort_values(REF_METHOD, ascending=False))

# %% colab={"base_uri": "https://localhost:8080/", "height": 902} id="f0e24dc0" executionInfo={"status": "ok", "timestamp": 1783157684054, "user_tz": -210, "elapsed": 1331, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}} outputId="1084b6b4-bae2-46f2-881e-d2fc6c1cd664"
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    per_ct_df.sort_values(REF_METHOD, ascending=False),
    annot=True, fmt='.2f', cmap='YlOrRd',
    vmin=0, vmax=1, ax=ax, linewidths=0.4,
)
ax.set_title('Within-CT niche weighted purity by cell type', fontsize=11)
ax.set_xlabel('Affinity method')
ax.set_ylabel('Cell type')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, 'r2_affinity_per_ct_heatmap.pdf'), bbox_inches='tight')
plt.show()

# %% id="irN4ilpbbRS7" executionInfo={"status": "ok", "timestamp": 1783157684057, "user_tz": -210, "elapsed": 2, "user": {"displayName": "Fatemeh Hashemi", "userId": "10225498037645406633"}}

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

# %% [markdown]
# # Train scProto — affinity comparison (arbf / mean product / BANKSY)
#
# Trains the full scProto pipeline (CVAE pretrain + prototype-UMAP) on **s28nsc**
# once per affinity graph:
#
# | Affinity | `affinity_type` | What it encodes |
# |---|---|---|
# | PCA only (baseline) | `arbf` | Transcriptomics only, adaptive RBF (SEACells kernel) |
# | Mean product | `mean_product` | `rbf(own PCA) x rbf(mean-neighbour PCA)` — soft AND logic, k=35 spatial context |
# | BANKSY | `banksy0.5` | `concat(own PCA, mean-neighbour PCA)` -> single RBF, alpha=0.5 |
#
# Each run trains its own encoder + prototypes end-to-end and saves checkpoints
# (pretrain checkpoint, UMAP checkpoint, metacells, metrics) under
# `MODEL_DIR/s28nsc/<model_name>/` — see the **Reload elsewhere** section at the
# bottom for how another notebook picks these back up for metric comparison
# without retraining.

# %% [markdown]
# ## Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# !pip install -q scarches SEACells faiss-gpu-cu12 scib-metrics

# %%
# %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py

# %% [markdown]
# ## Config
#
# Same dataset for all three runs so the resulting models are directly
# comparable. Hyperparameters mirror the established `s28nsc` recipe from
# `train_scproto.ipynb` (`LAMBDA_PROTO_UMAP_PRECON` + `nassoc_agg='max'`).

# %%
from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
from interpretable_ssl.evaluation.spatial_immune_task import NSCLC_EVAL_GROUPS

DS_ID = 's28nsc'

COMMON_KWARGS = dict(
    cvae_epochs=50,
    train_epochs=50,
    eval_freq=3,
    patience=6,
    batch_size=1024,
    umap_steps_per_epoch=500,
    niche_key='niches_3D',
    target_groups=NSCLC_EVAL_GROUPS,
    lambda_config=LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'},
)

# Set True on a re-run to skip training and just reload each saved UMAP checkpoint.
LOAD_UMAP = False

AFFINITIES = ['arbf', 'mean_product', 'banksy0.5']

trainers = {}
results = {}
mc_adatas = {}

# %% [markdown]
# ## Train — one cell per affinity
#
# Each cell is independent — skip/re-run any one without affecting the others.

# %% [markdown]
# ### arbf (PCA-only baseline)

# %%
t, res, mc_ad = run_mc_task(DS_ID, affinity_type='arbf', load_umap=LOAD_UMAP, **COMMON_KWARGS)
trainers['arbf'], results['arbf'], mc_adatas['arbf'] = t, res, mc_ad
print(res)

# %% [markdown]
# ### mean_product

# %%
t, res, mc_ad = run_mc_task(DS_ID, affinity_type='mean_product', load_umap=LOAD_UMAP, **COMMON_KWARGS)
trainers['mean_product'], results['mean_product'], mc_adatas['mean_product'] = t, res, mc_ad
print(res)

# %% [markdown]
# ### BANKSY (alpha=0.5)

# %%
t, res, mc_ad = run_mc_task(DS_ID, affinity_type='banksy0.5', load_umap=LOAD_UMAP, **COMMON_KWARGS)
trainers['banksy0.5'], results['banksy0.5'], mc_adatas['banksy0.5'] = t, res, mc_ad
print(res)

# %% [markdown]
# ## Compare metrics across affinities

# %%
import pandas as pd

metrics_df = pd.DataFrame(results).T
metrics_df

# %% [markdown]
# ## Visualize — UMAP per affinity
#
# Colored by cell type and (3D) niche, prototypes overlaid.

# %%
for name, t in trainers.items():
    print(f'--- {name} ---')
    fig, proto_labels = t.plot_umap_simple(
        color_key=['celltypes', 'niches_3D'],
        show_proto_nums=False,
    )

# %% [markdown]
# ## Reload elsewhere — for metric comparison in other notebooks
#
# Each run's artifacts live under `MODEL_DIR/s28nsc/<model_name>/`
# (`umap_checkpoint.pth`, `metacells.h5ad`, `metrics.json`, `clusters.npz`, ...).
# The model name is derived from the hyperparameters passed to `get_trainer` /
# `run_mc_task`, so **another notebook reloads a specific run by calling
# `run_mc_task` with the exact same arguments plus `load_umap=True`** — this
# skips training entirely and just loads the saved checkpoint:
#
# ```python
# from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
# from interpretable_ssl.evaluation.spatial_immune_task import NSCLC_EVAL_GROUPS
#
# COMMON_KWARGS = dict(
#     cvae_epochs=50, train_epochs=50, eval_freq=3, patience=6,
#     batch_size=1024, umap_steps_per_epoch=500,
#     niche_key='niches_3D', target_groups=NSCLC_EVAL_GROUPS,
#     lambda_config=LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'},
# )
#
# t_arbf, res_arbf, mc_arbf = run_mc_task(
#     's28nsc', affinity_type='arbf', load_umap=True, **COMMON_KWARGS
# )
# # same for 'mean_product' and 'banksy0.5'
# ```
#
# `mc_ad` / `mc_adatas[...]` is the metacell-level AnnData (K prototypes x genes);
# `t.train_ds.adata.obs['metacell_id']` has the per-cell prototype assignment.
# Both are also saved to disk (`metacells.h5ad`) so they can be loaded with
# `sc.read_h5ad(...)` directly if a fresh trainer isn't needed.

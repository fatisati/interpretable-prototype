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
# # Interpretable SSL — Example Notebook
#
# This notebook shows how to find metacells on your own single-cell or spatial transcriptomics data.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatisati/interpretable-prototype/blob/master/notebooks/example.ipynb)

# %% [markdown]
# ## 1. Install

# %%
# !pip install -q faiss-gpu-cu12
# !pip install -q git+https://github.com/fatisati/interpretable-prototype.git

# %% [markdown]
# ## 2. Mount Google Drive (if using Colab)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 3. Train
#
# - `MODEL_DIR`: where checkpoints are saved — use Google Drive so they persist across sessions
# - For **spatial data**, set `affinity_type='ctx_umap'` and make sure `adata.obsm['spatial']` exists

# %%
import os
os.environ['MODEL_DIR'] = '/content/drive/MyDrive/models/'

from interpretable_ssl.experiments.tasks import find_metacells

t, res, mc_adata = find_metacells(
    '/content/drive/MyDrive/data/your_data.h5ad',  # path to your h5ad file
    label_key='cell_type',       # adata.obs column with cell type labels (required, for evaluation only)
    batch_key='sample',          # adata.obs column with batch labels (optional)
    cvae_epochs=50,
    train_epochs=50,
    eval_freq=5,
    patience=20,
    # affinity_type='ctx_umap',  # uncomment for spatial data
    result_save_path='/content/drive/MyDrive/results/',
)

# %% [markdown]
# ## 4. Visualize
#
# Plot the embedding space with cells colored by label and prototypes (metacells) overlaid.

# %%
fig, proto_labels = t.plot_umap_simple(
    color_key=['cell_type'],  # add more columns to color by, e.g. ['cell_type', 'sample']
    show_proto_nums=False,
)

# %% [markdown]
# ## 5. Inspect results

# %%
# Evaluation metrics
print(res)

# %%
# Metacell gene expression: AnnData [K prototypes x genes]
print(mc_adata)

# %%
# Metacell ID per cell
print(t.train_ds.adata.obs['metacell_id'])

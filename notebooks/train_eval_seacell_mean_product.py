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
# # SEACells archetypal analysis on our own `mean_product` affinity graph
#
# Runs *only* the archetypal-analysis half of SEACells (Frank-Wolfe fit of
# `‖M − M·B·A‖²` with hard simplex constraints on `A`/`B`) on our own
# `mean_product` affinity graph (`rbf(own PCA) × rbf(mean-neighbour PCA)` —
# see `interpretable_ssl/augmenters/graph_generator.py:rbf_product`), instead
# of letting SEACells build its own PCA-RBF kernel.
#
# Both halves of the pipeline use our own graph, not SEACells':
# - **kernel**: `model.add_precomputed_kernel_matrix(aff)` — skips
#   `construct_kernel_matrix()` entirely.
# - **archetype seeding**: `waypoint_archetype_indices(aff, ...)` — a
#   diffusion-map + greedy-MaxMin selection computed directly from `aff`
#   (adapted from `SCProtoTrainer._init_prototypes_waypoint`), passed via
#   `model.fit(initial_archetypes=...)`. Skips SEACells' own
#   `initialize_archetypes()`, which otherwise needs
#   `ad.obsm[build_kernel_on]` and runs an unrelated palantir diffusion-map
#   call.
#
# The Frank-Wolfe fit itself is SEACells' real, tested implementation
# (`use_sparse=True` — keeps `K`/`A`/`B` sparse throughout), not a
# reimplementation — there's no existing rewrite of that part in this repo,
# and its per-iteration cost is bounded by `n_SEACells × n_cells`, not
# `n_cells²`, so there wasn't a case for writing a new one.
#
# All of the above is wrapped in a single call —
# `train_seacell_own_affinity()` in `seacell_train.py` — that mirrors
# `train_seacell()`'s pattern (see `train_eval_seacell_diffusion.ipynb`):
# dataset config comes from `DATASETS[ds_id]`, the affinity is loaded from
# `graph_dir` if already cached, and the result is loaded instead of
# recomputed if `MODEL_DIR/{ds_id}/seacell_mean_product/` already exists and
# `mode != 'train'` — safe to re-run.

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

# %%
DS_ID = 's28nsc'

AFFINITY_TYPE   = 'mean_product'
K_AFF           = 50     # affinity graph kNN (top-k pruning in rbf_product)
PER_SPACE_SIGMA = True   # stricter AND logic — see rbf_product docstring
N_WAYPOINT_EIGS = 10     # diffusion-map eigenvectors for archetype seeding
MAX_ITER        = 100    # outer Frank-Wolfe iterations

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# %% [markdown]
# ## Run
#
# Set `mode='eval'` on a re-run to skip straight to loading the saved result
# instead of recomputing.

# %%
ad, SEACell_ad, model = train_seacell_own_affinity(
    DS_ID, mode='train', affinity_type=AFFINITY_TYPE,
    k=K_AFF, per_space_sigma=PER_SPACE_SIGMA,
    n_waypoint_eigs=N_WAYPOINT_EIGS, max_iter=MAX_ITER,
    graph_dir=GRAPH_DIR,
)

# %% [markdown]
# ## Sanity checks

# %%
import numpy as np
import matplotlib.pyplot as plt

assignments = np.array(model.A_.argmax(axis=0)).ravel()
n_unique = len(np.unique(assignments))
n_archetypes = model.B_.shape[1]
print(f'{n_unique} unique SEACells from {n_archetypes} archetypes '
      f'(RSS: {model.RSS_iters[0]:.2f} -> {model.RSS_iters[-1]:.2f} '
      f'over {len(model.RSS_iters)} iterations)')

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(model.RSS_iters)
axes[0].set_title('Reconstruction error (RSS) over iterations')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel(r'$\|M - MBA\|$')
axes[0].grid(alpha=0.3)

sizes = SEACell_ad.obs['Pseudo-sizes'] if 'Pseudo-sizes' in SEACell_ad.obs else \
    ad.obs['SEACell'].value_counts()
axes[1].hist(sizes, bins=40)
axes[1].set_title('Cells per SEACell')
axes[1].set_xlabel('Cell count')
axes[1].set_ylabel('Number of SEACells')
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Metrics
#
# Reuses the standard SEACell eval pipeline (same one `train_eval_seacell_diffusion`
# uses for its `seacell_X_diffusion` row) — pass `build_kernel_on=AFFINITY_TYPE`
# so it looks under `seacell_mean_product/` instead of the default `seacell/`.

# %%
eval_seacell_task1(DS_ID, build_kernel_on=AFFINITY_TYPE, k_neighbors=K_AFF,
                   affinity_type=AFFINITY_TYPE, graph_dir=GRAPH_DIR)
eval_seacell_task2(DS_ID, build_kernel_on=AFFINITY_TYPE)
eval_seacell_task3(DS_ID, build_kernel_on=AFFINITY_TYPE)

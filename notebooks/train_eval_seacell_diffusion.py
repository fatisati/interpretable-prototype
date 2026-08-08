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
# # SEACells on a diffusion-compacted affinity graph — isolating the compaction question
#
# Context: `train_eval_sim_recon_diffusion.ipynb` compares scProto (+ various
# `sim_recon` targets) against SEACells(PCA), and consistently finds SEACells
# far ahead on cell-type/niche purity (see `files/sim_recon_competing_losses.md`).
# That comparison can't cleanly separate two different questions:
#
# 1. Does compacting the affinity graph to a low-dimensional diffusion
#    embedding lose fidelity that archetypal analysis needs?
# 2. Does scProto's neural net (five competing loss terms fighting over the
#    same `soft_assign` — see `files/sim_recon_competing_losses.md`) lose
#    fidelity on top of that?
#
# This notebook isolates (1): run SEACells' own archetypal analysis (same
# `SEACells.core.SEACells` call, same Frank-Wolfe optimizer) directly on a
# diffusion embedding of the same `arbf` affinity graph SEACells(PCA) already
# uses — no neural net, no competing losses, nothing else different.
#
# **Which diffusion embedding.** Per
# `files/sim_recon_global_vs_local_compaction.md`'s Eckart-Young derivation:
# weighting each eigenvector of `L_sym = D^-1/2 A D^-1/2` by `sqrt(eigenvalue)`
# (`diffusion_t=0.5`) makes the Gram matrix of the top-`n_eigs` weighted
# eigenvectors *exactly* the optimal rank-`n_eigs` reconstruction of the
# affinity matrix itself (not `t=1`, which reconstructs `L_sym²`, and not the
# unweighted `t=0` scProto's `sim_recon` uses by default, which is a
# Laplacian-eigenmap embedding with no such reconstruction property). So
# `t=0.5` is the only setting that makes "SEACells on this embedding" a fair
# test of "SEACells on a compacted-but-faithful version of its own kernel."
#
# See `files/sim_recon_investigation_index.md` for the full investigation
# this notebook is part of.

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
DIFFUSION_AFFINITY_TYPE = 'arbf'   # same kernel SEACells(PCA) itself is built on
DIFFUSION_N_EIGS = 1024
DIFFUSION_T = 0.5                  # sqrt(eigenvalue) — see markdown above for why

NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# %% [markdown]
# ## SEACells baselines
#
# Both calls skip straight to eval if a saved run is already found under
# `MODEL_DIR/{DS_ID}/seacell_X_pca` / `seacell_X_diffusion` — safe to re-run.

# %% [markdown]
# ### SEACells (PCA) — the existing baseline

# %%
train_seacell(DS_ID, mode='train', build_kernel_on='X_pca')

# %% [markdown]
# ### SEACells (diffusion, t=0.5) — the new comparison point
#
# Builds the `arbf` affinity (same PCA-based adaptive-RBF kernel
# `SEACells(PCA)` uses internally), computes its `t=0.5`-weighted
# 1024-dim diffusion embedding, stores it in `ad.obsm['X_diffusion']`, then
# hands that straight to the same `SEACells.core.SEACells(...)`
# construct-kernel / initialize-archetypes / fit pipeline as the PCA run
# above — the only thing that differs between this cell and the one above is
# which embedding the kernel is built on.

# %%
train_seacell(
    DS_ID, mode='train', build_kernel_on='X_diffusion',
    diffusion_affinity_type=DIFFUSION_AFFINITY_TYPE,
    diffusion_n_eigs=DIFFUSION_N_EIGS,
    diffusion_t=DIFFUSION_T,
)

# %% [markdown]
# ## Comparison
#
# scProto rows are loaded (not retrained) purely for context alongside the
# two SEACells variants — `arbf+diffusion_t0.5` is left out here since it
# hasn't been trained yet as of this notebook being written (see
# `train_eval_sim_recon_diffusion.ipynb` cell for that run); add it to
# MODEL_KEYWORDS below once it exists.

# %%
from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
from interpretable_ssl.evaluation.spatial_immune_task import NSCLC_EVAL_GROUPS

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

model_names = {}
try:
    t, res, mc_ad = run_mc_task(DS_ID, affinity_type='arbf', load_umap=True, **COMMON_KWARGS)
    model_names['arbf'] = t.get_model_name()
    t, res, mc_ad = run_mc_task(
        DS_ID, affinity_type='arbf', load_umap=True,
        **(COMMON_KWARGS | {'lambda_config': COMMON_KWARGS['lambda_config'] | {
            'lambda_sim_recon': 1.0, 'sim_recon_target': 'diffusion', 'sim_recon_n_eigs': 128,
        }})
    )
    model_names['arbf+diffusion'] = t.get_model_name()
except Exception as e:
    print(f"Skipping scProto reference rows (not trained/loadable yet): {e}")

MODEL_KEYWORDS = {
    'seacell':           'SEACells (PCA)',
    'seacell_X_diffusion': f'SEACells (diffusion t={DIFFUSION_T})',
    **{model_names[k]: v for k, v in {
        'arbf': 'scProto (arbf)',
        'arbf+diffusion': 'scProto + sim-recon/diffusion, t=0 (arbf)',
    }.items() if k in model_names},
}
MODEL_KEYWORDS

# %% [markdown]
# ### 1. Cell-type purity table

# %%
median_ct, q25_ct, q75_ct = fig_celltype_purity_table(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='celltype_purity',
)
format_purity_table(median_ct, q25_ct, q75_ct)

# %%
median_ct.round(3).style.background_gradient(cmap='YlGnBu', vmin=0, vmax=1)

# %% [markdown]
# ### 2. Niche purity heatmap — one panel per model

# %%
fig_all_celltype_niche_heatmap(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity',
    save_path=os.path.join(GRAPH_DIR, 'seacell_diffusion_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — SEACells(diffusion) vs. SEACells(PCA)
#
# (model − SEACells(PCA)) per (cell type, niche) cell. This is the direct
# answer to the compaction question: if diffusion compaction at `t=0.5` were
# lossless, this heatmap would be ~uniformly near zero.

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference='seacell',
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'seacell_diffusion_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'seacell_diffusion_tradeoff.pdf'),
)

# %% [markdown]
# ### 5. Niche purity summary table

# %%
median_niche, q25_niche, q75_niche = fig_celltype_purity_table(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity',
)
format_purity_table(median_niche, q25_niche, q75_niche)

# %%
median_niche.round(3).style.background_gradient(cmap='YlOrRd', vmin=0, vmax=1)

# %% [markdown]
# ### 6. Distribution per cell type (violin)

# %%
fig_purity_violin(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity',
    save_path=os.path.join(GRAPH_DIR, 'seacell_diffusion_purity_violin.pdf'),
)

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
# # scProto + `sim_recon_target='full'` with column-subsampling (`sim_recon_neg_sample`) vs SEACells
#
# `full`-target `sim_recon` reconstructs each cell's actual (raw-value) affinity row —
# unlike `diffusion` mode, it never rescales anything into a shared, unit-normalized
# eigenbasis, so a strong real neighbor stays numerically large and a true non-neighbor
# stays exactly 0. That's the property that makes plain MSE naturally prioritize getting
# real, important relationships right (see `files/sim_recon_global_vs_local_compaction.md`
# for why that property breaks for `diffusion` coordinates).
#
# The naive cost of `full` mode: for a single-batch, ~58k-cell dataset like `s28nsc`,
# reconstructing every cell against *every other cell in its batch* every step is
# expensive. `sim_recon_neg_sample` (`configs/defaults.py`, implemented in
# `trainers/scproto.py`'s `sim_recon_loss` block) compacts that *computation*, not the
# *representation*: always keep each row's true-neighbor columns, add a random sample of
# `sim_recon_neg_sample` zero-columns (freshly redrawn every step — nothing permanently
# discarded), and run class-balanced MSE on just that reduced column set. Same raw
# affinity values, same natural magnitude-based prioritization, far fewer columns
# touched per step. See `files/sim_recon_full_column_subsampling.md` for the full
# reasoning.
#
# **What this notebook tests**: does `full` + `sim_recon_neg_sample`, trained against
# the plain-`arbf` baseline, actually move purity/niche-purity toward SEACells — and how
# far. SEACells is loaded in **eval-only mode** here (no retraining) — reuses whatever
# was already trained via `train_eval_sim_recon.ipynb` / `train_eval_sim_recon_diffusion.ipynb`,
# since it's the same fixed reference point across all of these notebooks.
#
# `SIM_RECON_NEG_SAMPLE` below is a first guess (not yet empirically chosen — flagged as
# an open question in `files/sim_recon_investigation_index.md`). Tune from here based on
# training speed vs. `sim_recon` loss quality (watch `pred_std`/`target_std` — for `full`
# mode that's printed differently, see the epoch log's `sim_recon=...` line).

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
from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
from interpretable_ssl.evaluation.spatial_immune_task import NSCLC_EVAL_GROUPS
import pandas as pd

DS_ID = 's28nsc'
AFFINITY = 'arbf'

# Same first-guess scale as the other sim_recon notebooks — affinity values already
# live in ~[0,1], comparable to nassoc's own terms.
LAMBDA_SIM_RECON = 1.0

# Column-subsampling size: mean degree here is ~74.5 (real neighbors, always kept
# regardless of this number) out of ~58,423 possible columns per row. 1000 is a first
# guess — ~13x the real neighbor count, ~58x fewer columns than the full unsampled
# batch — not yet tuned. See the module docstring above and
# files/sim_recon_investigation_index.md's open-questions section.
SIM_RECON_NEG_SAMPLE = 1000

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

# Baseline: set True if you already trained plain arbf elsewhere (this notebook,
# train_eval_sim_recon.ipynb, train_eval_sim_recon_diffusion.ipynb, or
# train_scproto_spatial.ipynb all save/load under the same model-name scheme) and just
# want to reload it.
LOAD_BASELINE = True

# The negsample run is the new thing this notebook exists to produce — False trains it;
# flip to True on a re-run to just reload.
LOAD_NEGSAMPLE = False

trainers = {}
results = {}
mc_adatas = {}
model_names = {}  # label -> exact saved model directory name


def train_or_load(label, affinity_type, load, extra_lambda=None):
    """Run (or reload) one scProto config and record its exact model directory name."""
    kwargs = COMMON_KWARGS if not extra_lambda else COMMON_KWARGS | {
        'lambda_config': COMMON_KWARGS['lambda_config'] | extra_lambda
    }
    t, res, mc_ad = run_mc_task(DS_ID, affinity_type=affinity_type, load_umap=load, **kwargs)
    trainers[label], results[label], mc_adatas[label] = t, res, mc_ad
    model_names[label] = t.get_model_name()
    print(f'{label}: {res}')
    return t, res, mc_ad


# %% [markdown]
# ## Train / load — arbf baseline + full-target sim-recon with column-subsampling
#
# Each cell is independent — skip/re-run either without affecting the other.

# %% [markdown]
# ### Baseline (no sim-recon)

# %%
train_or_load('arbf', AFFINITY, load=LOAD_BASELINE)

# %% [markdown]
# ### +sim-recon (`full` target, column-subsampled)
#
# Everything else identical to the baseline above — a clean single-variable ablation.
# Watch the epoch log's `sim_recon=...` value: it should be a small, decreasing number
# (class-balanced weighted MSE over the sampled columns), not stuck near a trivial
# floor the way the `diffusion`-target collapse looked.

# %%
train_or_load('arbf+full_negsample', AFFINITY, load=LOAD_NEGSAMPLE,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'full',
                              'sim_recon_neg_sample': SIM_RECON_NEG_SAMPLE})

# %% [markdown]
# ## SEACells (PCA) — eval only, no retraining
#
# `mode='eval'` skips training entirely and just checks the saved run exists — this is
# the same fixed SEACells reference point used across `train_eval_sim_recon.ipynb` and
# `train_eval_sim_recon_diffusion.ipynb`. If no saved run exists yet, switch to
# `mode='train'` once to produce it, then back to `'eval'`.

# %%
train_seacell(DS_ID, mode='eval', build_kernel_on='X_pca')

# %% [markdown]
# ## Quick numeric comparison — the two in-memory scProto runs
#
# Straight from the metrics `run_mc_task` already returned — no disk lookup.

# %%
metrics_df = pd.DataFrame(results).T
metrics_df

# %% [markdown]
# ## Full comparison — cell-type purity vs. niche purity, incl. SEACells
#
# Same metric definitions and plots as the other `sim_recon` notebooks — reproduced
# here so this notebook is a complete, standalone record of the experiment.
# `NICHE_KEY='niches_2D'` here by choice; it doesn't need to match training's own
# `niche_key` (`niches_3D` above) — this just scores purity against whichever
# ground-truth niche column you point it at.

# %%
NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# Exact model directory names, captured right after training/loading above — exact
# match (not keyword substring), same reasoning as the other sim_recon notebooks:
# a baseline run's name is always a literal substring of its sim-recon counterpart's
# name, so fuzzy keyword matching would silently match the wrong run.
MODEL_KEYWORDS = {
    model_names['arbf']:                 'scProto (arbf)',
    model_names['arbf+full_negsample']:  f'scProto + sim-recon/full (negsample={SIM_RECON_NEG_SAMPLE})',
    'seacell':                           'SEACells (PCA)',
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_negsample_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — does column-subsampled full sim-recon beat the plain-arbf baseline?
#
# (model − arbf-baseline) per (cell type, niche) cell. Redder than the baseline is the
# direct answer to whether `full` + `sim_recon_neg_sample` closes any of the purity gap
# to SEACells.

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference=model_names['arbf'],
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_negsample_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_negsample_tradeoff.pdf'),
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_negsample_purity_violin.pdf'),
)

# %% [markdown]
# ## Visualize — UMAP per run
#
# Colored by cell type and (3D) niche, prototypes overlaid.

# %%
for name, t in trainers.items():
    print(f'--- {name} ---')
    fig, proto_labels = t.plot_umap_simple(
        color_key=['celltypes', 'niches_3D'],
        show_proto_nums=False,
    )

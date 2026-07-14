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
# # scProto + cell-cell similarity reconstruction (`lambda_sim_recon`) — train & eval
#
# Self-contained notebook for one experiment: does adding the cell-cell
# similarity reconstruction loss close the resolution gap between scProto and
# SEACells? Background: SEACells' archetypal analysis reconstructs each cell's
# full row of the affinity matrix (forcing archetypes to resolve genuine
# per-cell heterogeneity); scProto's existing losses (nassoc, proto_usage)
# only ever see a K×K summary, so a large, internally well-connected but
# heterogeneous prototype pays no penalty. `lambda_sim_recon` (off by default,
# see `configs/defaults.py`) adds that missing cell-level pressure: it decodes
# each prototype into a predicted target and reconstructs it through `S`, with
# both `S` and the prototypes kept trainable through this loss specifically
# (see `scproto.py`'s `sim_recon_loss` block for the full reasoning on why
# that differs from `proto_recon_loss`'s detached `S`).
#
# Two target modes, both trained and compared here (`sim_recon_target`):
# - **`full`** — reconstruct each cell's actual row of the affinity graph.
#   Most literal match to what SEACells' RSS objective does.
# - **`diffusion`** — regress to a small precomputed per-cell diffusion-map
#   coordinate instead of the full row. Cheaper (no O(n_cells) decoder
#   output), but the compression could in principle discard some of the
#   resolving signal the `full` target has — that's an open question, not
#   an assumption, which is why both get trained and compared side by side
#   rather than picking one up front.
#
# **Scope for this pass: `arbf` only.** This trains exactly 3 scProto configs
# (arbf baseline, +full, +diffusion) plus SEACells — one affinity at a time
# keeps each pass cheap and isolates whether either sim-recon target closes
# the gap with SEACells before spending Colab time on other affinities.
#
# Runs are matched for evaluation by **exact** saved model directory name
# (captured right after training/loading), not by keyword substring — a
# baseline run's name is always a literal substring of both its sim-recon
# counterparts' names, so fuzzy keyword matching (as used in
# `scproto_spatial_comparison.ipynb`, safe there because it only ever had one
# variant per affinity) would silently match the wrong run. Exact-name
# matching sidesteps that entirely.

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

# lambda_sim_recon=1.0 is a first guess — affinity values already live in
# ~[0,1], comparable scale to nassoc's own terms. Tune from here if purity
# doesn't move, or moves too aggressively at the cost of the other losses.
LAMBDA_SIM_RECON = 1.0
SIM_RECON_N_EIGS = 128  # only used by sim_recon_target='diffusion'

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

# Baseline: set True if you already trained arbf in
# train_scproto_spatial.ipynb and just want to reload that checkpoint here.
# Set False to train it fresh in this notebook (fully self-contained, just slower).
LOAD_BASELINE = True

# The sim-recon runs (both targets) are the new thing this notebook exists to
# produce — False trains them; flip to True on a re-run to just reload.
LOAD_SIMRECON = False

trainers = {}
results = {}
mc_adatas = {}
model_names = {}  # label -> exact saved model directory name (see note above on why exact, not keyword)


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
# ## Train / load — arbf, three configs
#
# Each cell is independent — skip/re-run any one without affecting the others.

# %% [markdown]
# ### Baseline (no sim-recon)

# %%
train_or_load('arbf', AFFINITY, load=LOAD_BASELINE)

# %% [markdown]
# ### +sim-recon (`full` target)
#
# Everything else identical to the baseline above — a clean single-variable
# ablation. Reconstructs each cell's actual affinity-graph row.

# %%
train_or_load('arbf+full', AFFINITY, load=LOAD_SIMRECON,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'full'})

# %% [markdown]
# ### +sim-recon (`diffusion` target)
#
# Same ablation, but regresses to a precomputed `SIM_RECON_N_EIGS`-dim
# diffusion-map coordinate per cell instead of the full affinity row —
# cheaper, and the question this notebook exists to answer is whether that
# compression costs any of the resolution benefit the `full` target gives.

# %%
train_or_load('arbf+diffusion', AFFINITY, load=LOAD_SIMRECON,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'diffusion',
                              'sim_recon_n_eigs': SIM_RECON_N_EIGS})

# %% [markdown]
# ## SEACells (PCA) baseline
#
# `train_seacell` skips training and returns immediately if a saved run is
# already found — safe to call every time.

# %%
train_seacell(DS_ID, mode='train', build_kernel_on='X_pca')

# %% [markdown]
# ## Quick numeric comparison — the three in-memory scProto runs
#
# Straight from the metrics `run_mc_task` already returned — no disk lookup,
# so this part can't be affected by the exact-name-matching note above.

# %%
metrics_df = pd.DataFrame(results).T
metrics_df

# %% [markdown]
# ## Full comparison — cell-type purity vs. niche purity, incl. SEACells
#
# Same metric definitions and plots as `scproto_spatial_comparison.ipynb`
# (see that notebook's markdown for the exact `celltype_purity` /
# `niche_purity` formulas) — reproduced here so this notebook is a complete,
# standalone record of the experiment. `NICHE_KEY='niches_2D'` here by choice;
# it doesn't need to match training's own `niche_key` (`niches_3D` above) —
# this just scores purity against whichever ground-truth niche column you
# point it at.

# %%
NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# Exact model directory names, captured right after training/loading above —
# see the top-of-notebook note on why this must be exact, not a keyword.
MODEL_KEYWORDS = {
    model_names['arbf']:           'scProto (arbf)',
    model_names['arbf+full']:      'scProto + sim-recon/full (arbf)',
    model_names['arbf+diffusion']: 'scProto + sim-recon/diffusion (arbf)',
    'seacell':                     'SEACells (PCA)',
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — does sim-recon beat the plain-arbf baseline?
#
# (model − arbf-baseline) per (cell type, niche) cell — same convention as
# `scproto_spatial_comparison.ipynb`. Two things to read off this one:
# - Are `+full`/`+diffusion` redder than the baseline — the actual
#   single-variable ablation this notebook exists to answer.
# - Is `+full` redder than `+diffusion` — the direct answer to whether the
#   diffusion compression costs resolution.

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference=model_names['arbf'],
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_tradeoff.pdf'),
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_niche_purity_violin.pdf'),
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

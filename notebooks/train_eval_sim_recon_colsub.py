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
# # scProto + `sim_recon` full-mode column-subsampling (`sim_recon_neg_sample`) — train & eval
#
# Companion to `train_eval_sim_recon.py`, which compared `sim_recon_target='full'` vs
# `'diffusion'` against the plain-arbf baseline. This notebook instead asks: does
# **column-subsampling within `full` mode** (`sim_recon_neg_sample`) give the same
# quality as unrestricted `full`, but faster?
#
# Background: `full` reconstructs each cell's entire affinity-graph row every step —
# decoding every column in the batch, most of which are zero (>99.8% sparse). That's
# what makes `full` slow. `sim_recon_neg_sample` (see `trainers/scproto.py`,
# `configs/defaults.py`) keeps every row's real neighbor columns (the informative
# ones) but decodes only a random sample of the zero columns each step instead of all
# of them — a fresh sample every step, so nothing is permanently discarded the way a
# fixed low-rank basis (`diffusion`) would be. The class-balanced weighting already in
# the loss recomputes itself from whatever ends up in the reduced target, so no extra
# reweighting logic was needed to add this.
#
# **Three configs compared, all `sim_recon_target='full'`:**
# - **`arbf`** — baseline, no `sim_recon`.
# - **`arbf+full`** — unrestricted `full`: every column, every step. The known-good
#   reference point.
# - **`arbf+full+colsub`** — same loss, `sim_recon_neg_sample` set so only a subset of
#   zero columns get decoded per step.
#
# Two things this notebook checks side by side: **(1)** is `arbf+full+colsub`'s purity
# as good as `arbf+full`'s (did subsampling cost any fidelity), and **(2)** is it
# actually faster (wall-clock time per run, captured directly, not inferred).
#
# Runs are matched for evaluation by **exact** saved model directory name, same
# convention as `train_eval_sim_recon.py` — a baseline run's name is always a literal
# substring of its variants' names, so exact-name matching (not keyword fuzzy-match)
# avoids picking up the wrong run.

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
import time

DS_ID = 's28nsc'
AFFINITY = 'arbf'

# Same lambda_sim_recon as train_eval_sim_recon.py, for a clean single-variable
# comparison against those results.
LAMBDA_SIM_RECON = 1.0

# How many zero columns to sample per step (in addition to each row's real
# neighbors), for the colsub run. Start here; if purity drops vs. arbf+full, this is
# too aggressive — raise it. If timing barely improves, this is too close to the
# batch's full column count — lower it.
SIM_RECON_NEG_SAMPLE = 2000

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

# Baseline: set True if you already trained arbf elsewhere and just want to reload
# that checkpoint here. Set False to train it fresh (self-contained, just slower).
LOAD_BASELINE = True

# The sim-recon runs (both full variants) are the new thing this notebook exists to
# produce — False trains them; flip to True on a re-run to just reload.
LOAD_SIMRECON = False

trainers = {}
results = {}
mc_adatas = {}
model_names = {}   # label -> exact saved model directory name
timings = {}       # label -> wall-clock seconds for the train_or_load call


def train_or_load(label, affinity_type, load, extra_lambda=None):
    """Run (or reload) one scProto config, timing it and recording its exact model directory name."""
    kwargs = COMMON_KWARGS if not extra_lambda else COMMON_KWARGS | {
        'lambda_config': COMMON_KWARGS['lambda_config'] | extra_lambda
    }
    t0 = time.time()
    t, res, mc_ad = run_mc_task(DS_ID, affinity_type=affinity_type, load_umap=load, **kwargs)
    elapsed = time.time() - t0
    trainers[label], results[label], mc_adatas[label] = t, res, mc_ad
    model_names[label] = t.get_model_name()
    timings[label] = elapsed
    print(f'{label}: {res}  [{elapsed:.0f}s]')
    return t, res, mc_ad


# %% [markdown]
# ## Train / load — arbf, three configs
#
# Each cell is independent — skip/re-run any one without affecting the others. Timing
# is only meaningful for cells actually trained this run (`load=False`) — a reloaded
# checkpoint's `timings[...]` reflects load time, not train time, so don't compare
# those two kinds of number to each other.

# %% [markdown]
# ### Baseline (no sim-recon)

# %%
train_or_load('arbf', AFFINITY, load=LOAD_BASELINE)

# %% [markdown]
# ### +sim-recon (`full`, unrestricted — every column, every step)
#
# The known-good reference point from `train_eval_sim_recon.py`.

# %%
train_or_load('arbf+full', AFFINITY, load=LOAD_SIMRECON,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'full'})

# %% [markdown]
# ### +sim-recon (`full`, column-subsampled)
#
# Identical loss and target semantics to `arbf+full` above — only the number of zero
# columns decoded per step differs (`sim_recon_neg_sample`). If this notebook's
# purity numbers match `arbf+full` closely, subsampling is free; if they diverge,
# `SIM_RECON_NEG_SAMPLE` is cutting too much signal.

# %%
train_or_load('arbf+full+colsub', AFFINITY, load=LOAD_SIMRECON,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'full',
                              'sim_recon_neg_sample': SIM_RECON_NEG_SAMPLE})

# %% [markdown]
# ## Timing comparison — did column-subsampling actually help?
#
# Direct wall-clock comparison for the two `full`-mode runs above (only meaningful if
# both were actually trained this session, i.e. `LOAD_SIMRECON=False`).

# %%
if 'arbf+full' in timings and 'arbf+full+colsub' in timings:
    t_full = timings['arbf+full']
    t_sub = timings['arbf+full+colsub']
    speedup = t_full / t_sub if t_sub > 0 else float('nan')
    print(f"arbf+full:        {t_full:.0f}s")
    print(f"arbf+full+colsub: {t_sub:.0f}s")
    print(f"speedup: {speedup:.2f}x")
else:
    print("Both full-mode runs need load=False (actually trained this session) for a fair timing comparison.")

# %% [markdown]
# ## SEACells (PCA) baseline
#
# `train_seacell` skips training and returns immediately if a saved run is already
# found — safe to call every time.

# %%
train_seacell(DS_ID, mode='train', build_kernel_on='X_pca')

# %% [markdown]
# ## Quick numeric comparison — the three in-memory scProto runs
#
# Straight from the metrics `run_mc_task` already returned — no disk lookup, so this
# part can't be affected by the exact-name-matching note above.

# %%
metrics_df = pd.DataFrame(results).T
metrics_df

# %% [markdown]
# ## Full comparison — cell-type purity vs. niche purity, incl. SEACells
#
# Same metric definitions and plots as `train_eval_sim_recon.py` / `scproto_spatial_
# comparison.ipynb` — reproduced here so this notebook is a complete, standalone
# record. `NICHE_KEY='niches_2D'` here by choice; it doesn't need to match training's
# own `niche_key` (`niches_3D` above).

# %%
NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# Exact model directory names, captured right after training/loading above — see the
# top-of-notebook note on why this must be exact, not a keyword.
MODEL_KEYWORDS = {
    model_names['arbf']:              'scProto (arbf)',
    model_names['arbf+full']:         'scProto + sim-recon/full (arbf)',
    model_names['arbf+full+colsub']:  'scProto + sim-recon/full+colsub (arbf)',
    'seacell':                        'SEACells (PCA)',
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_colsub_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — does colsub match `full`, and do either beat baseline?
#
# (model − arbf-baseline) per (cell type, niche) cell. Two things to read off this:
# - Are `+full`/`+full+colsub` redder than the baseline — the actual ablation.
# - Is `+full+colsub` close to `+full` (subsampling cost little/no fidelity) or
#   visibly paler (subsampling cut real signal — raise `SIM_RECON_NEG_SAMPLE`).

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference=model_names['arbf'],
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_colsub_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_colsub_tradeoff.pdf'),
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_colsub_niche_purity_violin.pdf'),
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

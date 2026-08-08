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
# # scProto + `sim_recon_target='diffusion'` — train & eval vs. SEACells
#
# Companion to `train_eval_sim_recon.ipynb`, which is now dedicated to the
# `full` target (reconstructing each cell's actual affinity-graph row). This
# notebook is dedicated to the `diffusion` target only (regressing to a
# compact precomputed per-cell diffusion-map/spectral-embedding coordinate
# instead) so the two targets don't keep clobbering each other's runs in one
# shared notebook.
#
# Background on `lambda_sim_recon` itself (why it exists at all — closing the
# per-cell resolution gap between scProto and SEACells' archetypal-analysis
# RSS objective) is in `train_eval_sim_recon.ipynb`'s intro; not repeated
# here.
#
# ## What changed in `_compute_sim_recon_diffusion_targets` right before this
# notebook was written (see `scproto.py` for the full reasoning):
# - **Target rescaled to O(1) RMS entry.** `eigsh` returns each eigenvector
#   unit-L2-norm over the whole batch, so raw entries shrink as
#   `~1/sqrt(N_batch)` — for a few thousand cells that's already tiny enough
#   that a decoder outputting near-zero for everything sits almost exactly on
#   the MSE floor. That looks like the loss "converging" instantly and reads
#   like vanishing gradient, but it's really just a target-scale artifact.
#   Fixed by rescaling to O(1) RMS entry regardless of batch size.
# - **Trivial leading eigenvector dropped.** The top eigenvalue (~1 for a
#   connected graph) corresponds to a `sqrt(degree)`-ish direction that's the
#   same shape for every graph of this type — not discriminative between
#   cells — so keeping it just wastes one of `n_eigs` target dimensions.
# - **`diffusion_t` (eigenvalue-weighting/diffusion-time) knob removed
#   entirely**, not just fixed. It briefly existed as a config option but was
#   never actually wired into the target-computation call, so every prior run
#   silently used unweighted eigenvectors regardless of what was configured.
#   Rather than fix the wiring, we removed the knob: `diffusion_t>0` would
#   upweight coarse/global eigenvectors over fine/local ones, but the
#   fine/local directions are exactly what let this loss catch a prototype
#   that's secretly gluing together two disconnected sub-communities — so
#   unweighted (every eigenvector equal) isn't a default waiting to be tuned
#   up, it's the only setting that makes sense for what this loss is for.
#
# **What to watch while training runs below:** the epoch progress line now
# prints `sim_recon=... [pred_std=... target_std=...]`. If `pred_std` stays
# near 0 while `target_std` doesn't, the decoder has collapsed to a
# near-constant output rather than actually predicting per-cell structure —
# a real regression, not just a slow-to-converge run.
#
# **Scope: `arbf` only**, same as the `full`-target notebook, so results are
# directly comparable across the two notebooks without an extra affinity-type
# variable in the way.

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

# Same first-guess scale as the `full`-target notebook — affinity values
# already live in ~[0,1], comparable to nassoc's own terms.
LAMBDA_SIM_RECON = 1.0
SIM_RECON_N_EIGS = 128  # dimensionality of the diffusion-coordinate target

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

# Baseline: set True if you already trained plain arbf elsewhere (this
# notebook, train_eval_sim_recon.ipynb, or train_scproto_spatial.ipynb all
# save/load under the same model-name scheme) and just want to reload it.
# Set False to train it fresh here (fully self-contained, just slower).
LOAD_BASELINE = True

# The diffusion sim-recon run is the new thing this notebook exists to
# produce — False trains it; flip to True on a re-run to just reload.
LOAD_SIMRECON = False

# sqrt(eigenvalue)-weighted variant (sim_recon_diffusion_t=0.5) — see the
# markdown at that training cell below for what this tests.
LOAD_SIMRECON_WEIGHTED = False

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
# ## Train / load — arbf baseline + diffusion sim-recon
#
# Each cell is independent — skip/re-run either without affecting the other.

# %% [markdown]
# ### Baseline (no sim-recon)

# %%
train_or_load('arbf', AFFINITY, load=LOAD_BASELINE)

# %% [markdown]
# ### +sim-recon (`diffusion` target)
#
# Everything else identical to the baseline above — a clean single-variable
# ablation. Watch the printed `pred_std`/`target_std` pair each epoch (see
# the intro above) to confirm the decoder is actually tracking the target
# rather than collapsing to a near-constant output.

# %%
train_or_load('arbf+diffusion', AFFINITY, load=LOAD_SIMRECON,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'diffusion',
                              'sim_recon_n_eigs': SIM_RECON_N_EIGS})

# %% [markdown]
# ### +sim-recon (`diffusion` target, sqrt(eigenvalue)-weighted: `sim_recon_diffusion_t=0.5`)
#
# Same target dimensionality as the run above, but each eigenvector is
# scaled by `eigenvalue**0.5` before the batch-size rescale
# (`_compute_sim_recon_diffusion_targets`, `trainers/scproto.py`). By the
# Eckart-Young theorem this makes per-cell MSE on the weighted coordinates
# mathematically equivalent (up to the rank-`n_eigs` truncation) to MSE on a
# reconstructed similarity matrix — as close as `diffusion` mode can get to
# behaving like `sim_recon_target='full'` (or SEACells' own RSS), at the
# cost of the same fine/rare-pattern sensitivity the unweighted (`t=0`) run
# above protects. See `files/sim_recon_global_vs_local_compaction.md` for
# the full reasoning — this cell is the empirical test of that dial, not a
# claim that it's strictly better.
#
# Compare this run's purity/niche-purity against `arbf+diffusion` (`t=0`)
# and SEACells below: closer to SEACells here, at whatever cost shows up in
# the per-cell-type purity table for rare types, is exactly the trade-off
# predicted.

# %%
train_or_load('arbf+diffusion_t0.5', AFFINITY, load=LOAD_SIMRECON_WEIGHTED,
               extra_lambda={'lambda_sim_recon': LAMBDA_SIM_RECON, 'sim_recon_target': 'diffusion',
                              'sim_recon_n_eigs': SIM_RECON_N_EIGS, 'sim_recon_diffusion_t': 0.5})

# %% [markdown]
# ### Diagnostic — how much of the diffusion target is prototype-explainable?
#
# The sim_recon decoder can only ever emit a per-prototype-constant profile
# (`soft_assign @ decoded`, rank <= NP prototypes) — its ceiling on any given
# eigen-index is however much of that dimension's variance sits *between*
# prototypes rather than within one. Eigen-index runs coarse -> fine; this
# checks where that between-prototype fraction decays into noise, which is
# the actual number of dimensions worth asking the decoder to reconstruct
# (vs. dimensions where "predict ~0" is already the MSE-optimal answer and
# just dilutes the aggregate pred_std/target_std readout).
#
# Cheap: one argmax pass over the already-trained model + a single
# vectorized scatter-add over (N, n_eigs) — a few seconds, not a retrain.

# %%
import numpy as np
import matplotlib.pyplot as plt

t_diff = trainers['arbf+diffusion']
target = t_diff._sim_recon_diffusion_target.numpy()  # (N, n_eigs)
assignments, _ = t_diff._get_assignments()            # (N,) hard prototype id per cell

K = int(assignments.max()) + 1
n_eigs = target.shape[1]

group_sum = np.zeros((K, n_eigs), dtype=np.float64)
np.add.at(group_sum, assignments, target)
group_counts = np.bincount(assignments, minlength=K).astype(np.float64)
group_means = group_sum / np.clip(group_counts[:, None], 1, None)

overall_mean = target.mean(axis=0)
between_var = np.average((group_means - overall_mean) ** 2, axis=0, weights=group_counts)
total_var = target.var(axis=0)
between_frac = between_var / (total_var + 1e-8)

plt.figure(figsize=(8, 4))
plt.plot(between_frac)
plt.xlabel('eigen-index (coarse -> fine)')
plt.ylabel('between-prototype variance fraction')
plt.title(f'Prototype-explainable fraction per diffusion dim (NP={K}, n_eigs={n_eigs})')
plt.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='5% floor')
plt.legend()
plt.show()

print('between_frac[:20]     =', np.round(between_frac[:20], 3))
print('between_frac[100:120] =', np.round(between_frac[100:120], 3))
print('between_frac[500:520] =', np.round(between_frac[500:520], 3))
# Read the cutoff off the curve above (where it drops under ~5-10% and stays
# there) rather than guessing a round number for sim_recon_n_eigs.

# %% [markdown]
# ## SEACells (PCA) baseline
#
# `train_seacell` skips training and returns immediately if a saved run is
# already found — safe to call every time.

# %%
train_seacell(DS_ID, mode='train', build_kernel_on='X_pca')

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
# Same metric definitions and plots as `scproto_spatial_comparison.ipynb` /
# `train_eval_sim_recon.ipynb` — reproduced here so this notebook is a
# complete, standalone record of the experiment. `NICHE_KEY='niches_2D'` here
# by choice; it doesn't need to match training's own `niche_key`
# (`niches_3D` above) — this just scores purity against whichever
# ground-truth niche column you point it at.

# %%
NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

# Exact model directory names, captured right after training/loading above —
# exact match (not keyword substring) because 'arbf' is a literal substring
# of 'arbf+diffusion''s saved name too.
MODEL_KEYWORDS = {
    model_names['arbf']:                'scProto (arbf)',
    model_names['arbf+diffusion']:      'scProto + sim-recon/diffusion (arbf)',
    model_names['arbf+diffusion_t0.5']: 'scProto + sim-recon/diffusion, t=0.5 (arbf)',
    'seacell':                          'SEACells (PCA)',
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_diffusion_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — does diffusion sim-recon beat the plain-arbf baseline?
#
# (model − arbf-baseline) per (cell type, niche) cell. Redder than the
# baseline is the direct answer to whether the diffusion-coordinate
# compression still buys any of the resolution gain the `full` target is
# meant to give (compare against `train_eval_sim_recon.ipynb`'s own
# baseline-vs-`full` diff heatmap for the other half of that comparison).

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference=model_names['arbf'],
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_diffusion_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_diffusion_tradeoff.pdf'),
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_diffusion_niche_purity_violin.pdf'),
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

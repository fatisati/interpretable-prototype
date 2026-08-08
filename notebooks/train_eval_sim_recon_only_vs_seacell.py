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
# # scProto (sim-recon ONLY) vs. SEACells — apples-to-apples reconstruction test
#
# `train_eval_sim_recon_diffusion.ipynb` adds `sim_recon` (diffusion target) as an
# *extra* term on top of the full scProto loss stack (UMAP contrastive + nassoc +
# proto_recon + proto_usage) — that's an ablation of "does sim_recon help scProto",
# and even with it on, scProto still trails SEACells.
#
# This notebook asks a narrower, more diagnostic question: **with every other loss
# turned off**, can `sim_recon` alone — the one scProto term that is structurally
# closest to SEACells' own objective (SEACells archetypal analysis is *itself* a
# reconstruction/RSS objective, nothing else) — get scProto to SEACells-level
# metacell quality? If a reconstruction-only scProto still trails SEACells by a lot,
# the gap is about the *mechanism* (soft-assignment + shared-per-prototype-decoder
# vs. archetypal analysis), not about scProto's other loss terms fighting the
# reconstruction signal. If it closes most of the gap, the other loss terms (or
# their interaction with sim_recon) are the more promising place to keep digging.
#
# Concretely, relative to `train_eval_sim_recon_diffusion.ipynb`'s config:
# - `lambda_umap = 0` — no contrastive positive/negative edge loss.
# - `lambda_nassoc = 0` — no normalized-association purity regularizer.
# - `lambda_proto_recon = 0`, `lambda_proto_usage = 0` — no gene-expression
#   reconstruction, no usage-balancing term.
# - `lambda_sim_recon = 1.0`, `sim_recon_target = 'diffusion'` — the only active loss.
# - `sim_recon_n_eigs = 1024` (vs. 128 in the additive-ablation notebook) — a much
#   higher-resolution diffusion-coordinate target, since this run leans on sim_recon
#   as its *only* source of structure and needs more of the spectrum to have a shot
#   at SEACells-level resolution.
#
# **What stays on:** `calibrate_eps=1`, `prot_init='waypoint'`,
# `umap_similarity='proto'`, `umap_proto_metric='dotp'`. These aren't losses — they
# control prototype initialization and the softmax temperature that turns
# encoder/prototype dot-products into the soft-assignment `S` that `sim_recon`
# itself depends on (`predicted = soft_assign @ decoded`). Without `umap_similarity
# ='proto'`, `sim_recon` can't fire at all (see `_run_umap_epoch` in `scproto.py`).
# Prototypes still receive gradient — only through `sim_recon_loss` now, since
# `lambda_umap * umap_loss` is still computed every step (just zeroed out) rather
# than skipped.
#
# **Caveat on `sim_recon_n_eigs=1024`:** the diffusion-coordinate target is computed
# *per biological batch* (`section`), not globally (see
# `_compute_sim_recon_diffusion_targets` docstring) — any section with
# <= `n_eigs + 2` cells gets an all-zero target and contributes no sim_recon
# supervision. Check the printed per-section sizes if `sim_recon_target_std` looks
# unexpectedly small. `eigsh(k=1024)` per section is also the slow part of setup —
# expect this to take noticeably longer than the `n_eigs=128` runs, and note it
# reruns on every `load_umap_checkpoint()` call too (not just on first training),
# since rebuilding the edge/training state is required either way.
#
# **SEACells is loaded, not retrained**, from whatever baseline
# `train_eval_sim_recon_diffusion.ipynb` (or an earlier run of this notebook)
# already produced at `build_kernel_on='X_pca'` — see the SEACells section below for
# how that's enforced.
#
# ## +anchor variant — does giving prototypes SEACells' own structure help further?
#
# With every other loss at 0, this notebook's base run has **no prototype
# position or anti-collapse mechanism at all** — `nassoc=0`, `proto_usage=0`,
# `umap=0` means `sim_recon_loss`'s own gradient is the *only* thing touching
# `soft_assign`/prototypes. SEACells' archetypes, by contrast, are a hard
# structural constraint: literally convex combinations of real cells
# (`files/seacells_kernel_archetypal_vs_scproto_losses.md`).
#
# `lambda_proto_anchor` (`files/proto_anchoring_vs_proto_usage.md`, option A)
# adds a soft version of that same constraint on top of the base run here:
# prototype stays a free, gradient-trained parameter (no `proto_decoupled`),
# but gets an added MSE pull toward a column-normalized `soft_assign`
# combination of this batch's cell embeddings — a latent-space analogue of
# SEACells' `B` matrix, computed fresh each minibatch. Since the base run
# already has nothing else shaping prototype position, this is the cleanest
# possible test of whether that structure alone moves scProto closer to
# SEACells.

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
SEACELL_BUILD_KERNEL_ON = 'X_pca'

LAMBDA_SIM_RECON = 1.0
SIM_RECON_N_EIGS = 1024  # much higher than the 128 used in the additive-ablation notebook

# Everything not listed here defaults to 0 already (see LAMBDA_PROTO_UMAP_PRECON in
# tasks.py vs. interpretable_ssl/configs/defaults.py) — spelled out explicitly below
# anyway so it's unambiguous at a glance that every other loss is off.
SIM_RECON_ONLY_LAMBDA = dict(
    lambda_umap=0,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=0,
    lambda_proto_usage=0,
    lambda_nassoc=0,
    lambda_r1r2=0,
    lambda_proto_attract=0,
    # --- kept on: prototype-similarity machinery sim_recon itself depends on ---
    calibrate_eps=1,
    umap_proto_metric='dotp',
    prot_init='waypoint',
    umap_similarity='proto',
    # --- the one active loss ---
    lambda_sim_recon=LAMBDA_SIM_RECON,
    sim_recon_target='diffusion',
    sim_recon_n_eigs=SIM_RECON_N_EIGS,
)
SIM_RECON_ONLY_LAMBDA

# %%
# First guess for lambda_proto_anchor — same "start at 1.0, same scale as the other
# λ=1 graph-topology terms" convention used for LAMBDA_SIM_RECON above. Not yet tuned.
LAMBDA_PROTO_ANCHOR = 1.0

SIM_RECON_ONLY_ANCHOR_LAMBDA = SIM_RECON_ONLY_LAMBDA | dict(
    lambda_proto_anchor=LAMBDA_PROTO_ANCHOR,
)
SIM_RECON_ONLY_ANCHOR_LAMBDA

# %%
COMMON_KWARGS = dict(
    cvae_epochs=50,
    train_epochs=50,
    eval_freq=3,
    patience=6,
    batch_size=1024,
    umap_steps_per_epoch=500,
    niche_key='niches_3D',
    target_groups=NSCLC_EVAL_GROUPS,
    lambda_config=SIM_RECON_ONLY_LAMBDA,
    trainer_kwargs={'experiment_name': 'sim_recon_only'},
)

# False trains fresh; flip to True on a re-run to just reload the saved checkpoint
# (still rebuilds the eigsh-based diffusion targets — see caveat above).
LOAD_SIM_RECON_ONLY = False

# The +anchor variant is the new thing this notebook exists to add — False trains it;
# flip to True on a re-run to just reload.
LOAD_SIM_RECON_ONLY_ANCHOR = False

# %% [markdown]
# ## Train / load — scProto, sim_recon-only (diffusion, n_eigs=1024)

# %%
t_sr, res_sr, mc_sr = run_mc_task(DS_ID, affinity_type=AFFINITY, load_umap=LOAD_SIM_RECON_ONLY, **COMMON_KWARGS)
model_name_sr = t_sr.get_model_name()
print(f'sim_recon_only: {res_sr}')

# %% [markdown]
# **Watch while training runs above:** the epoch progress line prints
# `sim_recon=... [pred_std=... target_std=...]`. If `pred_std` stays near 0 while
# `target_std` doesn't, the decoder has collapsed to a near-constant output instead
# of actually predicting per-cell structure — since this run has no other loss to
# fall back on, that collapse would mean the whole run learned nothing.

# %% [markdown]
# ## Train / load — scProto, sim_recon-only + proto_anchor (convex-in-latent pull)

# %%
t_sr_anchor, res_sr_anchor, mc_sr_anchor = run_mc_task(
    DS_ID, affinity_type=AFFINITY, load_umap=LOAD_SIM_RECON_ONLY_ANCHOR,
    **(COMMON_KWARGS | {
        'lambda_config': SIM_RECON_ONLY_ANCHOR_LAMBDA,
        'trainer_kwargs': {'experiment_name': 'sim_recon_only_anchor'},
    })
)
model_name_sr_anchor = t_sr_anchor.get_model_name()
print(f'sim_recon_only_anchor: {res_sr_anchor}')

# %% [markdown]
# Watch for the new `proto_anchor=...` term in this run's epoch log (added right
# after `proto_attract` when `lambda_proto_anchor>0` — see `_print_umap_epoch` in
# `scproto.py`), and compare `sim_recon`'s own `pred_std`/`target_std` against the
# base run above — same `sim_recon` config in both, only the prototype-position
# mechanism differs.

# %% [markdown]
# ### Diagnostic — how much of the diffusion target is prototype-explainable?
#
# Computed for both runs so the `between_frac` curves are directly comparable —
# if `+anchor` sits meaningfully above the base run, giving prototypes an explicit
# convex-combination pull helped; if not, `sim_recon`'s own gradient was already
# doing that job on its own.
#
# Same diagnostic as `train_eval_sim_recon_diffusion.ipynb`. The sim_recon decoder
# can only ever emit a per-prototype-constant profile (`soft_assign @ decoded`, rank
# <= NP prototypes) — its ceiling on any given eigen-index is however much of that
# dimension's variance sits *between* prototypes rather than within one. Cheap: one
# argmax pass over the trained model + a vectorized scatter-add, not a retrain.

# %%
import numpy as np
import matplotlib.pyplot as plt


def between_frac_for(trainer):
    target = trainer._sim_recon_diffusion_target.numpy()  # (N, n_eigs)
    assignments, _ = trainer._get_assignments()            # (N,) hard prototype id per cell
    K = int(assignments.max()) + 1
    n_eigs = target.shape[1]

    group_sum = np.zeros((K, n_eigs), dtype=np.float64)
    np.add.at(group_sum, assignments, target)
    group_counts = np.bincount(assignments, minlength=K).astype(np.float64)
    group_means = group_sum / np.clip(group_counts[:, None], 1, None)

    overall_mean = target.mean(axis=0)
    between_var = np.average((group_means - overall_mean) ** 2, axis=0, weights=group_counts)
    total_var = target.var(axis=0)
    return between_var / (total_var + 1e-8)


between_fracs = {
    'sim_recon_only': between_frac_for(t_sr),
    'sim_recon_only+anchor': between_frac_for(t_sr_anchor),
}

plt.figure(figsize=(8, 4))
for label, bf in between_fracs.items():
    plt.plot(bf, label=label)
plt.xlabel('eigen-index (coarse -> fine)')
plt.ylabel('between-prototype variance fraction')
plt.title(f'Prototype-explainable fraction per diffusion dim (NP=800, n_eigs={SIM_RECON_N_EIGS})')
plt.axhline(0.05, color='gray', linestyle='--', linewidth=1, label='5% floor')
plt.legend()
plt.show()

for label, bf in between_fracs.items():
    print(f'{label}: mean={bf.mean():.4f}, [:20]={np.round(bf[:20], 3)}')

# %% [markdown]
# ## SEACells (PCA) — load in eval mode, do not retrain
#
# `train_seacell(..., mode='eval')` only trains if no saved run exists yet at
# `build_kernel_on='X_pca'`; if one does (e.g. from
# `train_eval_sim_recon_diffusion.ipynb`), it's left untouched — this is the "don't
# accidentally refit SEACells with fresh randomness" guarantee. `eval_seacell_task1`
# then *only ever loads* the saved `seacell_sc.h5ad` from disk (never refits) to
# compute task-1 metrics and write `cell_assignments.csv`, which the comparison
# figures below need.

# %%
train_seacell(DS_ID, mode='eval', build_kernel_on=SEACELL_BUILD_KERNEL_ON)
seacell_res = eval_seacell_task1(DS_ID, build_kernel_on=SEACELL_BUILD_KERNEL_ON)
print(f'seacell: {seacell_res}')

# %% [markdown]
# ## Quick numeric comparison — task-1 metrics only, straight from memory
#
# Restricted to the keys both sides actually produce (`purity`, `niche_purity`,
# `batch_entropy`, `modularity`, ...) — `res_sr` has additional task-2/3/group
# entries `seacell_res` doesn't, dropped here for a clean side-by-side.

# %%
shared_keys = [k for k in seacell_res if k in res_sr]
metrics_df = pd.DataFrame({
    'scProto (sim_recon only, diffusion, n_eigs=1024)': {k: res_sr[k] for k in shared_keys},
    'scProto (sim_recon only +anchor)': {k: res_sr_anchor[k] for k in shared_keys},
    'SEACells (PCA)': {k: seacell_res[k] for k in shared_keys},
}).T
metrics_df

# %% [markdown]
# ## Full comparison — cell-type purity vs. niche purity
#
# Same metric definitions and plots as the other `train_eval_sim_recon*.ipynb`
# notebooks, reproduced here so this is a standalone record.

# %%
NICHE_KEY = 'niches_2D'
CELLTYPE_KEY = 'celltypes'
MIN_CELLS = 20

GRAPH_DIR = os.path.join(os.environ['CODE_DIR'], 'graphs')
os.makedirs(GRAPH_DIR, exist_ok=True)

MODEL_KEYWORDS = {
    model_name_sr:        'scProto (sim_recon only, diffusion)',
    model_name_sr_anchor: 'scProto (sim_recon only +anchor)',
    'seacell':             'SEACells (PCA)',
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_only_vs_seacell_niche_heatmap.pdf'),
)

# %% [markdown]
# ### 3. Difference heatmap — does sim_recon-only scProto close the gap to SEACells?
#
# (model − SEACells) per (cell type, niche) cell, reference = `seacell` this time
# (not the scProto baseline) — this is the direct "is sim_recon alone enough to
# match SEACells" readout the notebook exists to answer. Redder than SEACells means
# scProto wins that cell; bluer means SEACells still wins it.

# %%
fig_celltype_niche_heatmap_diff(
    DS_ID, MODEL_KEYWORDS, reference='seacell',
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS, metric='niche_purity', min_n=5,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_only_vs_seacell_niche_diff_heatmap.pdf'),
)

# %% [markdown]
# ### 4. Trade-off — cell-type purity vs. niche purity, per model

# %%
fig_celltype_niche_tradeoff(
    DS_ID, MODEL_KEYWORDS,
    celltype_key=CELLTYPE_KEY, niche_key=NICHE_KEY,
    min_cells=MIN_CELLS,
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_only_vs_seacell_tradeoff.pdf'),
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
    save_path=os.path.join(GRAPH_DIR, 'sim_recon_only_vs_seacell_niche_purity_violin.pdf'),
)

# %% [markdown]
# ## Visualize — UMAP per run
#
# Colored by cell type and (3D) niche, prototypes overlaid.

# %%
for name, t in {'sim_recon_only': t_sr, 'sim_recon_only+anchor': t_sr_anchor}.items():
    print(f'--- {name} ---')
    fig, proto_labels = t.plot_umap_simple(
        color_key=['celltypes', 'niches_3D'],
        show_proto_nums=False,
    )

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# !pip install -q scarches faiss-cpu scib-metrics
# !pip install --upgrade --force-reinstall git+https://github.com/dpeerlab/SEACells.git --quiet --no-deps
# !pip install numpy scipy --upgrade -q
# !pip install -q palantir harmonypy statsmodels leidenalg python-igraph
# !pip install -q "numpy==1.26.4" "scipy==1.13.1"
# !pip install --upgrade --force-reinstall numpy cupy-cuda12x
# !pip install "numpy<2.3"
# !pip install -q diptest

# %% [markdown]
# **Restart the runtime now** (numpy/scipy/anndata are C-extension linked; an in-process
# upgrade won't take effect on already-imported modules), then continue from the next cell.

# %%
_checks = {
    "numpy": "numpy", "scipy": "scipy", "anndata": "anndata", "scanpy": "scanpy",
    "scarches": "scarches", "scvi-tools": "scvi", "seacells": "SEACells",
    "palantir": "palantir", "scib-metrics": "scib_metrics",
    "statsmodels": "statsmodels", "faiss": "faiss", "leidenalg": "leidenalg",
    "diptest": "diptest",
}
_failed = []
for pkg, mod in _checks.items():
    try:
        __import__(mod)
    except ImportError as e:
        _failed.append((pkg, str(e)))
if _failed:
    print("FAILED imports (fix before continuing, then restart runtime again):")
    for pkg, err in _failed:
        print(f"  {pkg}: {err}")
else:
    print("All packages import cleanly.")

# %%
# %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py

# %%
import os
import gc
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import get_affinity_path
from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity, _ensure_X_ctx
from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
from interpretable_ssl.evaluation.collapse_diagnostics import (
    labels_from_trainer, joint_purity_per_metacell, collapse_summary, compare_runs,
    plot_collapse, metacell_sizes, metacell_tightness,
)

# %% [markdown]
# # D1 / D2: density-based prototype regularization on s28nsc + BANKSY(lambda=0.8)
#
# **Purpose.** The current `L_nassoc` diagonal measures *containment*
# (`edges_inside / volume`), which is additive under merging -- so merging two prototypes
# always raises it and hierarchical splits are structurally vetoed. Result: a few huge
# metacells plus a tail of near-empty ones. This notebook tests two independent fixes,
# **added one at a time**:
#
# | run | `nassoc_diag_norm` | `nassoc_diag_loss` | `nassoc_gamma` | expected outcome |
# |---|---|---|---|---|
# | A baseline | `volume` | `mse` | -- | **collapses to giants** = the merge bias |
# | B D1 only | **`pair`** | `mse` | -- | **fragments to tiny metacells** |
# | C D1+D2 | **`pair`** | **`cpm`** | **swept** | optimum lands in between |
#
# **Run B is expected to look bad, and that is the point.** On a kNN graph
# `density <= k/(size-1)`, so a target of 1 is unreachable and `pair`+`mse` is minimized by
# metacells of size 2. B fragmenting is direct evidence that swapping the denominator
# flipped the preference from "always merge" to "always split" -- i.e. that D1 does what it
# claims. `gamma` in run C is then what places the optimum between the two extremes.
# `cpm` rather than `hinge` for C: `hinge` goes silent above `gamma` (no gradient), so it
# only lifts the veto on splits; `cpm` keeps a gradient and its `pairs` weighting is what
# makes a big loose prototype dominate the loss and actively prefer a split. `cpm` gives
# empty prototypes zero weight, so `lambda_proto_usage` (already 0.1 in the base config)
# must stay on to refill dead slots.
#
# **Everything else is held identical to the existing rebuttal baseline**
# (`ctx_pca_product_experiment.ipynb`): same dataset, same affinity (`bk08`), same
# `cvae_epochs=50 / train_epochs=15`, same `num_prototypes`, same `LAMBDA_PROTO_UMAP_PRECON`.
#
# **Read before running.**
# - Run A **loads** an existing checkpoint; it never retrains. If the load fails, stop and
#   fix the path rather than training a fresh "baseline" that isn't the one we reported.
# - `gamma`'s absolute scale is arbitrary (it depends on the affinity scale and the
#   minibatch sampling rate), so `GAMMAS` is a log sweep. Read off the resulting mean
#   metacell size rather than trusting any single value.
# - Cells are ordered so a Colab disconnect loses one run, not the notebook.

# %%
# ---------------------------- CONFIG ----------------------------
DS_ID = 's28nsc'
CT_KEY = 'celltypes'
NICHE_KEY = 'niches_2D'
AFFINITY_TYPE = 'bk08'          # pybanksy, lambda=0.8, num_neighbours=32, n_components=50
K_GRAPH = 50
N_PROTOTYPES = 200              # matches ctx_pca_product_experiment.ipynb's bk08 runs

# held identical across A/B/C
BASE_KW = dict(
    cvae_epochs=50, train_epochs=15, eval_freq=5, patience=10,
    batch_size=1024, affinity_type=AFFINITY_TYPE, num_prototypes=N_PROTOTYPES,
    label_key=CT_KEY, niche_key=NICHE_KEY, skip_metrics=['task2', 'task3'],
)
BASELINE_EXP = 'proto_bk08_np200_ema15'       # the ALREADY-TRAINED run to load

GAMMAS = [0.01, 0.03, 0.1]      # D2 sweep; log-spaced, refine after seeing mean size
SKIP_IF_EXISTS = True

OUT_DIR = '/content/drive/MyDrive/codes/interpretable-prototype/files/tables/d1_d2_s28nsc_bk08'
FIG_DIR = '/content/drive/MyDrive/codes/interpretable-prototype/files/figures/d1_d2_s28nsc_bk08'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

summaries = {}      # run name -> summary row
labels_by_run = {}  # run name -> per-cell metacell id

# %%
# ---- CLAUDE.md section 9 check: confirm the new options actually reach the model name ----
# A missing ABBREVIATIONS entry is silently dropped, which would make two different runs
# share one model folder and clobber each other's checkpoints. Verify BEFORE training.
from interpretable_ssl.configs.defaults import get_defaults
from interpretable_ssl.constants import ABBREVIATIONS

_d = get_defaults()
for _k, _expect_default in [('nassoc_diag_norm', 'volume'), ('nassoc_gamma', 0.0),
                            ('nassoc_diag_loss', 'mse')]:
    assert _k in ABBREVIATIONS, f'{_k} missing from ABBREVIATIONS -> would be silently dropped'
    assert _d[_k] == _expect_default, f'{_k} default changed: {_d[_k]!r} != {_expect_default!r}'
print('config keys OK:', {k: ABBREVIATIONS[k] for k in
                          ['nassoc_diag_norm', 'nassoc_gamma', 'nassoc_diag_loss']})
# 4-char truncation rule: string values must differ within their first 4 chars
assert len({v[:4] for v in ['volume', 'pair']}) == 2
assert len({v[:4] for v in ['mse', 'nll', 'nll2', 'hinge', 'cpm']}) == 5
print('4-char truncation OK')

# %%
ds_conf = DATASETS[DS_ID]
adata = sc.read_h5ad(ds_conf['path'])
_ensure_X_ctx(adata)
print(adata)
print('\ncell types:\n', adata.obs[CT_KEY].value_counts())
print('\nniches:\n', adata.obs[NICHE_KEY].value_counts(dropna=False))

# %%
# The SAME affinity object every run trains against and every diagnostic is computed on.
affinity = load_or_build_affinity(
    adata, DS_ID, affinity_type=AFFINITY_TYPE, k=K_GRAPH, k_neighbors=K_GRAPH,
    bk=ds_conf.get('batch_key'))
print('affinity:', affinity.shape, 'nnz =', affinity.nnz if hasattr(affinity, 'nnz') else 'dense')

# %% [markdown]
# ## Run A -- baseline (load only, never retrain)

# %%
tA, _, _ = run_mc_task(
    DS_ID, **{**BASE_KW, 'train_epochs': 1, 'eval_freq': 1, 'patience': 1},
    lambda_config=dict(LAMBDA_PROTO_UMAP_PRECON),
    trainer_kwargs={'experiment_name': BASELINE_EXP},
    load_umap=True, skip_eval=True,
)
print('loaded baseline:', tA.get_model_name())
labels_by_run['A_baseline'] = labels_from_trainer(tA, adata)

# %%
_lab = labels_by_run['A_baseline']
_pur = joint_purity_per_metacell(_lab, adata, CT_KEY, NICHE_KEY)
summaries['A_baseline'] = collapse_summary(_lab, affinity=affinity, name='A_baseline', purity=_pur)
pd.Series(summaries['A_baseline']).to_frame('A_baseline')

# %%
plot_collapse(_lab, affinity, name='A baseline')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/A_baseline.png', dpi=120, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Baseline diagnosis, part 2 -- SEACells on the same graph
#
# **Nothing has trained yet.** Run A and this section only LOAD saved models, so by the end
# of this cell you can already see -- for scProto's current loss AND for both SEACells
# variants (bk08 and PCA) -- whether the collapse exists, on the same affinity and the same
# metrics. Only if it does is there any reason to run B/C/D.
#
# SEACells is not a competitor here: it picks archetypes rather than optimizing a
# containment objective, so it is the reference for what a healthy size/tightness
# distribution looks like on this exact graph.

# %%
# READ-ONLY: loads SEACells runs already on disk (cell_assignments.csv). It never trains,
# so it cannot cost a Colab session. If a tag isn't there yet, it is skipped -- train it in
# the notebook that owns it (seacells_train.ipynb / ctx_pca_product_experiment.ipynb)
# rather than here.
from interpretable_ssl.evaluation.paper_figures import _resolve_run_dir
from interpretable_ssl.evaluation.niche_program_recovery import load_cell_assignments

for sc_name, sc_tag in [('SEACells_bk08', AFFINITY_TYPE), ('SEACells_pca', 'X_pca')]:
    try:
        run_dir = _resolve_run_dir(DS_ID, f'seacell_{sc_tag}')
        lab = load_cell_assignments(run_dir, adata, NICHE_KEY)['metacell_id']
    except Exception as e:
        print(f'skip {sc_name}: no saved run ({type(e).__name__}: {e})')
        continue
    # tightness is computed against `affinity`, whose rows follow adata's cell order --
    # so the label vector MUST be reindexed to adata.obs_names, not left in csv order.
    lab = lab.reindex(adata.obs_names.to_numpy())
    if lab.isna().any():
        print(f'{sc_name}: {int(lab.isna().sum())} cells unassigned in the saved run; '
              f'they are excluded from its metacells but still count in the graph.')
    labels_by_run[sc_name] = lab
    pur = joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
    summaries[sc_name] = collapse_summary(lab, affinity=affinity, name=sc_name, purity=pur)
    plot_collapse(lab, affinity, name=sc_name)
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{sc_name}.png', dpi=120, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Run B -- D1 only (density denominator, nothing else changed)

# %%
lcB = dict(LAMBDA_PROTO_UMAP_PRECON)
lcB['nassoc_diag_norm'] = 'pair'          # <-- D1. The ONLY change vs run A.
expB = 'proto_bk08_np200_ema15_d1pair'

try:
    if not SKIP_IF_EXISTS:
        raise FileNotFoundError
    tB, _, _ = run_mc_task(
        DS_ID, **{**BASE_KW, 'train_epochs': 1, 'eval_freq': 1, 'patience': 1},
        lambda_config=lcB, trainer_kwargs={'experiment_name': expB},
        load_umap=True, skip_eval=True)
    print('loaded existing D1 run')
except FileNotFoundError:
    print('=== training D1 (nassoc_diag_norm=pair) ===')
    tB, _, _ = run_mc_task(DS_ID, **BASE_KW, lambda_config=lcB,
                           trainer_kwargs={'experiment_name': expB})
print('model name:', tB.get_model_name())
labels_by_run['B_d1_pair'] = labels_from_trainer(tB, adata)

# %%
_lab = labels_by_run['B_d1_pair']
_pur = joint_purity_per_metacell(_lab, adata, CT_KEY, NICHE_KEY)
summaries['B_d1_pair'] = collapse_summary(_lab, affinity=affinity, name='B_d1_pair', purity=_pur)
plot_collapse(_lab, affinity, name='B D1 (pair)')
plt.tight_layout(); plt.savefig(f'{FIG_DIR}/B_d1_pair.png', dpi=120, bbox_inches='tight')
plt.show()
pd.DataFrame([summaries['A_baseline'], summaries['B_d1_pair']]).set_index('run').T

# %%
del tB; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Run C -- D1 + D2 (density + CPM resolution knob), one gamma at a time

# %%
for g in GAMMAS:
    name = f'C_d1d2_g{g}'
    lcC = dict(LAMBDA_PROTO_UMAP_PRECON)
    lcC['nassoc_diag_norm'] = 'pair'
    lcC['nassoc_diag_loss'] = 'cpm'
    lcC['nassoc_gamma'] = g
    expC = f'proto_bk08_np200_ema15_d1d2_g{g}'

    try:
        if not SKIP_IF_EXISTS:
            raise FileNotFoundError
        tC, _, _ = run_mc_task(
            DS_ID, **{**BASE_KW, 'train_epochs': 1, 'eval_freq': 1, 'patience': 1},
            lambda_config=lcC, trainer_kwargs={'experiment_name': expC},
            load_umap=True, skip_eval=True)
        print(f'loaded existing gamma={g}')
    except FileNotFoundError:
        print(f'=== training D1+D2, gamma={g} ===')
        tC, _, _ = run_mc_task(DS_ID, **BASE_KW, lambda_config=lcC,
                               trainer_kwargs={'experiment_name': expC})
    print('model name:', tC.get_model_name())

    lab = labels_from_trainer(tC, adata)
    labels_by_run[name] = lab
    pur = joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
    summaries[name] = collapse_summary(lab, affinity=affinity, name=name, purity=pur)
    # save after EVERY gamma so a disconnect costs one gamma, not the sweep
    pd.DataFrame(list(summaries.values())).set_index('run').to_csv(
        f'{OUT_DIR}/collapse_summary.csv')

    plot_collapse(lab, affinity, name=f'C D1+D2 gamma={g}')
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{name}.png', dpi=120, bbox_inches='tight')
    plt.show()

    del tC; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Run D -- hinge control, ONE run at the best gamma from the C sweep
#
# Isolates a single question: is it enough to *lift* the target (hinge pays only below
# `gamma`, nothing above), or does the `pairs` weighting in `cpm` -- which lets a big loose
# prototype dominate the loss -- actually matter? Same `gamma`, same everything else, so the
# only difference is floor-vs-reward. Not swept: one run answers it.

# %%
BEST_GAMMA = None    # <-- set from the C sweep above (the gamma with the best size/tightness
                     #     trade-off), then run this cell. None = skip.

if BEST_GAMMA is not None:
    name = f'D_hinge_g{BEST_GAMMA}'
    lcD = dict(LAMBDA_PROTO_UMAP_PRECON)
    lcD['nassoc_diag_norm'] = 'pair'
    lcD['nassoc_diag_loss'] = 'hinge'
    lcD['nassoc_gamma'] = BEST_GAMMA
    expD = f'proto_bk08_np200_ema15_d1hinge_g{BEST_GAMMA}'
    try:
        if not SKIP_IF_EXISTS:
            raise FileNotFoundError
        tD, _, _ = run_mc_task(
            DS_ID, **{**BASE_KW, 'train_epochs': 1, 'eval_freq': 1, 'patience': 1},
            lambda_config=lcD, trainer_kwargs={'experiment_name': expD},
            load_umap=True, skip_eval=True)
        print('loaded existing hinge run')
    except FileNotFoundError:
        print(f'=== training hinge control, gamma={BEST_GAMMA} ===')
        tD, _, _ = run_mc_task(DS_ID, **BASE_KW, lambda_config=lcD,
                               trainer_kwargs={'experiment_name': expD})
    print('model name:', tD.get_model_name())

    lab = labels_from_trainer(tD, adata)
    labels_by_run[name] = lab
    pur = joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
    summaries[name] = collapse_summary(lab, affinity=affinity, name=name, purity=pur)
    pd.DataFrame(list(summaries.values())).set_index('run').to_csv(
        f'{OUT_DIR}/collapse_summary.csv')
    plot_collapse(lab, affinity, name=f'D hinge gamma={BEST_GAMMA}')
    plt.tight_layout(); plt.savefig(f'{FIG_DIR}/{name}.png', dpi=120, bbox_inches='tight')
    plt.show()
    del tD; gc.collect(); torch.cuda.empty_cache()
else:
    print('BEST_GAMMA not set -- skipping the hinge control. Set it from the C sweep first.')

# %% [markdown]
# ## Final comparison
#
# What to read, in order:
#
# - `largest_share`, `dip_p` -- did collapse happen
#   (`dip_p` small = the size distribution is two humps with a gap = the cliff)
# - `tightness_gap` (macro - micro) -- ~0 healthy; >>0 = the BIG metacells are the loose ones
# - `size_tightness_corr` -- expected ~-0.2; < -0.6 means the giants are pathological.
#   NaN means fewer than 3 metacells have >= 2 members, which is itself a collapse signal.
# - `purity_micro` / `purity_gap` -- joint (cell type | niche).
#   `purity_macro` is deliberately NOT reported: it rises with fragmentation (a singleton
#   metacell is 100% pure), so it invites exactly the cross-run comparison that misleads.

# %%
report = pd.DataFrame(list(summaries.values())).set_index('run')
cols = ['largest_share', 'dip_stat', 'dip_p',
        'tightness_gap', 'size_tightness_corr',
        'purity_micro', 'purity_gap']
report = report[[c for c in cols if c in report.columns]]
report.to_csv(f'{OUT_DIR}/collapse_summary.csv')
report.round(4).T

# %% [markdown]
# ## DGE -- niche-program recovery, same runs (s28nsc, K=200)
#
# Extends the collapse-diagnostics comparison above with the niche-DGE recovery metric used
# in `spatial_new_loss_dge.ipynb` (same library: `interpretable_ssl/evaluation/niche_dge.py`),
# so every arm in `labels_by_run` -- A/B/C and `SEACells_bk08` (and `SEACells_pca` if it
# loaded) -- gets scored on whether its metacells recover known niche-vs-rest transcriptional
# programs, not just on collapse shape.
#
# **This is an even comparison.** `SEACells_bk08`'s own `metrics.json` on disk has
# `"K_target": 200`, matching every scProto arm's `NP200` here -- unlike the `ss28nsc`
# notebook's SEACells comparison, K does not need to be checked or re-run.
#
# **Ground truth / pooled-DE cache paths are dataset-scoped on purpose.**
# `get_ground_truth` / `pooled_celltype_de` are plain load-if-exists wrappers keyed only by
# the path string you pass -- they do not know which dataset produced the file. `ss28nsc` is
# a documented subsample of this same section, so reusing its `files/celltype_niches_full.csv`
# here would silently score every arm against the wrong cells. Hence the `_s28nsc` suffix
# below, distinct from the `ss28nsc` notebook's paths.

# %%
from interpretable_ssl.evaluation import niche_dge as nd

DGE_GT_PATH = os.path.join(os.environ['CODE_DIR'], 'files', 'celltype_niches_full_s28nsc.csv')
DGE_POOLED_PATH = os.path.join(os.environ['CODE_DIR'], 'files', 'celltype_pooled_de_s28nsc.csv')
DGE_REFERENCE = 'B_d1_pair'      # primary new-loss candidate; change if comparing a different arm
DGE_FORCE_RERUN = False          # True only to deliberately recompute ground truth / per-arm DGE

ground_truth = nd.get_ground_truth(
    adata, DGE_GT_PATH, ct_key=CT_KEY, niche_key=NICHE_KEY, force=DGE_FORCE_RERUN,
)
print(f'{len(ground_truth)} (cell type, niche) pairs')

# %%
# Each arm writes its own CSV before the next starts, so a disconnect loses at most one arm.
dge_results = {}
for name, lab in labels_by_run.items():
    dge_results[name] = nd.run_niche_dge(
        name, lab, adata, ground_truth, OUT_DIR,
        ct_key=CT_KEY, niche_key=NICHE_KEY, force=DGE_FORCE_RERUN,
    )
    gc.collect()

# %%
dge_summary = nd.summary_table(dge_results)
dge_summary.to_csv(f'{OUT_DIR}/dge_summary.csv')
dge_summary.round(3)

# %%
fig = nd.plot_dge_summary(
    dge_summary, highlight=DGE_REFERENCE,
    out_path=f'{FIG_DIR}/dge_summary.png',
    title='Niche-program recovery, within cell type (s28nsc, K=200)',
)
plt.show()

# %% [markdown]
# ### Paired, per (cell type, niche)
#
# `pearson_r_gap_<arm>` = that arm minus `DGE_REFERENCE`. Negative gap = `DGE_REFERENCE` wins
# on that pair.

# %%
dge_paired = nd.compare_arms(dge_results, reference=DGE_REFERENCE, metric='pearson_r')
gap_cols = [c for c in dge_paired.columns if c.startswith('pearson_r_gap_')]
if gap_cols:
    dge_paired = dge_paired.sort_values(gap_cols[0], ascending=False)
dge_paired.to_csv(f'{OUT_DIR}/dge_paired.csv', index=False)
dge_paired.round(3).head(25)

# %% [markdown]
# ### Size-floor sweep -- is a remaining gap a size artifact or missing signal?
#
# Restricts the comparison to pairs where `DGE_REFERENCE`'s own niche-positive metacells
# clear a rising size floor. Rising toward the baselines = small-N pseudobulk noise. Flat at
# floor 30-50 = the signal genuinely is not there.

# %%
dge_sweep = nd.size_floor_sweep(dge_results, size_arm=DGE_REFERENCE)
dge_sweep.to_csv(f'{OUT_DIR}/dge_size_floor_sweep.csv', index=False)
print(dge_sweep.round(3).to_string(index=False))

# %%
fig = nd.plot_size_floor_sweep(
    dge_sweep, arms=list(dge_results), highlight=DGE_REFERENCE,
    out_path=f'{FIG_DIR}/dge_size_floor_sweep.png',
)
plt.show()

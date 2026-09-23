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
# # scProto with the new loss on spatial — niche-program recovery (DGE)
#
# **Question.** The plan1 rebuttal experiment found SEACells *ahead* of scProto on
# niche-program recovery, and the diagnosis was that scProto's niche-positive metacells
# were built from too few real cells to give a stable pseudobulk fold-change — i.e. the
# over-merging / fragmentation failure (a few giant prototypes plus many near-singletons).
# The corrected `nassoc` diagonal is meant to fix exactly that. So:
#
# > Does the new loss remove the collapse, and does removing it close the DGE gap?
#
# **Dataset: `ss28nsc` only** (NSCLC CosMx, one slide). One dataset, run fully, before any
# second tissue — same rule as the rest of this project.
#
# **What runs here**
#
# | Step | Cost | Cached as |
# |---|---|---|
# | 1. train scProto, new loss (`nassoc_diag_norm='pair'`) | training | checkpoint under `MODEL_DIR/ss28nsc/` |
# | 2. train scProto, old loss (`'volume'`), same everything else | training | ditto |
# | 3. collapse diagnostics on both | seconds | `tables/` |
# | 4. ground-truth niche DE from real single cells | minutes | `files/celltype_niches_full.csv` |
# | 5. DGE recovery per arm | minutes | `tables/ss28nsc/<arm>_per_pair.csv` |
# | 6. comparison + size-floor sweep + figures | seconds | `tables/`, `figures/` |
#
# **Resuming after a Colab disconnect.** Every step is load-if-exists. Re-run the notebook
# top to bottom: finished steps print `loaded from ...` and cost nothing. Set
# `LOAD_UMAP = True` to reload trained checkpoints instead of retraining, and
# `FORCE_RERUN = True` only to deliberately recompute the eval tables.
#
# **Read before running.** `bk08` (BANKSY, lambda=0.8) needs `X_bk08` precomputed on the
# **full tissue**. If `ss28nsc` is a filtered subset and `X_bk08` is not already in
# `adata.obsm`, the affinity code will compute it on the subset, where the neighbour-mean
# term sees only the surviving cells — that is not the real spatial context. The affinity
# cell prints a warning if this happens; do not report numbers from a run that warned.

# %% [markdown]
# ## Setup

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Multi-pass on purpose — collapsing this to one pin has been tried and fails.
# statsmodels: branch 2's weighted t-test. diptest: the collapse dip test.
# SEACells always --no-deps (prot_init='waypoint' needs its diffusion-map code).
# !pip install -q scarches faiss-gpu-cu12 scib-metrics
# !pip install git+https://github.com/dpeerlab/SEACells.git --quiet --no-deps
# !pip install numpy scipy --upgrade -q
# !pip install -q palantir harmonypy
# !pip install -q "numpy==1.26.4" "scipy==1.13.1"
# !pip install --upgrade --force-reinstall numpy cupy-cuda12x
# !pip install "numpy<2.3"
# !pip install -q statsmodels diptest pybanksy

# %% [markdown]
# **RESTART THE RUNTIME NOW** (Runtime -> Restart session) before running anything below.
# numpy/scipy/anndata are C-extension linked, so an in-process upgrade will not take
# effect on already-imported modules. Skipping this is the exact cause of
# `cannot import name '_center' from numpy._core.umath` on the next cell.

# %%
_checks = {
    "numpy": "numpy", "scipy": "scipy", "anndata": "anndata", "scanpy": "scanpy",
    "scarches": "scarches", "scvi-tools": "scvi", "seacells": "SEACells",
    "palantir": "palantir", "scib-metrics": "scib_metrics",
    "statsmodels": "statsmodels", "diptest": "diptest", "faiss": "faiss",
}
_failed = []
for pkg, mod in _checks.items():
    try:
        __import__(mod)
    except ImportError as e:
        _failed.append((pkg, str(e)))
if _failed:
    print("FAILED imports (fix, then restart the runtime again):")
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
import torch
import matplotlib.pyplot as plt

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.evaluation.spatial_immune_task import NSCLC_EVAL_GROUPS
from interpretable_ssl.evaluation import niche_dge as nd
from interpretable_ssl.evaluation import collapse_diagnostics as cd

# %% [markdown]
# ## Config
#
# The two training arms differ in **one** setting, `nassoc_diag_norm`, so any difference
# between them is attributable to the loss and nothing else.
#
# ```
# 'volume'  edges_inside / volume          old. volume is additive under merging, so
#                                          merging two prototypes always raises the score
#                                          -> hierarchical splits are penalized -> over-merging
# 'pair'    edges_inside / possible_pairs  new. the induced-subgraph edge density; pairs is
#                                          superadditive, so merging LOWERS it -> the loss
#                                          can finally rank a justified split above a merge
# ```

# %%
DS_ID = 'ss28nsc'
CT_KEY = 'celltypes'
NICHE_KEY = 'niches_2D'
AFFINITY = 'bk08'          # BANKSY lambda=0.8 + real SEACells RBF

NB_DIR = os.path.join(os.environ['CODE_DIR'], 'notebooks', 'spatial_new_loss_dge')
TABLE_DIR = os.path.join(NB_DIR, 'tables', DS_ID)
FIG_DIR = os.path.join(NB_DIR, 'figures')
GT_PATH = os.path.join(os.environ['CODE_DIR'], 'files', 'celltype_niches_full.csv')
POOLED_PATH = os.path.join(os.environ['CODE_DIR'], 'files', 'celltype_pooled_de.csv')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

FORCE_RERUN = False        # True only to deliberately recompute the eval tables
LOAD_UMAP = False          # True on a re-run: reload trained checkpoints, skip training

# Epochs follow the project minimum. Both arms use identical values, so the loss
# comparison is not confounded by training length.
COMMON_KWARGS = dict(
    cvae_epochs=50,
    train_epochs=20,
    eval_freq=2,
    patience=4,
    batch_size=1024,
    umap_steps_per_epoch=500,
    niche_key=NICHE_KEY,
    target_groups=NSCLC_EVAL_GROUPS,
)
BASE_LAMBDA = LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'}

NEW_LOSS = 'scProto (new loss, pair)'
OLD_LOSS = 'scProto (old loss, volume)'

trainers, labels = {}, {}

# %%
ds_conf = DATASETS[DS_ID]
adata = sc.read_h5ad(ds_conf['path'])
print(adata)
print('\ncell types:\n', adata.obs[CT_KEY].value_counts())
print('\nniches:\n', adata.obs[NICHE_KEY].value_counts(dropna=False))
print('\nX_bk08 precomputed:', 'X_bk08' in adata.obsm)

# %% [markdown]
# ## 1. Train — new loss
#
# One cell per arm so a disconnect costs one training run, not both.

# %%
t, res, _ = run_mc_task(
    DS_ID, affinity_type=AFFINITY, load_umap=LOAD_UMAP,
    lambda_config=BASE_LAMBDA | {'nassoc_diag_norm': 'pair'},
    **COMMON_KWARGS,
)
trainers[NEW_LOSS] = t
labels[NEW_LOSS] = nd.labels_from_trainer(t)
print(t.get_model_name())
print(res)

# %% [markdown]
# ## 2. Train — old loss (the control)

# %%
t, res, _ = run_mc_task(
    DS_ID, affinity_type=AFFINITY, load_umap=LOAD_UMAP,
    lambda_config=BASE_LAMBDA | {'nassoc_diag_norm': 'volume'},
    **COMMON_KWARGS,
)
trainers[OLD_LOSS] = t
labels[OLD_LOSS] = nd.labels_from_trainer(t)
print(t.get_model_name())
print(res)

# %% [markdown]
# ## 3. Baselines — load only, never retrain
#
# `seacell` is SEACells on PCA: zero spatial signal, the transcriptomics-only baseline.
# The pinned keyword is the exact scProto run the rebuttal's plan1 numbers came from
# (`aff-ctx`, old loss), kept so the new numbers sit next to the ones already reported.
# Any arm whose files are missing is dropped with a message rather than failing the cell.

# %%
BASELINES = {
    'SEACells (PCA)': 'seacell',
    'scProto (plan1 run, ctx)': (
        'proto_umap_prtInit-wayp_aff-ctx_lprec0.01_usim-prot_ecal1'
        '_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp'
    ),
}

for name, keyword in BASELINES.items():
    lab = nd.labels_from_run(DS_ID, keyword, adata.obs_names)
    if lab is None or lab.isna().all():
        print(f'SKIP {name}: no saved cell assignments for keyword {keyword!r}')
        continue
    labels[name] = lab.dropna()
    print(f'{name}: {labels[name].nunique()} metacells over {len(labels[name])} cells')

print('\narms:', list(labels))

# %% [markdown]
# ## 4. Collapse diagnostics — the prerequisite check
#
# This is the gate. If the new loss has not changed the size distribution, nothing
# downstream is worth reading.
#
# ```
# largest_share   fraction of all cells in the single biggest metacell. rising = collapsing
# dip_p           Hartigan dip on log10(sizes). SMALL p = two humps = the collapse cliff.
#                 HEALTHY is p > 0.05 (one smooth hump)
# purity_micro    joint (cell type | niche) purity, each CELL counting equally
# purity_gap      macro minus micro. >> 0 means the big metacells are the impure ones
# ```
#
# Tightness columns are NaN here: they need the affinity matrix, which is not loaded in
# this notebook. Sizes, the dip test and joint purity already separate the three regimes.

# %%
purity = {n: cd.joint_purity_per_metacell(lab, adata, CT_KEY, NICHE_KEY)
          for n, lab in labels.items()}

collapse = pd.DataFrame([
    cd.collapse_summary(lab, name=n, purity=purity[n]) for n, lab in labels.items()
]).set_index('run')
collapse['n_metacells'] = pd.Series({n: lab.nunique() for n, lab in labels.items()})
collapse['median_size'] = pd.Series(
    {n: cd.metacell_sizes(lab).median() for n, lab in labels.items()})

collapse.to_csv(os.path.join(TABLE_DIR, 'collapse_summary.csv'))
collapse.round(3)

# %%
# One panel per arm: metacell sizes, sorted, log y.
# A CLIFF (a few giants, then a drop to near-empty) is the collapse signature;
# a smooth PLATEAU is healthy.
fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 3.6))
for ax, (n, lab) in zip(np.atleast_1d(axes), labels.items()):
    cd.plot_collapse(lab, name=n, axes=[ax])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f'{DS_ID}_collapse.png'), bbox_inches='tight', dpi=200)
plt.show()

# %% [markdown]
# ## 5. Ground truth — niche-vs-rest DE on real single cells
#
# Wilcoxon rank-sum within each cell type, niche X vs. all other niches, on the real single
# cells. This is what every method is scored against, and it is method-independent — so it
# is computed once and shared by every arm.

# %%
ground_truth = nd.get_ground_truth(
    adata, GT_PATH, ct_key=CT_KEY, niche_key=NICHE_KEY, force=FORCE_RERUN,
)
print(f'{len(ground_truth)} (cell type, niche) pairs')

# %% [markdown]
# ## 6. DGE recovery per arm
#
# Each arm writes its own CSV before the next one starts, so a disconnect loses at most one
# arm. Re-running prints `loaded from ...` for the arms already done.

# %%
results = {}
for name, lab in labels.items():
    results[name] = nd.run_niche_dge(
        name, lab, adata, ground_truth, TABLE_DIR,
        ct_key=CT_KEY, niche_key=NICHE_KEY, force=FORCE_RERUN,
    )
    gc.collect()

# %% [markdown]
# ## 7. Compare
#
# `pearson_r` / `kendall_tau` (branch 1, no significance gate) is the headline. `tpr`
# (branch 2) is reported alongside and is expected lower — SuperCell's test was validated
# on far more metacells per group than exist here. Every mean carries its std across cell
# types.

# %%
summary = nd.summary_table(results)
summary.to_csv(os.path.join(TABLE_DIR, 'dge_summary.csv'))
summary.round(3)

# %%
fig = nd.plot_dge_summary(
    summary, highlight=NEW_LOSS,
    out_path=os.path.join(FIG_DIR, f'{DS_ID}_dge_summary.png'),
    title=f'Niche-program recovery, within cell type ({DS_ID})',
)
plt.show()

# %% [markdown]
# ### Paired, per (cell type, niche)
#
# `pearson_r_gap_<arm>` = that arm minus the new-loss arm. Negative gap = the new loss wins
# on that pair. Sorted so the new loss's worst pairs are at the top.

# %%
paired = nd.compare_arms(results, reference=NEW_LOSS, metric='pearson_r')
gap_cols = [c for c in paired.columns if c.startswith('pearson_r_gap_')]
if gap_cols:
    paired = paired.sort_values(gap_cols[0], ascending=False)
paired.to_csv(os.path.join(TABLE_DIR, 'dge_paired.csv'), index=False)
paired.round(3).head(25)

# %% [markdown]
# ### Size-floor sweep — is a remaining gap a size artifact or missing signal?
#
# Restricts the comparison to pairs where the new-loss arm's own niche-positive metacells
# clear a rising size floor. If its curve rises toward the baselines as the floor rises,
# the residual gap is small-N pseudobulk noise, not absent signal. Flat at floor 30-50
# means the signal genuinely is not there.

# %%
sweep = nd.size_floor_sweep(results, size_arm=NEW_LOSS)
sweep.to_csv(os.path.join(TABLE_DIR, 'dge_size_floor_sweep.csv'), index=False)
print(sweep.round(3).to_string(index=False))

# %%
fig = nd.plot_size_floor_sweep(
    sweep, arms=list(results), highlight=NEW_LOSS,
    out_path=os.path.join(FIG_DIR, f'{DS_ID}_size_floor_sweep.png'),
)
plt.show()

# %% [markdown]
# ## 8. R4 — the non-circular discovery check
#
# Everything above measures **recovery**: does a metacell reproduce a difference already
# visible in the single cells? That is a validity check, not a discovery claim.
#
# This section asks the question the paper actually needs: which niche programs are
# **invisible to standard pooled cell-type DE** (all cells of a type, niche ignored) but
# visible at the niche level? Those are the programs the paper claims existing analysis
# averages away — and the pairs listed here, ranked by how well the new-loss arm recovers
# them, are the candidates for the R4 figure.
#
# Optional and last: nothing above depends on it.

# %%
pooled = nd.pooled_celltype_de(adata, CT_KEY, path=POOLED_PATH, force=FORCE_RERUN)

candidates = nd.discovery_candidates(ground_truth, pooled, results, arm=NEW_LOSS)
candidates.to_csv(os.path.join(TABLE_DIR, 'discovery_candidates.csv'), index=False)
print(f'{len(candidates)} (cell type, niche) pairs have genes hidden from pooled DE')
candidates.head(15)

# %% [markdown]
# ## What to read off this notebook
#
# 1. **`collapse_summary.csv`** — did `dip_p` rise above 0.05 and `largest_share` fall
#    under the new loss? That is the prerequisite. If not, stop here and fix the loss.
# 2. **`dge_summary.csv`** — did the new loss close the gap to SEACells on `pearson_r`?
# 3. **`dge_size_floor_sweep.csv`** — if a gap remains, is it a metacell-size artifact?
# 4. **`discovery_candidates.csv`** — which (cell type, niche) pairs carry programs that
#    pooled DE cannot see. This is the shortlist the R4 figure gets built from.
#
# Interpretation of these numbers is a separate conversation — this notebook produces
# them, it does not decide what they mean.

# %%
del adata
gc.collect()
torch.cuda.empty_cache()

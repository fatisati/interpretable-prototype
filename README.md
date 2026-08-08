# Interpretable SSL

Train interpretable self-supervised metacell models on single-cell RNA-seq and spatial transcriptomics data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fatisati/interpretable-prototype/blob/master/notebooks/example.ipynb)

## Installation

```bash
pip install faiss-gpu-cu12  # or faiss-cpu if no GPU
pip install git+https://github.com/fatisati/interpretable-prototype.git
```

On **Google Colab**, set your model save directory before importing:

```python
import os
os.environ['MODEL_DIR'] = '/content/drive/MyDrive/models/'
```

## Quickstart

```python
from interpretable_ssl.experiments.tasks import find_metacells

t, res, mc_adata = find_metacells(
    '/path/to/your_data.h5ad',
    label_key='cell_type',
    cvae_epochs=50,
    train_epochs=50,
    eval_freq=5,
    patience=20,
)
print(res)
```

## Parameters

### Required

| Parameter | Description |
|---|---|
| `ds_id` | Path to your `.h5ad` file, or an `AnnData` object directly |
| `label_key` | Column in `adata.obs` with cell-type labels. Used for evaluation metrics only, not training. |
| `cvae_epochs` | Epochs for pre-training the encoder (e.g. 50) |
| `train_epochs` | Max epochs for UMAP training — early stopping applies (e.g. 50) |
| `eval_freq` | Evaluate quality every N epochs (e.g. 5) |
| `patience` | Stop if no improvement for this many eval steps (e.g. 20) |

### Optional

| Parameter | Default | Description |
|---|---|---|
| `batch_key` | `None` | Column in `adata.obs` with batch/sample labels. If not provided, all cells are treated as a single batch. |
| `niche_key` | `None` | Column in `adata.obs` with niche annotations. Only needed for spatial niche metrics. |
| `num_prototypes` | `n_cells // 100` | Number of metacells/prototypes. If not set, one prototype per 100 cells. |
| `affinity_type` | `'arbf'` | Graph construction method — see below. |
| `result_save_path` | `None` | Directory to save `metrics.json`. If not set, results are only returned. |
| `batch_size` | `256` | Training batch size. |

## Affinity types

The affinity graph controls which cells are considered neighbours during training.

- **`arbf`** — Adaptive RBF kernel on PCA embeddings. Builds a kNN graph in expression space where edge weights decay smoothly with distance. Good default for scRNA-seq datasets.

- **`ctx_umap`** — Spatial context UMAP. For each cell, averages the PCA embeddings of its k nearest spatial neighbors, then builds a UMAP graph on these spatially-smoothed embeddings. Connects cells that are transcriptionally similar *within their spatial context*. **Preferred for spatial transcriptomics data.** Requires `adata.obsm['spatial']` to be present.

## Full example

```python
from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON

t, res, mc_adata = find_metacells(
    '/path/to/your_data.h5ad',
    label_key='cell_type',
    cvae_epochs=50,
    train_epochs=50,
    eval_freq=5,
    patience=20,
    batch_size=512,
    batch_key='sample',
    niche_key='niche',        # optional, for spatial datasets
    num_prototypes=200,
    affinity_type='arbf',
    result_save_path='./results/',
)

print(res)
# {'purity': 0.87, 'modularity': 0.54, 'batch_entropy': 0.91, ...}

# mc_adata: AnnData [K prototypes × genes] with decoded metacell gene expression
# t.train_ds.adata.obs['metacell_id']: integer metacell ID per cell
```

## Output

`res` is a dict with the following keys:

- `purity` — average cell-type purity per metacell
- `batch_entropy` — how evenly batches are mixed per metacell
- `modularity` — graph modularity of the metacell partition
- `coverage` — fraction of cell types represented
- `dge_rbo_avg`, `dge_kendall_avg`, `dge_jaccard_avg` — differential gene expression ranking agreement
- `scgraph_corr_avg` — correlation between metacell and single-cell graphs

## Design notes / ongoing investigations

Deeper write-ups of specific design decisions and open questions (not
derivable from reading the code alone) live in `files/`:

- [`files/sim_recon_investigation_index.md`](files/sim_recon_investigation_index.md) —
  entry point for the `lambda_sim_recon` investigation: what it is, the
  collapse bugs found and fixed, `full` vs `diffusion` targets, the
  diffusion-map eigenvalue-decay measurement, and why compaction is done
  per-cell rather than via a shared global basis. Links out to the other
  `files/sim_recon_*.md` docs and the notebooks each one is backed by.

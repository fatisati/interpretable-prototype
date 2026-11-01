"""Local (macOS, no CUDA) smoke test of find_metacells on scanpy's pbmc3k.

Usage (from repo root):
    conda activate interpretable-ssl
    python notebooks/pbmc3k_local/run_pbmc3k_local.py            # CPU
    USE_MPS=1 python notebooks/pbmc3k_local/run_pbmc3k_local.py  # Apple GPU (opt-in)

Data: data/pbmc3k.h5ad, built by prepare_pbmc3k.py in this folder (2638 cells,
2000 HVGs, counts layer, louvain labels in obs['cell_type']).

This is a SMOKE TEST of the pipeline on this machine, not a result. It uses the
LAMBDA_PROTO_UMAP_PRECON preset and tiny epoch counts; it is not the 4-term
gradient-share-calibrated config from CLAUDE.md section 9d.
"""
import os, sys, time, json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('HOME_DIR',  ROOT + '/')
os.environ.setdefault('MODEL_DIR', os.path.join(ROOT, 'models') + '/')
os.environ.setdefault('DATA_DIR',  os.path.join(ROOT, 'data') + '/')
os.environ.setdefault('CODE_DIR',  ROOT + '/')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.chdir(ROOT)                      # get_affinity_path uses ./graphs
sys.path.insert(0, ROOT)

import interpretable_ssl               # noqa: anndata.read shim
from interpretable_ssl.utils import get_device
from interpretable_ssl.experiments.tasks import find_metacells

FORCE_RERUN = False
OUT = os.path.join(ROOT, 'notebooks', 'pbmc3k_local', 'tables')

print(f"device = {get_device()}")
t0 = time.time()
t, res, mc_adata = find_metacells(
    os.path.join(ROOT, 'data', 'pbmc3k.h5ad'),
    label_key='cell_type',
    batch_key=None,
    num_prototypes=30,
    cvae_epochs=10,
    train_epochs=4,
    eval_freq=2,
    patience=4,
    batch_size=256,
    umap_steps_per_epoch=None,
    load_pretrain=not FORCE_RERUN,
    result_save_path=OUT,
)
print(f"\nTOTAL WALL TIME {time.time()-t0:.0f}s on {get_device()}")
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
print(mc_adata)

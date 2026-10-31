"""Shared local-run environment: paths, anndata compat shims, imports.
Mirrors _env() in modal_scripts/train_d64_nb.py for this Mac checkout."""
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('HOME_DIR',  ROOT + '/')
os.environ.setdefault('MODEL_DIR', os.path.join(ROOT, 'models') + '/')
os.environ.setdefault('DATA_DIR',  os.path.join(ROOT, 'data') + '/')
os.environ.setdefault('CODE_DIR',  ROOT + '/')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.chdir(ROOT)                      # get_affinity_path uses ./graphs
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import interpretable_ssl  # noqa: F401  (anndata.read shim, SCIPY_ARRAY_API)

# old-anndata files encode Python None as an HDF5 dataset with encoding-type 'null'
import h5py
from anndata._io.specs.registry import _REGISTRY, IOSpec
try:
    @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
    def _read_null(elem, *a, **kw):
        return None
except TypeError:
    pass
try:
    import anndata
    anndata.settings.allow_write_nullable_strings = True
except Exception:
    pass

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
TAG = "arbf_ctb20_seed0"
RAW_H5 = os.path.join(ROOT, 'data', 'raw_figshare', 'Immune_ALL_human.h5ad')
HVG_H5 = os.path.join(ROOT, 'data', 'scpoli', 'Immune_ALL_human_hvg.h5ad')
SPLIT_DIR = os.path.join(ROOT, 'data', 'node_heldout')
TRAIN_H5 = os.path.join(SPLIT_DIR, f"{DS}_{TAG}_train.h5ad")
SPLIT_NPZ = os.path.join(SPLIT_DIR, f"{DS}_{TAG}_split.npz")

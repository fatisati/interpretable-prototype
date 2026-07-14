"""
Colab boilerplate — run this at the top of every notebook:
    %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py
"""
import sys, os, importlib
from importlib import reload

# Auto-reload any changed module before every cell — no need to re-run nb_setup
try:
    ip = get_ipython()
    ip.run_line_magic('load_ext', 'autoreload')
    ip.run_line_magic('autoreload', '2')
except Exception:
    pass

PROJECT_ROOT = '/content/drive/MyDrive/codes/interpretable-prototype'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import anndata
if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ['HOME_DIR']     = '/content/drive/MyDrive/'
os.environ['MODEL_DIR']    = '/content/drive/MyDrive/models/'
os.environ['DATA_DIR']     = '/content/drive/MyDrive/data/'
os.environ['CODE_DIR']     = '/content/drive/MyDrive/codes/interpretable-prototype/'
os.environ['ISLANDER_SRC'] = '/content/drive/MyDrive/codes/Islander/src'

os.chdir('/content/drive/MyDrive/codes/interpretable-prototype/')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

def reload_interpretable_ssl():
    for m in list(sys.modules):
        if m.startswith("interpretable_ssl") or m in ("constants", "seacell_train", "metaq_train", "sure_train"):
            del sys.modules[m]

# Make tasks available directly after %run nb_setup.py
reload_interpretable_ssl()
from interpretable_ssl.experiments.tasks import (
    get_trainer,
    run_mc_task,
    LAMBDA_PROTO_UMAP,
    LAMBDA_PROTO_UMAP_PRECON,
    LAMBDA_PARAM_UMAP,
    LAMBDA_RECON_ONLY,
    LAMBDA_PROTO_RECON_ONLY,
    LAMBDA_PROTO_CTX_UMAP,
)

from seacell_train import (
    train_seacell,
    eval_seacell_task1,
    eval_seacell_task2,
    eval_seacell_task3,
)

from metaq_train import train_metaq, eval_metaq_task1

from sure_train import (
    train_sure,
    eval_sure_task1,
    eval_sure_task2,
    eval_sure_task3,
)

from interpretable_ssl.evaluation.paper_figures import *
from interpretable_ssl.evaluation.metric_helpers.result_tables import *
from interpretable_ssl.augmenters.graph_generator import generate_affinity, save_affinity

print("nb_setup done. Available: get_trainer, run_mc_task, fig_*, LAMBDA_PROTO_UMAP, LAMBDA_PROTO_UMAP_PRECON, LAMBDA_PARAM_UMAP, LAMBDA_RECON_ONLY, train_sure, eval_sure_task1/2/3")
print("Configs:", {
    'LAMBDA_PROTO_UMAP': LAMBDA_PROTO_UMAP,
    'LAMBDA_PARAM_UMAP': LAMBDA_PARAM_UMAP,
    'LAMBDA_RECON_ONLY': LAMBDA_RECON_ONLY,
})

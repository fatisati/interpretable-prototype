"""[D46] Node-holdout training on Modal -- THE protocol (user decision 2026-09-24).

20% of cells (stratified by batch; the rebuttal's cached split) are absent from
EVERYTHING: PCA, affinity graph, pretraining, training. After training, the
held-out cells are placed by a FROZEN encoder forward pass (the rebuttal's
encode_and_assign_new_cells contract, asserted, not assumed) and a combined
clusters_full.npz over all N cells is written next to the run.

Assets expected on the volume (upload_nodeheldout_assets in the repo history):
  /vol/data/node_heldout/<ds>_arbf_nodeheldout20_seed0_train.h5ad
  /vol/data/node_heldout/<ds>_arbf_nodeheldout20_seed0_split.npz
  /vol/code/data/<ds>_arbf_nodeheldout20_seed0_train_label_encoder.pkl
  /vol/code/graphs/affinity_<ds>_arbf_nodeheldout20_seed0_train<N>_..._arbf.pkl
  /vol/models/<ds>_arbf_nodeheldout20_seed0_train/pretrain/.../pretrain_checkpoint.pth

Config: single source build_config (cohesion 9d base + parameterized
lam_umap / lam_proto_recon / lam_recon / grad_probe). probe() returns the
validated config the DEPLOYED code builds -- run it after every deploy, before
any spawn (a stale deployment already cost three GPU runs on d42).

Deploy + spawn:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal deploy modal_scripts/train_d46_nodeheldout.py
  then probe() and Function.spawn via modal.Function.from_name("scproto-d46", ...).
"""
import time

import modal

app = modal.App("scproto-d46")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in [
            "lung", "lung_fetal_donor", "lung_fetal_organoid", "brain", "breast",
            "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas",
        ]
    ])
)

volume = modal.Volume.from_name("scproto-data")

AUTHORITY = 1.5403
TAG = "arbf_nodeheldout20_seed0"
CONFIG_TAG = "d46nh"

# ds -> (full h5ad rel to DATA_DIR, batch_key, label_key, K)
DATASETS = {
    "pancreas": ("pancreas_hvg.h5ad", "tech", "celltype", 220),
    "lung": ("lung_hvg.h5ad", "batch", "cell_type", 300),
    "pbmc-immune": ("scpoli/Immune_ALL_human_hvg.h5ad", "study",
                    "final_annotation", 300),
}

ZERO_TERMS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl")


def build_config(lam_gram, lam_aff, lam_recon=0.0, lam_proto_recon=0.0,
                 lam_umap=0.0, grad_probe=0, umap_proto_sg=0):
    """THE loss config (CLAUDE.md 9d cohesion base), with the [D44]/[D45]
    parameterized terms. ONE definition used by run() AND probe()."""
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_umap": lam_umap,
        "lambda_proto_recon": lam_proto_recon,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": lam_recon,
        "proto_reinit_every": 0,
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 500.0,
        "grad_probe": grad_probe,      # [D45] gradient-norm probe cadence
        # [D48] 1 = community/UMAP gradient reaches the ENCODER only (detached
        # prototypes in its similarity); cohesion is sole owner of prototypes.
        "umap_proto_sg": umap_proto_sg,
        "latent_dims": 8,
    }


def validate(lc, lam_gram, lam_aff, span_aff, span_gram):
    for k in ZERO_TERMS:
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    assert lc["lambda_cohesion"] > 0 and lc["lambda_cohesion_aff"] > 0
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["cohesion_global"] == 2 and lc["coh_mem_init"] == "full"
    assert lc["latent_dims"] == 8
    auth = lam_gram * span_gram + lam_aff * span_aff
    assert abs(auth - AUTHORITY) < 0.03, f"authority drifted: {auth}"
    return {
        "config_tag": CONFIG_TAG, "protocol": "node-holdout", "tag": TAG,
        "authority": round(auth, 4),
        "all_zero_terms": {k: float(lc[k]) for k in ZERO_TERMS},
        "objective": {"lambda_cohesion(gram)": lc["lambda_cohesion"],
                      "lambda_cohesion_aff": lc["lambda_cohesion_aff"]},
        "lambda_umap": lc["lambda_umap"],
        "lambda_recon": lc["lambda_recon"],
        "lambda_proto_recon": lc["lambda_proto_recon"],
        "grad_probe": lc["grad_probe"],
        "umap_proto_sg": lc["umap_proto_sg"],
        "cohesion_self": lc["cohesion_self"],
        "latent_dims": lc["latent_dims"],
    }


def _env():
    import os
    import sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, *a, **kw):
            return None
    except TypeError:
        pass

    # The node-holdout train h5ad carries pandas nullable-string obs columns
    # (written by a newer anndata); any save in the pipeline (metacells.h5ad,
    # caches) hard-crashes without this opt-in. Killed the first gmA run AFTER
    # training completed -- turn it on before anything writes.
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


@app.function(image=image, timeout=900, volumes={"/vol": volume})
def probe(lam_gram: float, lam_aff: float, span_aff: float, span_gram: float,
          lam_recon: float = 0.0, lam_proto_recon: float = 0.0,
          lam_umap: float = 0.0, grad_probe: int = 0, umap_proto_sg: int = 0):
    """Validated config the DEPLOYED code builds. Run after EVERY deploy."""
    _env()
    return validate(build_config(lam_gram, lam_aff, lam_recon, lam_proto_recon,
                                 lam_umap, grad_probe, umap_proto_sg),
                    lam_gram, lam_aff, span_aff, span_gram)


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run(ds: str, lam_gram: float, lam_aff: float,
        span_aff: float, span_gram: float, epochs: int = 10,
        suffix: str = "", lam_recon: float = 0.0, lam_proto_recon: float = 0.0,
        lam_umap: float = 0.0, grad_probe: int = 0, umap_proto_sg: int = 0):
    import os
    _env()

    import numpy as np
    import pandas as pd
    import torch

    from interpretable_ssl.experiments.tasks import run_mc_task

    full_h5, bk, lk, K = DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{TAG}_train.h5ad"
    split_np = f"/vol/data/node_heldout/{ds}_{TAG}_split.npz"
    assert os.path.exists(train_h5), f"missing {train_h5}"
    assert os.path.exists(split_np), f"missing {split_np}"

    name = f"{CONFIG_TAG}{suffix}_{ds}"
    lc = build_config(lam_gram, lam_aff, lam_recon, lam_proto_recon,
                      lam_umap, grad_probe, umap_proto_sg)
    checked = validate(lc, lam_gram, lam_aff, span_aff, span_gram)
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}")

    print("\n" + "=" * 78)
    print(f"=== {name}  NODE-HOLDOUT  K={K}  LD=8  epochs={epochs}")
    print(f"    train h5ad: {train_h5}")
    print(f"    lam_gram={lam_gram:.2f} lam_aff={lam_aff:.2f} "
          f"lam_umap={lam_umap} lam_recon={lam_recon} "
          f"lam_proto_recon={lam_proto_recon} grad_probe={grad_probe}")
    print("=" * 78)

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=bk, label_key=lk,
        cvae_epochs=50, train_epochs=epochs, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"[{name}] trained in {time.time()-t0:.0f}s  "
          f"n_dead={metrics.get('n_dead')}")

    # ---- frozen-pass placement of the held-out 20% (rebuttal contract) ----
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        get_preprocessed_adata)
    ad_full = get_preprocessed_adata(ds)
    d = np.load(split_np, allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()

    assert trainer.check_conditions_compatible(trainer.model, ad_test), (
        "ad_test has batch/condition values the trained model never saw -- "
        "encode_adata would silently RETRAIN via scPoli's load_query_data "
        "instead of a frozen forward pass. Refusing to leak."
    )
    with torch.no_grad():
        z = trainer.encode_adata(ad_test, trainer.model, z_idx=1)
        assignments_test = trainer.model.prototypes(z).argmax(dim=1).cpu().numpy()

    assignments_train, _ = trainer._get_assignments()
    s_train = pd.Series(assignments_train,
                        index=trainer.train_ds.adata.obs_names)
    s_test = pd.Series(assignments_test, index=ad_test.obs_names)
    combined = pd.concat([s_train, s_test]).reindex(ad_full.obs_names)
    assert not combined.isna().any(), "missing assignment after combining"
    labels_full = combined.values.astype(int)

    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    out = os.path.join(trainer.get_dump_path(), "clusters_full.npz")
    np.savez(out, labels=labels_full, test_mask=test_mask)
    print(f"[{name}] saved combined assignments for all {len(ad_full)} cells "
          f"({int(test_mask.sum())} frozen-pass test cells): {out}")
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics

"""[D64] FINISH the NB arms WITHOUT retraining. (constants inlined from
train_d64_nb.py -- Modal mounts only the entry script, so a cross-import of
modal_scripts fails inside the container.)

Original header follows.

[D63] Four arms: proto_recon form 3 with scores DETACHED, under the
TWO-BUDGET calibration, crossed with CVAE-recon on/off and MSE/NB.

PRE-REGISTRATION: my-docs/adr/D63-two-budget-detached-mse-nb.md

WHY THE TWO-BUDGET CHANGE MATTERS (derived from the D62 probe, 2026-10-24):
With `coh_proto_sg=1` and `umap_proto_sg=1`, only TWO terms touch the prototypes
at all -- GM-KL and proto_recon. The old calibration set every lambda from the
ENCODER-path gradient, which for proto_recon with DETACHED scores is
structurally ZERO. Reading the probe's prototype-path norms instead:

    gnorm_proto   gmkl 0.0551     proto_recon 9.261
    deployed      lambda_gmkl 0.02228  ->  prototype pull 0.0012276
    D62 dose      lambda_pr 0.001236   ->  prototype pull 0.011447  = 9.3x GM-KL
    D62 high dose lambda_pr 0.002472   ->                            18.6x GM-KL

GM-KL is the ONLY force holding a prototype at its members' latent centroid, and
proto_recon was out-pulling it ~9-19x. That is a sufficient explanation for the
measured collapse (median nearest-prototype cosine 0.81 without proto_recon ->
0.9997 with it). D63 sets the prototype budget 50/50 instead.

ARMS (all: leiden init, cohesion gram+aff proto-detached, GM-KL, mix3 with
scores DETACHED and decoder LIVE, no community, 10 epochs, K=300):
    d63_mse-rec     MSE, CVAE recon ON
    d63_mse-norec   MSE, CVAE recon OFF
    d63_nb-rec      NB,  CVAE recon ON
    d63_nb-norec    NB,  CVAE recon OFF

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach \
       modal_scripts/train_d63_budgets.py --arms all
"""
import json
import os
import time

import modal

app = modal.App("scproto-d64-finish")

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
        for name in ["lung", "lung_fetal_donor", "lung_fetal_organoid", "brain",
                     "breast", "heart", "eye", "gut_fetal", "skin", "COVID",
                     "pancreas"]
    ])
)

volume = modal.Volume.from_name("scproto-data")

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
TAG = "arbf_ctb20_seed0"
TRAIN_EPOCHS = 10
SEED = 31

# ---------------------------------------------------------------------------
# ENCODER-path gradient norms (frozen full pass, lambda=1). MSE values measured
# by the D62/D63 probes; NB values filled by measure_grad_shares_d63nb.py.
# ---------------------------------------------------------------------------
GNORM_ENC = {
    "mse": {"cohesion_gram": 0.043067, "cohesion_aff": 0.02464,
            "recon": 11.834, "gmkl": 11.223},
    # [D64] MEASURED 2026-10-25, measure_grad_shares_d63nb.py (recon_loss='nb',
    # mix3, scores DETACHED, lambda=1): 402/402 rows, 0 errors, weights verified
    # frozen. CVs 0.00-0.09.
    "nb":  {"cohesion_gram": 0.042662, "cohesion_aff": 0.030624,
            "recon": 16.478, "gmkl": 13.681},
}
# PROTOTYPE-path gradient norms -- the second budget. proto_recon's value MUST
# come from the DETACHED probe (measure_grad_shares_d63.py), because detaching
# the scores removes one of the two gradient paths into the prototypes.
GNORM_PROTO = {
    # MEASURED 2026-10-24, measure_grad_shares_d63.py, scores DETACHED,
    # 402/402 rows, 0 errors, weights verified frozen. Routing verified:
    # cohesion/umap/recon put exactly 0 on the prototypes.
    # (with scores LIVE the D62 probe read 9.261 -- detaching removes one of
    #  the two gradient paths into the prototypes, hence the lower value.)
    "mse": {"gmkl": 0.055115, "proto_recon": 5.6654},
    # [D64] same probe, PROTOTYPE path. Routing verified: cohesion/umap/recon
    # put exactly 0 on the prototypes.
    "nb":  {"gmkl": 0.050154, "proto_recon": 8.7688},
}

# encoder budget: unchanged from D60 (its relative proportions are what worked)
SHARE_ENC = {"kernel": 1 / 8, "recon": 1 / 4, "gmkl": 1 / 4}
# prototype budget: the two terms that touch prototypes, split evenly
SHARE_PROTO = {"gmkl": 0.5, "proto_recon": 0.5}

# [D64] the user's 4 arms: ALL NB (recon_loss='nb', V1 per-cell size factor),
# crossed over prototype INIT and whether the CVAE recon term is on.
# arm -> (prot_init, cvae_recon_on); recon_loss is 'nb' throughout.
ARMS = {
    "d64_leid-rec":   ("leid",     True),
    "d64_leid-norec": ("leid",     False),
    "d64_wayp-rec":   ("waypoint", True),
    "d64_wayp-norec": ("waypoint", False),
}


def build_config(prot_init, recon_on, lik="nb"):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    ge, gp = GNORM_ENC[lik], GNORM_PROTO[lik]
    for d, nm in ((ge, "GNORM_ENC"), (gp, "GNORM_PROTO")):
        for k, v in d.items():
            assert v, f"{nm}[{lik}][{k}] not filled -- run the probe first"
    lam_gmkl = round(SHARE_ENC["gmkl"] / ge["gmkl"], 6)
    # prototype budget: pin proto_recon so its PROTOTYPE pull matches GM-KL's,
    # in the ratio SHARE_PROTO. lambda_gmkl is already fixed by the encoder
    # budget, so only proto_recon is free here.
    gmkl_proto_pull = lam_gmkl * gp["gmkl"]
    lam_pr = round((SHARE_PROTO["proto_recon"] / SHARE_PROTO["gmkl"])
                   * gmkl_proto_pull / gp["proto_recon"], 8)
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "prot_init": prot_init,
        "recon_loss": lik,
        "lambda_cohesion": round(SHARE_ENC["kernel"] / ge["cohesion_gram"], 6),
        "lambda_cohesion_aff": round(SHARE_ENC["kernel"] / ge["cohesion_aff"], 6),
        "lambda_recon": round(SHARE_ENC["recon"] / ge["recon"], 6) if recon_on else 0.0,
        "lambda_gmkl": lam_gmkl,
        "lambda_proto_recon": lam_pr,
        "proto_recon_form": "mix3",
        "proto_recon_dec_sg": 0,        # decoder TRAINABLE in the term
        "proto_recon_stopgrad": True,   # scores DETACHED (the D63 design)
        "lambda_umap": 0.0,
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "grad_probe": 25, "seed": SEED,
    }




def _env():
    import shutil
    import sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    # h5py + network volume = unreliable seeks; stage to container-local disk
    for rel in (f"node_heldout/{DS}_{TAG}_train.h5ad",
                f"node_heldout/{DS}_{TAG}_train_label_encoder.pkl",
                f"node_heldout/{DS}_{TAG}_split.npz",
                "scpoli/Immune_ALL_human_hvg.h5ad"):
        s_, d_ = f"/vol/data/{rel}", f"/tmp/data/{rel}"
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if not os.path.exists(d_) or os.path.getsize(d_) != os.path.getsize(s_):
            shutil.copyfile(s_, d_)
    os.environ["DATA_DIR"] = "/tmp/data/"


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def finish(arm: str):
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    prot_init, recon_on = ARMS[arm]
    lc = build_config(prot_init, recon_on, "nb")
    train_h5 = f"/tmp/data/node_heldout/{DS}_{TAG}_train.h5ad"

    # load_umap=True  -> load the SAVED stage-2 checkpoint, do NOT train
    # skip_eval=True  -> skip the scib eval that killed the original runs
    trainer, _, _ = run_mc_task(
        train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        cvae_epochs=50, train_epochs=TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": arm},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
        load_umap=True, skip_eval=True,
    )
    print(f"   [D64 finish] loaded checkpoint for {arm}: {trainer.get_model_name()}")

    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
    ad_full = get_preprocessed_adata(DS)
    d = np.load(f"/tmp/data/node_heldout/{DS}_{TAG}_split.npz", allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
    assert trainer.check_conditions_compatible(trainer.model, ad_test)

    trainer.model.eval()
    with torch.no_grad():
        z_te = trainer.encode_adata(ad_test, trainer.model, z_idx=1)
        z_te = z_te.cpu().numpy() if hasattr(z_te, "cpu") else np.asarray(z_te)
        a_te = trainer.model.prototypes(
            torch.as_tensor(z_te, device=trainer.device)).argmax(dim=1).cpu().numpy()
        z_tr = trainer.encode_adata(trainer.train_ds.adata, trainer.model, z_idx=1)
        z_tr = z_tr.cpu().numpy() if hasattr(z_tr, "cpu") else np.asarray(z_tr)
    a_tr, _ = trainer._get_assignments()

    combined = pd.concat([
        pd.Series(a_tr, index=trainer.train_ds.adata.obs_names),
        pd.Series(a_te, index=ad_test.obs_names)]).reindex(ad_full.obs_names)
    assert not combined.isna().any()
    lab = combined.values.astype(int)
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    P = trainer.model.get_prototypes().detach().cpu().numpy()
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12)
    Cm = Pn @ Pn.T
    np.fill_diagonal(Cm, -9)
    iu = np.triu_indices_from(Cm, 1)
    used = np.bincount(np.asarray(a_tr).astype(int), minlength=P.shape[0])
    summary = {
        "arm": arm, "prot_init": prot_init, "cvae_recon_on": recon_on,
        "recon_loss": "nb", "n_dead": int((used == 0).sum()),
        "median_nn_cos": float(np.median(Cm.max(1))),
        "frac_pairs_gt_0.99": float((Cm[iu] > 0.99).mean()),
        "model_name": trainer.get_model_name(),
    }
    dump = trainer.get_dump_path()
    np.savez(os.path.join(dump, "clusters_full.npz"), labels=lab, test_mask=test_mask)
    zf = np.zeros((len(ad_full), z_tr.shape[1]), dtype=np.float32)
    zf[ad_full.obs_names.get_indexer(list(trainer.train_ds.adata.obs_names))] = z_tr
    zf[test_mask] = z_te
    np.save(os.path.join(dump, "latent_z_full.npy"), zf)
    np.save(os.path.join(dump, "proto_vectors.npy"), P)
    with open(os.path.join(dump, "d64_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    volume.commit()
    print(f"[{arm}] {summary}")
    return summary


@app.local_entrypoint()
def main(arms: str = "all"):
    sel = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    for a in sel:
        try:
            print(f"OK {a}: {finish.remote(a)}", flush=True)
        except Exception as e:
            print(f"FAILED {a}: {type(e).__name__}: {str(e)[:200]}", flush=True)

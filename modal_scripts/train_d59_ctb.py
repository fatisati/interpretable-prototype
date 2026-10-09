"""[D59] Three runs on the (cell type x batch)-stratified split, pbmc-immune.

THE QUESTION
------------
D58 localised the problem: within-batch purity is fine (it was a PLACEMENT
artifact), cross-batch agreement is the real deficit and it is an OPTIMIZATION
failure -- 0.860 on TRAIN, 0.873 on test, against Leiden's 0.975 / 0.971.

Cause: no term both SEES cross-batch pairs and acts on the ASSIGNMENT.
  cohesion   excludes cross-batch pairs by design (CLAUDE.md 9d)
  community  sees them, but only 7.7% of train-graph edges are cross-batch
  recon      aligns z across batches, but nothing ties z-alignment to which
             prototype is picked -> measured cross-batch effect -0.006 (noise)

proto_recon is the candidate fix: each prototype is decoded under EACH batch
condition, so ONE prototype can reconstruct both batches' version of a cell
type, and spending two prototypes on the same biology wastes capacity under
fixed K. That is cross-batch pressure from reconstruction alone -- no labels, no
forced mixing, nothing pushed for a type present in only one batch.

It requires proto_recon_stopgrad=0. With the default True, scproto.py:1725 uses
`scores.detach()`, so proto_recon can only move prototypes and the decoder and
can NEVER change which prototype a cell picks -- the capacity pressure cannot
operate. That, plus D46 having tested it on single-batch s28nsc
(batch_key=None), is why its prior "FAILS" verdict does not apply here.

ARMS (identical split, identical pretrain checkpoint, identical everything else)
  scpoli_leiden       baseline: Leiden on the stage-1 train latent, test cells
                      placed by 15-NN cosine among train cells
  scproto_base        cohesion(gram+aff) + community + CVAE recon
  scproto_protorecon  cohesion(gram+aff) + community + proto_recon  (recon OFF)

Lambdas come from measure_grad_shares_ctb.py: share vector B (equal share per
SUBSYSTEM) divided by the frozen full-pass gnorm median on THIS split.

Run (probe must have completed first -- it also creates the pretrain checkpoint):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_d59_ctb.py --arms all
"""
import json
import os
import time

import modal

app = modal.App("scproto-d59-ctb")

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
# MEASURED frozen full-pass gnorm medians on THIS split, all lambdas = 1,
# proto_recon_stopgrad=0.  Filled in from measure_grad_shares_ctb.py.
# ---------------------------------------------------------------------------
GNORM = {
    # MEASURED 2026-10-06 by measure_grad_shares_ctb.py on this exact split:
    # 402/402 probe rows, 0 errors, weights VERIFIED frozen (prototype drift
    # 1.14e-08), CV 0.04-0.08. proto_recon measured for the FIRST time -- its
    # gradient is 4.3x recon's, so its lambda is correspondingly smaller.
    "cohesion_gram": 0.04306,
    "cohesion_aff":  0.024707,
    "umap":          7.1986,
    "recon":         12.048,
    "proto_recon":   51.881,
}

# share vector B: equal share per SUBSYSTEM (cohesion 1/3 split over its two
# kernels, community 1/3, the reconstruction subsystem 1/3)
SHARE_COH_KERNEL = 1 / 6
SHARE_COMMUNITY = 1 / 3
SHARE_RECONSTRUCTION = 1 / 3


def _lam(term, share):
    g = GNORM[term]
    assert g, (f"GNORM[{term}] is not filled in -- run "
               f"measure_grad_shares_ctb.py and paste its medians")
    return round(share / g, 6)


def lambdas(arm):
    """Returns (gram, aff, community, recon, proto_recon)."""
    gram = _lam("cohesion_gram", SHARE_COH_KERNEL)
    aff = _lam("cohesion_aff", SHARE_COH_KERNEL)
    comm = _lam("umap", SHARE_COMMUNITY)
    if arm == "scproto_base":
        return gram, aff, comm, _lam("recon", SHARE_RECONSTRUCTION), 0.0
    if arm == "scproto_protorecon":
        return gram, aff, comm, 0.0, _lam("proto_recon", SHARE_RECONSTRUCTION)
    raise ValueError(arm)


ZERO_ALWAYS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
               "lambda_swav", "lambda_kl")


def build_config(gram, aff, comm, recon, protorecon):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": gram, "lambda_cohesion_aff": aff,
        "lambda_umap": comm, "lambda_recon": recon,
        "lambda_proto_recon": protorecon,
        # nothing detached: gradient reaches BOTH the encoder (via scores) and
        # the prototypes. This is the whole point -- see module docstring.
        "proto_recon_stopgrad": 0,
        "proto_recon_hard": False,
        "proto_decoupled": False,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "umap_proto_sg": 1, "grad_probe": 25, "seed": SEED,
    }


def validate(lc, arm):
    for k in ZERO_ALWAYS:
        assert float(lc[k]) == 0.0, f"{arm}: {k} must be 0"
    assert lc["umap_proto_sg"] == 1
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    assert lc["proto_recon_stopgrad"] == 0, "the gradient must reach the assignment"
    # exactly ONE reconstruction term active
    r, pr = float(lc["lambda_recon"]), float(lc["lambda_proto_recon"])
    assert (r > 0) != (pr > 0), (
        f"{arm}: exactly one of recon / proto_recon must be active, got {r} / {pr}")
    assert float(lc["lambda_cohesion"]) > 0 and float(lc["lambda_cohesion_aff"]) > 0
    assert float(lc["lambda_umap"]) > 0
    return {"arm": arm, "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "community": lc["lambda_umap"], "recon": r, "proto_recon": pr,
            "proto_recon_stopgrad": lc["proto_recon_stopgrad"],
            "seed": lc["seed"], "train_epochs": TRAIN_EPOCHS}


def _env():
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
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


def _place_and_save(t, arm, metrics, train_s):
    """Frozen-pass placement of the held-out cells + save every artifact the
    local scorer needs. Writes clusters.npz AS WELL AS clusters_full.npz because
    score_all_baselines.load_labels() only ever looks for clusters.npz."""
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata

    ad_full = get_preprocessed_adata(DS)
    d = np.load(f"/vol/data/node_heldout/{DS}_{TAG}_split.npz", allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
    assert t.check_conditions_compatible(t.model, ad_test), "unseen conditions"

    with torch.no_grad():
        z_te = t.encode_adata(ad_test, t.model, z_idx=1)
        z_te = z_te.cpu().numpy() if hasattr(z_te, "cpu") else np.asarray(z_te)
        a_te = t.model.prototypes(
            torch.as_tensor(z_te, device=t.device)).argmax(dim=1).cpu().numpy()
        z_tr = t.encode_adata(t.train_ds.adata, t.model, z_idx=1)
        z_tr = z_tr.cpu().numpy() if hasattr(z_tr, "cpu") else np.asarray(z_tr)

    a_tr, _ = t._get_assignments()
    s_tr = pd.Series(a_tr, index=t.train_ds.adata.obs_names)
    s_te = pd.Series(a_te, index=ad_test.obs_names)
    combined = pd.concat([s_tr, s_te]).reindex(ad_full.obs_names)
    assert not combined.isna().any()
    lab = combined.values.astype(int)
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    dump = t.get_dump_path()
    np.savez(os.path.join(dump, "clusters_full.npz"), labels=lab, test_mask=test_mask)
    np.savez(os.path.join(dump, "clusters.npz"), labels=lab)
    # latent for BOTH halves, so the matched 15-NN placement can be evaluated
    zf = np.zeros((len(ad_full), z_tr.shape[1]), dtype=np.float32)
    zf[ad_full.obs_names.get_indexer(list(t.train_ds.adata.obs_names))] = z_tr
    zf[test_mask] = z_te
    np.save(os.path.join(dump, "latent_z_full.npy"), zf)
    with open(os.path.join(dump, "d59_arm_summary.json"), "w") as f:
        json.dump({"arm": arm, "train_seconds": round(train_s, 1),
                   "model_name": t.get_model_name(), "dump_path": dump,
                   "metrics": metrics}, f, indent=2, default=str)
    print(f"[{arm}] saved clusters(+full), latent_z_full, summary -> {dump}")
    return dump


@app.function(image=image, gpu="L4", timeout=21600, volumes={"/vol": volume})
def run_scproto(arm: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    gram, aff, comm, recon, pr = lambdas(arm)
    lc = build_config(gram, aff, comm, recon, pr)
    checked = validate(lc, arm)
    print(f"   [D59 CONFIG VERIFIED IN-RUN] {checked}")

    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    assert os.path.exists(train_h5), f"missing {train_h5}"
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        cvae_epochs=50, train_epochs=TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": f"d59_{arm}"},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    train_s = time.time() - t0
    assert getattr(trainer, "proto_recon_stopgrad", True) == 0, \
        "proto_recon_stopgrad did not reach the trainer"
    print(f"[{arm}] trained in {train_s:.0f}s  n_dead={metrics.get('n_dead')}")
    _place_and_save(trainer, arm, metrics, train_s)
    volume.commit()
    return {"arm": arm, "config": checked, "train_seconds": round(train_s, 1),
            "metrics": metrics}


@app.function(image=image, gpu="L4", timeout=14400, volumes={"/vol": volume})
def run_leiden():
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.neighbors import NearestNeighbors
    from interpretable_ssl.experiments.tasks import _resolve_ds_id, get_trainer
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        get_preprocessed_adata, build_latent_affinity, leiden_resolution_search)

    t0 = time.time()
    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    ds_id = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                           niche_key=None, num_prototypes=K)
    t = get_trainer(experiment_name="ctb_stage1_extract", cvae_epochs=50,
                    dataset_id=ds_id, l2norm=1, assignment_metric="dotp",
                    batch_size=1024, affinity_type="arbf")
    t.load_pretrain_checkpoint()          # created by the probe run
    ad_tr = t.train_ds.adata
    with torch.no_grad():
        z_tr = t.encode_adata(ad_tr, t.model, z_idx=1).cpu().numpy()

    ad_full = get_preprocessed_adata(DS)
    d = np.load(f"/vol/data/node_heldout/{DS}_{TAG}_split.npz", allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
    assert t.check_conditions_compatible(t.model, ad_test), "unseen conditions"
    with torch.no_grad():
        z_te = t.encode_adata(ad_test, t.model, z_idx=1).cpu().numpy()

    ad_tr.obsm["X_st1ctb"] = z_tr
    aff = build_latent_affinity(ad_tr, "X_st1ctb")
    lab_tr, _, nk = leiden_resolution_search(aff, K)
    print(f"[leiden] K={nk} on the train latent ({time.time()-t0:.0f}s)")

    zn_tr = z_tr / (np.linalg.norm(z_tr, axis=1, keepdims=True) + 1e-12)
    zn_te = z_te / (np.linalg.norm(z_te, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(zn_tr)
    _, idx = nn.kneighbors(zn_te)
    lab_te = np.array([np.bincount(lab_tr[row]).argmax() for row in idx])

    combined = pd.concat([pd.Series(lab_tr, index=ad_tr.obs_names),
                          pd.Series(lab_te, index=ad_test.obs_names)]
                         ).reindex(ad_full.obs_names)
    assert not combined.isna().any()
    lab = combined.values.astype(int)
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    out = f"/vol/models/{DS}/d59_leiden_ctb_K{K}"
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "clusters.npz"), labels=lab, test_mask=test_mask)
    np.savez(os.path.join(out, "clusters_full.npz"), labels=lab, test_mask=test_mask)
    zf = np.zeros((len(ad_full), z_tr.shape[1]), dtype=np.float32)
    zf[ad_full.obs_names.get_indexer(list(ad_tr.obs_names))] = z_tr
    zf[test_mask] = z_te
    np.save(os.path.join(out, "latent_z_full.npy"), zf)
    volume.commit()
    print(f"[leiden] saved {out}  K={nk}  ({time.time()-t0:.0f}s)")
    return {"arm": "scpoli_leiden", "K_found": int(nk), "dump_path": out}


@app.local_entrypoint()
def main(arms: str = "all"):
    sel = (["scpoli_leiden", "scproto_base", "scproto_protorecon"]
           if arms == "all" else [a.strip() for a in arms.split(",") if a.strip()])
    print(f"[D59] {DS} / {TAG}  train_epochs={TRAIN_EPOCHS}  arms={sel}")
    if any(a.startswith("scproto") for a in sel):
        print(f"[D59] gnorm medians in use: {GNORM}")
        for a in sel:
            if a.startswith("scproto"):
                print(f"        {a:<20} lambdas(gram,aff,comm,recon,protorecon) = "
                      f"{lambdas(a)}")
    calls = {}
    for a in sel:
        calls[a] = (run_leiden.spawn() if a == "scpoli_leiden"
                    else run_scproto.spawn(a))
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")
    outdir = os.path.join("files", "tables", "d59_ctb")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "d59_arms.json"), "w") as f:
        json.dump({"ok": out, "failed": failed, "gnorm": GNORM, "tag": TAG},
                  f, indent=2, default=str)
    print(f"\n[D59] wrote {outdir}/d59_arms.json")
    for a, r in out.items():
        print(f"  {a:<20} {r}")
    if failed:
        print(f"[D59] FAILED: {list(failed)}")

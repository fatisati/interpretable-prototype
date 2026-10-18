"""[D60] Three arms: GM-KL recipe with detached prototypes, (init x community).

PRE-REGISTRATION: my-docs/adr/D60-gmkl-recipe-detached-protos-three-arms.md
(written and committed BEFORE this script ran; bars there bind this run).

RECIPE (all arms):
  cohesion gram+aff   coh_proto_sg=1 -> assignments vs DETACHED protos,
                      gradient reaches the ENCODER only
  GM-KL               lambda_gmkl > 0: 0.5*(sig^2 - log sig^2 - 1)  [anti-collapse]
                      + 0.5*sum_k s_ik |mu_n - proto_k|^2           [centroid pull]
                      protos LIVE here -- their ONLY teacher
  recon               kept, freeze_batch_embedding=True (preservation role)
  community           arm d60_wayp_comm only, routed (umap_proto_sg=1)
  OFF                 proto_recon, nassoc, modularity, proto_usage, swav, plain kl

ARMS (identical split arbf_ctb20_seed0, identical pretrain checkpoint, seed 31):
  d60_leid_noc    prot_init='leid'  no community   vs baseline d60_epoch0_leid
  d60_wayp_noc    prot_init='wayp'  no community   vs baseline d59_epoch0
  d60_wayp_comm   prot_init='wayp'  + community    vs baseline d59_epoch0

SHARES: equal per ACTIVE subsystem (kernels 1/8 each, recon 1/4, gmkl 1/4,
community 1/4 or absent), lambda_i = share_i / median gnorm_i from the D60 probe
(measure_grad_shares_ctb.py, coh_proto_sg=1 + lambda_gmkl=1 -- the probe also
VERIFIED the routing: cohesion/community gnorm_proto == 0, gmkl > 0).

Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_d60_ctb.py --arms all
"""
import json
import os
import time

import modal

app = modal.App("scproto-d60-train")

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
# Frozen full-pass gnorm medians from the D60 probe (coh_proto_sg=1, gmkl=1).
# FILL FROM THE PROBE OUTPUT -- the asserts refuse to run on placeholders.
# ---------------------------------------------------------------------------
GNORM = {
    "cohesion_gram": None,
    "cohesion_aff": None,
    "umap": None,
    "recon": None,
    "gmkl": None,
}

SHARE = {"kernel": 1 / 8, "recon": 1 / 4, "gmkl": 1 / 4, "community": 1 / 4}


def _lam(term, share):
    g = GNORM[term]
    assert g, f"GNORM[{term}] not filled -- paste the D60 probe medians first"
    return round(share / g, 6)


ARMS = {
    # arm -> (prot_init, community_on)
    "d60_leid_noc": ("leid", False),
    "d60_wayp_noc": ("wayp", False),
    "d60_wayp_comm": ("wayp", True),
}

ZERO_ALWAYS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
               "lambda_swav", "lambda_kl", "lambda_proto_recon")


def build_config(prot_init, community_on):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "prot_init": {"leid": "leid", "wayp": "waypoint"}[prot_init],
        "lambda_cohesion": _lam("cohesion_gram", SHARE["kernel"]),
        "lambda_cohesion_aff": _lam("cohesion_aff", SHARE["kernel"]),
        "lambda_recon": _lam("recon", SHARE["recon"]),
        "lambda_gmkl": _lam("gmkl", SHARE["gmkl"]),
        "lambda_umap": _lam("umap", SHARE["community"]) if community_on else 0.0,
        "coh_proto_sg": 1,
        "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_proto_recon": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "grad_probe": 25, "seed": SEED,
    }


def validate(lc, arm):
    for k in ZERO_ALWAYS:
        assert float(lc[k]) == 0.0, f"{arm}: {k} must be 0"
    assert lc["coh_proto_sg"] == 1 and lc["umap_proto_sg"] == 1
    assert float(lc["lambda_gmkl"]) > 0, f"{arm}: gmkl is the protos' only teacher"
    assert float(lc["lambda_cohesion"]) > 0 and float(lc["lambda_cohesion_aff"]) > 0
    assert float(lc["lambda_recon"]) > 0
    assert lc["cohesion_self"] == 1.0 and lc["proto_reinit_every"] == 0
    return {"arm": arm, "prot_init": lc["prot_init"],
            "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "recon": lc["lambda_recon"], "gmkl": lc["lambda_gmkl"],
            "community": lc["lambda_umap"], "coh_proto_sg": lc["coh_proto_sg"],
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


@app.function(image=image, gpu="L4", timeout=21600, volumes={"/vol": volume})
def run_arm(arm: str):
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    prot_init, community_on = ARMS[arm]
    lc = build_config(prot_init, community_on)
    checked = validate(lc, arm)
    print(f"   [D60 CONFIG VERIFIED IN-RUN] {checked}")

    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    assert os.path.exists(train_h5)
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        cvae_epochs=50, train_epochs=TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": arm},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    train_s = time.time() - t0
    assert getattr(trainer, "coh_proto_sg", 0) == 1
    assert getattr(trainer, "lambda_gmkl", 0.0) > 0
    print(f"[{arm}] trained in {train_s:.0f}s  n_dead={metrics.get('n_dead')}")

    # --- post-run sigma check (P1 of the pre-registration), in-run and free ---
    captured = []
    hook = trainer.model.scpoli_cvae.encoder.register_forward_hook(
        lambda _m, _i, out: captured.append(out[1].detach().cpu()))
    with torch.no_grad():
        trainer.model.eval()
        trainer.encode_adata(trainer.train_ds.adata, trainer.model, z_idx=1)
    hook.remove()
    sig = np.exp(0.5 * torch.cat(captured, 0).numpy())
    sigma_median = float(np.median(sig))
    print(f"[{arm}] POST-RUN sigma median = {sigma_median:.4g} "
          f"(stage-1 was 0.3503; the broken regime was 2.7e-06)")

    # --- frozen-pass placement + artifacts for the scorer ---
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
    ad_full = get_preprocessed_adata(DS)
    d = np.load(f"/vol/data/node_heldout/{DS}_{TAG}_split.npz", allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
    assert trainer.check_conditions_compatible(trainer.model, ad_test)
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
    dump = trainer.get_dump_path()
    np.savez(os.path.join(dump, "clusters_full.npz"), labels=lab, test_mask=test_mask)
    np.savez(os.path.join(dump, "clusters.npz"), labels=lab)
    zf = np.zeros((len(ad_full), z_tr.shape[1]), dtype=np.float32)
    zf[ad_full.obs_names.get_indexer(list(trainer.train_ds.adata.obs_names))] = z_tr
    zf[test_mask] = z_te
    np.save(os.path.join(dump, "latent_z_full.npy"), zf)
    summary = {"arm": arm, "config": checked, "train_seconds": round(train_s, 1),
               "sigma_median_postrun": sigma_median,
               "model_name": trainer.get_model_name(), "dump_path": dump,
               "metrics": metrics}
    with open(os.path.join(dump, "d60_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    volume.commit()
    print(f"[{arm}] saved artifacts -> {dump}")
    return summary


@app.local_entrypoint()
def main(arms: str = "all"):
    sel = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    bad = [a for a in sel if a not in ARMS]
    assert not bad, f"unknown arms {bad}"
    print(f"[D60] {DS}/{TAG} epochs={TRAIN_EPOCHS} arms={sel}")
    for a in sel:
        pi, c = ARMS[a]
        lc = build_config(pi, c)
        print(f"  {a:<14} init={pi:<5} gram={lc['lambda_cohesion']:g} "
              f"aff={lc['lambda_cohesion_aff']:g} recon={lc['lambda_recon']:g} "
              f"gmkl={lc['lambda_gmkl']:g} comm={lc['lambda_umap']:g}")
    calls = {a: run_arm.spawn(a) for a in sel}
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")
    outdir = os.path.join("files", "tables", "d60_ctb")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "d60_arms.json"), "w") as f:
        json.dump({"ok": out, "failed": failed, "gnorm": GNORM}, f,
                  indent=2, default=str)
    print(f"\n[D60] wrote {outdir}/d60_arms.json")
    for a, s in out.items():
        print(f"  {a:<14} train={s['train_seconds']}s "
              f"n_dead={s['metrics'].get('n_dead')} "
              f"sigma_median={s['sigma_median_postrun']:.3g}")
    if failed:
        print(f"[D60] FAILED: {list(failed)}")

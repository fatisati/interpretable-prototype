"""[D63] Four arms: proto_recon form 3 with scores DETACHED, under the
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

app = modal.App("scproto-d64-train")

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


def validate(lc, arm, prot_init, recon_on, lik="nb"):
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_umap"):
        assert float(lc[k]) == 0.0, f"{arm}: {k} must be 0"
    assert lc["coh_proto_sg"] == 1 and lc["umap_proto_sg"] == 1
    assert lc["proto_recon_form"] == "mix3"
    assert lc["proto_recon_stopgrad"] is True
    assert lc["proto_recon_dec_sg"] == 0
    assert lc["recon_loss"] == lik and lc["prot_init"] == prot_init
    assert (float(lc["lambda_recon"]) > 0) == recon_on
    assert float(lc["lambda_gmkl"]) > 0 and float(lc["lambda_proto_recon"]) > 0
    assert lc["cohesion_self"] == 1.0 and lc["proto_reinit_every"] == 0
    return {"arm": arm, "recon_loss": lik, "prot_init": prot_init,
            "cvae_recon_on": recon_on,
            "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "recon": lc["lambda_recon"], "gmkl": lc["lambda_gmkl"],
            "proto_recon": lc["lambda_proto_recon"],
            "pr_stopgrad": lc["proto_recon_stopgrad"],
            "pr_dec_sg": lc["proto_recon_dec_sg"],
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


@app.function(image=image, gpu="L4", timeout=28800, volumes={"/vol": volume})
def run_arm(arm: str):
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    prot_init, recon_on = ARMS[arm]
    lik = "nb"
    lc = build_config(prot_init, recon_on, lik)
    checked = validate(lc, arm, prot_init, recon_on, lik)
    print(f"   [D64 CONFIG VERIFIED IN-RUN] {checked}")

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

    # --- the checks D61 lacked: prove the code and the config reached the RUN ---
    import inspect
    from interpretable_ssl.trainers import scproto as _sp
    assert "mix3" in inspect.getsource(_sp), "stale scproto.py on this container!"
    _mn = trainer.get_model_name()
    assert "prfm-mix3" in _mn, f"proto_recon_form missing from model name: {_mn}"
    if lik == "nb":
        assert "rlss-nb" in _mn, f"recon_loss missing from model name: {_mn}"
        # [D64] do NOT assert on a use_counts flag -- it is consumed inside the
        # dataset __init__ and is not stored under that name (verified: it reads
        # back as None while X is correctly integer counts). Check the DATA.
        _xa = trainer.train_ds.adata.X
        _mx = float(_xa[:200].toarray().max() if hasattr(_xa, "toarray")
                    else _xa[:200].max())
        assert _mx > 30, f"NB arm: X max {_mx:.1f} does not look like raw counts"
        print(f"   [D64] NB arm confirmed on raw counts (X[:200] max {_mx:.0f})")
    assert getattr(trainer, "proto_recon_stopgrad", None) is True
    assert getattr(trainer, "coh_proto_sg", 0) == 1
    print(f"   [D64 VERIFIED] model folder = {_mn}")
    print(f"[{arm}] trained in {train_s:.0f}s  n_dead={metrics.get('n_dead')}")

    # --- post-run sigma (anti-collapse guard) ---
    captured = []
    hook = trainer.model.scpoli_cvae.encoder.register_forward_hook(
        lambda _m, _i, out: captured.append(out[1].detach().cpu()))
    with torch.no_grad():
        trainer.model.eval()
        trainer.encode_adata(trainer.train_ds.adata, trainer.model, z_idx=1)
    hook.remove()
    sigma_median = float(np.median(np.exp(0.5 * torch.cat(captured, 0).numpy())))

    # --- THE collapse measure: prototype pairwise cosine ([D62] finding) ---
    P = trainer.model.get_prototypes().detach().cpu().numpy()
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-12)
    Cm = Pn @ Pn.T
    np.fill_diagonal(Cm, -9)
    nn_cos = Cm.max(1)
    iu = np.triu_indices_from(Cm, 1)
    collapse = {"median_nn_cos": float(np.median(nn_cos)),
                "frac_pairs_gt_0.99": float((Cm[iu] > 0.99).mean()),
                "max_nn_cos": float(nn_cos.max())}
    print(f"[{arm}] sigma={sigma_median:.4g}  COLLAPSE {collapse} "
          f"(healthy ~0.81 / 0.0%, collapsed ~0.9997 / 1-3%)")

    # --- frozen-pass placement + artifacts ---
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
    np.save(os.path.join(dump, "proto_vectors.npy"), P)

    summary = {"arm": arm, "config": checked, "train_seconds": round(train_s, 1),
               "sigma_median_postrun": sigma_median, "collapse": collapse,
               "model_name": _mn, "dump_path": dump, "metrics": metrics}
    with open(os.path.join(dump, "d64_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- RECIPE.md: the loss in plain English, next to the run ---
    shares_e = {"cohesion_gram": lc["lambda_cohesion"] * GNORM_ENC[lik]["cohesion_gram"],
                "cohesion_aff": lc["lambda_cohesion_aff"] * GNORM_ENC[lik]["cohesion_aff"],
                "recon": lc["lambda_recon"] * GNORM_ENC[lik]["recon"],
                "gmkl": lc["lambda_gmkl"] * GNORM_ENC[lik]["gmkl"]}
    te = sum(shares_e.values())
    shares_p = {"gmkl": lc["lambda_gmkl"] * GNORM_PROTO[lik]["gmkl"],
                "proto_recon": lc["lambda_proto_recon"] * GNORM_PROTO[lik]["proto_recon"]}
    tp = sum(shares_p.values())
    with open(os.path.join(dump, "RECIPE.md"), "w") as f:
        f.write(f"# {arm}\n\nreconstruction likelihood: **{lik}**  |  "
                f"CVAE recon: **{'ON' if recon_on else 'OFF'}**\n\n"
                f"## ENCODER-path budget (what teaches the encoder)\n\n")
        for k, v in shares_e.items():
            f.write(f"- {k}: lambda={lc.get({'cohesion_gram':'lambda_cohesion','cohesion_aff':'lambda_cohesion_aff','recon':'lambda_recon','gmkl':'lambda_gmkl'}[k])}, "
                    f"share={100*v/te:.1f}%\n")
        f.write("\n## PROTOTYPE-path budget (the only two terms touching prototypes)\n\n")
        for k, v in shares_p.items():
            f.write(f"- {k}: share={100*v/tp:.1f}%\n")
        f.write(f"\n## routing\n\n- cohesion: prototypes DETACHED (coh_proto_sg=1)\n"
                f"- proto_recon: form 3 (mix3), scores DETACHED, decoder LIVE\n"
                f"- community: OFF; nassoc/modularity/usage/swav/plain-KL: OFF\n"
                f"\n## outcome\n\n- dead prototypes: {metrics.get('n_dead')}\n"
                f"- sigma median: {sigma_median:.4g}\n- collapse: {collapse}\n")
    volume.commit()
    print(f"[{arm}] saved artifacts + RECIPE.md -> {dump}")
    return summary


@app.local_entrypoint()
def main(arms: str = "all"):
    sel = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    bad = [a for a in sel if a not in ARMS]
    assert not bad, f"unknown arms {bad}"
    print(f"[D64] {DS}/{TAG} epochs={TRAIN_EPOCHS} arms={sel} (all recon_loss=nb)")
    for a in sel:
        pi, ro = ARMS[a]
        lc = build_config(pi, ro, "nb")
        print(f"  {a:<16} init={pi:<8} recon={'ON ' if ro else 'OFF'} "
              f"gram={lc['lambda_cohesion']:g} aff={lc['lambda_cohesion_aff']:g} "
              f"recon={lc['lambda_recon']:g} gmkl={lc['lambda_gmkl']:g} "
              f"pr3={lc['lambda_proto_recon']:g}")
    calls = {a: run_arm.spawn(a) for a in sel}
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")
    outdir = os.path.join("files", "tables", "d64_nb")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "d64_arms.json"), "w") as f:
        json.dump({"ok": out, "failed": failed,
                   "gnorm_enc": GNORM_ENC, "gnorm_proto": GNORM_PROTO},
                  f, indent=2, default=str)
    print(f"\n[D64] wrote {outdir}/d63_arms.json")
    for a, s in out.items():
        print(f"  {a:<14} n_dead={s['metrics'].get('n_dead')} "
              f"sigma={s['sigma_median_postrun']:.3g} "
              f"median_nn_cos={s['collapse']['median_nn_cos']:.4f}")
    if failed:
        print(f"[D64] FAILED: {list(failed)}")

"""[D61] Two arms: D60 leid-noc recipe + proto_recon FORM 3 (the chimera charge),
differing ONLY in proto_recon_stopgrad (which gradient path fixes chimeras?).

PRE-REGISTRATION: my-docs/adr/D61-proto-recon-mix3-two-arms.md
(written and committed BEFORE this script ran; bars there bind this run).

  d61_pr3_sg   proto_recon_stopgrad=True   scores DETACHED: gradient reaches the
               PROTOTYPES only (through the decoder, whose params are detached)
  d61_pr3_lv   proto_recon_stopgrad=False  scores LIVE: gradient also flows
               through s_ik into the ENCODER (reassign/move misfit cells)

Both: leiden init, NO community, cohesion proto-detached (coh_proto_sg=1),
GM-KL, recon, proto_recon_form=mix3, proto_recon_dec_sg=1. Lambdas for the
four D60 terms are IDENTICAL to d60_leid_noc; lambda_proto_recon from the D61
probe at share 1/4 -- SAME lambda in both arms, so the flag is the only
difference.

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

app = modal.App("scproto-d62-train")

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
    # MEASURED 2026-10-18 by the D60 probe (coh_proto_sg=1, lambda_gmkl=1):
    # 402/402 rows, 0 errors, weights VERIFIED frozen (drift 1.14e-08), and the
    # ROUTING VERIFIED from the probe's own jsonl -- median gnorm_proto:
    # gram 0.0, aff 0.0, umap 0.0, recon 0.0, gmkl 0.0553 (the only teacher).
    "cohesion_gram": 0.04305,
    "cohesion_aff":  0.024688,
    "umap":          7.2616,
    "recon":         11.857,
    "gmkl":          11.221,   # CV 0.01 -- the most stable term measured
    # [D62] MEASURED 2026-10-23 by measure_grad_shares_d62.py with the REAL
    # mix3 code on the volume (verified by md5 AND by prfm-mix3/prsgFalse
    # appearing in the probe run-folder name). 402/402 rows, 0 errors, CV 0.08.
    # ROUTING VERIFIED: gnorm_proto cohesion/umap/recon = 0.0, gmkl 0.0551,
    # proto_recon 9.261.
    # NOTE the D61 probe measured the OLD agg form and got 51.489 -- only 1.8%
    # away, because at the calibrated epsilon the assignments are nearly
    # one-hot, so mix3 - agg (the responsibility-weighted spread of the
    # prototype decodings) is just 0.92 out of 383 = 0.24% of the value.
    "proto_recon":   50.575,
}

SHARE = {"kernel": 1 / 8, "recon": 1 / 4, "gmkl": 1 / 4, "community": 1 / 4,
         "proto_recon": 1 / 4}


def _lam(term, share):
    g = GNORM[term]
    assert g, f"GNORM[{term}] not filled -- paste the D60 probe medians first"
    return round(share / g, 6)


# [D62] arm -> mix3 SHARE. Everything else is D60_leid_noc EXACTLY (same four
# lambdas), scores LIVE, decoder LIVE, recon ON -- the configuration the D61
# evidence selected. The only variable is the proto_recon dose, chosen LOW so
# cohesion keeps nearly all of its D60 relative strength (the D61 confound):
#     share 1/16 -> cohesion 30.8% of total (D60: 33.3%, D61 was 25.0%)
#     share 1/8  -> cohesion 28.6%
ARMS_D62 = {"d62_pr16": 1/16, "d62_pr8": 1/8}

ARMS = {
    # arm -> (proto_recon_stopgrad, proto_recon_dec_sg, cvae_recon_on)
    # 2026-10-19 USER DECISIONS mid-run: (1) decoder TRAINABLE in the term
    # ("so it can reconstruct the rare states we are finding"), prototypes
    # trainable; (2) test whether the CVAE recon loss is needed AT ALL any
    # more -> the deployed matrix is the dec-live 2x2 below. The two dec-
    # DETACHED arms (sg/lv) ran first and are kept as the decoder-axis
    # reference only.
    "d61_pr3_sg": (True, 1, True),        # ran first: dec detached, recon on
    "d61_pr3_lv": (False, 1, True),       # ran first: dec detached, recon on
    "d61_pr3_sgdec": (True, 0, True),     # scores detached, dec LIVE, recon ON
    "d61_pr3_lvdec": (False, 0, True),    # scores live,     dec LIVE, recon ON
    "d61_pr3_sgdec_nr": (True, 0, False),   # scores detached, dec LIVE, recon OFF
    "d61_pr3_lvdec_nr": (False, 0, False),  # scores live,     dec LIVE, recon OFF
}

ZERO_ALWAYS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
               "lambda_swav", "lambda_kl")


def build_config_d62(mix3_share):
    """[D62] the selected recipe. Four D60 lambdas untouched + low-dose mix3."""
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "prot_init": "leid",
        "lambda_cohesion": _lam("cohesion_gram", SHARE["kernel"]),
        "lambda_cohesion_aff": _lam("cohesion_aff", SHARE["kernel"]),
        "lambda_recon": _lam("recon", SHARE["recon"]),
        "lambda_gmkl": _lam("gmkl", SHARE["gmkl"]),
        "lambda_proto_recon": _lam("proto_recon", mix3_share),
        "proto_recon_form": "mix3",
        "proto_recon_dec_sg": 0,          # decoder TRAINABLE (user decision)
        "proto_recon_stopgrad": False,    # scores LIVE  (D61 evidence)
        "lambda_umap": 0.0,
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "grad_probe": 25, "seed": SEED,
    }


def build_config(stopgrad, dec_sg, recon_on):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "prot_init": "leid",
        "lambda_proto_recon": _lam("proto_recon", SHARE["proto_recon"]),
        "proto_recon_form": "mix3",
        "proto_recon_dec_sg": dec_sg,
        "proto_recon_stopgrad": stopgrad,
        "lambda_cohesion": _lam("cohesion_gram", SHARE["kernel"]),
        "lambda_cohesion_aff": _lam("cohesion_aff", SHARE["kernel"]),
        "lambda_recon": _lam("recon", SHARE["recon"]) if recon_on else 0.0,
        "lambda_gmkl": _lam("gmkl", SHARE["gmkl"]),
        "lambda_umap": 0.0,
        "coh_proto_sg": 1,
        "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
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
    assert float(lc["lambda_recon"]) >= 0   # 0 is a deliberate [D61] ablation arm
    assert float(lc["lambda_proto_recon"]) > 0 and lc["proto_recon_form"] == "mix3"
    assert lc["proto_recon_dec_sg"] in (0, 1)
    assert lc["cohesion_self"] == 1.0 and lc["proto_reinit_every"] == 0
    return {"arm": arm, "prot_init": lc["prot_init"],
            "pr_dec_sg": lc["proto_recon_dec_sg"],
            "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "recon": lc["lambda_recon"], "gmkl": lc["lambda_gmkl"],
            "proto_recon": lc["lambda_proto_recon"],
            "pr_stopgrad": lc["proto_recon_stopgrad"],
            "coh_proto_sg": lc["coh_proto_sg"],
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

    lc = (build_config_d62(ARMS_D62[arm]) if arm in ARMS_D62
          else build_config(*ARMS[arm]))
    checked = validate(lc, arm)
    print(f"   [D61 CONFIG VERIFIED IN-RUN] {checked}")

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
    assert getattr(trainer, "proto_recon_form", "agg") == "mix3"
    if arm in ARMS_D62:
        # [D62] THE CHECK D61 LACKED: prove the new option reached the RUN, not
        # just the trainer attribute. D61's uploads never landed on the volume
        # (Git Bash mangled a leading-slash remote path), so it silently trained
        # proto_recon form 'agg'. An attribute assert cannot catch that --
        # init_attributes setattr's anything. These two can:
        import inspect
        from interpretable_ssl.trainers import scproto as _sp
        assert "mix3" in inspect.getsource(_sp), (
            "the scproto.py ON THIS CONTAINER has no mix3 branch -- stale volume copy")
        _mn = trainer.get_model_name()
        assert "prfm-mix3" in _mn, f"proto_recon_form missing from model name: {_mn}"
        print(f"   [D62 VERIFIED] mix3 code present; model folder = {_mn}")
        assert getattr(trainer, "proto_recon_dec_sg", None) == 0
        assert getattr(trainer, "proto_recon_stopgrad", None) is False
        assert float(getattr(trainer, "lambda_recon", -1)) > 0
        assert float(getattr(trainer, "lambda_cohesion", 0)) == 2.9036
    else:
        assert getattr(trainer, "proto_recon_dec_sg", None) == ARMS[arm][1]
        assert getattr(trainer, "proto_recon_stopgrad", None) == ARMS[arm][0]
        assert (float(getattr(trainer, "lambda_recon", -1)) > 0) == ARMS[arm][2]
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
    with open(os.path.join(dump, "d61_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    volume.commit()
    print(f"[{arm}] saved artifacts -> {dump}")
    return summary


@app.local_entrypoint()
def main(arms: str = "all"):
    ALL = dict(ARMS) | {k: None for k in ARMS_D62}
    sel = list(ARMS_D62) if arms == "all" else [a.strip() for a in arms.split(",")]
    bad = [a for a in sel if a not in ALL]
    assert not bad, f"unknown arms {bad}"
    print(f"[D61] {DS}/{TAG} epochs={TRAIN_EPOCHS} arms={sel}")
    for a in sel:
        lc = (build_config_d62(ARMS_D62[a]) if a in ARMS_D62
              else build_config(*ARMS[a]))
        g = {"cohesion_gram": lc["lambda_cohesion"],
             "cohesion_aff": lc["lambda_cohesion_aff"],
             "recon": lc["lambda_recon"], "gmkl": lc["lambda_gmkl"],
             "proto_recon": lc["lambda_proto_recon"]}
        shares = {k: v * GNORM[k] for k, v in g.items()}
        tot = sum(shares.values())
        print(f"  {a:<12} " + " ".join(f"{k.split('_')[-1]}={v:g}" for k, v in g.items()))
        print(f"  {'':<12} realized shares: " +
              " ".join(f"{k.split('_')[-1]}={100*v/tot:.1f}%" for k, v in shares.items())
              + f"  | cohesion combined="
              f"{100*(shares['cohesion_gram']+shares['cohesion_aff'])/tot:.1f}%")
    calls = {a: run_arm.spawn(a) for a in sel}
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")
    outdir = os.path.join("files", "tables", "d62_dose")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "d62_arms.json"), "w") as f:
        json.dump({"ok": out, "failed": failed, "gnorm": GNORM}, f,
                  indent=2, default=str)
    print(f"\n[D62] wrote {outdir}/d61_arms.json")
    for a, s in out.items():
        print(f"  {a:<14} train={s['train_seconds']}s "
              f"n_dead={s['metrics'].get('n_dead')} "
              f"sigma_median={s['sigma_median_postrun']:.3g}")
    if failed:
        print(f"[D61] FAILED: {list(failed)}")

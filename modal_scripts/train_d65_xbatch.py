"""[D65] Four arms: CROSS-BATCH SIGNATURE ALIGNMENT, diag vs diag+off, MSE vs NB.

PRE-REGISTRATION: my-docs/adr/D65-cross-batch-signature-alignment.md

WHY THIS TERM EXISTS (measured 2026-10-31/11-01, pbmc-immune / arbf_ctb20_seed0,
every arm against its OWN stage-1 init d60_epoch0_leid):

                        stage-1   protorecon  protorec   protorec   floor
                         only        OFF      +CVAE      no CVAE
  coh_aff_test_all      0.0784     0.0853     0.0773     0.0784    0.0012
  coh_aff_test_rare     0.1262     0.1376     0.1319     0.1217    0.0101
  AGREE% (>=10 cells)    83.1%      66.0%      61.8%      65.2%    ~3 pt
  r_cross (gene space)   0.702      0.603      0.581      0.578      --
  chimeras (r_cross<0.3)  0.0%       9.9%      13.4%       9.4%      --

Stage-2 training REMOVES cross-batch alignment its own init had, and the base
recipe does it with no proto_recon involved (0.702 -> 0.603, 0% -> 9.9%).
Root cause: NO term in the loss ever evaluated a cross-batch pair. Cohesion is
scored inside each bio-batch's own node set V[b] BY DESIGN, GM-KL is
batch-agnostic, CVAE-recon is per-cell. Nothing changed value when a prototype
became a chimera, so nothing defended it. This term is the first that does.

ARMS (all: d60_leid_noc recipe + the new term, leiden init, cohesion gram+aff
proto-detached, GM-KL, community OFF, proto_recon OFF, 10 epochs, K=300):
    d65_diag-mse    diagonal only,        recon_loss=mse
    d65_both-mse    diagonal + off-diag,  recon_loss=mse
    d65_diag-nb     diagonal only,        recon_loss=nb
    d65_both-nb     diagonal + off-diag,  recon_loss=nb

The cross-batch term is BIT-IDENTICAL across the mse/nb axis -- it reads
adata.layers['lognorm'] and never touches the decoder. mse/nb changes only the
pretrain and the CVAE-recon head, i.e. the latent space the assignments live in.

PRE-REGISTERED BARS (vs d60_leid_noc, the same recipe without this term)
  PRIMARY    r_cross >= 0.69 (ceiling ~0.70);  chimeras <= 3%;
             AGREE%(>=10) > 69.0 (floor ~3 pts)
  HARD BARS  coh_aff_test_all >= 0.0853;  pooled_test_all >= 0.8160;
             n_dead = 0;  median_nn_cos >= 0.75;  frac_pairs>0.99 = 0%
  ARM BOTH   if median_nn_cos or n_dead degrade vs arm DIAG, the off-diagonal
             is doing harm -> drop it, keep DIAG.
Expectation is RECOVERY, not a new win: the size-matched cell-type ceiling at
prototype group sizes is r=0.704 and the untrained init already sits at 0.694.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach \
       modal_scripts/train_d65_xbatch.py --arms all
"""
import json
import os
import time

import modal

app = modal.App("scproto-d65-train")

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
# ENCODER-path gradient norms at temporary lambda=1 (frozen full pass).
# The four existing terms are the D63/D64 measurements; xbatch / xbatch_off are
# FILLED BY measure_grad_shares_d65.py and must not be guessed.
# ---------------------------------------------------------------------------
# ALL SIX come from ONE probe run per likelihood. Do NOT mix in the D63/D64
# numbers for the four older terms: those were measured under a different probe
# config (proto_recon live, different init), and 9d-1's whole point is that a
# gnorm is only comparable within the snapshot it was taken in. Measured here
# vs D63's for mse: cohesion_gram 0.0532 vs 0.0431, recon 13.36 vs 11.83,
# gmkl 10.65 vs 11.22 -- same order, but mixing them would silently shift every
# realized share.
GNORM_ENC = {
    # MEASURED 2026-11-01, measure_grad_shares_d65.py --lik mse.
    # 402/402 rows, 0 probe errors, CV 0.01-0.12, weights verified frozen
    # (prototype drift 2.4e-8). ROUTING VERIFIED: cohesion_gram, cohesion_aff,
    # recon, xbatch and xbatch_off ALL put exactly 0.0 on the prototypes; only
    # gmkl does (0.03305), so there is a single budget to calibrate.
    "mse": {"cohesion_gram": 0.0531847, "cohesion_aff": 0.0300136,
            "recon": 13.3565, "gmkl": 10.6492,
            "xbatch": 0.0216031, "xbatch_off": 0.0417084},
    # MEASURED 2026-11-01, measure_grad_shares_d65.py --lik nb (fresh NB
    # pretrain -- recon_loss keys the checkpoint). 402/402 rows, 0 errors,
    # CV 0.00-0.11. Same routing verification: only gmkl (0.02753) touches
    # the prototypes. Cross-check that the term is likelihood-independent as
    # designed: xbatch's VALUE is -0.1192 here vs -0.1155 under mse (it reads
    # layers['lognorm'] in both; the small gap is the different pretrain's
    # latent, not the term).
    "nb":  {"cohesion_gram": 0.0513594, "cohesion_aff": 0.0347355,
            "recon": 16.6334, "gmkl": 13.6526,
            "xbatch": 0.025611, "xbatch_off": 0.0443404},
}

# [D65] user decision 2026-11-01: cross-batch takes 1/8; the other four keep
# their 1:1:2:2 ratio, renormalized to sum to 1. Arm BOTH holds the diagonal
# constant and adds the off-diagonal at its own share, so DIAG vs BOTH isolates
# the off-diagonal -- at the cost of BOTH carrying more total new force, which
# is the [D62] dilution confound and is recorded rather than hidden.
SHARES = {
    "diag": {"cohesion_gram": 0.1458, "cohesion_aff": 0.1458,
             "recon": 0.2917, "gmkl": 0.2917, "xbatch": 0.1250},
    "both": {"cohesion_gram": 0.1296, "cohesion_aff": 0.1296,
             "recon": 0.2593, "gmkl": 0.2593,
             "xbatch": 0.1111, "xbatch_off": 0.1111},
}

# arm -> (which share vector, recon likelihood)
ARMS = {
    "d65_diag-mse": ("diag", "mse"),
    "d65_both-mse": ("both", "mse"),
    "d65_diag-nb":  ("diag", "nb"),
    "d65_both-nb":  ("both", "nb"),
}


def build_config(variant, lik):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    ge, sh = GNORM_ENC[lik], SHARES[variant]
    for k in sh:
        assert ge.get(k), (
            f"GNORM_ENC[{lik}][{k}] is not filled -- run "
            f"`modal run modal_scripts/measure_grad_shares_d65.py --lik {lik}` "
            f"first and paste the median gnorm_enc. CLAUDE.md 9d-1: a lambda is "
            f"NEVER guessed and NEVER portable across datasets.")
    lam = {k: round(sh[k] / ge[k], 8) for k in sh}
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "prot_init": "leid",
        "recon_loss": lik,
        "lambda_cohesion": lam["cohesion_gram"],
        "lambda_cohesion_aff": lam["cohesion_aff"],
        "lambda_recon": lam["recon"],
        "lambda_gmkl": lam["gmkl"],
        # ---- [D65] the new term -------------------------------------------
        "lambda_xbatch": lam["xbatch"],
        "lambda_xbatch_off": lam.get("xbatch_off", 0.0),
        "xbatch_margin": 0.5,
        "xbatch_mem_hlife": 5.0,     # DERIVED: 33506 cells / ~6700 per step
        "xbatch_mem_mix": 0.5,
        "xbatch_proto_sg": 1,        # encoder-only; GM-KL owns the prototypes
        "xbatch_min_size": 1.0,
        "xbatch_weight_sg": 1,       # min(size,size) is a reliability weight only
        # ---- everything else genuinely OFF --------------------------------
        "lambda_proto_recon": 0.0,   # [D63] CLOSED at 0
        "lambda_umap": 0.0,          # community off ([D60] d60_leid_noc)
        "coh_proto_sg": 1, "umap_proto_sg": 1,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "grad_probe": 25, "seed": SEED,
    }


def validate(lc, arm, variant, lik):
    """CLAUDE.md 9d: ASSERT the zeros. LAMBDA_PROTO_UMAP_PRECON turns several
    terms ON, so starting from it and setting only the new keys leaves nassoc
    and proto_usage running."""
    for k in ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_umap", "lambda_proto_recon"):
        assert float(lc[k]) == 0.0, f"{arm}: {k} must be 0, got {lc[k]}"
    assert lc["coh_proto_sg"] == 1 and lc["umap_proto_sg"] == 1
    assert lc["xbatch_proto_sg"] == 1, "the new term must not touch prototypes"
    assert lc["cohesion_self"] == 1.0 and lc["proto_reinit_every"] == 0
    assert lc["recon_loss"] == lik
    assert float(lc["lambda_xbatch"]) > 0, "the diagonal term is the point"
    assert (float(lc["lambda_xbatch_off"]) > 0) == (variant == "both")
    return {"arm": arm, "variant": variant, "recon_loss": lik,
            "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "recon": lc["lambda_recon"], "gmkl": lc["lambda_gmkl"],
            "xbatch": lc["lambda_xbatch"], "xbatch_off": lc["lambda_xbatch_off"],
            "xb_margin": lc["xbatch_margin"], "xb_hlife": lc["xbatch_mem_hlife"],
            "xb_mix": lc["xbatch_mem_mix"], "xb_weight_sg": lc["xbatch_weight_sg"],
            "seed": lc["seed"], "train_epochs": TRAIN_EPOCHS}


def _stage_data_locally():
    """[D64] h5py's non-sequential seeks are unreliable against a network-backed
    Modal volume for a large file ('wrong B-tree signature' on key X, md5
    identical). Copy to container-local disk first."""
    import shutil
    src, dst = "/vol/data", "/tmp/data"
    need = [
        f"node_heldout/{DS}_{TAG}_train.h5ad",
        f"node_heldout/{DS}_{TAG}_train_label_encoder.pkl",
        f"node_heldout/{DS}_{TAG}_split.npz",
        "scpoli/Immune_ALL_human_hvg.h5ad",
    ]
    for rel in need:
        s_, d_ = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d_), exist_ok=True)
        if not os.path.exists(d_) or os.path.getsize(d_) != os.path.getsize(s_):
            shutil.copyfile(s_, d_)
        assert os.path.getsize(d_) == os.path.getsize(s_), f"short copy: {rel}"
    import h5py
    with h5py.File(os.path.join(dst, need[0]), "r") as f:
        _ = f["X"]["data"][:1000]
    print(f"   [D65] staged {len(need)} files to {dst}; h5py data read OK")
    return dst + "/"


def _env():
    import sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = _stage_data_locally()
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


# [D64] T4, not L4: the fatemehs-hashemig workspace allocates T4 instantly but
# never allocates L4 (measured 2026-10-31, modal_scripts/gpu_probe.py).
@app.function(image=image, gpu="T4", timeout=36000, volumes={"/vol": volume})
def run_arm(arm: str):
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    # [D61] STALE-COPY GUARD, before anything expensive. D61's six arms (~$6 of
    # GPU) silently trained the PREVIOUS scproto.py because `modal volume put`
    # with a leading-/ remote path uploaded nowhere; the ADR was invalidated.
    import inspect
    from interpretable_ssl.trainers import scproto as _sp
    assert "[D65] CROSS-BATCH SIGNATURE ALIGNMENT" in inspect.getsource(_sp), (
        "STALE scproto.py on this container -- the D65 term is not in the code "
        "about to run. Re-upload with a RELATIVE remote path, verify by md5.")

    variant, lik = ARMS[arm]
    lc = build_config(variant, lik)
    checked = validate(lc, arm, variant, lik)
    print(f"   [D65 CONFIG VERIFIED IN-RUN] {checked}")

    train_h5 = os.path.join(os.environ["DATA_DIR"],
                            f"node_heldout/{DS}_{TAG}_train.h5ad")
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

    # [CLAUDE.md 9 step 3, as a GUARD not a printout] the option must appear in
    # the model folder name, or two arms differing only in it share a folder.
    _mn = trainer.get_model_name()
    assert "lxb" in _mn, f"lambda_xbatch missing from model name: {_mn}"
    if variant == "both":
        assert "lxbof" in _mn, f"lambda_xbatch_off missing from name: {_mn}"
    if lik == "nb":
        assert "rlss-nb" in _mn, f"recon_loss missing from model name: {_mn}"
        _xa = trainer.train_ds.adata.X
        _mx = float(_xa[:200].toarray().max() if hasattr(_xa, "toarray")
                    else _xa[:200].max())
        assert _mx > 30, f"NB arm: X max {_mx:.1f} does not look like raw counts"
        print(f"   [D65] NB arm confirmed on raw counts (X[:200] max {_mx:.0f})")
    assert getattr(trainer, "xbatch_proto_sg", 0) == 1
    assert getattr(trainer, "coh_proto_sg", 0) == 1
    print(f"   [D65 VERIFIED] model folder = {_mn}")
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

    # --- prototype collapse ([D62] measure) ---
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
    d = np.load(os.path.join(os.environ["DATA_DIR"],
                             f"node_heldout/{DS}_{TAG}_split.npz"),
                allow_pickle=True)
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

    # --- [D65] the term's OWN measure, in-run, on the full dataset -----------
    # r_cross on the final assignments, same definition as
    # modal_scripts/view_proto_crossbatch_r.py so it is directly comparable to
    # the baselines (stage-1 0.702 / 0.0% chimeras, d60_leid_noc 0.603 / 9.9%).
    # NOTE this is the TRAINING SIGNAL for these arms, so it is a diagnostic
    # here, NOT the headline evidence -- AGREE% and cross_test_all are scored
    # offline and are not what the loss optimizes.
    try:
        Xl = ad_full.layers["lognorm"]
        Xl = Xl.toarray() if hasattr(Xl, "toarray") else np.asarray(Xl)
        Xl = np.asarray(Xl, dtype=np.float32)
        bts = np.asarray(ad_full.obs[BATCH_KEY]).astype(str)
        bgs = {b: Xl[bts == b].mean(0) for b in np.unique(bts)}

        def _r(a, b):
            a = a - a.mean(); b = b - b.mean()
            na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb_)) if na > 0 and nb_ > 0 else np.nan

        per = []
        for k in np.unique(lab):
            sg = {}
            for b in np.unique(bts):
                m = (lab == k) & (bts == b)
                if m.sum() >= 10:
                    sg[b] = Xl[m].mean(0) - bgs[b]
            bs_ = list(sg)
            if len(bs_) < 2:
                continue
            vs = [_r(sg[bs_[i]], sg[bs_[j]])
                  for i in range(len(bs_)) for j in range(i + 1, len(bs_))]
            vs = [v for v in vs if not np.isnan(v)]
            if vs:
                per.append(float(np.mean(vs)))
        per = np.array(per)
        xb_measure = {"n_protos_spanning": int(per.size),
                      "r_cross_median": float(np.median(per)),
                      "r_cross_p10": float(np.percentile(per, 10)),
                      "frac_below_0.3": float((per < 0.3).mean())}
    except Exception as e:
        xb_measure = {"error": f"{type(e).__name__}: {e}"}
    print(f"[{arm}] [D65] r_cross {xb_measure}  "
          f"(stage-1 init 0.702 / 0.0%; d60_leid_noc 0.603 / 9.9%)")

    summary = {"arm": arm, "config": checked, "train_seconds": round(train_s, 1),
               "sigma_median_postrun": sigma_median, "collapse": collapse,
               "xbatch_measure": xb_measure,
               "model_name": _mn, "dump_path": dump, "metrics": metrics}
    with open(os.path.join(dump, "d65_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # --- RECIPE.md: the loss in plain English, next to the run ---
    ge = GNORM_ENC[lik]
    key_of = {"cohesion_gram": "lambda_cohesion", "cohesion_aff": "lambda_cohesion_aff",
              "recon": "lambda_recon", "gmkl": "lambda_gmkl",
              "xbatch": "lambda_xbatch", "xbatch_off": "lambda_xbatch_off"}
    shares_e = {k: lc[key_of[k]] * ge[k] for k in SHARES[variant]}
    te = sum(shares_e.values())
    with open(os.path.join(dump, "RECIPE.md"), "w") as f:
        f.write(f"# {arm}\n\nreconstruction likelihood: **{lik}**  |  "
                f"cross-batch: **{'diagonal + off-diagonal' if variant == 'both' else 'diagonal only'}**\n\n"
                f"## ENCODER-path budget (every term's gradient reaches the encoder only)\n\n")
        for k, v in shares_e.items():
            f.write(f"- {k}: lambda={lc[key_of[k]]:g}, share={100 * v / te:.1f}%\n")
        f.write(
            f"\n## the cross-batch term\n\n"
            f"- sig[k][b] = soft-assignment-weighted mean lognorm of k's batch-b\n"
            f"  members, MINUS the mean over all batch-b cells (cancels the batch effect)\n"
            f"- diagonal (pull): maximize sum over k, batch pairs of\n"
            f"  min(size[k][b1], size[k][b2]) * pearson(sig[k][b1], sig[k][b2]) / n_cells\n"
            f"- off-diagonal (push): {'mean of max(0, r - ' + str(lc['xbatch_margin']) + ') over k != j, b1 != b2' if variant == 'both' else 'OFF'}\n"
            f"- EMA memory half-life {lc['xbatch_mem_hlife']} steps "
            f"(= the group revisit interval), mix {lc['xbatch_mem_mix']}\n"
            f"- min(size,size) weight DETACHED (reliability weight, not a capacity charge)\n"
            f"- prototypes DETACHED (xbatch_proto_sg=1): GM-KL owns the prototypes\n"
            f"- measure is CIDER/IDER (Hu, Ahmed & Yau, Genome Biology 2021);\n"
            f"  memory is MSTN (Xie et al., ICML 2018). Neither used it as a loss.\n"
            f"\n## routing\n\n- cohesion: prototypes DETACHED (coh_proto_sg=1)\n"
            f"- proto_recon: OFF ([D63] closed at 0)\n"
            f"- community: OFF; nassoc/modularity/usage/swav/plain-KL: OFF\n"
            f"\n## outcome\n\n- dead prototypes: {metrics.get('n_dead')}\n"
            f"- sigma median: {sigma_median:.4g}\n- collapse: {collapse}\n"
            f"- r_cross (TRAINING SIGNAL here, diagnostic only): {xb_measure}\n")
    volume.commit()
    print(f"[{arm}] saved artifacts + RECIPE.md -> {dump}")
    return summary


@app.local_entrypoint()
def main(arms: str = "all"):
    sel = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    bad = [a for a in sel if a not in ARMS]
    assert not bad, f"unknown arms {bad}"
    print(f"[D65] {DS}/{TAG} epochs={TRAIN_EPOCHS} arms={sel}")
    for a in sel:
        v, lik = ARMS[a]
        lc = build_config(v, lik)          # fails loudly if a gnorm is missing
        print(f"  {a:<15} {v:<5} {lik:<4} "
              f"gram={lc['lambda_cohesion']:g} aff={lc['lambda_cohesion_aff']:g} "
              f"recon={lc['lambda_recon']:g} gmkl={lc['lambda_gmkl']:g} "
              f"xb={lc['lambda_xbatch']:g} xboff={lc['lambda_xbatch_off']:g}")
    calls = {a: run_arm.spawn(a) for a in sel}
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")
    outdir = os.path.join("files", "tables", "d65_xbatch")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "d65_arms.json"), "w") as f:
        json.dump({"ok": out, "failed": failed, "gnorm_enc": GNORM_ENC,
                   "shares": SHARES}, f, indent=2, default=str)
    print(f"\n[D65] wrote {outdir}/d65_arms.json")
    print(f"\n{'arm':<15}{'n_dead':>8}{'nn_cos':>9}{'r_cross':>9}{'chimera':>9}")
    for a, s in out.items():
        xm = s.get("xbatch_measure", {})
        print(f"{a:<15}{s['metrics'].get('n_dead'):>8}"
              f"{s['collapse']['median_nn_cos']:>9.4f}"
              f"{xm.get('r_cross_median', float('nan')):>9.3f}"
              f"{xm.get('frac_below_0.3', float('nan')):>9.1%}")
    print("baselines:    stage-1 init r_cross 0.702 / 0.0% chimeras;  "
          "d60_leid_noc 0.603 / 9.9%")
    if failed:
        print(f"[D65] FAILED: {list(failed)}")

"""[D59] Sigma-collapse check: did stage 2 switch off the VAE's sampling noise?

PREMISE BEING TESTED
--------------------
scPoli's forward SAMPLES z = mu + sigma*eps, and stage 2's cohesion consumes that
sampled z (scproto.py:1472 -> encoder_out -> scpoli forward). But stage 2 runs
with lambda_kl = 0, so nothing stops the encoder from predicting sigma -> 0,
which silently turns the built-in augmentation off and makes the latent freely
carveable. If stage-2 sigmas collapsed relative to stage 1, the mechanism is
confirmed and re-arming the KL (the GM-KL recipe) has a measured justification.

METHOD: forward hook on scpoli_cvae.encoder captures (z1_mean, z1_log_var) for
every cell under (a) the stage-1 pretrain checkpoint and (b) the trained
d59_scproto_base umap checkpoint. Same cells, same code path as training.
T4 (the trainer hard-requires CUDA: scproto.py:779 calls .cuda() unconditionally
in build_model, so a pure-CPU container cannot build the model). Still just
forward passes -- minutes, pennies.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/check_sigma_collapse.py
"""
import modal

app = modal.App("scproto-sigma-check")

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
STAGE2_DIR = ("/vol/models/pbmc-immune_arbf_ctb20_seed0_train/"
              "d59_scproto_base_ds-pbmc_prtInit-wayp_frzbe1_aff-arbf_"
              "recon0.027667_lumap0.046305_usim-prot_cvae_e50_ecal1_pum-ema_"
              "nagg-max_lcoh3.870568_upsg1_gprb25_cglb2_cmini-full_cdia1.0_"
              "lcha6.745727_upm-dotp_v31")
STAGE2_PR_DIR = ("/vol/models/pbmc-immune_arbf_ctb20_seed0_train/"
                 "d59_scproto_protorecon_ds-pbmc_prtInit-wayp_frzbe1_aff-arbf_"
                 "lprec0.006425_lumap0.046305_usim-prot_cvae_e50_ecal1_pum-ema_"
                 "nagg-max_lcoh3.870568_upsg1_gprb25_cglb2_cmini-full_cdia1.0_"
                 "lcha6.745727_upm-dotp_v31")


@app.function(image=image, gpu="T4", timeout=7200,
              volumes={"/vol": volume})
def check():
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
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    import anndata
    anndata.settings.allow_write_nullable_strings = True

    import numpy as np
    import torch
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)

    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    t = get_trainer(experiment_name="sigchk", cvae_epochs=50,
                    dataset_id=resolved, l2norm=1, assignment_metric="dotp",
                    batch_size=1024, affinity_type="arbf", seed=31,
                    num_prototypes=_resolve_num_prototypes(K, resolved))

    captured = []
    enc = t.model.scpoli_cvae.encoder

    def hook(_m, _inp, out):
        # scpoli encoder returns (z1_mean, z1_log_var)
        captured.append(out[1].detach().cpu())

    h = enc.register_forward_hook(hook)
    ad = t.train_ds.adata          # the 26,806 train cells -- what training saw

    def sigma_stats(tag):
        captured.clear()
        with torch.no_grad():
            t.model.eval()
            t.encode_adata(ad, t.model, z_idx=1)
        lv = torch.cat(captured, 0).numpy()           # (n_cells, latent_dims)
        sig = np.exp(0.5 * lv)
        flat = sig.ravel()
        row = {
            "tag": tag, "n_cells": int(sig.shape[0]), "dims": int(sig.shape[1]),
            "sigma_median": float(np.median(flat)),
            "sigma_p10": float(np.percentile(flat, 10)),
            "sigma_p90": float(np.percentile(flat, 90)),
            "frac_sigma_lt_0.1": float((flat < 0.1).mean()),
            "frac_sigma_lt_0.01": float((flat < 0.01).mean()),
            "per_dim_median": [float(x) for x in np.median(sig, axis=0)],
        }
        print(f"[{tag}] sigma median={row['sigma_median']:.4g} "
              f"p10={row['sigma_p10']:.4g} p90={row['sigma_p90']:.4g} "
              f"frac<0.1={row['frac_sigma_lt_0.1']:.3f} "
              f"frac<0.01={row['frac_sigma_lt_0.01']:.3f}")
        print(f"[{tag}] per-dim median sigma: "
              f"{['%.3g' % x for x in row['per_dim_median']]}")
        return row

    # --- stage 1 ---
    t.load_pretrain_checkpoint()
    s1 = sigma_stats("stage1_pretrain")

    # --- stage 2: scproto_base (10 epochs of the 4-term loss, lambda_kl=0) ---
    results = [s1]
    for tag, d in [("stage2_base", STAGE2_DIR),
                   ("stage2_protorecon", STAGE2_PR_DIR)]:
        ckpt = os.path.join(d, "umap_checkpoint.pth")
        if not os.path.exists(ckpt):
            print(f"[{tag}] missing {ckpt}")
            continue
        ck = torch.load(ckpt, map_location="cpu")
        for attr in ("sim_decoder_trunk", "sim_decoder_out"):
            if hasattr(t.model, attr):
                delattr(t.model, attr)
        t.model.load_state_dict(ck["model_state_dict"])
        results.append(sigma_stats(tag))

    h.remove()
    return results


@app.local_entrypoint()
def main():
    print("[D59] sigma-collapse check: stage-1 vs stage-2 encoder variances")
    for r in check.remote():
        print(r["tag"], {k: v for k, v in r.items()
                         if k not in ("tag", "per_dim_median")})

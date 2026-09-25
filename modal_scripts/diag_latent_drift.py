"""[DIAG] Encoder drift vs assignment failure -- pancreas, d42cohonly.

THE DECISION, WRITTEN BEFORE RUNNING
====================================
Leiden-K220 on OUR OWN stage-1 latent block-agrees at 0.9825; the trained
d42cohonly prototype partition block-agrees at 0.9240. Two candidate causes:

  (a) ENCODER DRIFT: stage-2 training (cohesion only -- sims are fixed graph
      weights, gradient reaches the encoder only through the assignment
      softmax) degrades the latent's cross-batch alignment.
  (b) ASSIGNMENT: the latent stays aligned; the prototype dot-product
      assignment loses the small no-cross-edge blocks anyway.

TEST: encode all cells with the FINAL stage-2 checkpoint, build the identical
adaptive-RBF latent graph, run the identical Leiden-at-K, compute blockAgree.

  blockAgree(leiden on stage-2 latent) ~ 0.98  ->  (b) assignment is the problem
  blockAgree(leiden on stage-2 latent) ~ 0.92  ->  (a) encoder drifted

Also reports cross-batch 15-NN type-match for both latents, overall and for
the 1,245 cells the trained run mispaired -- a graph-free corroboration.

VERDICT (fill in after reading): [ ] encoder drift  [ ] assignment  [ ] mixed

Cost: one T4, ~15-25 min (two Leiden-at-K searches dominate). No training.
Run:  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/diag_latent_drift.py
"""
import modal

app = modal.App("scproto-diag-drift")

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

RUN_DIR = ("/vol/models/pancreas/"
           "d42cohonly_pancreas_ds-panc_NP220_prtInit-wayp_aff-arbf_lumap0.0_"
           "usim-prot_cvae_e50_ecal1_pum-ema_nagg-max_lcoh4.04_cglb2_"
           "cmini-full_cmhl500.0_cdia1.0_lcha11.99_upm-dotp_v31")
BKEY, LKEY, K = "tech", "celltype", 220


def block_agree(lab, ct, bat):
    import numpy as np
    tot = agree = 0
    for mc in np.unique(lab):
        m = lab == mc
        cts, cnts = np.unique(ct[m], return_counts=True)
        maj = cts[np.argmax(cnts)]
        for b in np.unique(bat[m]):
            bm = m & (bat == b)
            bc_, bn_ = np.unique(ct[bm], return_counts=True)
            tot += bm.sum()
            agree += bm.sum() * (bc_[np.argmax(bn_)] == maj)
    return float(agree) / float(tot)


def cross_batch_knn_match(z, ct, bat, k=15):
    """For each cell: k nearest OTHER-batch cells by cosine; fraction same type."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    # query more neighbours than k, then drop same-batch ones per cell
    nn = NearestNeighbors(n_neighbors=min(len(zn), 200), metric="cosine").fit(zn)
    _, idx = nn.kneighbors(zn)
    out = np.full(len(zn), np.nan)
    for i in range(len(zn)):
        nb = idx[i][bat[idx[i]] != bat[i]][:k]
        if len(nb):
            out[i] = (ct[nb] == ct[i]).mean()
    return out


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(run_dir: str = RUN_DIR, out_tag: str = "d42cohonly"):
    import json
    import os
    import sys
    import time

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

    import numpy as np
    import torch
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        get_stage1_latent, build_latent_affinity, leiden_resolution_search)

    t0 = time.time()
    t, ad, z1 = get_stage1_latent("pancreas")
    ct = np.asarray(ad.obs[LKEY])
    bat = np.asarray(ad.obs[BKEY])
    res = {}

    print(f"[diag] target run: {run_dir}")
    # ---- the trained run's own partition: mispaired mask for the subset stat ----
    cl = np.load(os.path.join(run_dir, "clusters.npz"))
    key = [k for k in ("assignments", "labels", "clusters", "arr_0")
           if k in cl.files][0]
    lab_run = cl[key]
    if lab_run.ndim > 1:
        lab_run = lab_run.argmax(1)
    mis = np.zeros(len(ct), bool)
    for mc in np.unique(lab_run):
        m = lab_run == mc
        cts, cnts = np.unique(ct[m], return_counts=True)
        maj = cts[np.argmax(cnts)]
        for b in np.unique(bat[m]):
            bm = m & (bat == b)
            bc_, bn_ = np.unique(ct[bm], return_counts=True)
            if bc_[np.argmax(bn_)] != maj:
                mis |= bm
    res["blockAgree_run_partition"] = block_agree(lab_run, ct, bat)
    res["n_mispaired_cells"] = int(mis.sum())
    print(f"[diag] run partition blockAgree={res['blockAgree_run_partition']:.4f} "
          f"mispaired={mis.sum()} ({time.time()-t0:.0f}s)")

    # ---- stage-1 latent: leiden + blockAgree (sanity: expect ~0.9825) ----
    ad.obsm["X_stage1z"] = z1
    aff1 = build_latent_affinity(ad, "X_stage1z")
    lab1, _, nk1 = leiden_resolution_search(aff1, K)
    res["blockAgree_leiden_stage1"] = block_agree(lab1, ct, bat)
    res["leiden_stage1_K"] = int(nk1)
    print(f"[diag] stage-1 leiden K={nk1} blockAgree="
          f"{res['blockAgree_leiden_stage1']:.4f} ({time.time()-t0:.0f}s)")

    # ---- load FINAL stage-2 weights; verify they actually load ----
    ck = torch.load(os.path.join(run_dir, "umap_checkpoint.pth"),
                    map_location=t.device)
    sd = ck["model_state_dict"]
    before = {k: v.detach().float().norm().item()
              for k, v in t.model.state_dict().items()}
    missing, unexpected = t.model.load_state_dict(sd, strict=False)
    after = {k: v.detach().float().norm().item()
             for k, v in t.model.state_dict().items()}
    changed = [k for k in before if abs(before[k] - after[k]) > 1e-8]
    print(f"[diag] loaded stage-2 ckpt (epoch {ck.get('epoch')}): "
          f"{len(changed)} tensors changed, missing={missing}, "
          f"unexpected={unexpected}")
    assert any("encoder" in k or "scpoli" in k for k in changed), (
        "encoder weights did NOT change on load -- would test the wrong model")
    res["ckpt_epoch"] = int(ck.get("epoch", -1))
    res["n_tensors_changed"] = len(changed)

    with torch.no_grad():
        z2 = t.encode_adata(ad, t.model, z_idx=1).cpu().numpy()
    res["z_drift_l2_mean"] = float(np.linalg.norm(z2 - z1, axis=1).mean())
    res["z1_norm_mean"] = float(np.linalg.norm(z1, axis=1).mean())

    # ---- stage-2 latent: identical graph + leiden + blockAgree ----
    ad.obsm["X_stage2z"] = z2
    aff2 = build_latent_affinity(ad, "X_stage2z")
    lab2, _, nk2 = leiden_resolution_search(aff2, K)
    res["blockAgree_leiden_stage2"] = block_agree(lab2, ct, bat)
    res["leiden_stage2_K"] = int(nk2)
    print(f"[diag] stage-2 leiden K={nk2} blockAgree="
          f"{res['blockAgree_leiden_stage2']:.4f} ({time.time()-t0:.0f}s)")

    # ---- graph-free corroboration: cross-batch 15-NN type match ----
    m1 = cross_batch_knn_match(z1, ct, bat)
    m2 = cross_batch_knn_match(z2, ct, bat)
    for tag, m in (("stage1", m1), ("stage2", m2)):
        res[f"xbatch_knn_match_{tag}_all"] = float(np.nanmean(m))
        res[f"xbatch_knn_match_{tag}_mispaired"] = float(np.nanmean(m[mis]))
    print(f"[diag] xbatch 15-NN type-match  all: "
          f"z1={res['xbatch_knn_match_stage1_all']:.4f} "
          f"z2={res['xbatch_knn_match_stage2_all']:.4f}   mispaired cells: "
          f"z1={res['xbatch_knn_match_stage1_mispaired']:.4f} "
          f"z2={res['xbatch_knn_match_stage2_mispaired']:.4f}")

    out = f"/vol/models/pancreas/diag_latent_drift_{out_tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    volume.commit()
    print(f"[diag] saved {out} ({time.time()-t0:.0f}s total)")
    return res


@app.local_entrypoint()
def main(run_dir: str = RUN_DIR, out_tag: str = "d42cohonly"):
    print(run.remote(run_dir, out_tag))

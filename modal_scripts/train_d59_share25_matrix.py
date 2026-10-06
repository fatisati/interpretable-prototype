"""[D59] Real training: unified 4-term loss, share=0.25 each, all 5 datasets.

WHY THIS RUN
------------
Resolves [D55]'s PENDING target-share vector (user decision: 0.25 for every
term, no exception for s28nsc) using the measurement just taken in
measure_grad_shares_d58.py. Per CLAUDE.md 9d/9d-1 (rewritten 2026-09-26):

    lambda_i = share_i / gnorm_i     share_i = 0.25 for all 4 terms
                                      gnorm_i measured 2026-09-26, d58 probe

MEASURED gnorm_enc -> DERIVED lambda (this run's actual config, rounded per
CLAUDE.md's filename-length rule):

    dataset        cohesion_gram  cohesion_aff  umap(community)  recon
    pancreas          0.64           1.10           0.0082        0.00054
    lung              0.23           0.33           0.0075        0.0034
    pbmc-immune       0.87           1.17           0.046         0.0027
    crcx              0.28           0.25           0.0045        0.0062
    s28nsc            2.36           3.02           0.0136        0.0017

s28nsc: explicit user decision to NOT exempt it this time (its two prior
"community fails" verdicts were measured under the OLD authority/span method,
where cohesion held a larger, privileged budget -- this is a genuinely
different relative dose under the new symmetric 4-way split, not a repeat).

ALL FOUR terms active on ALL FIVE datasets, nothing else:
    lambda_cohesion, lambda_cohesion_aff, lambda_umap (umap_proto_sg=1),
    lambda_recon (freeze_batch_embedding=True).
Zeroed: lambda_proto_recon, lambda_nassoc, lambda_modularity,
lambda_proto_usage, lambda_swav, lambda_kl. proto_reinit_every=0.
cohesion_self=cohesion_diag=1.0. coh_mem_hlife=0.0 (honest no-decay default --
no per-dataset coh_mem_drift measurement exists yet to derive c*N/uniq from).
num_prototypes = dataset default (K from CLAUDE.md 9d, not the legacy
K=200/K=16 numbers old scoreboards used). train_epochs=20 (CLAUDE.md 5's
minimal standard). grad_probe=25 kept ON during the real run so drift is
visible (D46 Result 3's own lesson: cohesion's gradient does NOT reliably
decay, but check anyway rather than assume).

DECISION, BEFORE RUNNING: expect all 5 to keep K_live close to their full K
(0 or near-0 dead) given the 25% share is a genuinely subordinate dose for
community/recon relative to the two cohesion terms combined (0.5 share here
vs 1.0 for the two cohesion terms) -- watch unused_proto from epoch 1 as the
[D39]-style failure signature. VERDICT per dataset: [ ] works [ ] fails
[ ] inconclusive-because___ -- to be filled after eval, not left open.

Node-heldout (pancreas/lung/pbmc-immune): frozen-pass placement of the 20%
held-out cells after training, same contract as train_d46_nodeheldout.py.
Full-graph (s28nsc/crcx): no holdout, straight full-graph run.

ALL FIVE launched as ONE parallel dispatch (spawn + get), not staged batches.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_d59_share25_matrix.py
"""
import time

import modal

app = modal.App("scproto-d59")

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

NH_TAG = "arbf_nodeheldout20_seed0"
CONFIG_TAG = "d59"

# ds -> (batch_key, label_key, K, lam_gram, lam_aff, lam_umap, lam_recon)
# CORRECTED 2026-09-26 [D58b]: d58's numbers were measured on a degenerate
# 2-edge dataset (train_epochs=1 -> weight_threshold=max_weight, see
# measure_grad_shares_d58b.py). These are from the FULL-PASS re-measurement
# (train_epochs=20 threshold, matching this run's own deployment).
NH_DATASETS = {
    "pancreas":    ("tech",  "celltype",         220, 1.554, 2.332, 0.1224, 0.01622),
    "lung":        ("batch", "cell_type",        300, 2.406, 4.429, 0.0467, 0.0209),  # [D58c] standard single-minibatch, real path
    "pbmc-immune": ("study", "final_annotation", 300, 5.345, 7.728, 0.0345, 0.0140),  # [D58c] standard single-minibatch, real path
}

# ds -> (affinity_type, K, lam_gram, lam_aff, lam_umap, lam_recon)
# PENDING [D58b] full-pass re-measurement for s28nsc/crcx -- placeholders
# below are the OLD (degenerate-probe) values and MUST NOT be used to launch
# training until replaced. Do not spawn run_full for these until updated.
FULL_DATASETS = {
    "s28nsc": ("bk02", 800, 22.026, 31.468, 0.0813, 0.0072),  # [D58c] standard single-minibatch, real path
    "crcx":   ("bk05", 670, 9.487, 10.141, 0.0392, 0.0509),  # [D58c] standard single-minibatch, real path
}

ZERO_TERMS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
              "lambda_swav", "lambda_kl", "lambda_proto_recon")


def build_config(lam_gram, lam_aff, lam_umap, lam_recon, grad_probe=25):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "lambda_umap": lam_umap,
        "lambda_recon": lam_recon,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_proto_recon": 0.0,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 0.0,
        "proto_reinit_every": 0,
        "grad_probe": grad_probe,
        "umap_proto_sg": 1,
    }


def validate(lc):
    for k in ZERO_TERMS:
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    assert lc["lambda_cohesion"] > 0 and lc["lambda_cohesion_aff"] > 0
    assert lc["lambda_umap"] > 0 and lc["lambda_recon"] > 0
    assert lc["umap_proto_sg"] == 1
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    return {"config_tag": CONFIG_TAG,
            "lambda_cohesion(gram)": lc["lambda_cohesion"],
            "lambda_cohesion_aff": lc["lambda_cohesion_aff"],
            "lambda_umap": lc["lambda_umap"], "umap_proto_sg": lc["umap_proto_sg"],
            "lambda_recon": lc["lambda_recon"], "grad_probe": lc["grad_probe"]}


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
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run_nodeheldout(ds: str):
    import os
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    bk, lk, K, lg, la, lu, lr = NH_DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad"
    split_np = f"/vol/data/node_heldout/{ds}_{NH_TAG}_split.npz"
    assert os.path.exists(train_h5), f"missing {train_h5}"
    assert os.path.exists(split_np), f"missing {split_np}"

    lc = build_config(lg, la, lu, lr)
    checked = validate(lc)
    name = f"{CONFIG_TAG}_{ds}"
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}  ds={ds} K={K} NODE-HELDOUT")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=bk, label_key=lk,
        cvae_epochs=50, train_epochs=10, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    print(f"[{name}] trained in {time.time()-t0:.0f}s  n_dead={metrics.get('n_dead')}")

    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
    ad_full = get_preprocessed_adata(ds)
    d = np.load(split_np, allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()

    assert trainer.check_conditions_compatible(trainer.model, ad_test), (
        "ad_test has batch/condition values the trained model never saw -- "
        "refusing to leak via scPoli's load_query_data retrain path."
    )
    with torch.no_grad():
        z = trainer.encode_adata(ad_test, trainer.model, z_idx=1)
        assignments_test = trainer.model.prototypes(z).argmax(dim=1).cpu().numpy()

    assignments_train, _ = trainer._get_assignments()
    s_train = pd.Series(assignments_train, index=trainer.train_ds.adata.obs_names)
    s_test = pd.Series(assignments_test, index=ad_test.obs_names)
    combined = pd.concat([s_train, s_test]).reindex(ad_full.obs_names)
    assert not combined.isna().any(), "missing assignment after combining"

    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True
    out = os.path.join(trainer.get_dump_path(), "clusters_full.npz")
    np.savez(out, labels=combined.values.astype(int), test_mask=test_mask)
    print(f"[{name}] saved combined assignments ({int(test_mask.sum())} frozen-pass "
          f"test cells): {out}")
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run_full(ds: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    aff, K, lg, la, lu, lr = FULL_DATASETS[ds]
    lc = build_config(lg, la, lu, lr)
    checked = validate(lc)
    name = f"{CONFIG_TAG}_{ds}"
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}  ds={ds} K={K} FULL-GRAPH aff={aff}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=10, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=aff, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    print(f"[{name}] trained in {time.time()-t0:.0f}s  n_dead={metrics.get('n_dead')}")
    print(f"[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics


@app.local_entrypoint()
def main(ds: str = ""):
    """ds='' -> launch every dataset with a non-pending lambda (skips any
    FULL_DATASETS entry still marked PENDING). ds='<name>' -> launch just
    that one dataset, alone, with its own separate stdout -- use this to get
    per-dataset log files instead of interleaved output when running several
    `modal run` invocations in parallel."""
    targets = [ds] if ds else list(NH_DATASETS) + list(FULL_DATASETS)

    calls = {}
    for d in targets:
        if d in NH_DATASETS:
            calls[d] = run_nodeheldout.spawn(d)
        elif d in FULL_DATASETS:
            if FULL_DATASETS[d][2] is None:
                print(f"SKIPPING {d}: lambda still PENDING -- update FULL_DATASETS first")
                continue
            calls[d] = run_full.spawn(d)
        else:
            raise ValueError(f"unknown dataset {d!r}")

    print(f"Spawned {len(calls)} run(s): {list(calls)}")
    with open("modal_scripts/.d59_calls.txt", "a") as f:
        for d, c in calls.items():
            f.write(f"{d}\t{c.object_id}\n")

    results = {}
    for d, c in calls.items():
        name, model_name, metrics = c.get()
        results[d] = metrics
        print(f"\n[{d}] DONE: {name}\n  model={model_name}\n  metrics={metrics}")

    print("\n" + "=" * 100)
    print("D59 -- unified 4-term loss, share=0.25 each")
    print("=" * 100)
    for d, m in results.items():
        print(f"{d:<14} n_dead={m.get('n_dead')}  {m}")

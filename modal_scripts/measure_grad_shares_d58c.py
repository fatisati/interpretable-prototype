"""[D58c] FIX #2: measure gnorm at the TRUE fresh state, correct edge threshold,
no contamination from a full distorting pass.

THE BUG IN d58b
----------------
d58b fixed the degenerate-2-edge bug (train_epochs=1 -> weight_threshold=max)
by running a FULL epoch (continue_train_umap_edges(epochs=1)) at the correct
threshold (train_epochs=20) before measuring. But at temporary lambda=1 for
all 4 terms, recon and community's raw gradients are 15-40x cohesion's, so
that whole measurement pass already pulled the model toward recon/community.
The measured gnorm came from an already-distorted state, not the clean
freshly-loaded pretrain checkpoint the REAL deployed run actually starts
from. Verified 2026-09-26: pbmc-immune's real run, read from its own
grad_probe.jsonl, showed 4.98%/6.27%/47.70%/41.05% share at its OWN step 0 --
nowhere near the intended 25% each -- because d58b's measured gnorm didn't
match the real run's actual starting gnorm.

THE FIX
-------
`umap_steps_per_epoch` decouples "how many steps run" from "what threshold
built the edge dataset" (trainers/scproto.py:1050-1052): EdgeDataset's
weight_threshold uses `train_epochs` (the OUTER epochs arg) regardless of
`umap_steps_per_epoch`, which only caps `n_samples` for THIS epoch's
DataLoader. So: train_epochs=10 (matches train_d59's real deployment on ALL
5 datasets now) for a representative, non-degenerate edge set, but
umap_steps_per_epoch=1 so only ONE step actually runs before the probe fires
(grad_probe=1 catches it) -- the true fresh state, no distortion, no
degenerate edges.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_d58c.py
"""
import json
import time

import modal

app = modal.App("scproto-d58c-probe")

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
REAL_TRAIN_EPOCHS = 10   # matches train_d59's ACTUAL deployment on all 5 datasets

NH_DATASETS = {
    "pancreas": ("tech", "celltype", 220),
    "lung": ("batch", "cell_type", 300),
    "pbmc-immune": ("study", "final_annotation", 300),
}
FULL_DATASETS = {
    "s28nsc": ("bk02", 800),
    "crcx": ("bk05", 670),
}


def build_probe_config():
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_proto_recon": 0.0,
        "lambda_umap": 1.0, "lambda_recon": 1.0,
        "lambda_cohesion": 1.0, "lambda_cohesion_aff": 1.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0,
        "proto_reinit_every": 0,
        "grad_probe": 1,             # fire at step 0
        "umap_proto_sg": 1,
    }


def _env():
    import os, sys
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


def _extract_first(dump_path):
    import os
    p = os.path.join(dump_path, "grad_probe.jsonl")
    if not os.path.exists(p):
        return {"error": f"no grad_probe.jsonl at {p}"}
    with open(p) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    for line in lines:
        row = json.loads(line)
        if "error" in row:
            continue
        terms = {k: v for k, v in row.items() if k != "step" and isinstance(v, dict)}
        if all(t in terms for t in ("cohesion_gram", "cohesion_aff", "umap", "recon")):
            return {"step": row.get("step", -1), **{t: terms[t]["gnorm_enc"] for t in
                     ("cohesion_gram", "cohesion_aff", "umap", "recon")}}
    return {"error": "no row had all 4 terms", "rows_seen": len(lines)}


@app.function(image=image, gpu="A100", timeout=1800, volumes={"/vol": volume})
def probe_nodeheldout(ds: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task
    bk, lk, K = NH_DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad"
    lc = build_probe_config()
    name = f"d58c_{ds}"
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=bk, label_key=lk,
        cvae_epochs=50, train_epochs=REAL_TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=1,       # <- only 1 step actually runs
    )
    shares = _extract_first(trainer.get_dump_path())
    print(f"[{ds}] FRESH-STATE gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.function(image=image, gpu="A100", timeout=1800, volumes={"/vol": volume})
def probe_full(ds: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task
    aff, K = FULL_DATASETS[ds]
    lc = build_probe_config()
    name = f"d58c_{ds}"
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=REAL_TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=aff, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=1,
    )
    shares = _extract_first(trainer.get_dump_path())
    print(f"[{ds}] FRESH-STATE gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.local_entrypoint()
def main(ds: str = ""):
    targets_nh = [ds] if ds else list(NH_DATASETS)
    targets_nh = [d for d in targets_nh if d in NH_DATASETS]
    targets_full = [ds] if ds else list(FULL_DATASETS)
    targets_full = [d for d in targets_full if d in FULL_DATASETS]

    calls = {}
    for d in targets_nh:
        calls[d] = probe_nodeheldout.spawn(d)
    for d in targets_full:
        calls[d] = probe_full.spawn(d)
    print(f"Spawned {len(calls)} parallel FRESH-STATE probes: {list(calls)}")

    results = {}
    for ds, c in calls.items():
        ds2, r = c.get()
        results[ds2] = r

    print("\n" + "=" * 100)
    print("FRESH-STATE gnorm_enc (step 0, correct threshold) -> lambda_i = 0.25/gnorm_i")
    print("=" * 100)
    SHARE = 0.25
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:<14} FAILED: {r}")
            continue
        print(f"\n{ds}  (step {r['step']}):")
        for term in ("cohesion_gram", "cohesion_aff", "umap", "recon"):
            g = r[term]
            lam = SHARE / g if g > 0 else float("inf")
            print(f"  {term:<14} gnorm_enc={g:.6e}   lambda = {lam:.5f}")

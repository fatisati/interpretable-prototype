"""[D58] Unified 4-term gradient-share probe -- CLAUDE.md 9d-1 (2026-09-26 rewrite).

WHY THIS RUN
------------
CLAUDE.md 9d/9d-1 was rewritten today: cohesion is no longer a privileged
authority/span anchor. All FOUR active terms (cohesion-gram, cohesion-aff,
community, recon) are now calibrated the SAME way:

    share_i = 0.25 for every term (user decision, this run resolves [D55]'s
              PENDING target-share vector)
    gnorm_i = raw encoder-gradient norm of term i, measured with the term
              turned ON (any nonzero lambda -- gnorm_enc is computed from the
              UNSCALED loss tensor, e.g. cohesion_loss = 1.0 - cohesion, via
              torch.autograd.grad(term_tensor, encoder_params), so it does NOT
              depend on which lambda value was used to turn the term on --
              verified by reading trainers/scproto.py's [D45] probe block)
    lambda_i = share_i / gnorm_i          -- no total-budget constant (Adam
              normalizes by its own running gradient magnitude, so an overall
              scale cancels in the update ratio -- authority=1.5403 RETIRED)

This script does ONLY the measurement: run each dataset for 1 short epoch with
grad_probe=1 (fires every step, so step 0 lands in grad_probe.jsonl) and all
four terms turned on at lambda=1 simultaneously, matching real deployed
conditions otherwise (umap_proto_sg=1 routing, freeze_batch_embedding=True,
cohesion_self=1.0, batch_size=1024, dataset-default K). Reads back
grad_probe.jsonl's step-0 record and prints the derived lambda_i per term per
dataset. Nothing here is a real training run -- no scoreboard number should be
read from it.

s28nsc EXCEPTION: community has a standing "do not re-run" verdict
(train_d50_s28nsc_routed.py: community alone AND recon alone both regressed
vs. the cohesion-only [D39]-E baseline, confirmed at two separate doses). This
script still MEASURES s28nsc's community gnorm (so nothing is assumed), but
the derived lambda_umap for s28nsc is reported and then forced to 0 in any
downstream training script -- not deployed live. See CLAUDE.md 9d.

Run (short, cheap -- T4, ~5 datasets x a few minutes each if pretrain
checkpoints already exist on the volume):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_grad_shares_d58.py
"""
import json
import time

import modal

app = modal.App("scproto-d58-probe")

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

# node-heldout datasets: ds -> K (dataset default)
NH_DATASETS = {
    "pancreas": ("tech", "celltype", 220),
    "lung": ("batch", "cell_type", 300),
    "pbmc-immune": ("study", "final_annotation", 300),
}

# full-graph datasets: ds -> (affinity_type, K)
FULL_DATASETS = {
    "s28nsc": ("bk02", 800),
    "crcx": ("bk05", 670),
}


def build_probe_config():
    """ALL FOUR terms on at lambda=1 (temporary, for measurement only).
    Everything else zeroed, exactly per CLAUDE.md 9d."""
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_proto_recon": 0.0,
        "lambda_umap": 1.0,          # temporary -- probing only
        "lambda_recon": 1.0,         # temporary -- probing only
        "lambda_cohesion": 1.0,      # temporary -- probing only
        "lambda_cohesion_aff": 1.0,  # temporary -- probing only
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 0.0,        # short probe -- no decay needed
        "proto_reinit_every": 0,
        "grad_probe": 1,             # fire every step -> step 0 always logged
        "umap_proto_sg": 1,          # [D48] required whenever lambda_umap > 0
        # latent_dims: not passed -> defaults.py's 8
    }


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


def _extract_shares(dump_path):
    """Read grad_probe.jsonl's first usable record, return {term: gnorm_enc}."""
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
        terms = {k: v for k, v in row.items()
                 if k != "step" and isinstance(v, dict)}
        if all(t in terms for t in ("cohesion_gram", "cohesion_aff", "umap", "recon")):
            return {t: terms[t]["gnorm_enc"] for t in
                    ("cohesion_gram", "cohesion_aff", "umap", "recon")}
    return {"error": "no row had all 4 terms", "rows_seen": len(lines)}


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def probe_nodeheldout(ds: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    bk, lk, K = NH_DATASETS[ds]
    train_h5 = f"/vol/data/node_heldout/{ds}_{NH_TAG}_train.h5ad"
    lc = build_probe_config()
    name = f"d58p_{ds}"
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=bk, label_key=lk,
        cvae_epochs=50, train_epochs=1, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    shares = _extract_shares(trainer.get_dump_path())
    print(f"[{ds}] gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def probe_full(ds: str):
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task

    aff, K = FULL_DATASETS[ds]
    lc = build_probe_config()
    name = f"d58p_{ds}"
    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=1, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=aff, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    shares = _extract_shares(trainer.get_dump_path())
    print(f"[{ds}] gnorm_enc -> {shares}  ({time.time()-t0:.0f}s)")
    volume.commit()
    return ds, shares


@app.local_entrypoint()
def main():
    results = {}
    for ds, r in probe_nodeheldout.starmap([(d,) for d in NH_DATASETS]):
        results[ds] = r
    for ds, r in probe_full.starmap([(d,) for d in FULL_DATASETS]):
        results[ds] = r

    print("\n" + "=" * 100)
    print("MEASURED gnorm_enc (raw, lambda-independent) and DERIVED lambda_i = 0.25 / gnorm_i")
    print("=" * 100)
    SHARE = 0.25
    for ds, r in results.items():
        if "error" in r:
            print(f"{ds:<14} FAILED: {r}")
            continue
        print(f"\n{ds}:")
        for term, g in r.items():
            lam = SHARE / g if g > 0 else float("inf")
            print(f"  {term:<14} gnorm_enc={g:.6e}   lambda = {SHARE}/{g:.6e} = {lam:.4f}")

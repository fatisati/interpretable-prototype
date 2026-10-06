"""[D58] Ablation + dose sweep on pbmc-immune, around share vector B.

PLAN: my-docs/doing/ablation-plan.txt   DECISION: my-docs/adr/ (D58, to be written)
BASE LAMBDAS: my-docs/experiments/base-lambdas-gradnorm.md  (run tag d57, post-fix)

WHAT THIS IS
------------
One dose point at 0 per term IS that term's ablation arm, so the ablation and the
sweep are the same experiment. Ratios per term: {0, 1x, 2x} (user decision).

Base = share vector B, equal share per SUBSYSTEM (cohesion 1/3 split evenly over
its two kernels, community 1/3, recon 1/3), with
    lambda_i = share_i / median_gradnorm_i
and gradnorm measured by measure_grad_shares_d60.py (tag d57): one full frozen
pass, weights VERIFIED bit-identical, reproducibility verified by a duplicate arm
that matched on every printed digit.

    pbmc-immune base B:  gram 3.794   aff 6.399   community 0.04307   recon 0.02042

Each of the FOUR terms is swept individually (gram and aff separately, so we can
see which kernel does the work), everything else held at base.

BUDGET IS A HARD CAP ($5, user decision). Arms are therefore launched in waves:
wave 1 is the reference arm alone, to MEASURE the real per-arm cost before
committing the rest. Pass --arms to pick a wave.

TRAPS HANDLED
-------------
- `seed` is NOT in interpretable_ssl/constants.py ABBREVIATIONS, so two runs that
  differ ONLY in seed get the SAME model folder and clobber each other
  (CLAUDE.md 9). Every arm here therefore gets a DISTINCT experiment_name,
  including the seed arms. (Adding 'seed' to ABBREVIATIONS would be the real fix
  -- behaviour-preserving, since the default 31 would still be omitted from names
  -- but library code is not changed mid-sweep.)
- PRETRAIN_PARAM_KEYS excludes seed, so all arms share ONE CVAE checkpoint. That
  is both cheap and the right scope: the sweep varies stage-2 lambdas, so the
  noise floor should measure stage-2 variance.
- validate() must ALLOW a zero lambda for the term being ablated, unlike d59's
  which asserts all four > 0.
- train_epochs=10, matching d59 AND the edge threshold the lambdas were
  calibrated against (weight_threshold = max_weight / n_epochs).

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_d58_ablation.py --arms ref
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_d58_ablation.py --arms wave1
"""
import json
import time

import modal

app = modal.App("scproto-d58-ablation")

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

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
NH_TAG = "arbf_nodeheldout20_seed0"
TRAIN_EPOCHS = 10
GPU = "L4"          # measured against T4 by the reference arm; see module docstring

# base B for pbmc-immune, from d57 (gram, aff, community, recon)
B_GRAM, B_AFF, B_COMM, B_RECON = 3.794, 6.399, 0.04307, 0.02042

# arm -> (gram, aff, community, recon, seed)
ARMS = {
    # --- reference + noise floor ---
    "base_s31":  (B_GRAM,   B_AFF,   B_COMM,     B_RECON,   31),
    "base_s32":  (B_GRAM,   B_AFF,   B_COMM,     B_RECON,   32),
    "base_s33":  (B_GRAM,   B_AFF,   B_COMM,     B_RECON,   33),
    # --- ablations: dose 0 for each term, one at a time ---
    "gram0":     (0.0,      B_AFF,   B_COMM,     B_RECON,   31),
    "aff0":      (B_GRAM,   0.0,     B_COMM,     B_RECON,   31),
    "coh0":      (0.0,      0.0,     B_COMM,     B_RECON,   31),
    "comm0":     (B_GRAM,   B_AFF,   0.0,        B_RECON,   31),
    "recon0":    (B_GRAM,   B_AFF,   B_COMM,     0.0,       31),
    # --- 2x dose points ---
    "comm2x":    (B_GRAM,   B_AFF,   2 * B_COMM, B_RECON,   31),
    "recon2x":   (B_GRAM,   B_AFF,   B_COMM,     2 * B_RECON, 31),
    "gram2x":    (2 * B_GRAM, B_AFF, B_COMM,     B_RECON,   31),
    "aff2x":     (B_GRAM,   2 * B_AFF, B_COMM,   B_RECON,   31),
}

WAVES = {
    # the reference arm alone -- MEASURES real per-arm cost before committing budget
    "ref":   ["base_s31"],
    # priority order if the budget is tight: noise floor, then the four ablations,
    # then the 2x points
    "wave1": ["base_s32", "base_s33", "gram0", "aff0", "comm0", "recon0", "comm2x"],
    "wave2": ["coh0", "recon2x", "gram2x", "aff2x"],
    "all":   list(ARMS),
}

ZERO_ALWAYS = ("lambda_nassoc", "lambda_modularity", "lambda_proto_usage",
               "lambda_swav", "lambda_kl", "lambda_proto_recon")


def build_config(gram, aff, comm, recon, seed):
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": gram,
        "lambda_cohesion_aff": aff,
        "lambda_umap": comm,
        "lambda_recon": recon,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_proto_recon": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0,
        "proto_reinit_every": 0,
        "umap_proto_sg": 1,
        "grad_probe": 25,      # cheap at 25; records the REALIZED shares per arm
        "seed": seed,
    }


def validate(lc, arm):
    """Unlike d59's validate, a swept term IS allowed to be 0 -- that is the
    ablation. Everything NOT part of the 4-term design must still be exactly 0."""
    for k in ZERO_ALWAYS:
        assert float(lc[k]) == 0.0, f"{arm}: {k} must be 0, got {lc[k]}"
    assert lc["umap_proto_sg"] == 1, f"{arm}: routing must stay on"
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["proto_reinit_every"] == 0
    n_active = sum(1 for k in ("lambda_cohesion", "lambda_cohesion_aff",
                               "lambda_umap", "lambda_recon") if float(lc[k]) > 0)
    assert n_active >= 2, f"{arm}: only {n_active} active terms -- not an ablation arm"
    return {"arm": arm, "gram": lc["lambda_cohesion"], "aff": lc["lambda_cohesion_aff"],
            "community": lc["lambda_umap"], "recon": lc["lambda_recon"],
            "seed": lc["seed"], "train_epochs": TRAIN_EPOCHS}


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


@app.function(image=image, gpu=GPU, timeout=21600, volumes={"/vol": volume})
def run_arm(arm: str):
    import os
    _env()
    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import run_mc_task

    gram, aff, comm, recon, seed = ARMS[arm]
    lc = build_config(gram, aff, comm, recon, seed)
    checked = validate(lc, arm)
    # DISTINCT experiment_name per arm -- including the seed arms, because `seed`
    # is not in ABBREVIATIONS and would otherwise collide (see module docstring).
    name = f"d58_{arm}"
    print(f"   [D58 CONFIG VERIFIED IN-RUN] {checked}")

    train_h5 = f"/vol/data/node_heldout/{DS}_{NH_TAG}_train.h5ad"
    split_np = f"/vol/data/node_heldout/{DS}_{NH_TAG}_split.npz"
    assert os.path.exists(train_h5), f"missing {train_h5}"
    assert os.path.exists(split_np), f"missing {split_np}"

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
        cvae_epochs=50, train_epochs=TRAIN_EPOCHS, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=True,
        umap_steps_per_epoch=None,
    )
    train_s = time.time() - t0
    print(f"[{arm}] trained in {train_s:.0f}s  n_dead={metrics.get('n_dead')}")

    # --- place the held-out 20% by a frozen-encoder forward pass (same contract
    #     as train_d59_share25_matrix.py) ---
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata
    ad_full = get_preprocessed_adata(DS)
    d = np.load(split_np, allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()

    assert trainer.check_conditions_compatible(trainer.model, ad_test), (
        "ad_test has batch/condition values the trained model never saw -- "
        "refusing to leak via scPoli's load_query_data retrain path.")
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
    dump = trainer.get_dump_path()
    lab_full = combined.values.astype(int)
    np.savez(os.path.join(dump, "clusters_full.npz"),
             labels=lab_full, test_mask=test_mask)
    # ALSO write clusters.npz with the SAME full-length labels. score_all_baselines.
    # load_labels() only ever looks for "clusters.npz" -- it does NOT know about
    # clusters_full.npz -- so a run that saves only clusters_full.npz is SILENTLY
    # skipped by the scorer ("no usable per-cell labels"). That is exactly why
    # d59_v2 / d59_v3 never made it into the immune scoreboard despite holding
    # correct 33,506-length labels. Overwriting is intended: the trainer's own
    # clusters.npz holds TRAIN-ONLY assignments, and for an NH run the scorable
    # artifact is the combined train+frozen-pass-placed-test labelling.
    np.savez(os.path.join(dump, "clusters.npz"), labels=lab_full)

    summary = {"arm": arm, "config": checked, "train_seconds": round(train_s, 1),
               "gpu": GPU, "model_name": trainer.get_model_name(),
               "dump_path": dump, "metrics": metrics}
    with open(os.path.join(dump, "d58_arm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[{arm}] saved clusters_full.npz ({int(test_mask.sum())} frozen-pass test cells)")
    print(f"[{arm}] model: {trainer.get_model_name()}")
    print(f"[{arm}] >>> {metrics}")
    volume.commit()
    return summary


@app.local_entrypoint()
def main(arms: str = "ref"):
    sel = WAVES.get(arms)
    if sel is None:
        sel = [a.strip() for a in arms.split(",") if a.strip()]
    bad = [a for a in sel if a not in ARMS]
    assert not bad, f"unknown arms: {bad}. known: {list(ARMS)}"

    print(f"[D58] dataset={DS}  gpu={GPU}  train_epochs={TRAIN_EPOCHS}")
    print(f"[D58] base B: gram={B_GRAM} aff={B_AFF} community={B_COMM} recon={B_RECON}")
    print(f"[D58] launching {len(sel)} arm(s): {sel}")
    for a in sel:
        g, af, c, r, s = ARMS[a]
        print(f"        {a:<10} gram={g:<9g} aff={af:<9g} comm={c:<9g} "
              f"recon={r:<9g} seed={s}")

    calls = {a: run_arm.spawn(a) for a in sel}
    out, failed = {}, {}
    for a, c in calls.items():
        try:
            out[a] = c.get()
        except Exception as e:
            failed[a] = f"{type(e).__name__}: {e}"
            print(f"[{a}] FAILED: {failed[a]}")

    print("\n" + "=" * 96)
    print("[D58] ARM SUMMARY")
    print("=" * 96)
    print(f"{'arm':<10}{'train_s':>9}{'n_dead':>8}  lambdas (gram/aff/comm/recon)")
    tot = 0.0
    for a, s in out.items():
        cfg = s["config"]
        tot += s["train_seconds"]
        print(f"{a:<10}{s['train_seconds']:>9.0f}{str(s['metrics'].get('n_dead')):>8}  "
              f"{cfg['gram']:g} / {cfg['aff']:g} / {cfg['community']:g} / {cfg['recon']:g}")
    print(f"\ntotal train time this wave: {tot/3600:.2f} GPU-h on {GPU}")
    with open("d58_arms.json", "w") as f:
        json.dump({"ok": out, "failed": failed, "gpu": GPU}, f, indent=2, default=str)
    print("[D58] wrote d58_arms.json")
    if failed:
        print(f"[D58] {len(failed)} FAILED: {list(failed)}")

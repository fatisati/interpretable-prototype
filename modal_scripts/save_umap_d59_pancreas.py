"""Generate umap_cells.csv / umap_protos.csv for the d59_share25_pancreas run
(the current best pancreas scProto model -- "scProto NH (4-term share25)" in
files/tables/node_heldout_scoreboard/RESULTS.md and pancreas_scoreboard.csv,
produced by train_d59_share25_matrix.py::run_nodeheldout('pancreas')).

WHY THIS RUN
------------
train_d59_share25_matrix.py trains and commits clusters_full.npz but never
calls trainer.save_umap_data() -- so paper_figures.fig_umap() has nothing to
read for this model yet. This script reloads the ALREADY-TRAINED checkpoint
(no retraining: find_metacells(load_umap=True, skip_eval=True), the tested
pattern from eval_salvage_nousage_dge.py) and calls save_umap_data() once.

SCOPE DECISION (user, 2026-09-26): train-only. save_umap_data() encodes
self.train_ds.adata, i.e. the 80% node-heldout TRAIN split only -- the
held-out 20% test cells are NOT projected/added here. This is cheaper and is
what save_umap_data() does with zero extra code; the tradeoff (scProto panel
shows ~80% of cells vs. the SEACells baseline panel's 100%) should be noted
in the figure caption, not silently hidden.

DECISION, BEFORE RUNNING: expect this to just load + one forward pass + one
UMAP fit (CPU-side sc.tl.umap on ~35-40k pancreas train cells) -- no gradient
steps, so this should finish in a couple minutes on a T4. VERDICT: [ ] works
[ ] fails [ ] inconclusive-because___

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/save_umap_d59_pancreas.py

Then pull the two files back locally, e.g.:
  modal volume get scproto-data models/d59_pancreas*/umap_cells.csv  <local dir>
  modal volume get scproto-data models/d59_pancreas*/umap_protos.csv <local dir>
(exact model-folder name is printed by this script as trainer.get_model_name())
"""
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
# must match train_d59_share25_matrix.py's NH_DATASETS["pancreas"] exactly, so
# get_model_name() resolves to the SAME folder the original run wrote to.
BATCH_KEY, LABEL_KEY, K = "tech", "celltype", 220
LAM_GRAM, LAM_AFF, LAM_UMAP, LAM_RECON = 1.554, 2.332, 0.1224, 0.01622


def _build_config():
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": LAM_GRAM,
        "lambda_cohesion_aff": LAM_AFF,
        "lambda_umap": LAM_UMAP,
        "lambda_recon": LAM_RECON,
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
        "grad_probe": 25,
        "umap_proto_sg": 1,
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


@app.function(image=image, gpu="T4", timeout=1800, volumes={"/vol": volume})
def save_umap_pancreas():
    import os
    _env()
    # NOT using find_metacells(load_umap=True, skip_eval=True) here: that path
    # calls t.load_umap_checkpoint() WITHOUT first calling
    # t.model.freeze_batch_embedding() (tasks.py only calls freeze inside its
    # `if not load_umap:` branch) -- so the freshly-built optimizer has a
    # DIFFERENT param-group structure (batch-embedding params still trainable)
    # than the one saved in this run's checkpoint (trained WITH
    # freeze_batch_embedding=True per CLAUDE.md 9d), and
    # optimizer.load_state_dict() raises a param-group-size mismatch. Confirmed
    # by running it: that is exactly what happened here 2026-09-26.
    # Fix: replicate find_metacells's trainer construction (tasks.py:192-253)
    # ourselves, call freeze_batch_embedding() BEFORE load_umap_checkpoint(),
    # matching the order the original training run used.
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes,
    )

    name = f"{CONFIG_TAG}_pancreas"
    train_h5 = f"/vol/data/node_heldout/pancreas_{NH_TAG}_train.h5ad"
    assert os.path.exists(train_h5), f"missing {train_h5}"

    lc = _build_config()
    lc["affinity_type"] = "arbf"
    ds_id = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                           niche_key=None, num_prototypes=K)
    resolved_k = _resolve_num_prototypes(K, ds_id)

    trainer = get_trainer(
        experiment_name=name,
        cvae_epochs=50,
        affinity_type=lc["affinity_type"],
        dataset_id=ds_id,
        l2norm=1,
        assignment_metric="dotp",
        batch_size=1024,
        umap_steps_per_epoch=None,
        umap_similarity=lc.get("umap_similarity", "proto"),
        freeze_batch_embedding=1,
        freeze_decoder=0,
        num_prototypes=resolved_k,
        **{k: v for k, v in lc.items() if k not in ("umap_similarity", "affinity_type")},
    )
    trainer.load_pretrain_checkpoint()
    if hasattr(trainer.model, "freeze_batch_embedding"):
        trainer.model.freeze_batch_embedding()
    trainer.load_umap_checkpoint()

    print(f"[{name}] loaded checkpoint (no retrain): {trainer.get_model_name()}")
    print(f"[{name}] dump path: {trainer.get_dump_path()}")

    trainer.save_umap_data()
    volume.commit()
    print(f"[{name}] umap_cells.csv / umap_protos.csv written and committed.")
    return trainer.get_dump_path()


@app.local_entrypoint()
def main():
    path = save_umap_pancreas.remote()
    print(f"done: {path}")

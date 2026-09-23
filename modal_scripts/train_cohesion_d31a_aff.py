"""[D31a] The AFF (1-hop, ratio-association) cohesion with the new memory.

DECISION BEFORE RUNNING
-----------------------
d30_A_aff is the saved CONTROL (metrics.json on the volume): coverage 0.889,
mean purity 0.636, niche purity 0.423, 16 unused, 184 live, median 122 -- the
aff arm already beats gram on coverage/purity but loses on DGE kendall
(0.114 vs 0.252). These two arms ask ONE question: do the memory fixes
([D32] honest init, [D31] age decay) move the aff arm?

  d31a_ainit  aff + coh_mem_init=full            -- expect same-or-better end
              metrics vs d30_A_aff; drop -> [D32] interacts badly with aff.
  d31a_afull  aff + init full + hlife=500        -- stability under decay.

WATCH the same churn signals as d31: [reinit] revived N per cycle (the gram
arms churn at ~176-180), unused per epoch, median size. If aff churns
identically, the churn is fully similarity-independent -> reinit policy /
missing size control, per the standing [D20]/[D240] diagnosis.

LAMBDA: 44.6 = 1.5403 / span(aff, self-included)=0.034573 -- d30's measured
value, unchanged by the memory knobs (scale-preserving, unit-tested).
"""
import time
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph",
        "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git", extra_options="--no-deps")
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

SPAN_AFF_SELF = 0.034573            # measured (d30 header), unchanged by D31/D32
AUTHORITY = 20.0 * 0.077017         # 1.5403 -- d26_A's total, held fixed
LAM_AFF = round(AUTHORITY / SPAN_AFF_SELF, 1)     # 44.6 -- THE measured lambda

# (name, coh_mem_init, coh_mem_hlife)
ARMS = [
    ("d31a_ainit", "full", 0.0),    # [D32] only; control = saved d30_A_aff
    ("d31a_afull", "full", 500.0),  # [D32] + [D31]
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, mem_init: str, hlife: float):
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

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        # --- identical to d30_B_gram except the two memory knobs under test ---
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        # aff arm drives lambda_cohesion_aff with lambda_cohesion=0 (d30_A
        # pattern): the 1-hop term runs on its own path, gram never computed.
        "lambda_cohesion": 0.0,
        "lambda_cohesion_aff": LAM_AFF,   # 44.6 -- measured span, see header
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 0.5,             # [D30] founder gate 0.25
        "cohesion_global": 2,             # [D29] remembered profiles
        "coh_mem_init": mem_init,         # [D32] under test
        "coh_mem_hlife": hlife,           # [D31] under test
        "proto_reinit_every": 2,
        "proto_reinit_rel": 0.1,
        "proto_reinit_target": "wrst",
        "latent_dims": 16,
    }
    assert lc["cohesion_self"] == 0.5 and lc["cohesion_global"] == 2
    assert lc["lambda_cohesion_aff"] == 44.6 and lc["lambda_cohesion"] == 0.0

    print(f"\n{'='*78}\n=== {name}\n"
          f"    d30_A_aff base, lambda_cohesion_aff={LAM_AFF} (measured span)\n"
          f"    coh_mem_init={mem_init}  coh_mem_hlife={hlife:g}\n"
          f"    WATCH: median size | n_unused per epoch | [reinit] revived N |\n"
          f"           coh_mem_drift / coh_mem_age (new)\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name})
    print(f"\n[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    results = list(run.starmap(ARMS))
    print("\n" + "=" * 104)
    print("[D31a] aff + memory knobs -- control is the SAVED d30_A_aff run")
    print("=" * 104)
    print(f"{'arm':<12}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<12}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 104)
    print("Verdict fields to fill after reading the logs (per run discipline):")
    print("  d31_ctrl : works / fails / inconclusive-because-...   drift was: ___")
    print("  d31_init : works / fails / inconclusive-because-...")
    print("  d31_full : works / fails / inconclusive-because-...")
    print("Then: eval_all_vs_seacells.py on the winning arm for the frozen scoreboard")
    print("(cohesion, n_dead_proto, joint_purity, n_covered_states, mean DGE pearson).")

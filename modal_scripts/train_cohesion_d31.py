"""[D31]+[D32] VALIDATE the age-weighted, honestly-initialized assignment memory.

DECISION BEFORE RUNNING (per the run discipline)
------------------------------------------------
Three arms, all identical to d30_B_gram (the measured-lambda gram arm) except the
two new memory knobs. What we expect, and how it changes decisions:

  d31_ctrl  hlife=0, init unif  -- pure refactor control. The [D31] code path at
            hlife=0 is unit-tested bit-identical to [D29], so this must land where
            d30_B did (up to run noise). It also logs coh_mem_drift/coh_mem_age for
            the first time: THE number that decides whether any half-life is needed.
            drift ~0 after epoch 2 -> [D31] decay stays OFF for good (verdict:
            not needed); sustained drift -> pick hlife ~ the timescale drift shows.
  d31_init  init full           -- [D32] only. Expect: profiles/sizes honest from
            step 0 (no first-epoch dilution), so early cohesion_term starts at a
            realistic level instead of drifting up; end metrics same-or-better.
            If end metrics DROP, [D32] is wrong somewhere -> investigate before use.
  d31_full  init full + hlife=500 (half-life = one epoch) -- the full mechanism.
            Expect: bounded values, no NaN, metrics comparable to ctrl unless the
            drift measured in ctrl is large, in which case this arm should be
            cleaner on dead protos / churn.

THE SERIOUS-FAILURE WATCH (user's): if prototypes die again AND reinit revives
them only for them to die again (revival count pinned high every epoch, unused
sawtoothing like d26's 120-150), that is a REAL formulation problem, not noise --
stop tuning, diagnose why. Watch, per epoch, in this order:
  1. MEDIAN metacell size (the usage loss died here: 86 -> 11)
  2. n_unused_protos per epoch (d26 sawtoothed 120-150; d29 was 111-146 at ep4;
     d30's founder gate 0.25 is supposed to have fixed recruitment)
  3. [reinit] revived N (d26 pinned at ~170 = churn, never healing)
  4. coh_mem_drift / coh_mem_age (new: is staleness even real?)

LAMBDA -- measured, not swept (the [D26]/[D30] span method). The term's authority
is lambda * span, span = value(good partition) - value(random) measured on the
full graph. d30 measured span(gram, self-included) = 0.076776 on s28nsc/bk02 and
held d26_A's total authority 1.5403 -> lambda = 1.5403/0.076776 = 20.1. [D31]/[D32]
do not change the term's scale (every quantity stays the same weighted mean,
bounded [0,1] -- unit-tested), so the span and therefore THE LAMBDA CARRY OVER.

SCOREBOARD (frozen, user's): cohesion, n_dead_proto, joint_purity,
n_covered_states, mean DGE (pearson) -- better/comparable to SEACells. This run
reports what run_mc_task emits (purity, niche_purity, coverage, modularity,
unused); DGE + the SEACells head-to-head need the existing
eval_all_vs_seacells.py pass on the winning arm afterwards.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d31.py
Cost: ~21 min per arm on a T4, 3 arms in parallel ~ $0.45.
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

SPAN_GRAM_SELF = 0.076776           # measured (d30 header), unchanged by D31/D32
AUTHORITY = 20.0 * 0.077017         # 1.5403 -- d26_A's total, held fixed
LAM_GRAM = round(AUTHORITY / SPAN_GRAM_SELF, 1)   # 20.1 -- THE measured lambda

# (name, coh_mem_init, coh_mem_hlife)
ARMS = [
    ("d31_ctrl", "unif", 0.0),      # refactor control + drift measurement
    ("d31_init", "full", 0.0),      # [D32] only
    ("d31_full", "full", 500.0),    # [D32] + [D31], half-life = one epoch
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
        "lambda_cohesion": LAM_GRAM,      # 20.1 -- measured span, see header
        "lambda_cohesion_aff": 0.0,
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
    assert lc["lambda_cohesion"] == 20.1

    print(f"\n{'='*78}\n=== {name}\n"
          f"    d30_B_gram base, lambda_cohesion={LAM_GRAM} (measured span)\n"
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
    print("[D31]+[D32] memory validation -- all arms = d30_B_gram + the two new knobs")
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

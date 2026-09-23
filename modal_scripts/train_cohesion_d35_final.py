"""[D35] THE single most-correct cohesion configuration, no crutches.

DECISION BEFORE RUNNING
-----------------------
One arm only. Every fix from this session's investigation combined, nothing
masking the result:

  sim = aff (1-hop, sim[i,j] = affinity[i,j])          -- user's choice
  cohesion_self = 1.0 (SIMPLE formula: mean similarity to a proto's members,
                        self-loop = 1, no discount)      -- user's choice.
                        Was 0.5 in d30 as a recruitment-gate workaround; that
                        workaround may have been compensating for [D35]'s
                        norm bug, not a real gate problem. This run tests
                        that directly.
  cohesion_diag = 1.0   -- sim(c,c) = 1 exactly (RBF kernel self-affinity)
  cohesion_global = 2   -- [D29] remembered full-graph profiles
  coh_mem_init = 'full' -- [D32] memory starts from a real inference pass,
                            not neutral-but-diluted uniform rows
  coh_mem_hlife = 500   -- [D31] one-epoch half-life age decay on the memory
  [D35] every prototype rescaled to unit norm after EVERY optimizer step
        (scproto.py, right after optimizer.step()) -- the actual fix: dotp
        assignment scores dot(z, proto_k) = |proto_k|*cos(angle), z is
        already normalized every forward pass, prototypes were NOT -- so an
        established prototype's norm could drift and win cells on magnitude
        alone, independent of fit. This is the mechanism suspected to cause
        the dead-proto churn seen in every prior arm (d29/d30/d31/d31a),
        including cases where a reinit-teleported prototype, placed EXACTLY
        at its founding cell's embedding (cosine similarity 1, the
        theoretical max), still failed to win that cell.

  proto_reinit_every = 0  -- NO reinit. If cohesion + the self term + the
                             now-normalized geometry are doing their job,
                             prototypes should not need to be rescued at all.
  early_stop = False      -- NO early stopping ([D34] passthrough). Runs the
                             full fixed 20 epochs; the LAST checkpoint is what
                             gets evaluated -- no proxy-metric cherry-picking
                             of an earlier, luckier epoch.

WHAT WE EXPECT, PER EPOCH (both now printed live -- [D34] added the cohesion
line, unused_proto already existed):
    n_unused_protos:  should trend toward 0 and STAY there, unforced, purely
                       from the loss (no reinit to hide a bad trend). If it
                       instead grows or stays high through 20 full epochs
                       with no reinit, the D35 norm fix was NOT sufficient,
                       and the remaining cause needs to be identified.
    cohesion_aff:     should trend upward, ideally toward SEACells' own
                       coh_aff (measured post-hoc by eval_all_vs_seacells.py,
                       not known ahead of the run -- do not assume the Leiden
                       reference 0.0384 is SEACells' number, it isn't).

LAMBDA: 44.6 = AUTHORITY(1.5403) / SPAN_AFF_SELF(0.034573). This span was
itself measured "WITH sim[c][c]=1" (the d30 header's own words) -- i.e. AT
cohesion_self=1, which is what this run actually uses. d30's own aff arm
used this same lambda while training at self=0.5, a small mismatch; this run
removes it.

VERDICT (fill in after the run, per the run-discipline rule):
  [ ] works -- dead protos -> 0 and stay there, cohesion improves
  [ ] fails -- dead protos persist/grow despite D35 -- root cause NOT the norm bug
  [ ] inconclusive-because-...
  [ ] abandoned, not worth the time

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d35_final.py
Cost: one T4 run, 20 fixed epochs (no early stop) -- longer than prior arms,
budget ~30-40 min, a few dollars.
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

SPAN_AFF_SELF = 0.034573            # measured AT cohesion_self=1 (d30 header)
AUTHORITY = 20.0 * 0.077017         # 1.5403 -- d26_A's total, held fixed
LAM_AFF = round(AUTHORITY / SPAN_AFF_SELF, 1)   # 44.6


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run():
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
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_cohesion": 0.0,
        "lambda_cohesion_aff": LAM_AFF,    # 44.6 -- measured span, self=1
        "cohesion_sim": "gram",            # irrelevant here (lambda_cohesion=0,
                                            # gram path never computed), left at
                                            # default so nothing else changes
        "cohesion_form": "perc",           # mean over members, THE simple form
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,              # sim(c,c) = 1 exactly
        "cohesion_self": 1.0,              # [user] SIMPLE: no discount, self
                                            # counts like any other member
        "cohesion_global": 2,              # [D29] remembered profiles
        "coh_mem_init": "full",            # [D32] honest init
        "coh_mem_hlife": 500.0,            # [D31] one-epoch half-life
        "proto_reinit_every": 0,           # NO reinit -- test the loss alone
        "latent_dims": 16,
    }
    assert lc["lambda_cohesion_aff"] == 44.6 and lc["lambda_cohesion"] == 0.0
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["cohesion_global"] == 2 and lc["proto_reinit_every"] == 0

    print(f"\n{'='*78}\n=== d35_final -- THE single most-correct configuration\n"
          f"    aff sim, cohesion_self=1.0 (simple), diag=1.0, lambda_aff={LAM_AFF}\n"
          f"    memory: D29 remembered + D32 honest init + D31 hlife=500\n"
          f"    D35 per-step prototype renorm ON (scproto.py, joint path)\n"
          f"    NO reinit, NO early stop -- 20 fixed epochs, last checkpoint kept\n"
          f"    WATCH: unused_proto -> should reach 0 and STAY; cohesion_aff -> up\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": "d35_final"},
        early_stop=False)
    print(f"\n[d35_final] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[d35_final] >>> metrics: {metrics}")
    volume.commit()
    return metrics


@app.local_entrypoint()
def main():
    m = run.remote()
    print("\n" + "=" * 90)
    print("[D35] single-arm final result")
    print("=" * 90)
    for k in ('purity', 'niche_purity', 'coverage', 'modularity', 'n_unused_protos'):
        if k in m:
            print(f"  {k:20s} {m[k]}")
    print("\nFill verdict (per run discipline): works / fails / inconclusive-because- / abandoned")
    print("Next: eval_all_vs_seacells.py for joint_purity / dge_r / n_covered / coh_aff vs SEACells")

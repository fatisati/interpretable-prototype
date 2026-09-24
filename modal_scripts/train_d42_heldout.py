"""[D42] The full scProto model on the HELD-OUT-EDGE graphs: pancreas, lung, immune.

WHY THE HELD-OUT GRAPH: it is the setup the baselines already sit in, so nothing
has to be re-run to get a fair table. 20% of each affinity graph's edges were
hidden (FRAC_HELDOUT=0.20, MIN_VISIBLE_DEGREE=5, SEED=0, affinity 'arbf'); the
model trains on the visible 80% and every metric is ALSO reported on the hidden
20%, which no model had a training signal for. Verified exact: the two cached
graphs partition the edge set with no overlap and no loss --
pancreas 462,058 + 115,515 = 577,573, lung 979,170 + 244,792, immune
1,036,350 + 259,088, all exactly 20.0% test.

A NAMING TRAP THIS SCRIPT AVOIDS. affinity_type is truncated to 4 characters in
the model folder name (CLAUDE.md 9), so 'arbf' and 'arbf_heldout20_seed0' BOTH
render as 'aff-arbf' and collide into one directory. The earlier rebuttal
notebook hit this: its held-out scProto run and the canonical full-graph run
share a folder, so that older checkpoint can no longer be attributed to either
graph. Every run here therefore carries an explicit experiment_name 'd42_<ds>'.

LOSS CONFIG -- CLAUDE.md 9c base, plus the two things CLAUDE.md asks for on top:
  9c:   nassoc_diag_norm='pair', nassoc_diag_loss='cpm', nassoc_size_scale='full',
        nassoc_cpm_norm='gful'   (on top of LAMBDA_PROTO_UMAP_PRECON, i.e. the
        community term, proto_recon, usage and nassoc are all ON -- this is the
        FULL model, not the cohesion-only ablation)
  the note at the top of CLAUDE.md: the nassoc OFF-DIAGONAL denominator is
        min(vol_p, vol_q)  ->  nassoc_offdiag_norm='min'
  the cohesion term, which is the current development, with multi-batch cohesion
        now unconditional (no flag): each cell is scored against V[b], the node
        set of its own bio-batch graph.

nassoc_gamma IS AN ASSUMPTION, FLAGGED. CLAUDE.md 9b marks the derivation
UNSETTLED and says explicitly not to treat 0.15 as principled, so gamma is
MEASURED per dataset here from the move rule
    gamma = (same-label weighted degree) / (target_size - 1),  target_size = N/K
taking the p25 across cells -- "the minimum average affinity a cell needs to a
group to be worth joining it", so most cells clear it. Measured on the
heldout-train graphs: pancreas 0.198, lung 0.121, immune 0.108 (means were
0.252 / 0.168 / 0.158). If this choice is wrong the diagonal is mis-weighted,
so it is called out rather than buried.

cohesion_self = 1.0 and cohesion_diag = 1.0 -- UNCHANGED from the
batch-agnostic formula, deliberately. The only thing multi-batch changes is the
set of cells each cohesion is computed against.

LAMBDAS: measured, never swept. lambda = AUTHORITY/span with AUTHORITY = 1.5403
held fixed from the s28nsc arms, spans measured on the SAME heldout-train graphs
under the V[b] denominator (leiden at K minus random at K, so K is controlled).

METRICS come from modal_scripts/score_all_baselines.py afterwards, so scProto and
every baseline go through identical code on identical graphs.

Run:
  modal deploy modal_scripts/train_d42_heldout.py
  python modal_scripts/spawn_d42.py
"""
import time

import modal

app = modal.App("scproto-d42")

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

AUTHORITY = 1.5403
HELDOUT = "arbf_heldout20_seed0"

# ds -> (K, span_aff, span_gram, gamma_p25)
# spans + gamma measured on the heldout-train graphs, 2026-09-24
CFG = {
    "pancreas":    (220, 0.064251, 0.190749, 0.198),
    "lung":        (300, 0.029265, 0.137383, 0.121),
    "pbmc-immune": (300, 0.022884, 0.088139, 0.108),
}


def jobs_for(datasets):
    out = []
    for ds in datasets:
        K, sa, sg, gamma = CFG[ds]
        out.append((ds, K, 0.5 * AUTHORITY / sg, 0.5 * AUTHORITY / sa, gamma,
                    sa, sg))
    return out


@app.function(image=image, gpu="T4", timeout=21600, volumes={"/vol": volume})
def run(ds: str, K: int, lam_gram: float, lam_aff: float, gamma: float,
        span_aff: float, span_gram: float):
    import os
    import sys

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")        # get_affinity_path's './graphs' is CWD-relative

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, *a, **kw):
            return None
    except TypeError:
        pass

    from interpretable_ssl.experiments.tasks import (run_mc_task,
                                                     LAMBDA_PROTO_UMAP_PRECON)

    name = f"d42_{ds}"
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        # --- CLAUDE.md 9c: the scProto base config ---
        "nassoc_diag_norm": "pair",
        "nassoc_diag_loss": "cpm",
        "nassoc_gamma": gamma,
        "nassoc_size_scale": "full",
        "nassoc_cpm_norm": "gful",
        # --- CLAUDE.md header: off-diagonal denominator is min(vol_p, vol_q) ---
        "nassoc_offdiag_norm": "min",
        # --- the cohesion term (multi-batch handling is unconditional now) ---
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 1.0,      # UNCHANGED from the batch-agnostic formula
        "cohesion_global": 2,      # [D29] full-graph assignment memory
        "coh_mem_init": "full",    # [D32]
        "coh_mem_hlife": 500.0,    # [D31]
        "latent_dims": 8,
    }
    auth = lam_gram * span_gram + lam_aff * span_aff
    assert abs(auth - AUTHORITY) < 0.03, f"authority drifted: {auth}"
    assert lc["cohesion_self"] == 1.0 and lc["cohesion_diag"] == 1.0
    assert lc["lambda_umap"] == 1, "this is the FULL model -- community stays on"

    print("\n" + "=" * 78)
    print(f"=== {name}   K={K}  graph={HELDOUT}  LD=8")
    print(f"    FULL model: community ON, nassoc CPM ON (gamma={gamma}, "
          f"offdiag=min), cohesion ON")
    print(f"    lambda_cohesion(gram)={lam_gram:.2f}  "
          f"lambda_cohesion_aff={lam_aff:.2f}  authority={auth:.4f}")
    print(f"    cohesion_self=1.0 cohesion_diag=1.0 (unchanged); multi-batch "
          f"cohesion is unconditional -> V[b]")
    print("=" * 78)

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=HELDOUT, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"\n[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics

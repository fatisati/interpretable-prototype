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

LOSS CONFIG -- CLAUDE.md 9d, the cohesion config. Community OFF, nassoc OFF,
modularity OFF, usage OFF: cohesion is the ONLY objective, so nothing can
outvote the term under test. cohesion_self=1.0 and cohesion_diag=1.0 per 9d.
Multi-batch is unconditional (no flag): each cell scored against V[b].

SUPERSEDED NOTE -- what 9c used to say:
  9c:   nassoc_diag_norm='pair', nassoc_diag_loss='cpm', nassoc_size_scale='full',
        nassoc_cpm_norm='gful'   (on top of LAMBDA_PROTO_UMAP_PRECON, i.e. the
        community term, proto_recon, usage and nassoc are all ON -- this is the
        FULL model, not the cohesion-only ablation)
  the note at the top of CLAUDE.md: the nassoc OFF-DIAGONAL denominator is
        min(vol_p, vol_q)  ->  nassoc_offdiag_norm='min'
  the cohesion term, which is the current development, with multi-batch cohesion
        now unconditional (no flag): each cell is scored against V[b], the node
        set of its own bio-batch graph.

nassoc_gamma IS NO LONGER USED -- nassoc is OFF (see the config below).
The measurement is kept here only as a record. ORIGINAL NOTE: CLAUDE.md 9b marks the derivation
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
# Baked into experiment_name so the run FOLDER records which config ran.
# Bump it whenever the loss config changes -- a stale deployment then
# shows up as a wrong tag instead of silently producing wrong results.
CONFIG_TAG = "cohonly"

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
        # ROUND the lambdas before they reach the model name. Full-precision
        # floats render as 'lcoh4.03750478377344' and pushed the run-folder name
        # past the 255-byte filename limit -- the pancreas run died with
        # OSError [Errno 36] File name too long AFTER training. Two decimals
        # costs nothing numerically (the authority assert still passes to 0.03)
        # and saves ~25 characters per lambda.
        out.append((ds, K, round(0.5 * AUTHORITY / sg, 2),
                    round(0.5 * AUTHORITY / sa, 2), gamma, sa, sg))
    return out


ZERO_TERMS = ("lambda_nassoc",
              "lambda_modularity", "lambda_proto_usage", "lambda_swav",
              "lambda_kl")
# [D44] lambda_umap is deliberately NOT in ZERO_TERMS any more. Diagnosis
# (2026-09-24): the purity gap is GROUP-LEVEL cross-batch block mispairing
# (blockAgree 0.9240 vs Leiden-on-stage1's 0.9825 -- on OUR OWN pretrain
# latent), and BOTH recon anchors at share-0.5 lambdas failed to move it
# (0.9217 / 0.9142) while costing rare coverage. The submitted NeurIPS model
# had NO purity gap on pancreas (0.98, best in table) and its config differed
# by exactly this term: the community/UMAP loss is a PAIRWISE cell-cell term
# (pull edge-connected cells' assignment profiles together, push sampled
# negatives apart) -- the only term that actively maintains the stage-1
# cross-batch structure through stage 2. Cohesion cannot: its sims are fixed
# graph weights and it has no cell-pair coupling and no repulsion.
# [D43] lambda_recon is deliberately NOT in ZERO_TERMS any more. Diagnosis:
# stage-2 cohesion training DEGRADES the cross-batch alignment stage 1 built.
# Measured on pancreas -- within-batch purity 0.9819 (the best of every method)
# but overall 0.9114, so the whole deficit is cross-batch; and Leiden on the
# stage-1 latent ALONE reaches gap 0.0119 at the same ~6 batches/metacell
# against scProto's 0.0706. Cohesion has no cross-batch term by construction
# (0.00% penalty for a bad cross-batch merge, measured), so nothing opposes the
# drift. Recon is what CREATED the batch-corrected latent, so keeping it in
# stage 2 re-imposes that constraint.


def build_config(lam_gram, lam_aff, lam_recon=0.0, lam_proto_recon=0.0,
                 lam_umap=0.0):
    """THE loss config -- CLAUDE.md 9d. Cohesion is the ONLY objective.

    ONE definition, used by run() AND returned by probe(), so the probe reports
    what the deployed code actually builds rather than a hand-written summary
    that can drift from it. A stale deployment already cost three GPU runs.

    nassoc is OFF: nassoc and cohesion are both DIAGONAL terms and cohesion
    replaces nassoc (CLAUDE.md 9b -- CPM's charge is a function of the
    assignment column sums only, so it cannot move a cell for similarity
    reasons). Running both competes and drags in nassoc_gamma, whose derivation
    9b marks UNSETTLED.
    """
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    return dict(LAMBDA_PROTO_UMAP_PRECON) | {
        # every non-cohesion loss term is ZERO unless explicitly parameterized
        # [D44] community/UMAP loss -- raw magnitude measured on pancreas
        # (epoch_trace of the [D43] magnitude run): umap=2.62 vs cohesion loss
        # side 4.04*(1-0.128)+11.99*(1-0.057)=14.84, so lam_umap=1.0 makes it
        # ~15% of the total loss (anchor, not competitor) AND is verbatim the
        # submitted NeurIPS value that produced purity 0.98 on this dataset.
        "lambda_umap": lam_umap,
        # [D43] proto_recon: each prototype is decoded SEPARATELY INTO EVERY
        # BATCH it spans (decoded_all[cond]) and must reconstruct its members
        # there. So a prototype whose batch-A block and batch-B block are
        # DIFFERENT cell types cannot satisfy both -> a direct penalty on
        # GROUP-LEVEL cross-batch mispairing, which is the measured failure
        # (7.6% of cell mass on pancreas) and exactly cohesion's blind spot
        # (0.00% penalty, measured). CVAE recon cannot do this: it is per-cell
        # and never asks which cells share a prototype.
        "lambda_proto_recon": lam_proto_recon,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": lam_recon,   # [D43] CVAE reconstruction, see below
        "proto_reinit_every": 0,
        # the objective
        "lambda_cohesion": lam_gram,        # gram, 2-hop
        "lambda_cohesion_aff": lam_aff,     # aff,  1-hop
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,               # sim(c,c) = 1
        "cohesion_self": 1.0,               # ALWAYS 1.0 (CLAUDE.md 9d)
        "cohesion_global": 2,               # [D29] full-graph memory
        "coh_mem_init": "full",             # [D32]
        "coh_mem_hlife": 500.0,             # [D31]
        "latent_dims": 8,
    }


def validate(lc, lam_gram, lam_aff, span_aff, span_gram):  # noqa: C901
    """Every check that must hold before a GPU is touched. Returns a summary."""
    for k in ZERO_TERMS:
        assert float(lc[k]) == 0.0, f"{k} must be 0, got {lc[k]}"
    assert float(lc["lambda_recon"]) >= 0.0
    assert float(lc["lambda_proto_recon"]) >= 0.0
    assert float(lc["lambda_umap"]) >= 0.0
    assert lc["lambda_cohesion"] > 0 and lc["lambda_cohesion_aff"] > 0
    assert lc["cohesion_self"] == 1.0, "cohesion_self must be 1.0"
    assert lc["cohesion_diag"] == 1.0, "cohesion_diag must be 1.0"
    assert lc["cohesion_global"] == 2 and lc["coh_mem_init"] == "full"
    assert lc["latent_dims"] == 8, f"latent_dims must be 8, got {lc['latent_dims']}"
    auth = lam_gram * span_gram + lam_aff * span_aff
    assert abs(auth - AUTHORITY) < 0.03, f"authority drifted: {auth}"
    return {
        "config_tag": CONFIG_TAG,
        "graph": HELDOUT,
        "authority": round(auth, 4),
        "all_zero_terms": {k: float(lc[k]) for k in ZERO_TERMS},
        "objective": {"lambda_cohesion(gram)": lc["lambda_cohesion"],
                      "lambda_cohesion_aff": lc["lambda_cohesion_aff"]},
        "cohesion_self": lc["cohesion_self"],
        "cohesion_diag": lc["cohesion_diag"],
        "cohesion_global": lc["cohesion_global"],
        "coh_mem_init": lc["coh_mem_init"],
        "coh_mem_hlife": lc["coh_mem_hlife"],
        "latent_dims": lc["latent_dims"],
        "lambda_recon": lc["lambda_recon"],
        "lambda_proto_recon": lc["lambda_proto_recon"],
        "lambda_umap": lc["lambda_umap"],
    }


@app.function(image=image, timeout=900, volumes={"/vol": volume})
def probe(lam_gram: float = 4.04, lam_aff: float = 11.99,
          span_aff: float = 0.064251, span_gram: float = 0.190749,
          lam_recon: float = 0.0, lam_proto_recon: float = 0.0,
          lam_umap: float = 0.0):
    """Return the validated config the DEPLOYED code builds. CPU-only, ~free.

    Run after every deploy and BEFORE spawning. This is the check that catches a
    stale deployment; without it, the wrong config produces plausible-looking
    results and nothing in the run status reveals it.
    """
    import os, sys
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    return validate(build_config(lam_gram, lam_aff, lam_recon,
                                 lam_proto_recon, lam_umap),
                    lam_gram, lam_aff, span_aff, span_gram)


@app.function(image=image, gpu="A100", timeout=21600, volumes={"/vol": volume})
def run(ds: str, K: int, lam_gram: float, lam_aff: float, gamma: float,
        span_aff: float, span_gram: float, lam_recon: float = 0.0,
        freeze_be: bool = False, epochs: int = 10, suffix: str = "",
        lam_proto_recon: float = 0.0, lam_umap: float = 0.0):
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

    from interpretable_ssl.experiments.tasks import run_mc_task

    name = f"d42{CONFIG_TAG}{suffix}_{ds}"
    lc = build_config(lam_gram, lam_aff, lam_recon, lam_proto_recon, lam_umap)
    checked = validate(lc, lam_gram, lam_aff, span_aff, span_gram)
    print(f"   [CONFIG VERIFIED IN-RUN] {checked}")

    print("\n" + "=" * 78)
    print(f"=== {name}   K={K}  graph={HELDOUT}  LD=8")
    print(f"    CLAUDE.md 9d cohesion config: community OFF, nassoc OFF, "
          f"modularity OFF, usage OFF -- cohesion is the ONLY objective")
    print(f"    lambda_cohesion(gram)={lam_gram:.2f}  "
          f"lambda_cohesion_aff={lam_aff:.2f}  authority={checked['authority']}")
    print(f"    cohesion_self=1.0 cohesion_diag=1.0 (unchanged); multi-batch "
          f"cohesion is unconditional -> V[b]")
    print("=" * 78)

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=epochs, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=HELDOUT, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False, freeze_batch_embedding=freeze_be,
    )
    print(f"\n[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    print(f"[{name}] TERM MAGNITUDES for lambda calibration: "
          f"recon={metrics.get('recon')}  cohesion={metrics.get('cohesion_term')}  "
          f"cohesion_aff={metrics.get('cohesion_aff_term')}  loss={metrics.get('loss')}")
    volume.commit()
    return name, trainer.get_model_name(), metrics

"""[D30] LOWER THE FOUNDER GATE: cohesion_self = 0.5, on top of the D29 remembered
profiles. The one-line story of every run so far:

    d26  no self          -> nothing pays a cell to stay anywhere -> 120-150 dead, churn
    d27  self = 1, batch  -> founder bonus 8.6x inflated by the batch artifact -> tiny
    d28  self = 1, global -> artifact removed, invisible protos mute -> MORE dead (134)
    d29  self = 1, remembered profiles -> profiles visible, but the founder gate is
         sim > 0.5 and the MEDIAN cell's best-neighbour cosine is 0.472 -> founders
         still cannot recruit -> still 111-146 dead at epoch 4

THE GATE. A singleton scores exactly cohesion_self (its only pair is itself). A cell
joining it gains 2*sim*a against a dilution of cohesion_self*a, so it joins iff
sim > cohesion_self/2. Measured best-neighbour gram cosine on bk02: p5 0.312, p25 0.391,
p50 0.472, p75 0.569 -- at a gate of 0.5, 58% of cells can NEVER follow any founder.
cohesion_self = 0.5 puts the gate at 0.25, below the p5, so ~97% of cells can follow
their best-matching founder while unrelated cells (typical cosine ~0.08) are still
rejected 3x over. The large-m join rule (sim > sbar/2) is untouched, and a live
singleton is still worth 0.5 > 0 against an empty slot, so the anti-collapse remains.
Derived from the measured graph, not swept (CLAUDE.md 9b). Verified: w=1/w=0 reproduce
the old branches exactly; founder-join gradients match (2*sim - w) - w on every pair.

LAMBDAS UNCHANGED: the self part contributes K_live/n to BOTH ends of the span (leiden
and random both have ~200 live prototypes), so it cancels and the measured spans -- and
the authorities -- carry over as they are.

--- the D29 header follows ---

[D29] REMEMBER each prototype's profile, instead of only what the batch shows.

THE IDEA (user's). Keep every cell's last-known assignment (N x K). Each step, swap this
batch's rows for the live gradient-carrying ones and leave the rest as detached context:

    S_eff = S_remembered, with the batch's rows replaced

Then both similarities become exact full-graph quantities, computed without needing any
of a prototype's members to be in the batch:

    aff   numer[c][k] = sum over ALL j of aff[c][j]*assign[j][k]   = (A_rows @ S_eff)[c][k]
    gram  numer[c][k] = dot( R[c], profile[k] ),  profile[k] = sum over ALL j of
                        assign[j][k]*R[j]                          = (R_rows @ (R^T S_eff))

The inner (R^T @ S_eff) IS the per-prototype affinity profile -- one length-N vector per
prototype, summed over its members and weighted by assignment. size[k] = S_eff.sum(0) is
then EXACT, not an EMA.

WHY THIS BEATS [D28]. D28 divided by a global size, so an invisible prototype gave
numerator 0 -> the cell earned nothing ("no evidence, no reward"). Correct but mute. Here
the profile survives from earlier batches, so the cell gets a REAL similarity. Verified
on s28nsc/bk02: the smallest prototype has 1 cell and 0 in the batch, yet its profile
norm is 1.0000 (remembered) and cells score 0.0015 against it -- not 0, not the phantom
1.0 the original loss handed out.

EXACTNESS. Both numerators match the full-graph truth to 4.6e-05 / 3.8e-05 (float32
noise on 58k-length dot products). No p-rescaling is needed anywhere, because nothing was
subsampled away.

GRADIENT. Non-batch rows are detached -- they are context. So a cell moves under its own
gradient while its prototype-mates hold still. That is the classic Lloyd/EM assignment
step, and it is a deliberate change to the optimisation, not only to the estimate.

COST / RISK. ~47MB for the remembered table plus one extra sparse matmul per step. The
risk is STALENESS: a non-batch row is as old as the last batch that contained it, about
8.6 steps here. If it matters it shows up as instability, and the fix is a periodic full
refresh (one forward pass per epoch).

Lambdas unchanged: the spans were always measured on the full graph, and this computes on
exactly that scale.

--- the D28 header follows ---

[D28] The d27 arms again, but with the objective estimated at FULL-GRAPH scale.

WHY RERUN. A training step sees only p = n_batch/N = 6757/58423 = 0.116 of the cells,
and that broke the estimate in two ways -- both measured, both fixed by
cohesion_global=1:

 (1) INVISIBLE PROTOTYPES. A prototype with S cells is absent from a batch with
     probability (1-p)^S: 29% at S=10, 54% at S=5, 88% at S=1. The loss cannot tell an
     invisible prototype from an empty one, so a cell joining it was scored as if it were
     FOUNDING one -- within = a*a over size = a, gradient 1.0, the MAXIMUM -- when the
     truth is its mean similarity to members it cannot see, about 0.08. A 12x
     overpayment, aimed precisely at the small prototypes. On the real d27_A partition:
     11.1 of 200 prototypes invisible in a typical batch, 35 invisible in >10% of them.
     The even-sized modularity baseline never hits this at all (P = 4e-16 at S=288).

 (2) SCALE MISMATCH. within[k] mixes a PAIR count (cross terms, needing BOTH cells in
     the batch, surviving at rate p*p) with a CELL count (self terms, one per cell,
     surviving at rate p). Dividing both by size[k] leaves cross unbiased but inflates
     self by 1/p = 8.6x. Measured: the self term is 4.3% of the full-graph objective but
     24.9% of the batch objective -- so training pushed toward small prototypes ~9x
     harder than the objective actually asks. Same class as the possible_pairs bug
     (CLAUDE.md 9b): a quadratic and a linear quantity compared across scales.

THE FIX divides by a DETACHED global size (EMA of batch_size/p) and rescales each part
to its full-graph rate (self/p, cross/p*p). Verified: batch estimate 1.25x the
full-graph truth -> 1.004x, and the invisible-prototype gradient 1.000 -> 1.7e-06.

WHY IT MATTERS FOR THE CONCLUSION. At FIXED K=200 the objective already prefers 200 real
groups over slivers by 6.5x (measured: 0.0816 with 200 groups vs 0.0126 with 190
singletons). So the sliver behaviour in d27 was NOT what the objective wants -- it was
the batch distortion. These arms test the intended objective for the first time.

The lambdas are unchanged and are now exactly right: the spans were always measured on
the FULL graph, and the loss finally computes on that same scale.

--- the d27 header follows ---

[D27] SELF-SIMILARITY AS THE ANTI-COLLAPSE. Does sim[c][c]=1 stop the churn?

THE QUESTION
------------
d26 (community + cohesion, pcex, self EXCLUDED) churned badly: unused_proto sawtoothed to
120-150 every two epochs, revivals pinned at ~170, purity 0.8066 against the modularity
baseline's 0.8686. These arms change exactly ONE thing -- put the self-pair back -- and
ask whether the collapse goes away.

WHAT WE COMPUTE, IN BOTH MODES
------------------------------
  GRAM  descriptor[c] = affinity_row[c] + cohesion_diag * e_c      closed neighbourhood
        v[c]          = descriptor[c] / norm(descriptor[c])
        sim[c][j]     = dot(v[c], v[j])             -> sim[c][c] = 1 EXACTLY (measured)

  AFF   sim[c][j]     = affinity[c][j]   for c != j
        sim[c][c]     = cohesion_diag               -> [D27] was 0, now 1

  COHESION ('perc' with cohesion_self=1: self in BOTH numerator and divisor)
        numer[c][k] = sum over ALL j, INCLUDING j == c, of sim[c][j]*assign[j][k]
        denom[k]    = sum over ALL j, INCLUDING j == c, of assign[j][k]   = size[k]
        cohesion    = (1/n_cells) * sum over c,k of assign[c][k]*numer[c][k]/denom[k]
        loss        = 1 - cohesion

[D27] fixed the AFF mode. The raw affinity carries setdiag(0) -- the no-self-loop
convention every graph-cut consumer needs -- so sim[c][c] was 0 there and cohesion_self
could not act on the 1-hop term at all. No second matrix is needed: sim[c][c]=d
contributes exactly d*assign[c][k] to numer, i.e. one elementwise add. Verified against a
literal double-loop implementation (|diff| 0.00e+00) and a singleton scores exactly 1.0.
The zero-weight diagnostics keep the RAW zero-diagonal affinity, since they are graph-cut
quantities and must stay comparable with the nassoc numbers.

SEPARATE PATHS. lambda_cohesion drives the term named by cohesion_sim; lambda_cohesion_aff
is an INDEPENDENT additive 1-hop term. So the aff arm below sets lambda_cohesion=0 and
drives everything through lambda_cohesion_aff -- the 1-hop term runs completely on its own,
without touching the gram path (which is then never even computed).

WHY THIS IS AN ANTI-COLLAPSE TERM, AND WHAT SHAPE IT HAS
--------------------------------------------------------
For a hard assignment in a prototype of size m the self term is a bonus of exactly 1/m per
cell, so every prototype contributes exactly 1 and the whole self part equals
K_live/n_cells (verified exactly: 0.125 == 5/40). A live prototype therefore ALWAYS beats
an empty one -- that is the anti-collapse. Measured share of the GRADIENT by size:

    m          1     2     3     8    12    50   150
    share   111%  108%   92%   58%   63%   50%   13%

so it is everything below ~3 cells and nearly nothing by 150: pressure exactly where
prototypes die, invisible where they are healthy. Measured share of the VALUE at these
settings is 4.4% (gram) / 9.9% (aff) -- NOT the 71% in decisions.md D17, which was
n_batch=1024 under the old 'glob' scaling (the self part scales as K_live/n).

THE RISK THIS RUN EXISTS TO TEST
--------------------------------
A singleton scores exactly 1, the MAXIMUM, so adding a second cell DILUTES it:
    d(score)/d(mass) at m=1     ~ 2*sim - 1      ~ -0.84   (repels)
    d(score)/d(mass) at large m ~ 2*sim - avgsim ~ +0.08   (attracts)
Empty slots attract (+3.4e-06 measured) but m=1 repels, so m=1 may be a stable trap --
mechanically the same sliver failure as the usage loss (median 86 -> 11, commit 69c9f64).
READ MEDIAN METACELL SIZE FIRST. That is the metric that caught the usage loss.

LAMBDAS -- measured, not swept (same method as [D26])
-----------------------------------------------------
span = value(good partition) - value(random) = how much the term can actually MOVE. The
loss VALUE is the wrong thing to match, because 1-cohesion is nearly a constant. Measured
on s28nsc/bk02, full graph, hard assignments, WITH sim[c][c]=1:

    term            leiden      bad30     random       SPAN
    gram self=1   0.081618   0.045014   0.004842    0.076776
    aff  self=1   0.038446   0.021620   0.003873    0.034573

Both rank leiden > bad30 > random, so both carry real signal. Holding the SAME total
authority as d26_A (20.0 * 0.077017 = 1.5403):

    gram : 1.5403 / 0.076776 = 20.1
    aff  : 1.5403 / 0.034573 = 44.6

so the two arms are comparable to each other AND to d26_A, which differed ONLY in the
self term.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_cohesion_d30.py
Cost: ~21 min per arm on a T4, in parallel. 2 arms ~ $0.28.
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

SPAN_GRAM_SELF = 0.076776       # measured, see the header
SPAN_AFF_SELF = 0.034573
AUTHORITY = 20.0 * 0.077017     # 1.5403 -- d26_A's total, held fixed

# (name, which similarity, lambda_gram, lambda_aff)
ARMS = [
    ("d30_A_aff",  "aff",  0.0, round(AUTHORITY / SPAN_AFF_SELF, 1)),    # 44.6
    ("d30_B_gram", "gram", round(AUTHORITY / SPAN_GRAM_SELF, 1), 0.0),   # 20.1
    # BOTH: same TOTAL authority as the two single arms, split 50/50 between the 2-hop
    # and 1-hop terms, so the comparison is "does mixing help" and not "is more cohesion
    # better" -- which two freely chosen lambdas would confound.
    ("d30_C_both", "both",
     round(0.5 * AUTHORITY / SPAN_GRAM_SELF, 1),                              # 10.0
     round(0.5 * AUTHORITY / SPAN_AFF_SELF, 1)),                              # 22.3
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, which: str, lam_gram: float, lam_aff: float):
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
        # --- everything except community + cohesion is OFF ---
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        # --- ONE similarity per arm. The aff arm drives lambda_cohesion_aff with
        #     lambda_cohesion=0, so the 1-hop term runs on its own SEPARATE path and the
        #     gram descriptor is never even computed. d26 showed the two are largely
        #     redundant (adding aff cost 0.058 purity for +0.011 niche purity).
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "cohesion_sim": "gram",
        "cohesion_form": "perc",          # [D27] self in BOTH numerator and divisor
        "cohesion_row_norm": "l2",        # degree-neutral (+0.044 vs +0.464 raw)
        "cohesion_diag": 1.0,             # sim[c][c] = 1 in BOTH modes
        "cohesion_self": 0.5,             # [D30] THE change under test: founder gate 0.25
        "cohesion_global": 2,             # [D29] THE change under test: remember profiles
        # --- reinit is the only dead-slot repair here; 'wrst' is what the best measured
        #     arm used ([D20] records 'splt' as neutral-to-harmful) ---
        "proto_reinit_every": 2,
        "proto_reinit_rel": 0.1,
        "proto_reinit_target": "wrst",
        "latent_dims": 16,
    }
    assert lc["lambda_modularity"] == 0.0 and lc["lambda_nassoc"] == 0.0
    assert lc["lambda_proto_usage"] == 0.0
    assert lc["cohesion_diag"] == 1.0 and lc["cohesion_form"] == "perc"
    assert lc["cohesion_self"] == 0.5 and lc["cohesion_global"] == 2
    if which != "both":
        assert (lam_gram > 0) != (lam_aff > 0), "exactly one similarity per arm"
    else:
        assert lam_gram > 0 and lam_aff > 0, "the both-arm needs both lambdas"

    auth = lam_gram * SPAN_GRAM_SELF + lam_aff * SPAN_AFF_SELF
    lam = f"gram={lam_gram} aff={lam_aff}"
    print(f"\n{'='*78}\n=== {name}\n"
          f"    community + cohesion ONLY. modularity/nassoc/usage OFF, reinit wrst/2.\n"
          f"    sim={which}  {lam}  perc, cohesion_self=1, sim[c][c]=1.0\n"
          f"    AUTHORITY (lambda * measured span) = {auth:.4f}  "
          f"(target {AUTHORITY:.4f}, same as d26_A)\n"
          f"    WATCH 1: MEDIAN metacell size -- the usage loss died here (86 -> 11)\n"
          f"    WATCH 2: unused_proto per epoch -- d26 sawtoothed to 120-150\n"
          f"    WATCH 3: [reinit] revived N -- d26 was pinned at ~170 every time\n"
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
    print("[D30] founder gate 0.25 + remembered profiles + sim[c][c]=1, 'perc', self in num AND div")
    print("=" * 104)
    print(f"{'arm':<18}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<18}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 104)
    print("REFERENCES (same dataset / graph / K / community term)")
    print(f"{'d26_A pcex gram':<18}{0.8066:>10.4f}{0.4670:>11.4f}{0.5023:>12.4f}"
          f"{0.722:>10.3f}   self EXCLUDED -- churned, 120-150 unused")
    print(f"{'d26_B pcex both':<18}{0.7485:>10.4f}{0.4778:>11.4f}{0.4483:>12.4f}"
          f"{0.667:>10.3f}   self EXCLUDED, + the 1-hop term")
    print(f"{'modularity l=10':<18}{0.8686:>10.4f}{0.4939:>11.4f}{0.3087:>12.4f}"
          f"{'--':>10}   K_live 200, median 288  <- baseline")
    print(f"{'modu + coh l=20':<18}{0.8710:>10.4f}{0.4963:>11.4f}{0.3063:>12.4f}"
          f"{'--':>10}   K_live 200, median 293  <- best so far")
    print("\nRead MEDIAN SIZE first (the usage loss died on it: 86 -> 11), then K_live,")
    print("then compare against d26_A -- which differs ONLY in the self term.")

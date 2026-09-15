"""TWO DIAGNOSTICS + TRAINING AT THE CORRECTED GAMMA. One launch.

WHY THIS RUN EXISTS
-------------------
SEACells uses the SAME K=200 on the SAME 58,423 cells, so it faces the identical
size floor (N/K = 292 cells/metacell) -- and it still reaches DGE 0.849 / 19 states /
internal_fraction 0.4465. That REFUTES "we need more prototypes". K=200 is provably
enough; the gap is the loss or the assignment head, nothing else.

Arithmetic done beforehand from measured quantities (m = total graph weight = 770,420,
validated: it reproduces d7fix's possible_pairs count to 0.05%):

    cpm_total = internal_fraction * m  -  gamma * total_possible_pairs

    partition              int_frac   edges_inside   total_possible_pairs
    SEACells                0.4465      343,992          11,653,000
    scProto L16_g015_uOFF   0.2926      225,425           8,577,000

    gamma = 0.024 :  scProto  +19,577   SEACells  +64,320   -> SEACells wins 3.3x
    gamma = 0.039 :  BREAK-EVEN
    gamma = 0.15  :  scProto -1,061,125 SEACells -1,403,958 -> scProto "wins" by 343k

=> Above gamma ~0.039 OUR OWN LOSS RANKS SEACELLS' (BETTER) PARTITION LOWER. Every gamma
   swept so far at latent 16 (0.15, 0.20) sat in that misspecified regime. Below 0.039 the
   loss prefers the right answer and the optimizer simply never found it.

WHAT THIS SCRIPT DOES
---------------------
A) VERIFY the crossover directly from raw label vectors instead of summary statistics.
B) PROBE whether our architecture can even EXPRESS SEACells' partition. Our head is
   softmax over dot(z, prototype_p) -- a linear softmax with no bias -- so the metacells
   it can represent are exactly convex polytopes in 16-d latent space. Fitting multinomial
   logistic regression (fit_intercept=False) from our z to SEACells' 200 labels is an
   EXACT expressivity test. IN-SAMPLE fit is the correct measure here: the question is
   representational capacity, not generalization.
       probe reaches int_frac ~0.45  -> architecture is fine, LOSS/optimizer is the problem
       probe caps out near ~0.29     -> a 200-way linear softmax in 16-d CANNOT express it,
                                        no amount of loss tuning will get there
C) TRAIN latent 16, usage OFF, at gamma in {0.020, 0.026, 0.032} -- the first time the
   satisfiable/correctly-specified window has been run at latent 16.
       satisfiable bound: gamma <= mean_wdeg * int_frac / (N/K - 1)
                                =  26.374 * 0.2926 / 291  =  0.0265
   0.020 is inside it, 0.026 sits exactly on it, 0.032 is just above it but still well
   below the 0.039 crossover.

NOTE: the 16-d pretrain checkpoint already exists from train_latent16_gamma_usage.py, so
no run here pays a fresh cvae_epochs=50.

NOTE (naming, CLAUDE.md section 9): lambda_proto_usage does NOT appear in the generated
model name (see the logged name ..._nagm0.2_nass-full_...), i.e. it is missing from
ABBREVIATIONS. The usage ON/OFF runs did not clobber each other only because their
experiment_name prefixes differ. Worth fixing separately; not fixed here so existing
folder names stay reproducible.

Run (detached -- a dropped local connection reaps a non-detached ephemeral app):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/diag_and_train_correct_gamma.py
Budget: ~$0.4 on a T4 (3 arms x ~400-700s + a few min of diagnostics).
volume.commit() after EVERY arm so a stop loses at most one.
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

# The reference run whose checkpoint supplies z for the probe, and whose labels are the
# scProto side of the crossover check. Best scProto run to date: DGE 0.834 / 16 states.
REF_RUN = dict(name="lf_L16_g015_uOFF", gamma=0.15, usage=0.0)

# gamma in the satisfiable + correctly-specified window (see header).
GAMMAS = [0.020, 0.026, 0.032]

GAMMA_GRID = [0.010, 0.015, 0.020, 0.024, 0.026, 0.030, 0.032, 0.035,
              0.039, 0.045, 0.05, 0.07, 0.10, 0.15, 0.20]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def diag_and_train():
    import os
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"

    import sys
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    import numpy as np, pandas as pd, scipy.sparse as sp, pickle
    import scanpy as sc
    from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.evaluation import niche_dge as nd
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        labels_from_trainer, joint_purity_per_metacell, collapse_summary,
    )

    ds_conf = DATASETS["s28nsc"]
    adata = sc.read_h5ad(ds_conf["path"])
    GT = os.path.join("/vol/code", "files", "celltype_niches_full_s28nsc.csv")
    ground_truth = nd.get_ground_truth(adata, GT, ct_key="celltypes", niche_key="niches_2D")
    TABLE_DIR = "/vol/code/notebooks/s28nsc_dge_lite/tables"
    os.makedirs(TABLE_DIR, exist_ok=True)

    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo()
    _m = float(_Atri.sum())
    _row, _col, _dat = _Atri.row, _Atri.col, _Atri.data
    print(f"[graph] m (total edge weight, upper triangle) = {_m:,.0f}")
    print(f"[graph] mean weighted degree = {2.0 * _m / adata.n_obs:.3f}")

    def partition_stats(label_ints):
        """edges_inside, sum(possible_pairs), internal_fraction, density, sizes.
        label_ints: int array over adata.obs_names order, -1 = unassigned."""
        li = np.asarray(label_ints, dtype=np.int64)
        same = (li[_row] == li[_col]) & (li[_row] >= 0)
        e = float(_dat[same].sum())
        sizes = np.bincount(li[li >= 0], minlength=max(int(li.max()) + 1, 1)).astype(float)
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        return {
            "edges_inside": e, "total_possible_pairs": pairs,
            "internal_fraction": e / _m, "aggregate_density": e / pairs if pairs else float("nan"),
            "n_alive": int(len(alive)),
            "median_size": float(np.median(alive)), "max_size": float(alive.max()),
            "size_ratio_max_over_median": float(alive.max() / np.median(alive)),
        }

    def to_ints(labels_series):
        lab = pd.Series(labels_series).reindex(adata.obs_names).to_numpy()
        ok = ~pd.isna(lab)
        li = np.full(len(lab), -1, dtype=np.int64)
        codes = pd.Categorical(pd.Series(lab[ok]).astype(str)).codes
        li[ok] = codes
        return li

    results = {"diagnostics": {}, "runs": {}}

    # ------------------------------------------------------------------ SEACells labels
    sea_li = None
    try:
        mdir = "/vol/models/s28nsc"
        cands = sorted(d for d in os.listdir(mdir)
                       if d.startswith("seacell") and
                       os.path.exists(os.path.join(mdir, d, "seacell_sc.h5ad")))
        print(f"[seacells] candidate dirs: {cands}")
        pick = next((d for d in cands if "bk02" in d), cands[0] if cands else None)
        print(f"[seacells] using: {pick}")
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import _reconstruct_from_delta
        sad = sc.read_h5ad(os.path.join(mdir, pick, "seacell_sc.h5ad"))
        if sad.uns.get("_seacell_delta"):
            sad = _reconstruct_from_delta(sad, os.path.join(mdir, pick))
        sea_lab = pd.Series(sad.obs["SEACell"].astype(str).values, index=sad.obs_names)
        sea_li = to_ints(sea_lab)
        st = partition_stats(sea_li)
        print(f"[seacells] {st}")
        results["diagnostics"]["seacells_stats"] = st
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[seacells] COULD NOT LOAD LABELS: {type(e).__name__}: {e}")
        print("[seacells] diagnostics A/B will be SKIPPED; training still runs.")

    # --------------------------------------------- reference scProto run: labels + latent z
    ref_li, ref_z = None, None
    try:
        lc_ref = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'full', 'nassoc_cpm_norm': 'gful',
            'nassoc_gamma': REF_RUN["gamma"], 'lambda_nassoc': 1000, 'nassoc_exact': 1,
            'lambda_proto_usage': REF_RUN["usage"], 'latent_dims': 16,
        }
        # find_metacells returns THREE values; its docstring says two (stale) -- use *_.
        tr_ref, *_ = find_metacells(
            "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
            batch_size=1024, affinity_type="bk02", num_prototypes=200,
            label_key="celltypes", niche_key="niches_2D",
            lambda_config=lc_ref, load_pretrain=True,
            load_umap=True, skip_eval=True,          # load checkpoint, touch nothing
            trainer_kwargs={"experiment_name": REF_RUN["name"]},
        )
        print(f"[ref] loaded {tr_ref.get_model_name()}")
        ref_li = to_ints(labels_from_trainer(tr_ref, adata).dropna())
        rt = partition_stats(ref_li)
        print(f"[ref] {rt}")
        results["diagnostics"]["ref_stats"] = rt

        # EXACTLY the path labels_from_trainer uses (collapse_diagnostics.py:205) so the
        # probe operates on the same z the real assignment head sees.
        import torch
        tr_ref.model.eval()
        with torch.no_grad():
            ref_z = tr_ref.encode_adata(adata, tr_ref.model, z_idx=1).cpu().numpy()
        print(f"[ref] latent z shape: {ref_z.shape}")
        # Does the real head carry a bias? fit_intercept must match it, or the probe is
        # not testing OUR architecture.
        _b = getattr(tr_ref.model.prototypes, "bias", None)
        HEAD_HAS_BIAS = _b is not None
        print(f"[ref] assignment head bias present: {HEAD_HAS_BIAS} "
              f"-> probe uses fit_intercept={HEAD_HAS_BIAS}")
        results["diagnostics"]["head_has_bias"] = bool(HEAD_HAS_BIAS)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[ref] COULD NOT LOAD: {type(e).__name__}: {e}")

    # ------------------------------------------------- DIAGNOSTIC A: the CPM crossover
    if sea_li is not None and ref_li is not None:
        s, r = results["diagnostics"]["seacells_stats"], results["diagnostics"]["ref_stats"]
        print("\n" + "=" * 78)
        print("DIAGNOSTIC A -- cpm_total = edges_inside - gamma * total_possible_pairs")
        print("  (computed from RAW LABEL VECTORS, not summary statistics)")
        print("=" * 78)
        print(f"{'gamma':>8}{'scProto':>16}{'SEACells':>16}{'our loss prefers':>20}")
        rows = []
        for g in GAMMA_GRID:
            cp_r = r["edges_inside"] - g * r["total_possible_pairs"]
            cp_s = s["edges_inside"] - g * s["total_possible_pairs"]
            pref = "SEACells" if cp_s > cp_r else "scProto (WRONG)"
            print(f"{g:>8.3f}{cp_r:>16,.0f}{cp_s:>16,.0f}{pref:>20}")
            rows.append({"gamma": g, "cpm_scproto": cp_r, "cpm_seacells": cp_s, "prefers": pref})
        denom = s["total_possible_pairs"] - r["total_possible_pairs"]
        cross = (s["edges_inside"] - r["edges_inside"]) / denom if denom else float("nan")
        print(f"\n>>> EXACT CROSSOVER gamma = {cross:.5f}")
        print(f">>> gamma BELOW it : loss prefers SEACells' partition -> optimizer/architecture failure")
        print(f">>> gamma ABOVE it : loss prefers OUR worse partition  -> MISSPECIFIED objective")
        results["diagnostics"]["crossover_gamma"] = float(cross)
        results["diagnostics"]["cpm_grid"] = rows

    # ----------------- DIAGNOSTIC B: can a linear softmax in 16-d express SEACells' answer?
    if sea_li is not None and ref_z is not None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import adjusted_rand_score
            print("\n" + "=" * 78)
            print("DIAGNOSTIC B -- expressivity probe: z (16-d) -> SEACells' 200 labels")
            print("  Our head is softmax(dot(z, prototype)) = linear softmax, no bias.")
            print("  fit_intercept=False matches it EXACTLY. In-sample fit is the correct")
            print("  measure: this asks what the architecture CAN represent, not how it")
            print("  generalizes.")
            print("=" * 78)
            ok = sea_li >= 0
            X, y = ref_z[ok], sea_li[ok]
            t0 = time.time()
            clf = LogisticRegression(
                fit_intercept=results["diagnostics"].get("head_has_bias", False),
                max_iter=300, n_jobs=-1)
            clf.fit(X, y)
            pred = clf.predict(X)
            acc = float((pred == y).mean())
            ari = float(adjusted_rand_score(y, pred))
            probe_li = np.full(len(sea_li), -1, dtype=np.int64)
            probe_li[ok] = pred
            pst = partition_stats(probe_li)
            print(f"[probe] fit in {time.time()-t0:.0f}s")
            print(f"[probe] in-sample accuracy vs SEACells labels : {acc:.4f}")
            print(f"[probe] ARI vs SEACells labels                : {ari:.4f}")
            print(f"[probe] partition it produces                 : {pst}")
            print(f"\n{'':<34}{'int_frac':>10}{'density':>10}{'n_alive':>9}")
            print(f"{'SEACells (target)':<34}{s['internal_fraction']:>10.4f}"
                  f"{s['aggregate_density']:>10.5f}{s['n_alive']:>9}")
            print(f"{'BEST LINEAR SOFTMAX ON OUR z':<34}{pst['internal_fraction']:>10.4f}"
                  f"{pst['aggregate_density']:>10.5f}{pst['n_alive']:>9}")
            print(f"{'what our trained loss achieved':<34}{r['internal_fraction']:>10.4f}"
                  f"{r['aggregate_density']:>10.5f}{r['n_alive']:>9}")
            gap_closed = ((pst["internal_fraction"] - r["internal_fraction"]) /
                          (s["internal_fraction"] - r["internal_fraction"]))
            print(f"\n>>> the probe closes {gap_closed*100:.0f}% of the internal_fraction gap.")
            print(">>> near 100% -> architecture is FINE, the loss/optimizer is at fault.")
            print(">>>  near 0%  -> a 200-way linear softmax in 16-d CANNOT express SEACells'")
            print(">>>              partition; the assignment head must change.")
            results["diagnostics"]["probe"] = {
                "accuracy": acc, "ari": ari, "partition": pst,
                "int_frac_gap_closed": float(gap_closed)}
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[probe] FAILED: {type(e).__name__}: {e}")

    # ------------------------------------------------ TRAIN at the corrected gamma window
    for gamma in GAMMAS:
        name = f"lf_L16_cg{str(gamma).replace('0.', '')}_uOFF"
        print(f"\n{'='*78}\n=== {name}  latent 16  gamma={gamma}  usage OFF  "
              f"(satisfiable bound 0.0265, crossover 0.039)\n{'='*78}")
        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'full', 'nassoc_cpm_norm': 'gful',
            'nassoc_gamma': gamma, 'lambda_nassoc': 1000, 'nassoc_exact': 1,
            'lambda_proto_usage': 0.0, 'latent_dims': 16,
        }
        try:
            trainer, metrics, *_ = find_metacells(
                "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="bk02", num_prototypes=200,
                label_key="celltypes", niche_key="niches_2D",
                lambda_config=lc, load_pretrain=True,
                skip_metrics=["task2", "task3"],
                trainer_kwargs={"experiment_name": name},
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            results["runs"][name] = {"gamma": gamma, "error": str(e)}
            volume.commit(); continue

        el = time.time() - t0
        print(f"[{name}] done in {el:.0f}s  model={trainer.get_model_name()}")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        st = partition_stats(to_ints(labels))
        pur = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        crow = collapse_summary(labels, name=name, purity=pur)
        dge = nd.run_niche_dge(name, labels, adata, ground_truth, TABLE_DIR,
                               ct_key="celltypes", niche_key="niches_2D", force=True)
        sm = dge["summary"].iloc[0]
        # [9b] the achieved density vs the gamma it was asked for -- does it track now that
        # gamma is inside the satisfiable window? Above 0.0265 it provably cannot.
        print(f"[{name}] [9b] asked density={gamma}  achieved={st['aggregate_density']:.5f}  "
              f"ratio={st['aggregate_density']/gamma:.2f}")
        print(f"[{name}] >>> DGE={sm['pearson_r']:.3f} n_pairs={sm['n_pairs']:.0f} "
              f"used={st['n_alive']}/200 density={st['aggregate_density']:.5f} "
              f"int_frac={st['internal_fraction']:.4f} med={st['median_size']:.0f} "
              f"max/med={st['size_ratio_max_over_median']:.1f} joint_pur={crow['purity_micro']:.3f}")
        results["runs"][name] = {"gamma": gamma, "elapsed": el, "metrics": metrics,
                                 "stats": st, "collapse": crow,
                                 "dge": dge["summary"].to_dict(orient="records")}
        volume.commit()

    print("\n\n===== CORRECTED-GAMMA SUMMARY =====")
    print(f"{'run':<22}{'gamma':>7}{'DGE':>8}{'n_pairs':>9}{'used':>8}{'density':>10}"
          f"{'int_frac':>10}{'med':>6}{'max/med':>9}{'jointP':>8}")
    for nm, rr in results["runs"].items():
        if "error" in rr:
            print(f"{nm:<22}{rr['gamma']:>7}   FAILED"); continue
        d = rr["stats"]; sm = rr["dge"][0]
        print(f"{nm:<22}{rr['gamma']:>7}{sm['pearson_r']:>8.3f}{sm['n_pairs']:>9.0f}"
              f"{str(d['n_alive'])+'/200':>8}{d['aggregate_density']:>10.5f}"
              f"{d['internal_fraction']:>10.4f}{d['median_size']:>6.0f}"
              f"{d['size_ratio_max_over_median']:>9.1f}{rr['collapse']['purity_micro']:>8.3f}")
    print(f"{'-- references --':<22}")
    print(f"{'SEACells bk02':<22}{'--':>7}{0.849:>8.3f}{19:>9.0f}{'200/200':>8}{0.02952:>10.5f}"
          f"{0.4465:>10.4f}{258:>6.0f}{3.6:>9.1f}{0.504:>8.3f}")
    print(f"{'lf_L16_g015_uOFF':<22}{0.15:>7}{0.834:>8.3f}{16:>9.0f}{'200/200':>8}{0.02628:>10.5f}"
          f"{0.2926:>10.4f}{293:>6.0f}{1.3:>9.1f}{0.438:>8.3f}")
    print(f"{'8-d g024 uOFF':<22}{0.024:>7}{0.821:>8.3f}{13:>9.0f}{'200/200':>8}{0.02449:>10.5f}"
          f"{0.3000:>10.4f}{268:>6.0f}{2.4:>9.1f}{float('nan'):>8.3f}")
    return results


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = diag_and_train.remote()
    print("wall (s):", time.time() - t0)
    import json; print(json.dumps(r, indent=2, default=str))

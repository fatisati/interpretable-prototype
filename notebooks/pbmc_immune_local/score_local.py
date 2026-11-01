"""Score local arms (and the pulled Modal runs) with the D63 metric cross-product:
cohesion aff/gram x {train,test} x {rare,common,all}; within/pooled/cross purity;
rare homogeneity / F1 / coverage. Needs the FULL-dataset arbf graph (built here if missing).
Adapted from modal_scripts/score_d63_budgets.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _local_env import *  # noqa
import numpy as np, pandas as pd, scanpy as sc
from interpretable_ssl.evaluation.metacell_scoreboard import cohesion_per_cell, rare_metrics_paper, load_affinity
from interpretable_ssl.augmenters.graph_generator import load_or_build_affinity

LOCAL = os.path.join(ROOT, "notebooks", "pbmc_immune_local", "tables")
MODAL = os.path.join(ROOT, "files", "volume_runs", DS)
MIN_CELLS = 3
ARMS = [  # name -> clusters_full.npz dir
    ("modal_epoch0_leid", os.path.join(MODAL, "d60_epoch0_leid")),
    ("modal_d60_leid_noc", os.path.join(MODAL, "d60_leid_noc")),
    ("modal_d63_mse-rec", os.path.join(MODAL, "d63_mse-rec")),
    ("modal_d63_mse-norec", os.path.join(MODAL, "d63_mse-norec")),
] + [(f"local_{a}_{tag}", os.path.join(LOCAL, a, tag))
     for a in sorted(os.listdir(LOCAL)) if os.path.isdir(os.path.join(LOCAL, a))
     for tag in ("epoch0", "final") if os.path.exists(os.path.join(LOCAL, a, tag, "clusters_full.npz"))]

ad = sc.read_h5ad(HVG_H5)
n = ad.n_obs
cts = np.asarray(ad.obs[LABEL_KEY]).astype(str); bts = np.asarray(ad.obs[BATCH_KEY]).astype(str)
bcodes = pd.factorize(bts)[0]
if 'X_pca' not in ad.obsm:
    sc.tl.pca(ad, n_comps=50)
A = load_or_build_affinity(ad, DS, "arbf", bk=BATCH_KEY, graph_dir=os.path.join(ROOT, "graphs"))
import scipy.sparse as sp
A = sp.csr_matrix(A)
names = np.asarray(ad.obs_names).astype(str)
d = np.load(SPLIT_NPZ, allow_pickle=True)
test = np.isin(names, [str(x) for x in d["test_ids"]]); train = ~test
freq = pd.Series(cts).value_counts(); qcut = np.quantile(freq.values, 0.25)
rare = np.isin(cts, list(freq[freq <= qcut].index))
sl = {"all": np.ones(n, bool), "rare": rare, "common": ~rare}; halves = {"train": train, "test": test}
print(f"{DS}: N={n:,} train={train.sum():,} test={test.sum():,} rare={rare.sum():,} ({100*rare.mean():.1f}%)")


def per_cell_indicators(lab, half):
    within = np.full(n, np.nan); pooled = np.full(n, np.nan); cross = np.full(n, np.nan)
    idx = np.where(half)[0]; l, c, b = lab[idx], cts[idx], bts[idx]
    for mc in np.unique(l):
        m = l == mc
        if m.sum() < MIN_CELLS: continue
        vals, cnt = np.unique(c[m], return_counts=True); gmaj = vals[np.argmax(cnt)]
        pooled[idx[m]] = (c[m] == gmaj).astype(float)
        for bb in np.unique(b[m]):
            s = m & (b == bb)
            if s.sum() < MIN_CELLS: continue
            v2, c2 = np.unique(c[s], return_counts=True); bmaj = v2[np.argmax(c2)]
            within[idx[s]] = (c[s] == bmaj).astype(float); cross[idx[s]] = float(bmaj == gmaj)
    return within, pooled, cross


rows, cache = [], {}
for arm, dirn in ARMS:
    p = os.path.join(dirn, "clusters_full.npz")
    if not os.path.exists(p): print(f"  --  {arm}: missing {p}"); continue
    z = np.load(p, allow_pickle=True); lab = z["labels"].astype(int)
    if "test_mask" in z and not np.array_equal(z["test_mask"], test):
        print(f"  !!  {arm}: its test_mask differs from the local split in {int((z['test_mask'] != test).sum())} cells")
    r = {"arm": arm, "n_mc": int(len(np.unique(lab)))}
    ca = cohesion_per_cell(lab, A, bcodes, sim="aff", _cache=cache); cg = cohesion_per_cell(lab, A, bcodes, sim="gram", _cache=cache)
    for hn, hm in halves.items():
        for sn, sm in sl.items():
            m = hm & sm; r[f"coh_aff_{hn}_{sn}"] = float(ca[m].mean()); r[f"coh_gram_{hn}_{sn}"] = float(cg[m].mean())
        w, po, x = per_cell_indicators(lab, hm)
        for sn, sm in sl.items():
            for mname, vec in (("within", w), ("pooled", po), ("cross", x)):
                r[f"{mname}_{hn}_{sn}"] = float(np.nanmean(vec[hm & sm]))
        rm = rare_metrics_paper(lab[hm], cts[hm], bts[hm])
        for k in ("rare_homogeneity_mean", "rare_f1_mean", "rare_coverage_mean"):
            if k in rm: r[f"{k}_{hn}"] = float(rm[k])
    rows.append(r)
    print(f"  OK  {arm:<28} coh_te {r['coh_aff_test_all']:.4f} (rare {r['coh_aff_test_rare']:.4f}) within_te {r['within_test_all']:.4f} cross_te {r['cross_test_all']:.4f} pooled_te {r['pooled_test_all']:.4f}")
df = pd.DataFrame(rows).set_index("arm"); df.to_csv(os.path.join(LOCAL, "scoreboard_local.csv"))
KEY = ["n_mc", "coh_aff_train_all", "coh_aff_test_all", "coh_aff_test_rare", "within_test_all", "within_test_rare",
       "cross_test_all", "pooled_test_all", "pooled_test_rare", "rare_homogeneity_mean_test", "rare_f1_mean_test", "rare_coverage_mean_test"]
print("\n=== KEY METRICS ===\n" + df[[k for k in KEY if k in df.columns]].round(4).to_string())

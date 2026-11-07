"""[D64] Score the NB arms on the TRAIN half.

The NB arms finished TRAINING but died in the trainer's internal scib eval, so
only the trainer's own clusters.npz (key 'assignments', 26,806 train cells) was
saved -- no held-out placement. Everything computable from train assignments is
computed here: cohesion, within/cross purity, batch-vote agreement, gene-space
signature correlation, batch entropy, rare metrics.

CAVEAT, stated in the output: the NB arms sit on an NB-pretrained encoder, a
different stage-1 latent from the MSE arms. Absolute values are comparable;
"delta vs its own baseline" is NOT available until an NB epoch-0 arm exists.
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from modal_scripts.score_all_baselines import DATASETS, load_labels  # noqa
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa
    cohesion_per_cell, rare_metrics_paper, load_affinity, read_obs)

DS, TAG = "pbmc-immune", "arbf_ctb20_seed0"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
NB = ["d64_leid-rec", "d64_leid-norec", "d64_wayp-rec", "d64_wayp-norec"]
REF = [("epoch0_leid", "d60_epoch0_leid"), ("leiden_st1", "d59_leiden_ctb"),
       ("d60_leid_noc", "d60_leid_noc"), ("d63_mse-rec", "d63_mse-rec")]


def main():
    n, h5, bk, lk, nk, K = DATASETS[DS]
    h5p = os.path.join(os.environ["DATA_DIR"], h5)
    # cohesion needs the full-dataset affinity graph; if it is not available
    # locally, every OTHER metric (which is what answers the cross-batch
    # question) is still computed.
    _gp = (f"{os.environ.get('GRAPHS_DIR','graphs')}/"
           f"affinity_{DS}{n}_ncomp50_kneighbors50_arbf.pkl")
    A = load_affinity(_gp) if os.path.exists(_gp) else None
    if A is None:
        print(f"NOTE: no affinity graph at {_gp} -- cohesion columns skipped")
    obs = read_obs(h5p, [bk, lk])
    cts = np.asarray(obs[lk]).astype(str)
    bts = np.asarray(obs[bk]).astype(str)
    bcodes = pd.factorize(bts)[0]
    import anndata
    names = np.asarray(anndata.read_h5ad(h5p, backed="r").obs_names).astype(str)
    d = np.load(os.path.join(REPO, "node_heldout_data", f"{DS}_{TAG}_split.npz"),
                allow_pickle=True)
    tid = set(str(x) for x in d["test_ids"])
    test = np.array([x in tid for x in names]); train = ~test
    freq = pd.Series(cts).value_counts()
    rare = np.isin(cts, list(freq[freq <= np.quantile(freq.values, 0.25)].index))
    nb_n = len(np.unique(bts)); logB = np.log(nb_n)

    # gene-space signatures need expression
    ad = anndata.read_h5ad(h5p)
    X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
    X = X.astype(np.float32)
    bg = {b: X[bts == b].mean(0) for b in np.unique(bts)}

    def rpearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        den = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a @ b) / den) if den > 0 else np.nan

    def score(lab):
        """lab: full-length, -1 where the cell has no assignment."""
        m0 = (lab >= 0) & train
        r = {}
        if A is not None:
            ca = cohesion_per_cell(lab.clip(min=0), A, bcodes, sim="aff")
            r["coh_train_all"] = float(ca[m0].mean())
            r["coh_train_rare"] = float(ca[m0 & rare].mean())
        # within / cross purity on the train half
        wi, cr, po = [], [], []
        agree = dis = 0
        for mc in np.unique(lab[m0]):
            mm = m0 & (lab == mc)
            if mm.sum() < 3:
                continue
            v, c = np.unique(cts[mm], return_counts=True)
            gmaj = v[np.argmax(c)]
            po.append(((cts[mm] == gmaj).mean(), mm.sum(), rare[mm].mean()))
            majs = []
            for b in np.unique(bts[mm]):
                s = mm & (bts == b)
                if s.sum() < 10:
                    continue
                v2, c2 = np.unique(cts[s], return_counts=True)
                bmaj = v2[np.argmax(c2)]
                majs.append(bmaj)
                wi.append(((cts[s] == bmaj).mean(), s.sum(), rare[s].mean()))
                cr.append((float(bmaj == gmaj), s.sum()))
            if len(majs) >= 2:
                agree += int(len(set(majs)) == 1); dis += int(len(set(majs)) > 1)
        def wmean(t, rare_only=False):
            if not t: return np.nan
            vals = np.array([x[0] for x in t]); w = np.array([x[1] for x in t], float)
            if rare_only:
                rr = np.array([x[2] for x in t]); k = rr > 0.5
                if k.sum() == 0: return np.nan
                vals, w = vals[k], w[k]
            return float(np.average(vals, weights=w))
        r["within_train_all"] = wmean(wi)
        r["within_train_rare"] = wmean(wi, True)
        r["pooled_train_rare"] = wmean(po, True)
        r["cross_train_all"] = float(np.average([x[0] for x in cr],
                                                weights=[x[1] for x in cr]))
        r["AGREE_pct"] = 100.0 * agree / max(agree + dis, 1)
        # batch entropy
        es, ws = [], []
        for mc in np.unique(lab[m0]):
            mm = m0 & (lab == mc)
            if mm.sum() < 3: continue
            _, c = np.unique(bts[mm], return_counts=True)
            p = c / c.sum(); es.append(-(p * np.log(p)).sum() / logB); ws.append(mm.sum())
        r["batch_entropy"] = float(np.average(es, weights=ws))
        # gene-space signature correlation across batches
        sig, vals = {}, []
        for mc in np.unique(lab[m0]):
            mm = m0 & (lab == mc)
            for b in np.unique(bts[mm]):
                s = mm & (bts == b)
                if s.sum() >= 10:
                    sig[(mc, b)] = X[s].mean(0) - bg[b]
        isrr = {}
        for mc in np.unique(lab[m0]):
            bs = [b for b in np.unique(bts) if (mc, b) in sig]
            if len(bs) < 2: continue
            v = [rpearson(sig[(mc, bs[i])], sig[(mc, bs[j])])
                 for i in range(len(bs)) for j in range(i + 1, len(bs))]
            mm = m0 & (lab == mc)
            isrr[mc] = (float(np.mean(v)), rare[mm].mean() > 0.5)
            vals.append(np.mean(v))
        r["sig_r_cross"] = float(np.mean(vals))
        rv = [x[0] for x in isrr.values() if x[1]]
        r["sig_r_cross_rare"] = float(np.mean(rv)) if rv else np.nan
        rm = rare_metrics_paper(lab[m0], cts[m0], bts[m0])
        for k_ in ("rare_homogeneity_mean", "rare_f1_mean", "rare_coverage_mean"):
            r[k_.replace("_mean", "")] = float(rm.get(k_, np.nan))
        return r

    rows = {}
    for a in NB:
        p = os.path.join(RUNS, f"{a}_nbtrain", "clusters.npz")
        if not os.path.exists(p):
            print(f"  -- {a}: missing"); continue
        asg = np.load(p, allow_pickle=True)["assignments"].astype(int)
        lab = np.full(n, -1, dtype=int)
        lab[train] = asg               # train adata preserves full-adata order
        rows[a] = score(lab)
        print(f"  OK {a}")
    for nm, folder in REF:
        lab = load_labels(os.path.join(RUNS, folder), n)
        if lab is None:
            print(f"  -- {nm}: missing"); continue
        rows[nm] = score(np.asarray(lab).astype(int))
        print(f"  OK {nm}")

    df = pd.DataFrame(rows).T
    out = os.path.join(REPO, "files", "tables", "d64_nb")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, f"{DS}_d64_train.csv"))
    print(f"\nsaved {out}/{DS}_d64_train.csv\n")
    print("TRAIN-HALF metrics (NB arms have no held-out placement yet)\n")
    print(df.round(4).to_string())


if __name__ == "__main__":
    main()

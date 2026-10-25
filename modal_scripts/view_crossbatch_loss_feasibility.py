"""Feasibility of the proposed cross-batch signature loss, measured not argued.

Proposed term, per prototype k spanning >= 2 batches:
    sig(k,b) = mean_expr(k's batch-b members) - mean_expr(all batch-b cells)
    penalty  = sum over batch pairs of ( 1 - pearson_r( sig(k,b1), sig(k,b2) ) )

The decisive question is NOT whether the quantity is meaningful at full-data
group sizes (view_proto_crossbatch_r.py already showed it is, r~0.70), but
whether it is estimable from the handful of cells a (prototype, batch) group
contributes to one MINIBATCH. This script measures the SPLIT-HALF RELIABILITY
of sig(k,b) as a function of the number of cells used:

    take a (k,b) group, split it into two DISJOINT random subsets of size m,
    build a signature from each, correlate them.

Both halves are the same cells, same batch, same prototype -- identical
biology by construction -- so this r is the CEILING the loss could ever see
between two different batches at that sample size. If it is ~0 at minibatch m,
the per-step term is noise and a running memory is mandatory, not optional.
"""
import os, sys
from collections import Counter
import numpy as np, pandas as pd, anndata

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
ARM = "d60_leid_noc"          # current best arm (cohesion + GM-KL + CVAE, no proto_recon)
SIZES = (3, 5, 10, 20, 40, 80)
RNG = np.random.default_rng(0)

n, h5, bk, lk, nk, K = DATASETS[DS]
ad = anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"], h5))
X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
X = np.asarray(X, dtype=np.float32)
cts = np.asarray(ad.obs[lk]).astype(str)
bts = np.asarray(ad.obs[bk]).astype(str)
batches = np.unique(bts)
print(f"{DS}: {X.shape[0]} cells x {X.shape[1]} genes, K={K}, {len(batches)} batches\n")

print("batch sizes")
for b in batches:
    print(f"  {b:<12} {int((bts == b).sum()):>7,}")
bg = {b: X[bts == b].mean(0) for b in batches}

lab = load_labels(os.path.join(RUNS, ARM), n)
print(f"\narm = {ARM}   n_prototypes_used = {len(np.unique(lab))}")


def r(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (da * db)) if da > 0 and db > 0 else np.nan


# ---- 1. occupancy: how big is a (prototype, batch) group, full data vs minibatch
groups = {}
for k in np.unique(lab):
    for b in batches:
        idx = np.where((lab == k) & (bts == b))[0]
        if len(idx):
            groups[(k, b)] = idx
sz = np.array(sorted(len(v) for v in groups.values()))
print(f"\n(prototype, batch) groups, FULL data: {len(sz)} non-empty")
print(f"  median {np.median(sz):.0f}   mean {sz.mean():.1f}   "
      f">=10 cells: {(sz >= 10).sum()}   >=3: {(sz >= 3).sum()}")

# what one minibatch sees. §9d: uniq cells per batch = batch_size*(1+neg_rate)*2, deduped
for uniq in (6700,):
    frac = uniq / n
    exp = sz * frac
    print(f"\none MINIBATCH (~{uniq:,} unique cells = {100*frac:.0f}% of data):")
    print(f"  expected cells per group: median {np.median(exp):.1f}  mean {exp.mean():.1f}")
    print(f"  groups expected to reach >=10 cells: {(exp >= 10).sum()} of {len(sz)}"
          f"   >=3 cells: {(exp >= 3).sum()}")

# ---- 2. split-half reliability of sig(k,b) vs number of cells
print("\nSPLIT-HALF RELIABILITY of sig(k,b)  (same cells, same batch, same prototype")
print("-> this is the CEILING for cross-batch r at that sample size)\n")
print(f"{'m cells/half':>13}{'n groups':>11}{'median r':>11}{'25th pct':>11}{'75th pct':>11}")
for m in SIZES:
    vals = []
    for (k, b), idx in groups.items():
        if len(idx) < 2 * m:
            continue
        for _ in range(5):                       # 5 random splits per group
            pick = RNG.permutation(idx)[:2 * m]
            s1 = X[pick[:m]].mean(0) - bg[b]
            s2 = X[pick[m:]].mean(0) - bg[b]
            v = r(s1, s2)
            if not np.isnan(v):
                vals.append(v)
    if vals:
        v = np.array(vals)
        print(f"{m:>13}{len(v)//5:>11}{np.median(v):>11.3f}"
              f"{np.percentile(v, 25):>11.3f}{np.percentile(v, 75):>11.3f}")

# ---- 3. the degenerate-minimum check: can the model escape by going batch-pure?
print("\nDEGENERATE MINIMUM CHECK")
print("the penalty only applies to prototypes spanning >=2 batches, so making a")
print("prototype batch-PURE sets its penalty to 0. How reachable is that?\n")
for minc in (3, 10):
    spans = []
    for k in np.unique(lab):
        nb = sum(1 for b in batches if ((lab == k) & (bts == b)).sum() >= minc)
        spans.append(nb)
    spans = np.array(spans)
    print(f"  at >= {minc:>2} cells/batch: prototypes spanning 1 batch = {(spans <= 1).sum()}"
          f" / {len(spans)}   median span = {np.median(spans):.0f} batches")
    # how much mass sits in the smallest batch of each spanning prototype
    drops = []
    for k in np.unique(lab):
        cnt = np.array([((lab == k) & (bts == b)).sum() for b in batches])
        live = cnt[cnt >= minc]
        if len(live) >= 2:
            drops.append(live.min() / cnt.sum())
    if drops:
        print(f"      fraction of a prototype's cells in its SMALLEST live batch:"
              f" median {np.median(drops):.3f}")
        print(f"      -> moving that fraction below threshold removes a batch pair"
              f" and its penalty entirely")

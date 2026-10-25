"""Is the cross-batch DE-signature correlation HIGH ENOUGH to be worth a loss?

The proposed term rewards  r( sig(k,b1), sig(k,b2) ).  It can only ever push a
prototype up to whatever a PERFECT prototype would score -- i.e. what the same
CELL TYPE scores across two batches.  If that ceiling is low, the loss has no
signal to chase and the whole idea dies cheaply, here, before any GPU time.

Measures, for three signature definitions:
    plain   sig(g,b) = mean lognorm over group - mean lognorm over batch b
    cdr     same, but on a matrix with cellular detection rate regressed out
            per batch per gene (CIDER's covariate), which also batch-centres it
    cdr+w   cdr, scored with a VARIANCE-TREND-WEIGHTED Pearson correlation
            (limma-trend's idea: a gene's variance read off a smooth
             variance-vs-mean curve, used to downweight noisy genes)

against three contrasts:
    SAME type, different batch    <- the ceiling a perfect prototype reaches
    DIFF type, different batch    <- the null; how low a chimera should score
    size-matched                  <- cell types are far bigger than prototypes,
                                     and r rises with group size, so the honest
                                     ceiling subsamples types to prototype size

and reports the current prototypes on the same scale for comparison.
"""
import os, sys
import numpy as np, pandas as pd, anndata

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels

DS = "pbmc-immune"
RUNS = os.path.join(REPO, "files", "volume_runs", DS)
MINC = 10
MATCH_SIZES = (10, 20, 30)
RNG = np.random.default_rng(0)

n, h5, bk, lk, nk, K = DATASETS[DS]
ad = anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"], h5))
X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
X = np.asarray(X, dtype=np.float32)
cts = np.asarray(ad.obs[lk]).astype(str)
bts = np.asarray(ad.obs[bk]).astype(str)
batches = np.unique(bts)
print(f"{DS}: {X.shape[0]:,} cells x {X.shape[1]:,} genes, {len(batches)} batches, "
      f"{len(np.unique(cts))} cell types\n")

# ---------- preprocessing: CDR residualization + variance trend, per batch ----------
cdr = (X > 0).sum(1).astype(np.float32)          # genes detected per cell
cdr = (cdr - cdr.mean()) / cdr.std()
print(f"cellular detection rate: mean {(X > 0).sum(1).mean():.0f} genes/cell, "
      f"range {(X > 0).sum(1).min()}-{(X > 0).sum(1).max()}")

R = np.zeros_like(X)                              # CDR-residualized, batch-centred
var_trend = {}                                    # per batch, per gene
for b in batches:
    m = bts == b
    Xb, cb = X[m], cdr[m]
    cb = cb - cb.mean()
    denom = float(cb @ cb)
    slope = (cb @ Xb) / denom if denom > 0 else np.zeros(Xb.shape[1], np.float32)
    R[m] = Xb - Xb.mean(0) - np.outer(cb, slope)  # residual, zero-mean within batch
    # variance trend: per-gene variance smoothed against mean expression
    v = R[m].var(0); mu = Xb.mean(0)
    order = np.argsort(mu); w = 201
    vs = pd.Series(v[order]).rolling(w, center=True, min_periods=1).median().to_numpy()
    tr = np.empty_like(vs); tr[order] = vs
    var_trend[b] = np.maximum(tr, 1e-8)
    frac = 1 - R[m].var(0).sum() / max(Xb.var(0).sum(), 1e-12)
    print(f"  {b:<10} n={int(m.sum()):>6,}  variance explained by CDR: {100*frac:5.2f}%")


def pear(a, b, w=None):
    if w is None:
        a = a - a.mean(); b = b - b.mean()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan
    sw = w.sum()
    a = a - (w @ a) / sw; b = b - (w @ b) / sw
    va, vb = float(w @ (a * a)), float(w @ (b * b))
    return float((w @ (a * b)) / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


def sigs(idx, b, mode):
    """signature of a cell group (indices idx, all inside batch b)."""
    if mode == "plain":
        return X[idx].mean(0) - X[bts == b].mean(0)
    return R[idx].mean(0)                          # already batch-centred


def score(i1, b1, i2, b2, mode):
    s1, s2 = sigs(i1, b1, mode), sigs(i2, b2, mode)
    if mode == "cdr+w":
        w = 1.0 / np.sqrt(var_trend[b1] * var_trend[b2])
        return pear(s1, s2, w)
    return pear(s1, s2)


# ---------- group index ----------
def group_index(labels):
    g = {}
    for t in np.unique(labels):
        for b in batches:
            idx = np.where((labels == t) & (bts == b))[0]
            if len(idx) >= MINC:
                g[(t, b)] = idx
    return g


gt = group_index(cts)
types = sorted({t for t, _ in gt})
print(f"\ncell-type groups with >={MINC} cells: {len(gt)}  over {len(types)} types")
print(f"  group sizes: median {np.median([len(v) for v in gt.values()]):.0f}  "
      f"mean {np.mean([len(v) for v in gt.values()]):.0f}")

MODES = ("plain", "cdr", "cdr+w")

# ---------- 1. SAME type across batches = the ceiling; DIFF type = the null -------
print("\n" + "=" * 78)
print("CEILING: same cell type, different batch   |   NULL: different type, diff batch")
print("=" * 78)
print(f"{'signature':<10}{'contrast':<22}{'n pairs':>9}{'median r':>11}"
      f"{'10th pct':>11}{'25th pct':>11}{'75th pct':>11}")
res = {}
for mode in MODES:
    same, diff = [], []
    keys = list(gt)
    for a in range(len(keys)):
        for c in range(a + 1, len(keys)):
            (t1, b1), (t2, b2) = keys[a], keys[c]
            if b1 == b2:
                continue
            v = score(gt[(t1, b1)], b1, gt[(t2, b2)], b2, mode)
            if np.isnan(v):
                continue
            (same if t1 == t2 else diff).append(v)
    res[mode] = (np.array(same), np.array(diff))
    for name, v in (("SAME type/diff batch", np.array(same)),
                    ("diff type/diff batch", np.array(diff))):
        print(f"{mode:<10}{name:<22}{len(v):>9}{np.median(v):>11.3f}"
              f"{np.percentile(v,10):>11.3f}{np.percentile(v,25):>11.3f}"
              f"{np.percentile(v,75):>11.3f}")
    s, d = res[mode]
    print(f"{'':<10}{'--> separation':<22}{'':>9}{np.median(s)-np.median(d):>11.3f}"
          f"   (same-median minus diff-median)")

# ---------- 2. size-matched ceiling: types subsampled to prototype size ----------
print("\n" + "=" * 78)
print("SIZE-MATCHED CEILING -- cell types subsampled to prototype-scale groups")
print("(prototype (k,batch) groups have median 10 cells, mean 24)")
print("=" * 78)
print(f"{'signature':<10}{'m cells/group':>15}{'n pairs':>10}{'median r':>11}"
      f"{'10th pct':>11}{'null med':>11}")
for mode in MODES:
    for m in MATCH_SIZES:
        same, diff = [], []
        keys = [k for k in gt if len(gt[k]) >= m]
        sub = {k: RNG.permutation(gt[k])[:m] for k in keys}
        for a in range(len(keys)):
            for c in range(a + 1, len(keys)):
                (t1, b1), (t2, b2) = keys[a], keys[c]
                if b1 == b2:
                    continue
                v = score(sub[keys[a]], b1, sub[keys[c]], b2, mode)
                if np.isnan(v):
                    continue
                (same if t1 == t2 else diff).append(v)
        if same:
            print(f"{mode:<10}{m:>15}{len(same):>10}{np.median(same):>11.3f}"
                  f"{np.percentile(same,10):>11.3f}{np.median(diff):>11.3f}")

# ---------- 3. where the prototypes actually sit, on the same scale ----------
print("\n" + "=" * 78)
print("WHERE OUR PROTOTYPES SIT (same signature definitions, same >=10 rule)")
print("=" * 78)
print(f"{'arm':<20}{'signature':<10}{'n protos':>10}{'median r':>11}"
      f"{'10th pct':>11}{'frac<0.3':>10}")
for arm in ("d60_epoch0_leid", "d60_leid_noc", "d63_mse-rec"):
    lab = load_labels(os.path.join(RUNS, arm), n)
    if lab is None:
        print(f"{arm:<20} labels not found"); continue
    gp = group_index(lab)
    for mode in MODES:
        per = []
        for k in np.unique(lab):
            bs = [b for b in batches if (k, b) in gp]
            if len(bs) < 2:
                continue
            vs = [score(gp[(k, b1)], b1, gp[(k, b2)], b2, mode)
                  for i, b1 in enumerate(bs) for b2 in bs[i + 1:]]
            vs = [v for v in vs if not np.isnan(v)]
            if vs:
                per.append(np.mean(vs))
        per = np.array(per)
        print(f"{arm:<20}{mode:<10}{len(per):>10}{np.median(per):>11.3f}"
              f"{np.percentile(per,10):>11.3f}{(per<0.3).mean():>10.1%}")

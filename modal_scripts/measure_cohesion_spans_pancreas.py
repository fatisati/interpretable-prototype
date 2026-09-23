"""[D41] Derive lambda_cohesion / lambda_cohesion_aff for PANCREAS, multi-batch.

CPU only, runs locally in ~1-2 min on the cached pancreas/arbf graph.

WHY A SEPARATE SCRIPT FROM measure_cohesion_spans.py
----------------------------------------------------
Spans are NOT portable across graphs (same reason nassoc_gamma is not,
CLAUDE.md 9b), so pancreas needs its own. But two further things differ here and
both change the number, so reusing the s28nsc spans would be wrong twice over:

  1. the graph is MASKED -- cross-batch affinity is zero ([D41] premise), which
     removes 11.6% of pancreas/arbf's edges and hits rare types hardest;
  2. the denominator is PER-BATCH, not global, so the value itself is on a
     different scale (a batch-mixed partition scores ~5.9x higher than under the
     global denominator).

This measures the span of the term EXACTLY as the trainer will compute it:
cohesion_form='perc', cohesion_self=0, cohesion_diag=1, coh_batch='locl'.

WHY cohesion_self=0 AND NOT 1 (the s28nsc arms used 1) -- measured, not assumed
------------------------------------------------------------------------------
The self-pair contributes 1/size_in_batch per cell. Under the GLOBAL denominator
that is 1/size, identical for a batch-pure and a batch-spread prototype, so it
carries no batch preference. Under the PER-BATCH denominator a batch-spread
prototype has size_in_batch = size/n_bio, so its self part is n_bio times
LARGER -- the term pays the model to smear every prototype thinly across
batches. Decomposed on this graph at K=220:

    partition     median size_b    TOTAL    self part   PAIR part
    leiden            191.0       0.13034    0.01343     0.11691
    random-220         10.0       0.12969    0.12007     0.00962

The pair part discriminates 12x; the self part hands random a 0.107 bonus that
cancels it, collapsing the span from 0.107 to 0.0006 and implying an absurd
lambda of 2396. So coh_batch='locl' REQUIRES cohesion_self=0. This is a scale
bug that going per-batch introduces (CLAUDE.md 9b: a term linear in cells
compared against a denominator that shrank by n_bio). cohesion_diag stays 1: it
defines the closed-neighbourhood gram descriptor, and for the aff term it is
added and then removed again, so it is a no-op there.

THE RULE (unchanged from [D26]/[D39])
-------------------------------------
    span[term] = value(good partition) - value(random partition)
    lambda[term] = AUTHORITY / span[term]

Weight by DYNAMIC RANGE, not by loss value. cohesion_aff sits near 0 (most
aff[c][j]=0), so 1-cohesion_aff ~ 0.998 is a near-CONSTANT: at equal lambda it
looks comparable in the loss while contributing almost nothing to the gradient.
Span is how much the term can actually MOVE the answer.

AUTHORITY = 1.5403 is held fixed from the s28nsc arms so that "does mixing the
two similarities help" is not confounded with "is more cohesion better".

Run:  python modal_scripts/measure_cohesion_spans_pancreas.py
"""
import os
import pickle
import time

import numpy as np
import scipy.sparse as sp

AUTHORITY = 1.5403          # held fixed across every arm, from the s28nsc runs
K = 220                     # pancreas num_prototypes (dataset_configs.py)
PKL = "graphs/affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl"
H5 = os.path.join(os.environ.get("DATA_DIR", ""), "pancreas_hvg.h5ad")
BATCH_KEY, LABEL_KEY = "tech", "celltype"

t0 = time.time()


def read_obs_cat(h5, key):
    """Read one categorical obs column straight from HDF5.

    Avoids anndata entirely: these files predate the current reader and carry
    `encoding-type: null` entries it has no reader for (CLAUDE.md 11).
    """
    import h5py
    with h5py.File(h5, "r") as f:
        g = f[f"obs/{key}"]
        cats = np.array([c.decode() if isinstance(c, bytes) else c
                         for c in g["categories"][:]])
        return cats[g["codes"][:]]


# ---------------------------------------------------------------- the graph ---
with open(PKL, "rb") as f:
    obj = pickle.load(f)
A = obj["affinity"] if isinstance(obj, dict) and "affinity" in obj else obj
if isinstance(obj, dict) and not sp.issparse(A):
    for k, v in obj.items():
        if sp.issparse(v):
            A = v
            break
A = sp.csr_matrix(A).astype(np.float32)
A = ((A + A.T) / 2.0).tocsr()
A.setdiag(0)
A.eliminate_zeros()
N = A.shape[0]

tech = read_obs_cat(H5, BATCH_KEY)
ct = read_obs_cat(H5, LABEL_KEY)
_, bcode = np.unique(tech.astype(str), return_inverse=True)
bcode = bcode.astype(np.int64)
n_bio = int(bcode.max()) + 1

coo = A.tocoo()
same = bcode[coo.row] == bcode[coo.col]
A_masked = sp.csr_matrix((coo.data[same], (coo.row[same], coo.col[same])),
                         shape=A.shape)
print(f"graph: N={N:,}  nnz={A.nnz:,}  bio batches={n_bio}")
print(f"[D41] masked: nnz {A.nnz:,} -> {A_masked.nnz:,} "
      f"({100.0*(A.nnz-A_masked.nnz)/A.nnz:.1f}% cross-batch dropped)  "
      f"({time.time()-t0:.0f}s)")


# ----------------------------------------------------------- the descriptors ---
def gram_rows(M, delta=1.0):
    """L2-normalised CLOSED-neighbourhood rows R, so Gram = R @ R.T (diag 1)."""
    cM = (M + delta * sp.eye(M.shape[0], format="csr", dtype=np.float32)).tocsr()
    n = np.sqrt(cM.multiply(cM).sum(axis=1)).A.ravel()
    n[n == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / n) @ cM)


def to_S(lab):
    _, codes = np.unique(lab, return_inverse=True)
    return sp.csr_matrix((np.ones(len(lab), np.float32),
                          (np.arange(len(lab)), codes)),
                         shape=(len(lab), codes.max() + 1))


def cohesion(S, sim, per_batch, cohesion_self=0.0):
    """The objective exactly as the trainer computes it, hard assignments.

        numer[c][p] = sum over j of sim[c][j]*S[j][p]      (+ self term for aff)
        den[c][p]   = size_in_batch[bio(c)][p]   if per_batch else size[p]
        value       = (1/N) * sum_c sum_p S[c][p] * numer[c][p] / max(den,1)

    sim='aff'  -> the masked affinity, with sim(c,c)=1 added (cohesion_diag=1,
                  since the stored matrix carries the setdiag(0) convention).
    sim='gram' -> cos of closed-neighbourhood rows; its diagonal is already
                  exactly 1 under the L2 norm.

    cohesion_self weights the i==j SELF-pairs in the numerator. It MUST be 0
    here -- see the header. At 1 the self part is 1/size_in_batch, which a
    batch-SPREAD partition maximises (size_in_batch = size/n_bio), so it pays
    the model to smear every prototype thinly across batches. Measured on this
    graph: random-220 scores 0.120 on the self part against Leiden's 0.013,
    cancelling a pair signal that discriminates 12x and collapsing the span
    from 0.107 to 0.0006.
    """
    M = A_masked
    Sd = S.toarray()
    if sim == "aff":
        numer = np.asarray((M @ S).todense()) + Sd      # cohesion_diag=1
    else:
        R = gram_rows(M, 1.0)
        numer = np.asarray((R @ (R.T @ S)).todense())   # gram diag is 1 already
    # drop (1 - cohesion_self) of the self-pairs; sim(c,c)=1 in both modes
    numer = numer - (1.0 - cohesion_self) * Sd
    if per_batch:
        size_bp = np.zeros((n_bio, S.shape[1]), np.float64)
        np.add.at(size_bp, bcode, S.toarray())
        den = size_bp[bcode]
    else:
        den = np.tile(np.asarray(S.sum(0)).ravel(), (N, 1))
    return float((Sd * (numer / np.maximum(den, 1.0))).sum() / N)


# ------------------------------------------------------------- the partitions ---
import igraph as ig
import leidenalg

src, dst = A_masked.nonzero()
m = src < dst
g = ig.Graph(n=N, edges=list(zip(src[m].tolist(), dst[m].tolist())))
g.es["weight"] = np.asarray(A_masked[src[m], dst[m]]).ravel().tolist()

# tune CPM resolution to land near K communities -- the span is defined at the
# K the run will actually use, and cohesion depends on K.
lo, hi, lab_leiden, best = 1e-4, 2.0, None, None
for _ in range(18):
    mid = (lo * hi) ** 0.5
    lab = np.asarray(leidenalg.find_partition(
        g, leidenalg.CPMVertexPartition, weights="weight",
        resolution_parameter=mid, seed=0).membership)
    nk = lab.max() + 1
    if best is None or abs(nk - K) < abs(best[1] - K):
        best, lab_leiden = (mid, nk), lab
    if nk < K:
        lo = mid
    else:
        hi = mid
print(f"Leiden: resolution={best[0]:.5f} -> {best[1]} communities "
      f"(target {K})  ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(0)
PARTS = [
    ("leiden",          lab_leiden),
    ("random-K",        rng.integers(0, K, N)),
    ("celltype",        ct),
    ("celltype x tech", np.array([f"{a}|{b}" for a, b in zip(ct, tech)])),
    ("one group",       np.zeros(N, np.int64)),
]

print("\n" + "=" * 86)
print(f"{'partition':<18}{'K':>6}{'aff [D41]':>12}{'gram [D41]':>12}"
      f"{'aff global':>12}{'gram global':>13}")
print("=" * 86)
vals = {}
for name, lab in PARTS:
    S = to_S(np.asarray(lab))
    row = {s + ("_d41" if pb else "_glob"): cohesion(S, s, pb)
           for s in ("aff", "gram") for pb in (True, False)}
    vals[name] = row
    print(f"{name:<18}{S.shape[1]:>6}{row['aff_d41']:>12.6f}{row['gram_d41']:>12.6f}"
          f"{row['aff_glob']:>12.6f}{row['gram_glob']:>13.6f}")

print("=" * 86)
print("\nSPAN = value(leiden) - value(random-K), and the lambda it implies")
print(f"AUTHORITY = {AUTHORITY} (held fixed from the s28nsc arms)\n")
spans = {}
for term in ("aff", "gram"):
    s = vals["leiden"][f"{term}_d41"] - vals["random-K"][f"{term}_d41"]
    spans[term] = s
    print(f"  {term:<5} span = {s:.6f}   ->  lambda_{term}_only = "
          f"{AUTHORITY/s:>8.2f}")
print(f"\n  span_gram / span_aff = {spans['gram']/spans['aff']:.3f}")
print("\nBOTH arm, authority split 50/50 (the split that won on s28nsc/bk02):")
print(f"  lambda_gram = {0.5*AUTHORITY/spans['gram']:.2f}   "
      f"lambda_aff = {0.5*AUTHORITY/spans['aff']:.2f}")
print(f"\n(check: 0.5*{AUTHORITY} split reproduces authority "
      f"{0.5*AUTHORITY/spans['gram']*spans['gram'] + 0.5*AUTHORITY/spans['aff']*spans['aff']:.4f})")
print(f"\ndone in {time.time()-t0:.0f}s")

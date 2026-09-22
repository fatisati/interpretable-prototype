"""[D26] Derive lambda_cohesion and lambda_cohesion_aff ANALYTICALLY from the graph.

CPU only, ~70s on s28nsc/bk02 (N=58,423). Run it again on any new graph -- these spans
are NOT portable across graphs, for the same reason nassoc_gamma is not (CLAUDE.md 9b).

MEASURED, s28nsc/bk02, K=200, pcex + l2, hard assignments on the full graph:
    term            leiden      bad30      random       SPAN
    gram delta=0   0.080057   0.042534   0.001492    0.078566
    gram delta=1   0.078437   0.041548   0.001419    0.077017
    aff            0.035154   0.018151   0.000448    0.034706
=> span_gram/span_aff = 2.219, so lambda_aff = 2.219*lambda_gram for equal authority.

The two terms must be weighted by their DYNAMIC RANGE, not by their loss value.
cohesion_aff sits near 0 (most aff[c][j]=0), so 1-cohesion_aff ~ 0.998 is a near
CONSTANT: at equal lambda it looks comparable in the loss while contributing almost
nothing to the gradient. Loss value is the wrong thing to match (CLAUDE.md 9b).

    span[term] = value(good partition) - value(random partition)

is how much the term can actually MOVE, so equal-span lambdas give the two terms
comparable authority over the answer.
"""
import pickle, time, numpy as np, scipy.sparse as sp, torch

P = "graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl"
t0 = time.time()
with open(P, "rb") as f:
    obj = pickle.load(f)
A = obj["affinity"] if isinstance(obj, dict) and "affinity" in obj else obj
if isinstance(obj, dict) and not sp.issparse(A):
    for k, v in obj.items():
        if sp.issparse(v):
            A = v; print("  using dict key:", k); break
A = sp.csr_matrix(A).astype(np.float32)
A = ((A + A.T) / 2.0).tocsr()          # symmetrise
A.setdiag(0); A.eliminate_zeros()
N = A.shape[0]
print(f"graph: N={N:,} nnz={A.nnz:,} mean_deg={A.sum()/N:.4f} "
      f"max_entry={A.data.max():.4f} ({time.time()-t0:.0f}s)")

# ---- descriptor matrices, exactly as scproto.py builds them ------------------
def gram_desc(delta):
    cA = A.copy()
    if delta: cA = cA + delta * sp.eye(N, format="csr", dtype=np.float32)
    n = np.sqrt(cA.multiply(cA).sum(axis=1)).A.ravel(); n[n == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / n) @ cA)

# ---- partitions --------------------------------------------------------------
K = 200
import igraph as ig, leidenalg
src, dst = A.nonzero(); m = src < dst
g = ig.Graph(n=N, edges=list(zip(src[m].tolist(), dst[m].tolist())))
g.es["weight"] = np.asarray(A[src[m], dst[m]]).ravel().tolist()
print(f"igraph built ({time.time()-t0:.0f}s), running Leiden...")
part = leidenalg.find_partition(g, leidenalg.CPMVertexPartition,
                                weights="weight", resolution_parameter=0.0069, seed=0)
lab_leiden = np.asarray(part.membership)
print(f"Leiden: {lab_leiden.max()+1} communities ({time.time()-t0:.0f}s)")

rng = np.random.default_rng(0)
lab_random = rng.integers(0, K, N)
# degraded: move 30% of cells to a random OTHER community
lab_bad = lab_leiden.copy()
sel = rng.choice(N, int(0.3 * N), replace=False)
lab_bad[sel] = rng.integers(0, lab_leiden.max() + 1, sel.size)

def to_S(lab):
    u = {v: i for i, v in enumerate(np.unique(lab))}
    S = sp.csr_matrix((np.ones(len(lab), np.float32),
                       (np.arange(len(lab)), [u[v] for v in lab])),
                      shape=(len(lab), len(u)))
    return S

# ---- cohesion under cohesion_form='pcex', on the FULL graph -------------------
# pcex: for each cell c,  sum_k S[c,k] * (sum_{j!=c} sim[c,j] S[j,k]) / (size[k]-S[c,k])
# Done sparsely: num = (Sim @ S) with the self term removed, den = size - S.
def cohesion(Sim, S, chunk=4000):
    size = np.asarray(S.sum(0)).ravel()
    tot = 0.0
    Sd = S.toarray() if S.shape[1] < 400 else None
    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        num = np.asarray((Sim[lo:hi] @ S).todense())          # (chunk, K)
        selfsim = np.asarray(Sim[lo:hi, lo:hi].diagonal()).ravel()
        Sb = Sd[lo:hi] if Sd is not None else np.asarray(S[lo:hi].todense())
        num = num - selfsim[:, None] * Sb                      # drop the self term
        den = np.maximum(size[None, :] - Sb, 1e-8)
        tot += float((Sb * (num / den)).sum())
    return tot / N

rows = []
for delta, tag in [(0.0, "gram delta=0"), (1.0, "gram delta=1")]:
    D = gram_desc(delta)
    # Gram is N x N dense -- never materialise it. Sim @ S == D @ (D.T @ S).
    class LazyGram:
        def __init__(self, D): self.D = D
        def __getitem__(self, k):
            if isinstance(k, tuple): return self.D[k[0]] @ self.D[k[1]].T
            return _Row(self.D, k)
    class _Row:
        def __init__(self, D, k): self.D, self.k = D, k
        def __matmul__(self, S): return sp.csr_matrix(self.D[self.k] @ (self.D.T @ S))
    for name, lab in [("leiden", lab_leiden), ("bad30", lab_bad), ("random", lab_random)]:
        v = cohesion(LazyGram(D), to_S(lab))
        rows.append((tag, name, v)); print(f"  {tag:14s} {name:8s} {v:.6f}  ({time.time()-t0:.0f}s)")

for name, lab in [("leiden", lab_leiden), ("bad30", lab_bad), ("random", lab_random)]:
    v = cohesion(A, to_S(lab))
    rows.append(("aff", name, v)); print(f"  {'aff':14s} {name:8s} {v:.6f}  ({time.time()-t0:.0f}s)")

print("\n" + "=" * 70)
print(f"{'term':<16}{'leiden':>11}{'bad30':>11}{'random':>11}{'SPAN':>11}")
for tag in ["gram delta=0", "gram delta=1", "aff"]:
    d = {n: v for t, n, v in rows if t == tag}
    span = d["leiden"] - d["random"]
    print(f"{tag:<16}{d['leiden']:>11.6f}{d['bad30']:>11.6f}{d['random']:>11.6f}{span:>11.6f}")


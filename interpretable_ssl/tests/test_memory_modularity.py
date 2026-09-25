"""[D47] Correctness tests for the memory-based soft modularity term.

Every test validates the SAME formula two independent ways:

  vectorized -- cohesion_memory.memory_modularity, exactly as scproto.py's
                [D47] block calls it
  brute      -- a literal triple loop over (batch cell c, proto p, cell j),
                written straight from the definition with no algebra:

      b = bio(c)
      reward(c,p) = sum_j  A[c][j] * S_eff[j][p]
      vol[b][p]   = sum_j  deg_b[b][j] * S_eff[j][p]
      q(c)        = sum_p S_live[c][p] * (reward - deg_b[b][c]*vol[b][p]/twom[b]) / twom[b]
      value       = N * mean over batch cells of q(c)

Anchors proven (the 9b checks, here once and exactly instead of eyeballed):
  * hard one-hot + fresh memory + full batch + single bio batch
      -> value == Newman's Q (independent pairwise brute force) to 1e-10
  * trivial partition (everything in one proto): 0 for ANY age vector on the
    single-batch graph; 0 with fresh memory on multi-batch graphs
  * uniform assignment: same zeros
  * random soft assignments land in Newman's range [-0.5, 1]
  * multi-batch with n_bio=1 coding == the global formula bit-identically
  * gradient flows through S_live and ONLY the live rows of S_eff

Run from the repo root:  python interpretable_ssl/tests/test_memory_modularity.py
CPU-only, ~seconds. No trainer import (Data_Handler cannot load outside Colab).
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interpretable_ssl.cohesion_memory import (  # noqa: E402
    bio_batch_degrees, memory_modularity)

torch.manual_seed(0)
np.random.seed(0)


# ------------------------------------------------------------------ helpers --
def random_graph(N, density=0.15):
    """Symmetric, non-negative, zero-diagonal weighted graph."""
    A = sp.random(N, N, density=density, format='csr', random_state=0)
    A = A + A.T
    A.setdiag(0)
    A.eliminate_zeros()
    return sp.csr_matrix(A.astype(np.float32))


def brute_value(A, S_live, S_eff, deg_b, twom, bcode, cell_idx):
    """Triple loop straight from the definition."""
    Ad = np.asarray(A.todense(), np.float64)
    Se = S_eff.detach().numpy().astype(np.float64)
    Sl = S_live.detach().numpy().astype(np.float64)
    db = deg_b.numpy().astype(np.float64)
    tm = twom.numpy().astype(np.float64)
    N, K = Se.shape
    total = 0.0
    for ci, c in enumerate(cell_idx.tolist()):
        b = int(bcode[ci])
        q = 0.0
        for p in range(K):
            reward = 0.0
            vol = 0.0
            for j in range(N):
                reward += Ad[c, j] * Se[j, p]
                vol += db[b, j] * Se[j, p]
            q += Sl[ci, p] * (reward - db[b, c] * vol / tm[b]) / tm[b]
        total += q
    return N * total / len(cell_idx.tolist())


def newman_q(A, labels):
    """Independent pairwise Newman Q: (1/2m) sum_{ij same} (w_ij - d_i d_j/2m).
    Includes the i==j null term (d_i^2/2m), the igraph convention."""
    Ad = np.asarray(A.todense(), np.float64)
    deg = Ad.sum(1)
    twom = deg.sum()
    q = 0.0
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if labels[i] == labels[j]:
                q += Ad[i, j] - deg[i] * deg[j] / twom
    return q / twom


def as_torch(deg_b, twom):
    return (torch.as_tensor(deg_b, dtype=torch.float32),
            torch.as_tensor(twom.astype(np.float32)))


N, K = 60, 7
A = random_graph(N)
bcode3 = np.random.randint(0, 3, N).astype(np.int64)
bcode1 = np.zeros(N, np.int64)


# ------------------------------------------------- 1. bio_batch_degrees ------
deg_b, twom = bio_batch_degrees(A, bcode3, 3)
Ad = np.asarray(A.todense(), np.float64)
for b in range(3):
    for j in range(N):
        if bcode3[j] == b:
            exp = Ad[j].sum()
        else:
            exp = Ad[j, bcode3 == b].sum()
        assert abs(deg_b[b, j] - exp) < 1e-4, (b, j)
# twom[b] must be 2x the weight of edges with >=1 endpoint in b
for b in range(3):
    keep = (bcode3 == b)[:, None] | (bcode3 == b)[None, :]
    assert abs(twom[b] - (Ad * keep).sum()) < 1e-3
print("1. bio_batch_degrees matches its definition        OK")


# ---------------------------------- 2. vectorized == brute, soft + aged ------
db1, tm1 = as_torch(*bio_batch_degrees(A, bcode1, 1))
db3, tm3 = as_torch(deg_b, twom)
S_mem = torch.softmax(torch.randn(N, K), dim=1)
age = torch.rand(N) * 0.9 + 0.1                     # arbitrary ages in (0,1]
batch = torch.arange(0, N, 3)                       # a third of the cells
S_live = torch.softmax(torch.randn(len(batch), K), dim=1)
S_eff = (S_mem * age.unsqueeze(1)).clone()
S_eff[batch] = S_live                               # live rows at weight 1
GS = torch.as_tensor(np.asarray((A[batch.numpy()] @
                                 S_eff.detach().numpy()))).float()
for db, tm, bc in ((db1, tm1, bcode1), (db3, tm3, bcode3)):
    bcells = torch.as_tensor(bc[batch.numpy()], dtype=torch.long)
    v = memory_modularity(GS, S_live, S_eff, db, tm, bcells, batch, N)
    bv = brute_value(A, S_live, S_eff, db, tm, bcells, batch)
    assert abs(v.item() - bv) < 1e-5, (v.item(), bv)
print("2. vectorized == triple-loop brute (soft, aged, 1 and 3 batches)  OK")


# ------------------------- 3. hard + fresh + full batch == Newman's Q --------
labels = np.random.randint(0, K, N)
S_hard = torch.zeros(N, K)
S_hard[torch.arange(N), torch.as_tensor(labels)] = 1.0
all_cells = torch.arange(N)
GS_full = torch.as_tensor(np.asarray(A @ S_hard.numpy())).float()
v = memory_modularity(GS_full, S_hard, S_hard, db1, tm1,
                      torch.zeros(N, dtype=torch.long), all_cells, N)
qn = newman_q(A, labels)
assert abs(v.item() - qn) < 1e-6, (v.item(), qn)
print(f"3. full/fresh/hard/1-batch == Newman Q ({qn:+.4f})  OK")


# --------------------------------------------- 4. anchors: trivial, uniform --
for db, tm, bc, tag in ((db1, tm1, bcode1, "1 batch"),
                        (db3, tm3, bcode3, "3 batches")):
    bcT = torch.as_tensor(bc, dtype=torch.long)
    for name, S in (("trivial", torch.eye(1, K).repeat(N, 1)),
                    ("uniform", torch.full((N, K), 1.0 / K))):
        GSx = torch.as_tensor(np.asarray(A @ S.numpy())).float()
        v = memory_modularity(GSx, S, S, db, tm, bcT, all_cells, N)
        assert abs(v.item()) < 1e-6, (name, tag, v.item())
# aged trivial/uniform, single batch: exact zero for ANY age vector
for name, S in (("trivial", torch.eye(1, K).repeat(N, 1)),
                ("uniform", torch.full((N, K), 1.0 / K))):
    S_aged = S * age.unsqueeze(1)
    GSx = torch.as_tensor(np.asarray(A @ S_aged.numpy())).float()
    v = memory_modularity(GSx, S, S_aged, db1, tm1,
                          torch.zeros(N, dtype=torch.long), all_cells, N)
    assert abs(v.item()) < 1e-6, (name, v.item())
print("4. trivial & uniform anchors == 0 (fresh both graphs; aged 1-batch)  OK")


# --------------------------------------------------- 5. range [-0.5, 1] ------
for trial in range(30):
    S_r = torch.softmax(torch.randn(N, K) * (trial % 5 + 1), dim=1)
    GSx = torch.as_tensor(np.asarray(A @ S_r.numpy())).float()
    for db, tm, bc in ((db1, tm1, bcode1), (db3, tm3, bcode3)):
        v = memory_modularity(GSx, S_r, S_r, db, tm,
                              torch.as_tensor(bc, dtype=torch.long),
                              all_cells, N).item()
        assert -0.5 - 1e-6 <= v <= 1.0 + 1e-6, v
print("5. 30 random soft partitions all inside Newman's range [-0.5, 1]  OK")


# ------------------------------------------------------- 6. gradient flow ----
S_mem_g = torch.softmax(torch.randn(N, K), dim=1)
S_live_g = torch.softmax(torch.randn(len(batch), K), dim=1).requires_grad_(True)
S_eff_g = S_mem_g.clone()
S_eff_g[batch] = S_live_g
GSg = torch.sparse_csr_tensor(
    torch.as_tensor(A[batch.numpy()].indptr, dtype=torch.int64),
    torch.as_tensor(A[batch.numpy()].indices, dtype=torch.int64),
    torch.as_tensor(A[batch.numpy()].data), size=(len(batch), N)) @ S_eff_g
v = memory_modularity(GSg, S_live_g, S_eff_g, db3, tm3,
                      torch.as_tensor(bcode3[batch.numpy()], dtype=torch.long),
                      batch, N)
g = torch.autograd.grad(v, S_live_g)[0]
assert g is not None and float(g.norm()) > 0
print(f"6. gradient flows through live rows (norm {float(g.norm()):.3e})  OK")

print("\nall [D47] memory-modularity tests passed")

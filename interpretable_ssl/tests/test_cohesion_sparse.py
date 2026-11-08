"""Equivalence of interpretable_ssl.cohesion_sparse.CohesionGraph against the LEGACY
scipy-slice + torch-CSR formulas that scproto.py used per step until 2026-10-31.

Checks every product the cohesion block consumes -- values AND gradients w.r.t. the
live assignment rows -- on cpu and, when available, on mps. Shapes mimic the real
problem (sparse symmetric graph, multiple bio batches, sorted unique batch indices).

Run:  python -I interpretable_ssl/tests/test_cohesion_sparse.py      (no pytest needed)
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from interpretable_ssl.cohesion_sparse import CohesionGraph  # noqa: E402

N, K, NBIO, NB, DEG = 3000, 25, 4, 700, 20


def _graph(seed=0):
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(N), DEG)
    cols = rng.integers(0, N, N * DEG)
    A = sp.csr_matrix((rng.random(N * DEG).astype(np.float32), (rows, cols)), shape=(N, N))
    A = A + A.T                                   # symmetric, like an RBF kernel
    A.setdiag(0); A.eliminate_zeros()
    A = sp.csr_matrix(A, dtype=np.float32)
    B = A + 1.0 * sp.eye(N, format='csr', dtype=np.float32)   # cohesion_diag = 1
    rn = np.sqrt(B.multiply(B).sum(axis=1)).A.ravel(); rn[rn == 0] = 1
    dg = np.asarray(B.multiply(B).sum(axis=1)).ravel() / rn ** 2
    bcode = rng.integers(0, NBIO, N)
    onehot = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), bcode)), shape=(N, NBIO))
    inV = np.asarray(((A > 0).astype(np.float32) @ onehot).todense()) > 0
    inV |= (bcode[:, None] == np.arange(NBIO)[None, :])
    cidx = np.sort(rng.choice(N, NB, replace=False))
    return A, B, rn, dg, bcode, inV.T.astype(np.float32), cidx


# ---------------- legacy formulas, verbatim shape of scproto.py (pre-2026-10-31) ---------
def _to_t(_sp):
    _sp = _sp.tocsr()
    return torch.sparse_csr_tensor(torch.from_numpy(_sp.indptr.astype(np.int64)),
                                   torch.from_numpy(_sp.indices.astype(np.int64)),
                                   torch.from_numpy(_sp.data.astype(np.float32)), size=_sp.shape)


def legacy(A, B, rn, dg, bcode, inV, cidx, S_eff, Sc, scl=1.0):
    out = {}
    AffB = A[cidx][:, cidx]
    out['aff_within'] = torch.sparse.mm(_to_t(AffB), Sc)
    out['deg_b'] = torch.as_tensor(np.asarray(AffB.sum(axis=1)).ravel(), dtype=torch.float32)
    out['aff_rows'] = torch.sparse.mm(_to_t(A[cidx]), S_eff)
    rnb = torch.as_tensor(rn[cidx], dtype=torch.float32).unsqueeze(1)
    rall = torch.as_tensor(rn, dtype=torch.float32).unsqueeze(1)
    Bt = _to_t(B[cidx])
    # multi-batch
    BT_full = _to_t(B.T)
    cell_b = torch.as_tensor(bcode[cidx])
    inV_t = torch.as_tensor(inV)
    GS = torch.zeros(NB, K)
    for b in torch.unique(cell_b).tolist():
        sel = (cell_b == b)
        PV_b = torch.sparse.mm(BT_full, (S_eff * inV_t[b].unsqueeze(1)) / rall)
        idx_b = cidx[sel.numpy()]
        GS[sel] = torch.sparse.mm(_to_t(B[idx_b]), PV_b) / (rnb[sel] * scl)
    out['gram_multi'] = GS
    PV = torch.sparse.mm(_to_t(B.T), S_eff / rall)
    out['gram_global'] = torch.sparse.mm(Bt, PV) / (rnb * scl)
    out['gram_batch'] = torch.sparse.mm(Bt, torch.sparse.mm(_to_t(B[cidx].T), Sc / rnb)) / (rnb * scl)
    out['aff_full'] = torch.sparse.mm(_to_t(A), S_eff)
    return out


def modern(G, bcode, inV, cidx, S_eff, Sc, scl=1.0):
    dev = G.device
    ci = torch.as_tensor(cidx, device=dev)
    S_eff, Sc = S_eff.to(dev), Sc.to(dev)
    out = {}
    out['aff_within'], out['deg_b'] = G.aff_within_mm(ci, Sc)
    out['aff_rows'], _deg2 = G.aff_rows_mm(ci, S_eff, want_degree=True)
    assert torch.allclose(_deg2, out['deg_b']), 'degree from aff_rows_mm differs'
    out['gram_multi'] = G.gram_rows_global_multi(ci, S_eff, torch.as_tensor(inV, device=dev),
                                                 torch.as_tensor(bcode[cidx], device=dev), scl)
    out['gram_global'] = G.gram_rows_global(ci, S_eff, scl)
    out['gram_batch'] = G.gram_rows_batch(ci, Sc, scl)
    out['aff_full'] = G.aff_full_mm(S_eff)
    return out


def _run(device):
    torch.manual_seed(0)
    A, B, rn, dg, bcode, inV, cidx = _graph()
    S_mem = torch.softmax(torch.randn(N, K) * 2, 1)
    Sc0 = torch.softmax(torch.randn(NB, K) * 2, 1)

    def build(S_live):
        S_eff = S_mem.to(S_live.device).clone(); S_eff[torch.as_tensor(cidx, device=S_live.device)] = S_live
        return S_eff

    Sc_l = Sc0.clone().requires_grad_(True)
    L = legacy(A, B, rn, dg, bcode, inV, cidx, build(Sc_l), Sc_l, scl=1.7)
    G = CohesionGraph(A, B, rn, dg, device)
    Sc_m = Sc0.clone().to(device).requires_grad_(True)
    M = modern(G, bcode, inV, cidx, build(Sc_m), Sc_m, scl=1.7)
    worst = 0.0
    for k in L:
        a, b = L[k], M[k].detach().cpu()
        err = (a - b).abs().max().item() / max(a.abs().max().item(), 1e-12)
        worst = max(worst, err)
        assert err < 1e-4, f"{device}: {k} rel err {err:.2e}"
    # gradients of a scalar built from every gradient-carrying product
    def scalar(o, S): return sum((o[k] * (S if o[k].shape[0] == NB else build(S))).sum()
                                  for k in ('aff_within', 'aff_rows', 'gram_multi', 'gram_global', 'gram_batch'))
    scalar(L, Sc_l).backward(); scalar(M, Sc_m).backward()
    gl, gm = Sc_l.grad, Sc_m.grad.cpu()
    gerr = (gl - gm).abs().max().item() / gl.abs().max().item()
    assert gerr < 1e-4, f"{device}: grad rel err {gerr:.2e}"
    assert torch.allclose(G.rn.cpu(), torch.as_tensor(rn, dtype=torch.float32))
    assert torch.allclose(G.dg.cpu(), torch.as_tensor(dg, dtype=torch.float32))
    print(f"[{device}] all products match legacy: worst value rel err {worst:.1e}, grad rel err {gerr:.1e}")


def test_cpu():
    _run('cpu')


def test_mps():
    if torch.backends.mps.is_available():
        _run('mps')
    else:
        print("[mps] not available, skipped")


if __name__ == '__main__':
    test_cpu(); test_mps()

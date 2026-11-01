"""Device-resident sparse graph operands for the cohesion terms (scproto.py, [D14]/[D26]/[D29]/[D41]).

Why this exists (2026-10-31): the per-step cohesion path used to slice the scipy
affinity on the host every step (``A[cidx][:, cidx]``, ``B[cidx]``, one ``B[idx_b]``
per bio batch), rebuild torch CSR tensors from the slices, and run one full-graph
sparse product PER BIO BATCH for the gram term. That is host-bound work (1.8 s/step
on a 28-core CPU for pbmc-immune, 10 studies), and it cannot run on Apple MPS at all
because MPS has no CSR kernels -- it does have COO ``torch.sparse.mm`` (measured 9 ms
vs 320 ms CPU for the full 2.05M-edge product at K=300).

This module keeps the SAME arithmetic and changes only where the operands live and how
the row subsets are formed:

  * the two graphs (raw affinity ``A`` for the 1-hop term, closed-neighbourhood ``B``
    for the gram term) are uploaded ONCE as COO tensors, together with their CSR row
    pointers, row norms and gram diagonal;
  * a step's row subset ``M[cidx]`` is formed on the device from the row pointers
    (``repeat_interleave`` over per-row edge counts) -- no scipy, no host copy;
  * the multi-batch gram profile ``PV_b = B^T (S_eff * inV[b]) / rn`` is computed for
    all bio batches in ONE product against an ``(N, n_bio*K)`` block, and the per-cell
    row is gathered by the cell's own batch code -- same numbers as the per-batch loop.

Every public method is checked against the legacy scipy/CSR formulas in
``interpretable_ssl/tests/test_cohesion_sparse.py`` (values and gradients, cpu and mps).
"""
import numpy as np
import torch


def _coo_from_csr(csr, device):
    """Upload one scipy CSR matrix as (COO sparse tensor, rowptr, col, val) on ``device``."""
    import scipy.sparse as sp
    csr = sp.csr_matrix(csr, copy=True).astype(np.float32)
    csr.sort_indices()
    rowptr = torch.from_numpy(csr.indptr.astype(np.int64)).to(device)
    col = torch.from_numpy(csr.indices.astype(np.int64)).to(device)
    val = torch.from_numpy(csr.data.astype(np.float32)).to(device)
    row = torch.repeat_interleave(torch.arange(csr.shape[0], device=device),
                                  rowptr[1:] - rowptr[:-1])
    coo = torch.sparse_coo_tensor(torch.stack([row, col]), val, csr.shape,
                                  device=device, is_coalesced=True)
    return coo, rowptr, col, val


class CohesionGraph:
    """COO copies of the cohesion graphs plus row-subset products, all on one device.

    Args:
        A_aff: scipy CSR (N, N) raw affinity, zero diagonal  (scproto ``_coh_A_aff``)
        B:     scipy CSR (N, N) descriptor matrix, symmetric  (scproto ``_coh_B``)
        rn:    (N,) row normaliser of B                        (scproto ``_coh_rn``)
        dg:    (N,) gram diagonal ||B[c]||^2 / rn[c]^2         (scproto ``_coh_dg_np``)
    """

    def __init__(self, A_aff, B, rn, dg, device):
        self.device = torch.device(device)
        self.N = A_aff.shape[0]
        self.A, self.A_ptr, self.A_col, self.A_val = _coo_from_csr(A_aff, self.device)
        self.B, self.B_ptr, self.B_col, self.B_val = _coo_from_csr(B, self.device)
        # B^T as its own COO (B is symmetric in every current config, but the legacy
        # code multiplies by B.T explicitly, so keep the exact operand).
        self.BT = torch.sparse_coo_tensor(
            torch.stack([self.B_col, self.B.indices()[0]]), self.B_val,
            (self.N, self.N), device=self.device).coalesce()
        self.rn = torch.as_tensor(np.asarray(rn), dtype=torch.float32, device=self.device)
        self.dg = torch.as_tensor(np.asarray(dg), dtype=torch.float32, device=self.device)
        self._mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)

    # ---- row subsets -------------------------------------------------------------
    def _rows(self, ptr, col, val, cidx):
        """Edges of rows ``cidx`` (sorted long tensor): (local_row, col, val)."""
        counts = ptr[cidx + 1] - ptr[cidx]
        total = int(counts.sum().item())              # the one host sync per subset
        starts = ptr[cidx]
        offsets = torch.cumsum(counts, 0) - counts
        ar = torch.arange(total, device=self.device)
        eid = (torch.repeat_interleave(starts - offsets, counts) + ar)
        local_row = torch.repeat_interleave(
            torch.arange(cidx.numel(), device=self.device), counts)
        return local_row, col[eid], val[eid]

    def _sub(self, ptr, col, val, cidx):
        """M[cidx] as a coalesced (nb, N) COO tensor."""
        r, c, v = self._rows(ptr, col, val, cidx)
        return torch.sparse_coo_tensor(torch.stack([r, c]), v, (cidx.numel(), self.N),
                                       device=self.device, is_coalesced=True)

    def _sub_within(self, ptr, col, val, cidx):
        """M[cidx][:, cidx] as (nb, nb) COO, plus the per-row sum of M[cidx][:, cidx]."""
        r, c, v = self._rows(ptr, col, val, cidx)
        self._mask.zero_(); self._mask[cidx] = True
        keep = self._mask[c]
        r, c, v = r[keep], c[keep], v[keep]
        lc = torch.searchsorted(cidx, c)
        nb = cidx.numel()
        W = torch.sparse_coo_tensor(torch.stack([r, lc]), v, (nb, nb),
                                    device=self.device, is_coalesced=True)
        deg = torch.zeros(nb, dtype=v.dtype, device=self.device).index_add_(0, r, v)
        return W, deg

    def _subT(self, ptr, col, val, cidx):
        """(M[cidx])^T as (N, nb) COO."""
        r, c, v = self._rows(ptr, col, val, cidx)
        return torch.sparse_coo_tensor(torch.stack([c, r]), v, (self.N, cidx.numel()),
                                       device=self.device).coalesce()

    # ---- 1-hop (raw affinity) ---------------------------------------------------
    def aff_rows_mm(self, cidx, D, want_degree=False):
        """(A[cidx] @ D): the [D29] full-memory 1-hop numerator, D is (N, K).

        With want_degree=True also returns the within-batch degree
        A[cidx][:, cidx].sum(1) from the same edge subset (diagnostics), else None.
        """
        r, c, v = self._rows(self.A_ptr, self.A_col, self.A_val, cidx)
        sub = torch.sparse_coo_tensor(torch.stack([r, c]), v, (cidx.numel(), self.N),
                                      device=self.device, is_coalesced=True)
        out = torch.sparse.mm(sub, D)
        deg = None
        if want_degree:
            self._mask.zero_(); self._mask[cidx] = True
            keep = self._mask[c]
            deg = torch.zeros(cidx.numel(), dtype=v.dtype, device=self.device
                              ).index_add_(0, r[keep], v[keep])
        return out, deg

    def aff_within_mm(self, cidx, Db):
        """(A[cidx][:, cidx] @ Db, within-batch degree): batch-only 1-hop numerator."""
        W, deg = self._sub_within(self.A_ptr, self.A_col, self.A_val, cidx)
        return torch.sparse.mm(W, Db), deg

    def aff_within_degree(self, cidx):
        return self._sub_within(self.A_ptr, self.A_col, self.A_val, cidx)[1]

    def aff_full_mm(self, D):
        """A @ D over the whole graph (diagnostics only)."""
        return torch.sparse.mm(self.A, D)

    # ---- 2-hop (gram of closed-neighbourhood rows) -------------------------------
    def gram_rows_global(self, cidx, S_eff, scale=1.0):
        """Legacy single-batch [D29] path:
            PV = B^T @ (S_eff / rn);  GS = B[cidx] @ PV / (rn[cidx] * scale)
        """
        PV = torch.sparse.mm(self.BT, S_eff / self.rn.unsqueeze(1))
        Bs = self._sub(self.B_ptr, self.B_col, self.B_val, cidx)
        return torch.sparse.mm(Bs, PV) / (self.rn[cidx].unsqueeze(1) * scale)

    def gram_rows_global_multi(self, cidx, S_eff, inV, bcode_cells, scale=1.0):
        """[D41] multi-batch path, all bio batches in one product:
            PV_b = B^T @ ((S_eff * inV[b]) / rn)          for every b
            GS[c] = (B[c] @ PV_{bio(c)}) / (rn[c] * scale)
        inV is (n_bio, N) 0/1, bcode_cells is (nb,) the bio batch of each cidx cell.
        """
        N, K = S_eff.shape
        nbio = inV.shape[0]
        blk = (S_eff.unsqueeze(1) * inV.t().unsqueeze(2))          # (N, nbio, K)
        blk = (blk / self.rn.view(N, 1, 1)).reshape(N, nbio * K)
        PV = torch.sparse.mm(self.BT, blk)                          # (N, nbio*K)
        Bs = self._sub(self.B_ptr, self.B_col, self.B_val, cidx)
        rows = torch.sparse.mm(Bs, PV).view(cidx.numel(), nbio, K)
        rows = rows[torch.arange(cidx.numel(), device=self.device), bcode_cells]
        return rows / (self.rn[cidx].unsqueeze(1) * scale)

    def gram_rows_batch(self, cidx, Sc, scale=1.0):
        """Legacy batch-only path:  B[cidx] @ (B[cidx]^T @ (Sc / rn_b)) / (rn_b * scale)."""
        rnb = self.rn[cidx].unsqueeze(1)
        Bs = self._sub(self.B_ptr, self.B_col, self.B_val, cidx)
        BsT = self._subT(self.B_ptr, self.B_col, self.B_val, cidx)
        return torch.sparse.mm(Bs, torch.sparse.mm(BsT, Sc / rnb)) / (rnb * scale)

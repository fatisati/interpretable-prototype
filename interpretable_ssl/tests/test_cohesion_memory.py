"""[D31] Correctness tests for the age-weighted assignment memory.

Every test validates the SAME formula two independent ways:

  vectorized -- assemble_membership + the sparse-matmul pipeline exactly as
                scproto.py's [D29]/[D31] block computes it
  brute      -- a literal triple loop over (batch cell c, proto k, cell j),
                written straight from the definition with no algebra at all:

      w[j]      = 1                                if j in batch
                  0.5 ** (age[j] / half_life)      otherwise (1 if decay off)
      assign(j) = live softmax row                 if j in batch
                  remembered row                   otherwise
      size[k]   = clamp( sum_j w[j]*assign(j)[k],  min=1 )
      num[k]    = sum_{c in batch} sum_j  live[c][k] * sim(c,j) * w[j]*assign(j)[k]
      value     = sum_k num[k]/size[k]  /  N

  for sim(c,j) = affinity[c][j]                    ('aff', 1-hop)
  and sim(c,j) = cos(aff_row[c], aff_row[j])       ('gram', 2-hop, l2 row norm)

Limits proven: half_life=0 is bit-identical to [D29]; full batch = exact dense
objective for any half_life; a cell absent k steps contributes exactly
0.5**(k/half_life) of its weight; gradients match finite differences and only
flow through the live batch rows.

Run from the repo root:  python interpretable_ssl/tests/test_cohesion_memory.py
CPU-only, ~seconds. No trainer import (Data_Handler cannot load outside Colab).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from interpretable_ssl.cohesion_memory import (age_weights, assemble_membership,
                                               batch_drift, batch_sizes)

torch.set_default_dtype(torch.float64)


# ---------------------------------------------------------------- fixtures ---
def make_case(seed, N=48, K=4, nb=16, half_life=7.0, step=23):
    """Random affinity (symmetric, nonneg, zero diagonal), memory, live batch."""
    rng = np.random.default_rng(seed)
    A = rng.random((N, N)) * (rng.random((N, N)) < 0.25)   # sparse-ish
    A = np.triu(A, 1)
    A = A + A.T                                            # symmetric, diag 0
    A = torch.tensor(A)

    S_mem = torch.softmax(torch.tensor(rng.normal(size=(N, K))) * 2.0, dim=1)
    batch_idx = torch.tensor(rng.choice(N, nb, replace=False))
    logits = torch.tensor(rng.normal(size=(nb, K)) * 2.0, requires_grad=True)
    last_seen = torch.tensor(rng.integers(-1, step, size=N))
    return A, S_mem, batch_idx, logits, last_seen, half_life, step


def weights_brute(batch_idx, last_seen, step, half_life, N):
    w = np.ones(N)
    if half_life and half_life > 0:
        for j in range(N):
            age = max(0.0, step - int(last_seen[j]))
            w[j] = 0.5 ** (age / half_life)
    for c in batch_idx.tolist():
        w[c] = 1.0
    return w


def value_brute(sim, S_mem, batch_idx, S_live, last_seen, step, half_life):
    """The literal definition. sim is a dense (N,N) numpy similarity."""
    N, K = S_mem.shape
    w = weights_brute(batch_idx, last_seen, step, half_life, N)
    bpos = {int(j): i for i, j in enumerate(batch_idx.tolist())}
    assign = S_mem.detach().numpy().copy()
    live = S_live.detach().numpy()
    for j, i in bpos.items():
        assign[j] = live[i]

    size = np.zeros(K)
    for k in range(K):
        for j in range(N):
            size[k] += w[j] * assign[j][k]
        size[k] = max(size[k], 1.0)

    value = 0.0
    for k in range(K):
        num = 0.0
        for c in bpos:
            for j in range(N):
                num += live[bpos[c]][k] * sim[c][j] * w[j] * assign[j][k]
        value += num / size[k]
    return value / N


def value_vectorized(A, S_mem, batch_idx, S_live, last_seen, step, half_life,
                     sim='aff'):
    """The pipeline exactly as scproto's [D29]/[D31] block runs it."""
    S_eff, size_raw, _ = assemble_membership(
        S_mem, S_live, batch_idx, last_seen=last_seen, step=step,
        half_life=half_life)
    size_g = size_raw.clamp(min=1.0)
    if sim == 'aff':
        GS = A[batch_idx] @ S_eff                       # (nb, K)
    else:                                               # 'gram', l2 row norm
        rn = A.pow(2).sum(dim=1).sqrt().clamp(min=1e-30)
        PV = A.T @ (S_eff / rn.unsqueeze(1))            # profile per proto
        GS = (A[batch_idx] @ PV) / rn[batch_idx].unsqueeze(1)
    cpair = (GS * S_live).sum(0)                        # (K,)
    return (cpair / size_g).sum() / float(S_mem.shape[0])


# ------------------------------------------------------------------- tests ---
def test_off_is_bit_identical_to_d29():
    """half_life=0 must reproduce [D29]'s clone+swap+sum exactly."""
    A, S_mem, bidx, logits, seen, _, step = make_case(0)
    S_live = torch.softmax(logits, dim=1)
    S_eff, size_raw, w = assemble_membership(S_mem, S_live, bidx,
                                             last_seen=seen, step=step,
                                             half_life=0.0)
    ref = S_mem.clone()
    ref[bidx] = S_live
    assert w is None
    assert torch.equal(S_eff, ref)
    assert torch.equal(size_raw, ref.sum(0))
    print("PASS  off (half_life=0) == [D29], bit-identical")


def test_brute_force_aff():
    for seed in range(5):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        S_live = torch.softmax(logits, dim=1)
        v = value_vectorized(A, S_mem, bidx, S_live, seen, step, hl, 'aff')
        b = value_brute(A.numpy(), S_mem, bidx, S_live, seen, step, hl)
        assert abs(v.item() - b) < 1e-10, (seed, v.item(), b)
    print("PASS  vectorized == triple-loop brute force ('aff', 5 seeds)")


def test_brute_force_gram():
    for seed in range(5):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        S_live = torch.softmax(logits, dim=1)
        An = A.numpy()
        rn = np.sqrt((An ** 2).sum(axis=1)).clip(min=1e-30)
        gram = (An @ An.T) / np.outer(rn, rn)           # cos of affinity rows
        v = value_vectorized(A, S_mem, bidx, S_live, seen, step, hl, 'gram')
        b = value_brute(gram, S_mem, bidx, S_live, seen, step, hl)
        assert abs(v.item() - b) < 1e-10, (seed, v.item(), b)
    print("PASS  factored gram path == explicit cos-matrix brute force (5 seeds)")


def test_decay_semantics():
    """A cell absent k steps contributes exactly 0.5**(k/H) of its weight."""
    N, K, H = 6, 3, 4.0
    S_mem = torch.zeros(N, K)
    S_mem[:, 0] = 1.0                                   # everyone in proto 0
    bidx = torch.tensor([0, 1])
    S_live = torch.full((2, K), 1.0 / K)
    for k_absent in [0, 1, 4, 8, 40]:
        step = 100
        seen = torch.full((N,), step - k_absent)
        _, size_raw, w = assemble_membership(S_mem, S_live, bidx,
                                             last_seen=seen, step=step,
                                             half_life=H)
        expect_w = 0.5 ** (k_absent / H)
        assert abs(w[2].item() - expect_w) < 1e-12
        # size in proto 0: 2 live cells at 1/K each + 4 remembered at expect_w
        expect_size = 2 * (1.0 / K) + 4 * expect_w
        assert abs(size_raw[0].item() - expect_size) < 1e-12
    print("PASS  absence of k steps scales a row by exactly 0.5**(k/H)")


def test_full_batch_is_exact():
    """batch = all cells -> the exact dense objective, decay on or off."""
    for hl in [0.0, 7.0]:
        A, S_mem, _, _, _, _, step = make_case(3)
        N, K = S_mem.shape
        logits = torch.tensor(
            np.random.default_rng(9).normal(size=(N, K)) * 2.0)
        S_live = torch.softmax(logits, dim=1)
        bidx = torch.arange(N)
        seen = torch.full((N,), step)                   # all just seen
        v = value_vectorized(A, S_mem, bidx, S_live, seen, step, hl, 'aff')
        S = S_live.numpy()
        size = np.maximum(S.sum(axis=0), 1.0)
        within = np.einsum('ik,ij,jk->k', S, A.numpy(), S)
        exact = float((within / size).sum() / N)
        assert abs(v.item() - exact) < 1e-10, (hl, v.item(), exact)
    print("PASS  full batch == exact dense objective (decay on and off)")


def test_gradient_matches_finite_difference():
    """Autograd through the live rows == finite differences of the brute value;
    and the memory receives no gradient at all."""
    A, S_mem, bidx, logits, seen, hl, step = make_case(1, N=30, nb=8)
    S_mem.requires_grad_(False)

    def f(lg):
        S_live = torch.softmax(lg, dim=1)
        return value_vectorized(A, S_mem, bidx, S_live, seen, step, hl, 'aff')

    v = f(logits)
    v.backward()
    g = logits.grad.clone()
    eps = 1e-6
    rng = np.random.default_rng(4)
    for _ in range(6):
        i = int(rng.integers(logits.shape[0]))
        j = int(rng.integers(logits.shape[1]))
        lp = logits.detach().clone(); lp[i, j] += eps
        lm = logits.detach().clone(); lm[i, j] -= eps
        fd = (f(lp).item() - f(lm).item()) / (2 * eps)
        assert abs(fd - g[i, j].item()) < 1e-6, (i, j, fd, g[i, j].item())
    assert S_mem.grad is None
    print("PASS  autograd == finite differences on live rows; memory grad-free")


def test_bounds():
    """Affinity in [0,1], softmax rows -> value stays in [0,1]."""
    for seed in range(50):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        S_live = torch.softmax(logits, dim=1)
        v = value_vectorized(A, S_mem, bidx, S_live, seen, step, hl, 'aff').item()
        assert -1e-12 <= v <= 1.0 + 1e-12, (seed, v)
    print("PASS  value in [0,1] over 50 random cases")


def test_drift_diagnostic():
    N, K = 5, 2
    S_mem = torch.tensor([[1.0, 0.0]] * N)
    fresh = torch.tensor([[1.0, 0.0],      # unchanged        -> L1 = 0
                          [0.0, 1.0],      # fully moved      -> L1 = 2
                          [0.5, 0.5]])     # half moved       -> L1 = 1
    bidx = torch.tensor([0, 1, 2])
    seen = torch.tensor([10, 10, -1, -1, -1])   # cell 2 never seen -> excluded
    drift, age = batch_drift(S_mem, fresh, bidx, seen, step=15)
    assert abs(drift - 1.0) < 1e-12          # mean of {0, 2}
    assert abs(age - 5.0) < 1e-12
    d0, a0 = batch_drift(S_mem, fresh, bidx, torch.full((N,), -1), step=15)
    assert d0 == 0.0 and a0 == 0.0           # nothing seen before
    print("PASS  drift diagnostic: L1 as defined, never-seen rows excluded")


def test_uniform_prior_fades():
    """Never-visited rows (last_seen=-1) decay as training proceeds."""
    seen = torch.tensor([-1])
    H = 10.0
    w_early = age_weights(seen, step=1, half_life=H)[0].item()
    w_late = age_weights(seen, step=100, half_life=H)[0].item()
    assert abs(w_early - 0.5 ** (2 / H)) < 1e-12
    assert abs(w_late - 0.5 ** (101 / H)) < 1e-12
    assert w_late < w_early < 1.0
    print("PASS  uniform-init prior fades with age instead of persisting")


def test_full_init_recipe():
    """[D32] the init assignment softmax(z @ protos / eps): rows are a proper
    distribution and each cell prefers its nearest prototype by dot product."""
    rng = np.random.default_rng(7)
    z = torch.tensor(rng.normal(size=(40, 8)))
    protos = torch.tensor(rng.normal(size=(5, 8)))
    eps = 0.1
    S = torch.softmax((z @ protos.T) / eps, dim=1)
    assert torch.allclose(S.sum(dim=1), torch.ones(40))
    assert torch.equal(S.argmax(dim=1), (z @ protos.T).argmax(dim=1))
    # rows are valid memory: age 0 (last_seen = step) -> weight exactly 1
    w = age_weights(torch.zeros(40, dtype=torch.long), step=0, half_life=9.0)
    assert torch.all(w == 1.0)
    print("PASS  [D32] init rows are distributions, nearest-proto, weight 1")


def test_model_name_registration():
    """coh_mem_hlife / coh_mem_init reach the model folder name ([9] step 3)."""
    from interpretable_ssl.model_name import generate_model_name
    defaults = {'model_name_version': 6, 'experiment_name': 'exp',
                'version': 31, 'coh_mem_hlife': 0.0, 'coh_mem_init': 'unif'}
    on = generate_model_name(defaults, {'experiment_name': 'exp',
                                        'coh_mem_hlife': 500.0,
                                        'coh_mem_init': 'full'})
    off = generate_model_name(defaults, {'experiment_name': 'exp'})
    assert 'cmhl500' in on, on
    assert 'cmini-full' in on, on
    assert 'cmhl' not in off and 'cmini' not in off, off
    print(f"PASS  model name carries the options: '{on}' vs '{off}'")



# ------------------------------------------------- [D41] multi-batch cohesion ---
# THE ONLY CHANGE from the batch-agnostic formula is the SET OF CELLS each cell's
# cohesion is computed against:
#
#   for each batch b:
#       bio_graph_b = all edges with AT LEAST ONE endpoint in b
#       V[b]        = node set of bio_graph_b
#                   = {j : bio(j)==b} U {j : j has an edge into b}
#
#   for cell c, b = bio(c):
#       numer[c][p] = sum over j in V[b] of  w[j]*ass[j][p]*sim[c][j]
#       denom[c][p] = sum over j in V[b] of  w[j]*ass[j][p]
#       cohesion_c  = sum over p of ass[c][p] * numer[c][p]/denom[c][p]
#
# Nothing is masked: cross-batch similarities keep their real values.
def build_inV(A, bcode, n_bio):
    """inV[b][j] = 1 if bio(j)==b OR j has an edge into batch b."""
    N = A.shape[0]
    bc = torch.as_tensor(bcode)
    adj = (A > 0).to(A.dtype)
    onehot = torch.zeros(N, n_bio, dtype=A.dtype)
    onehot[torch.arange(N), bc] = 1.0
    touch = (adj @ onehot) > 0                      # (N, n_bio)
    own = bc.unsqueeze(1) == torch.arange(n_bio).unsqueeze(0)
    return (touch | own).to(A.dtype).T              # (n_bio, N)


def value_brute_d41(sim, S_mem, batch_idx, S_live, last_seen, step, half_life,
                    bcode, n_bio, self_weight=1.0):
    """The literal loop from the spec. Slow triple loop, no algebra."""
    N, K = S_mem.shape
    w = weights_brute(batch_idx, last_seen, step, half_life, N)
    bpos = {int(j): i for i, j in enumerate(batch_idx.tolist())}
    assign = S_mem.detach().numpy().copy()
    live = S_live.detach().numpy()
    for j, i in bpos.items():
        assign[j] = live[i]

    inV = build_inV(torch.as_tensor(sim), bcode, n_bio).numpy()
    total = 0.0
    for c in bpos:
        b = bcode[c]
        for k in range(K):
            numer = 0.0
            denom = 0.0
            for j in range(N):
                if inV[b][j] == 0:
                    continue
                s_cj = sim[c][j]
                if j == c:
                    s_cj = s_cj * self_weight
                numer += w[j] * assign[j][k] * s_cj
                denom += w[j] * assign[j][k]
            total += live[bpos[c]][k] * numer / max(denom, 1e-8)
    return total / float(len(bpos))


def value_vectorized_d41(A, S_mem, batch_idx, S_live, last_seen, step, half_life,
                         bcode, n_bio, sim="aff", self_weight=1.0):
    """The pipeline exactly as scproto's [D41] block runs it."""
    bc = torch.as_tensor(bcode, dtype=torch.long)
    S_eff, _, _ = assemble_membership(S_mem, S_live, batch_idx,
                                      last_seen=last_seen, step=step,
                                      half_life=half_life)
    inV = build_inV(A, bcode, n_bio)
    den_bp = inV @ S_eff                                  # (n_bio, K)
    den = den_bp[bc[batch_idx]].clamp(min=1e-8)           # (nb, K)
    if sim == "aff":
        # restriction is automatic: every neighbour of c is in V[bio(c)]
        GS = A[batch_idx] @ S_eff
    else:
        # gram: the restriction is NOT automatic -- a 2-hop neighbour can sit
        # outside V[b] -- so build one profile per batch from that V[b] only.
        rn = A.pow(2).sum(dim=1).sqrt().clamp(min=1e-30)
        GS = torch.zeros(len(batch_idx), S_mem.shape[1], dtype=A.dtype)
        cell_b = bc[batch_idx]
        for b in torch.unique(cell_b).tolist():
            selb = cell_b == b
            PV_b = A.T @ ((S_eff * inV[b].unsqueeze(1)) / rn.unsqueeze(1))
            GS[selb] = (A[batch_idx[selb]] @ PV_b) / rn[batch_idx[selb]].unsqueeze(1)
    num = GS * S_live
    if self_weight != 1.0:
        dg = A[batch_idx, batch_idx] if sim == "aff" else torch.ones(len(batch_idx),
                                                                    dtype=A.dtype)
        num = num - (1.0 - self_weight) * (S_live.pow(2) * dg.unsqueeze(1))
    return (num / den).sum() / float(len(batch_idx))


def make_bcode(seed, N, n_bio):
    rng = np.random.default_rng(1000 + seed)
    bc = rng.integers(0, n_bio, size=N)
    bc[:n_bio] = np.arange(n_bio)               # every batch non-empty
    return bc


def test_d41_brute_force_aff():
    """Vectorised == the literal triple loop from the spec, 1-hop similarity."""
    for seed in range(5):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        A = A.clone()
        A[torch.arange(A.shape[0]), torch.arange(A.shape[0])] = 1.0  # sim(c,c)=1
        S_live = torch.softmax(logits, dim=1)
        bc = make_bcode(seed, A.shape[0], 3)
        got = value_vectorized_d41(A, S_mem, bidx, S_live, seen, step, hl, bc, 3,
                                   sim="aff").item()
        want = value_brute_d41(A.numpy(), S_mem, bidx, S_live, seen, step, hl,
                               bc, 3)
        assert abs(got - want) < 1e-9, (seed, got, want)
    print("PASS  [D41] aff  == brute-force triple loop over V[b]")


def test_d41_brute_force_gram():
    """Vectorised == the literal triple loop, 2-hop (gram) similarity."""
    for seed in range(5):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        S_live = torch.softmax(logits, dim=1)
        bc = make_bcode(seed, A.shape[0], 3)
        rn = A.pow(2).sum(dim=1).sqrt().clamp(min=1e-30)
        R = A / rn.unsqueeze(1)
        gram = (R @ R.T).numpy()
        got = value_vectorized_d41(A, S_mem, bidx, S_live, seen, step, hl, bc, 3,
                                   sim="gram").item()
        # brute force needs inV built from the SAME adjacency the code uses (A),
        # not from the dense gram -- pass A's structure explicitly
        inV_ref = build_inV(A, bc, 3).numpy()
        N, K = S_mem.shape
        w = weights_brute(bidx, seen, step, hl, N)
        bpos = {int(j): i for i, j in enumerate(bidx.tolist())}
        assign = S_mem.detach().numpy().copy()
        live = S_live.detach().numpy()
        for j, i in bpos.items():
            assign[j] = live[i]
        total = 0.0
        for c in bpos:
            b = bc[c]
            for k in range(K):
                numer = denom = 0.0
                for j in range(N):
                    if inV_ref[b][j] == 0:
                        continue
                    numer += w[j] * assign[j][k] * gram[c][j]
                    denom += w[j] * assign[j][k]
                total += live[bpos[c]][k] * numer / max(denom, 1e-8)
        want = total / float(len(bpos))
        assert abs(got - want) < 1e-9, (seed, got, want)
    print("PASS  [D41] gram == brute-force triple loop over V[b]")


def test_d41_single_batch_is_bit_identical():
    """THE constraint: ONE bio batch must behave exactly as before.

    With one batch every edge has an endpoint in b, so V[b] is ALL cells and the
    denominator is the global size. Identical to the [D29]/[D37] path, not merely
    close -- asserted to 1e-12, both similarities.
    """
    for seed in range(5):
        A, S_mem, bidx, logits, seen, hl, step = make_case(seed)
        S_live = torch.softmax(logits, dim=1)
        bc = np.zeros(A.shape[0], dtype=np.int64)            # ONE bio batch
        for sim in ("aff", "gram"):
            d41 = value_vectorized_d41(A, S_mem, bidx, S_live, seen, step, hl,
                                       bc, 1, sim=sim).item()
            glob = value_vectorized(A, S_mem, bidx, S_live, seen, step, hl,
                                    sim=sim).item() * float(
                S_mem.shape[0]) / float(len(bidx))
            assert abs(d41 - glob) < 1e-12, (seed, sim, d41, glob)
    print("PASS  [D41] one bio batch == the batch-agnostic path (<1e-12, both sims)")


def test_d41_inV_definition():
    """V[b] is the node set of 'all edges with at least one endpoint in b'."""
    for seed in range(4):
        A, *_ = make_case(seed)
        bc = make_bcode(seed, A.shape[0], 3)
        inV = build_inV(A, bc, 3).numpy()
        N = A.shape[0]
        for b in range(3):
            for j in range(N):
                own = bc[j] == b
                touches = bool(((A[j] > 0).numpy() & (bc == b)).any())
                assert bool(inV[b][j]) == bool(own or touches), (b, j)
        # every own-batch cell is in V[b]
        for b in range(3):
            assert inV[b][bc == b].all()
    print("PASS  [D41] V[b] == nodes of edges with >=1 endpoint in b (own U touching)")


def test_d41_registration():
    """coh_batch must reach the model name, or two runs clobber each other."""
    from interpretable_ssl.constants import ABBREVIATIONS
    assert 'coh_batch' in ABBREVIATIONS, "coh_batch missing from ABBREVIATIONS"
    vals = ['off', 'locl']
    assert len({v[:4] for v in vals}) == len(vals), "values not 4-char-distinct"
    print(f"PASS  [D41] registered: abbrev={ABBREVIATIONS['coh_batch']!r}")

if __name__ == '__main__':
    test_off_is_bit_identical_to_d29()
    test_brute_force_aff()
    test_brute_force_gram()
    test_decay_semantics()
    test_full_batch_is_exact()
    test_gradient_matches_finite_difference()
    test_bounds()
    test_drift_diagnostic()
    test_uniform_prior_fades()
    test_full_init_recipe()
    test_model_name_registration()
    test_d41_brute_force_aff()
    test_d41_brute_force_gram()
    test_d41_single_batch_is_bit_identical()
    test_d41_inV_definition()
    test_d41_registration()
    print("\nall tests passed")

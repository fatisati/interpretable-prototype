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
                                               batch_drift)

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
    print("\nall tests passed")

"""[D65] Numerical validation of the cross-batch signature alignment term.

CLAUDE.md 9b: a new loss term is not "working" because the run finished and the
number went down. These tests check the three things that have silently broken
before -- the formula itself, the scale/range, and whether the incentive points
where the design says it does.

Mirrors the vectorized implementation in trainers/scproto.py exactly, so a
divergence between the two is a real bug in one of them.

Run:  python -m pytest interpretable_ssl/tests/test_xbatch_signature.py -q
 or:  python interpretable_ssl/tests/test_xbatch_signature.py
"""
import numpy as np
import torch
import torch.nn.functional as F

G = 40          # genes
K = 6           # prototypes
NBIO = 3        # batches
N = 240         # cells
MIN_SIZE = 1.0


def _fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand((N, G), generator=g) * 4.0          # lognorm-like, >= 0
    bcode = torch.arange(N) % NBIO
    S = F.softmax(torch.randn((N, K), generator=g) * 2.0, dim=1)
    bg = torch.stack([X[bcode == b].mean(0) for b in range(NBIO)])
    return X, bcode, S, bg


# ---------------------------------------------------------------- signature
def _sig_bruteforce(X, bcode, S, bg):
    """sig[b][k] = sum_i in b s[i][k] x[i] / sum_i in b s[i][k]  -  bg[b]"""
    sig = torch.zeros((NBIO, K, G))
    size = torch.zeros((NBIO, K))
    for b in range(NBIO):
        for k in range(K):
            num = torch.zeros(G)
            den = 0.0
            for i in range(N):
                if int(bcode[i]) == b:
                    num = num + S[i, k] * X[i]
                    den = den + float(S[i, k])
            size[b, k] = den
            sig[b, k] = num / max(den, 1e-6) - bg[b]
    return sig, size


def _sig_vectorized(X, bcode, S, bg):
    """Exactly the scproto.py path."""
    M = F.one_hot(bcode, num_classes=NBIO).to(S.dtype)        # (N, nbio)
    size = M.t() @ S                                          # (nbio, K)
    ws = torch.stack([(S * M[:, b].unsqueeze(1)).t() @ X
                      for b in range(NBIO)])                  # (nbio, K, G)
    sig = ws / size.unsqueeze(-1).clamp(min=1e-6) - bg.unsqueeze(1)
    return sig, size


def test_signature_matches_bruteforce():
    X, bcode, S, bg = _fixture()
    s1, z1 = _sig_bruteforce(X, bcode, S, bg)
    s2, z2 = _sig_vectorized(X, bcode, S, bg)
    assert torch.allclose(z1, z2, atol=1e-5), (z1 - z2).abs().max()
    assert torch.allclose(s1, s2, atol=1e-4), (s1 - s2).abs().max()


def test_correlation_matches_numpy():
    """The normalize-then-dot shortcut must equal np.corrcoef."""
    X, bcode, S, bg = _fixture()
    sig, _ = _sig_vectorized(X, bcode, S, bg)
    c = sig - sig.mean(dim=-1, keepdim=True)
    u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    for k in range(K):
        got = float((u[0, k] * u[1, k]).sum())
        want = float(np.corrcoef(sig[0, k].numpy(), sig[1, k].numpy())[0, 1])
        assert abs(got - want) < 1e-5, (k, got, want)


def test_group_mean_in_background_only_rescales():
    """The group is inside its own batch background, which scales sig by
    (1 - fraction). Pearson is scale-invariant, so r must be UNCHANGED."""
    X, bcode, S, bg = _fixture()
    sig_in, _ = _sig_vectorized(X, bcode, S, bg)          # bg includes the group

    def _r(sig):
        c = sig - sig.mean(dim=-1, keepdim=True)
        u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return (u[0] * u[1]).sum(-1)

    # background EXCLUDING the group, computed per (b, k)
    sig_ex = torch.zeros_like(sig_in)
    for b in range(NBIO):
        m = bcode == b
        for k in range(K):
            w = S[m, k]
            gm = (w.unsqueeze(1) * X[m]).sum(0) / w.sum()
            rest = ((1 - w).unsqueeze(1) * X[m]).sum(0) / (1 - w).sum()
            sig_ex[b, k] = gm - rest
    assert torch.allclose(_r(sig_in), _r(sig_ex), atol=1e-3), \
        (_r(sig_in) - _r(sig_ex)).abs().max()


# ---------------------------------------------------------------- the reward
def _reward(X, bcode, S, bg, weight_sg=True):
    sig, size = _sig_vectorized(X, bcode, S, bg)
    c = sig - sig.mean(dim=-1, keepdim=True)
    u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    valid = size >= MIN_SIZE
    tot = torch.zeros(())
    for b1 in range(NBIO):
        for b2 in range(b1 + 1, NBIO):
            v = (valid[b1] & valid[b2]).to(u.dtype)
            w = torch.minimum(size[b1], size[b2])
            if weight_sg:
                w = w.detach()
            tot = tot + (w * (u[b1] * u[b2]).sum(-1) * v).sum()
    return tot / float(N)


def test_abandoning_a_batch_is_punished_not_rewarded():
    """THE design check. Under the old penalty form sum(1 - r), dropping a batch
    removes pairs and so LOWERS the cost -- the model goes batch-pure, which is
    the opposite of integration. As a size-weighted REWARD it must lose value."""
    X, bcode, S, bg = _fixture()
    base = float(_reward(X, bcode, S, bg))

    # push prototype 0's batch-2 cells onto prototype 1 (abandon one batch)
    S2 = S.clone()
    m = (bcode == 2)
    S2[m, 1] = S2[m, 1] + S2[m, 0]
    S2[m, 0] = 0.0
    after = float(_reward(X, bcode, S2, bg))
    assert after < base, f"abandoning a batch RAISED the reward: {base} -> {after}"

    # and the old penalty form would have rewarded exactly that move
    def _penalty(S_):
        sig, size = _sig_vectorized(X, bcode, S_, bg)
        c = sig - sig.mean(dim=-1, keepdim=True)
        u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        valid = size >= MIN_SIZE
        tot = 0.0
        for b1 in range(NBIO):
            for b2 in range(b1 + 1, NBIO):
                v = valid[b1] & valid[b2]
                r = (u[b1] * u[b2]).sum(-1)
                tot += float((1.0 - r)[v].sum())
        return tot
    assert _penalty(S2) < _penalty(S), (
        "the penalty form was expected to REWARD abandoning a batch; if this "
        "fails the fixture is not exercising the failure mode")


def test_reward_is_bounded():
    """|loss| <= (n_batches - 1) / 2 after dividing by n_cells. 9b range check."""
    for seed in range(5):
        X, bcode, S, bg = _fixture(seed)
        assert abs(float(_reward(X, bcode, S, bg))) <= (NBIO - 1) / 2.0 + 1e-6


def test_identical_composition_scores_one():
    """A prototype holding the SAME mixture in two batches must score r = 1:
    this is the 'consistency, not purity' property, asserted rather than assumed."""
    g = torch.Generator().manual_seed(3)
    prof = torch.rand((K, G), generator=g) * 4.0
    bcode = torch.arange(N) % NBIO
    X = torch.zeros((N, G))
    S = torch.zeros((N, K))
    for i in range(N):
        # k must NOT be i % K: with K=6 and NBIO=3, i%6 determines i%3, so every
        # prototype would sit in exactly ONE batch and nothing would span.
        k = (i // NBIO) % K
        S[i, k] = 1.0
        X[i] = prof[k]                      # same profile in every batch
    bg = torch.stack([X[bcode == b].mean(0) for b in range(NBIO)])
    sig, _ = _sig_vectorized(X, bcode, S, bg)
    c = sig - sig.mean(dim=-1, keepdim=True)
    u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r = (u[0] * u[1]).sum(-1)
    assert torch.allclose(r, torch.ones(K), atol=1e-4), r


def test_chimera_scores_low():
    """A prototype that is profile A in batch 0 and profile B in batch 1 must
    score far below a coherent one -- the thing the term exists to detect."""
    g = torch.Generator().manual_seed(4)
    prof = torch.rand((K, G), generator=g) * 4.0
    bcode = torch.arange(N) % NBIO
    X = torch.zeros((N, G))
    S = torch.zeros((N, K))
    for i in range(N):
        # k must NOT be i % K: with K=6 and NBIO=3, i%6 determines i%3, so every
        # prototype would sit in exactly ONE batch and nothing would span.
        k = (i // NBIO) % K
        S[i, k] = 1.0
        # prototype 0 is a CHIMERA: profile 0 in batch 0, profile 3 in batch 1
        X[i] = prof[3] if (k == 0 and int(bcode[i]) == 1) else prof[k]
    bg = torch.stack([X[bcode == b].mean(0) for b in range(NBIO)])
    sig, _ = _sig_vectorized(X, bcode, S, bg)
    c = sig - sig.mean(dim=-1, keepdim=True)
    u = c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    r = (u[0] * u[1]).sum(-1)
    assert float(r[0]) < 0.5, f"chimera scored {float(r[0])}"
    assert float(r[1:].min()) > 0.9, f"coherent prototypes scored {r[1:]}"


# ---------------------------------------------------------------- gradients
def test_gradient_reaches_assignments():
    """If this is zero the term cannot move a single cell and is decorative."""
    X, bcode, S, bg = _fixture()
    S = S.clone().requires_grad_(True)
    (-_reward(X, bcode, S, bg)).backward()
    assert S.grad is not None and float(S.grad.norm()) > 1e-8, S.grad


def test_weight_detach_changes_the_gradient():
    """xbatch_weight_sg is a real switch, not cosmetic: live, the size path adds
    a constant ~r pull to join any coherent spanning prototype regardless of fit
    (the 9b capacity-charge defect). Detached, only the r path survives."""
    X, bcode, S, bg = _fixture()
    a = S.clone().requires_grad_(True)
    (-_reward(X, bcode, a, bg, weight_sg=True)).backward()
    b = S.clone().requires_grad_(True)
    (-_reward(X, bcode, b, bg, weight_sg=False)).backward()
    assert not torch.allclose(a.grad, b.grad, atol=1e-6), \
        "detaching min(size,size) had no effect on the gradient"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")

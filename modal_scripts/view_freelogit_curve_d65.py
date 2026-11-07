"""[D65] FREE-LOGIT study: the ceiling of the signature objective, and the
cohesion-vs-alignment trade-off curve.

WHY THIS EXISTS
The [D65] arms left one question unanswerable: the signature term's own
quantity FELL every epoch (0.574 -> 0.449) while cohesion rose. That is
consistent with two OPPOSITE diagnoses --

    (a) the term is simply TOO SMALL      -> raise its weight
    (b) the term CONFLICTS with cohesion  -> no weight works, change the design

and a dose sweep in full training costs ~90 min and ~$1.50 per point.

This removes the encoder entirely: the assignment matrix becomes a free
parameter optimized directly against the same two objectives. One point then
costs minutes instead of 90, so the whole trade-off curve can be drawn, and it
separates "is the objective reachable at all" (here) from "can the encoder
reach it" (full training). CLAUDE.md 9d uses the same device: free-logit
optimisation gave 0 dead / 1 fragment where the trained encoder gave 5 dead /
54 fragments at an identical gate, which is how the founder gate was cleared.

READING THE CURVE
    a KNEE              alignment is cheap up to that weight -> pick it, run it
    straight diagonal   genuine conflict; no weight is good, redesign the term
    flat r_cross        the OBJECTIVE's own ceiling; not a weight problem and
                        not an encoder problem

Run:  python modal_scripts/view_freelogit_curve_d65.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
AFF = "graphs/affinity_pbmc-immune33506_ncomp50_kneighbors50_arbf.pkl"
STEPS = 120
LR = 0.05
EPS = 0.05                      # softmax temperature on the free logits
# 0 = cohesion only (the scProto objective); 1e9 = signature ALONE (the ceiling)
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 31
torch.set_num_threads(os.cpu_count() or 4)

print(f"[D65 free-logit] device={DEV} threads={torch.get_num_threads()} "
      f"K={K} steps={STEPS} weights={len(WEIGHTS)}")

# ---------------------------------------------------------------- data
import anndata                                                       # noqa: E402

ad = anndata.read_h5ad(os.path.join(os.environ["DATA_DIR"],
                                    "scpoli/Immune_ALL_human_hvg.h5ad"))
N = ad.n_obs
X = ad.layers["lognorm"] if "lognorm" in ad.layers else ad.X
X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)
G = X.shape[1]
bts = np.asarray(ad.obs[BATCH_KEY]).astype(str)
cts = np.asarray(ad.obs[LABEL_KEY]).astype(str)
ub, bcode = np.unique(bts, return_inverse=True)
NB = len(ub)
print(f"  {N:,} cells x {G:,} genes, {NB} batches")

with open(AFF, "rb") as f:
    A = pickle.load(f)
A = A["affinity"] if isinstance(A, dict) and "affinity" in A else A
A = sp.csr_matrix(A).astype(np.float32)
A.setdiag(0); A.eliminate_zeros()
print(f"  affinity {A.shape}, nnz {A.nnz:,}")

Xt = torch.as_tensor(X, device=DEV)
bt = torch.as_tensor(bcode, dtype=torch.long, device=DEV)
# cells of each batch, as index tensors. The batches PARTITION the cells, so
# slicing per batch makes the signature matmul cost K*N*G in total rather than
# K*N*G*NB -- a 5x saving here, and it is exact, not an approximation.
bidx = [torch.as_tensor(np.where(bcode == b)[0], device=DEV) for b in range(NB)]
bg = torch.stack([Xt[i].mean(0) for i in bidx])              # (NB, G)

Ac = A.tocoo()
At = torch.sparse_coo_tensor(
    np.vstack([Ac.row, Ac.col]), Ac.data.astype(np.float32),
    (N, N), device=DEV).coalesce()

# per-batch node set V[b]: a cell is in V[b] if it IS in b or touches b --
# the same denominator cohesion uses in training (CLAUDE.md 9d).
onehot = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), bcode)),
                       shape=(N, NB))
touch = (A > 0).astype(np.float32) @ onehot
inV = (np.asarray(touch.todense()) > 0) | (bcode[:, None] == np.arange(NB))
inVt = torch.as_tensor(inV.T.astype(np.float32), device=DEV)   # (NB, N)

freq = pd.Series(cts).value_counts()
rare_types = set(freq[freq <= np.quantile(freq.values, 0.25)].index)
print(f"  rare types: {sorted(rare_types)}")

# start every arm from the stage-1 leiden assignment, i.e. where real training
# actually begins (d60_epoch0_leid). INIT_P is the assigned prototype's initial
# probability: sharp enough to carry the init, soft enough for gradients.
INIT_P = 0.7
INIT_LOGIT = float(EPS * np.log(INIT_P / (1 - INIT_P) * (K - 1)))
# logit clamp: with L in [-CLAMP, CLAMP] the largest reachable assignment
# probability is  exp(2*CLAMP/EPS) / (exp(2*CLAMP/EPS) + K - 1). CAP_P = 0.8
# keeps the effective number of prototypes per cell near the ~3 the real
# training trace reports (effk 3.1), and -- the reason it exists -- keeps the
# softmax off its saturated corner where the gradient is exactly zero.
CAP_P = 0.8
CLAMP = float(0.5 * EPS * np.log(CAP_P / (1 - CAP_P) * (K - 1)))
_z = np.load(os.path.join(REPO, "files", "volume_runs", DS,
                          "d60_epoch0_leid", "clusters_full.npz"))
init_lab = _z["labels"].astype(np.int64)
assert init_lab.shape[0] == N, (init_lab.shape, N)
init_lab = np.clip(init_lab, 0, K - 1)
_ev0 = None
print(f"  init from stage-1 leiden: {len(np.unique(init_lab))} prototypes, "
      f"INIT_LOGIT={INIT_LOGIT:.4f} (p_assigned={INIT_P})")


# ---------------------------------------------------------------- objectives
def cohesion_of(S):
    GS = torch.sparse.mm(At, S) + S                 # +S = cohesion_diag 1.0
    den = (inVt @ S)[bt].clamp(min=1e-8)            # (N, K) per-batch size
    return ((GS * S) / den).sum() / N


def xbatch_of(S):
    sizes, us = [], []
    for b in range(NB):
        i = bidx[b]
        Sb = S[i]                                   # (n_b, K)
        sz = Sb.sum(0)                              # (K,)
        ws = Sb.t() @ Xt[i]                         # (K, G)
        sig = ws / sz.unsqueeze(-1).clamp(min=1e-6) - bg[b].unsqueeze(0)
        c = sig - sig.mean(-1, keepdim=True)
        us.append(c / c.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        sizes.append(sz)
    tot = torch.zeros((), device=DEV)
    for b1 in range(NB):
        for b2 in range(b1 + 1, NB):
            v = ((sizes[b1] >= 1.0) & (sizes[b2] >= 1.0)).float()
            w = torch.minimum(sizes[b1], sizes[b2]).detach()
            tot = tot + (w * (us[b1] * us[b2]).sum(-1) * v).sum()
    return tot / N


# ---------------------------------------------------------------- evaluation
# group means via ONE sparse matmul instead of 1500 masked numpy means
def eval_labels(lab):
    gid = lab.astype(np.int64) * NB + bcode
    M = sp.csr_matrix((np.ones(N, np.float32), (gid, np.arange(N))),
                      shape=(K * NB, N))
    cnt = np.asarray(M.sum(1)).ravel()
    sums = M @ X                                                 # (K*NB, G)
    with np.errstate(invalid="ignore"):
        means = sums / np.maximum(cnt, 1)[:, None]
    bgm = np.stack([X[bcode == b].mean(0) for b in range(NB)])
    per, per_rare = [], []
    for k in range(K):
        rows = [b for b in range(NB) if cnt[k * NB + b] >= 10]
        if len(rows) < 2:
            continue
        S_ = np.stack([means[k * NB + b] - bgm[b] for b in rows])
        S_ = S_ - S_.mean(1, keepdims=True)
        nrm = np.linalg.norm(S_, axis=1)
        if (nrm <= 0).any():
            continue
        U = S_ / nrm[:, None]
        Cm = U @ U.T
        iu = np.triu_indices(len(rows), 1)
        per.append(float(Cm[iu].mean()))
        mk = cts[lab == k]
        if mk.size:
            vals, c2 = np.unique(mk, return_counts=True)
            if vals[c2.argmax()] in rare_types:
                per_rare.append(per[-1])
    per = np.asarray(per)
    return {"r_cross": float(np.median(per)) if per.size else float("nan"),
            "r_rare": float(np.median(per_rare)) if per_rare else float("nan"),
            "frac_lt_0.3": float((per < 0.3).mean()) if per.size else float("nan"),
            "n_span": int(per.size)}


# ---------------------------------------------------------------- the sweep
print(f"\n{'w_sig':>10}{'cohesion':>11}{'r_cross':>10}{'r_rare':>9}"
      f"{'chimera':>9}{'n_span':>8}{'n_used':>8}{'effk':>7}{'min':>6}")
print("-" * 71)
rows = []
for w in WEIGHTS:
    torch.manual_seed(SEED)
    # Start from the SAME place real training starts: the stage-1 leiden
    # assignment. Random init would measure "what is reachable from scratch",
    # which is a different and much harder question than "from where training
    # begins, how far can each objective go". INIT_P sets the assigned
    # prototype's initial probability -- sharp enough to carry the init, soft
    # enough that gradients still flow.
    L = torch.zeros((N, K), device=DEV)
    L[torch.arange(N), torch.as_tensor(init_lab, device=DEV)] = INIT_LOGIT
    L = (L + torch.zeros_like(L).normal_(0, 1e-3)).requires_grad_(True)
    opt = torch.optim.Adam([L], lr=LR)
    only_sig = w >= 1e8
    t0 = time.time()
    traj = []
    for it in range(STEPS):
        opt.zero_grad()
        S = F.softmax(L / EPS, dim=1)
        xb = xbatch_of(S)
        cz = cohesion_of(S)
        loss = -xb if only_sig else -(cz + w * xb)
        loss.backward()
        opt.step()
        # CLAMP THE LOGITS. Without this the softmax saturates to one-hot
        # within a few steps and the gradient dies -- measured: mean max
        # probability hits 1.000 and cohesion then does not move at all for
        # the remaining steps. The clamp caps the assignment sharpness at
        # roughly the sharpness the real model operates at (effk ~ 3 in the
        # training trace), so this measures "the best reachable AT the model's
        # own sharpness", which is the question we actually care about.
        with torch.no_grad():
            L.clamp_(-CLAMP, CLAMP)
        if it % 25 == 0 or it == STEPS - 1:
            traj.append((it, float(cz), float(xb)))
    with torch.no_grad():
        S = F.softmax(L / EPS, dim=1)
        coh = float(cohesion_of(S))
        lab = S.argmax(1).cpu().numpy()
        maxp = float(S.max(1).values.mean())
        effk = float((1.0 / (S ** 2).sum(1)).mean())
    ev = eval_labels(lab)
    rows.append({"w": (None if only_sig else w), "sig_only": only_sig,
                 "cohesion": coh, **ev, "n_used": int(len(np.unique(lab))),
                 "mean_max_p": maxp, "mean_effk": effk, "traj": traj})
    tag = "SIG-ONLY" if only_sig else f"{w:.2f}"
    print(f"{tag:>10}{coh:>11.4f}{ev['r_cross']:>10.4f}{ev['r_rare']:>9.4f}"
          f"{ev['frac_lt_0.3']:>9.1%}{ev['n_span']:>8}"
          f"{len(np.unique(lab)):>8}{effk:>7.2f}{(time.time()-t0)/60:>6.1f}")
    print(f"           traj (step, cohesion, xbatch): "
          + "  ".join(f"{t[0]}:{t[1]:.4f}/{t[2]:.3f}" for t in traj[:6]))
    out = os.path.join(REPO, "files", "tables", "d65_xbatch")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "freelogit_curve.json"), "w") as f:
        json.dump({"steps": STEPS, "lr": LR, "eps": EPS, "K": K,
                   "rows": rows}, f, indent=2)

print("\nreference points from REAL training (same r_cross definition):")
print("  stage-1 init   r_cross 0.702   chimeras  0.0%")
print("  scProto        r_cross 0.603   chimeras  9.9%")
print("  + signature    r_cross 0.635   chimeras  5.7%")
print("\nREAD: knee -> alignment is cheap up to that weight, pick it.")
print("      straight diagonal -> genuine conflict, weight cannot fix it.")
print("      flat r_cross -> objective ceiling; not a weight or encoder problem.")

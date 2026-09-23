# scVI at its own defaults — findings (Reviewer F5RB, this round)

From `notebooks/scvi_stage2_pancreas.ipynb`, run 04 Aug 2026, all three datasets complete.
New code: `models/scvi_backbone.py`, `trainers/scvi_proto.py`, `experiments/scvi_stage2.py`.

---

## 1. What he asked

> *"restricting all methods to d=8 ... replacing scVI's standard count likelihood with a
> Gaussian likelihood may disadvantage methods under nonstandard settings."*

So: scVI at **its own defaults** (d=10, ZINB, raw counts, 50 epochs), then change only the
downstream step — cluster its latent (Leiden / SEACells), or let scProto's Stage 2 keep
training that same encoder. Same weights, one process, same graph, same K, same metric code.

---

## 2. Modularity — holds on 3/3

Per-batch mean ± std, canonical graph.

| | Pancreas (n=8) | Lung (n=15) | Immune (n=5) |
|---|---|---|---|
| **scProto Stage 2 (scVI)** | **0.616 ± 0.082** | **0.650 ± 0.039** | **0.618 ± 0.059** |
| Leiden (scVI) | 0.379 ± 0.122 | 0.522 ± 0.077 | 0.364 ± 0.183 |
| SEACells (scVI) | 0.296 ± 0.078 | 0.327 ± 0.053 | 0.240 ± 0.130 |
| the paper's own scProto (scPoli) | 0.615 | 0.654 | 0.631 |

**Two facts to cite:** with scPoli gone entirely, Stage 2 lands on the paper's own number
(0.616/0.650/0.618 vs 0.615/0.654/0.631) — his Stage-1 hypothesis is answered directly;
and clustering the identical latent reaches only 0.24–0.52.

**Frozen-encoder control** (whole-graph modularity — a *different* statistic from the
per-batch table, do not mix them):

| | Pancreas | Lung | Immune |
|---|---|---|---|
| frozen scVI latent, zero Stage-2 steps | 0.394 | 0.342 | 0.251 |
| after Stage 2 on that same encoder | **0.699** | **0.727** | **0.674** |

Identical encoder, graph and K. +0.31 / +0.39 / **+0.42**.

Coverage: 14/14, 17/17, 15/16 — at or above the paper's own run (0.93 / 0.88 / 0.94).

---

## 3. Rare-cell metrics — level overall

Paired one-sided Wilcoxon, scProto Stage 2 as reference, Bonferroni. `x/n` = batches won.

**Rare F1**

| Method | Pancreas | Lung | Immune |
|---|---|---|---|
| scProto Stage 2 | 0.614 ± 0.218 (ref) | 0.557 ± 0.255 (ref) | **0.897 ± 0.030** (ref) |
| SEACells (scVI) | 0.639 ± 0.196, ns (3/8) | **0.682 ± 0.162**, ns (1/15) | 0.892 ± 0.049, ns (3/5) |
| Leiden (scVI) | 0.258 ± 0.276, **p=.012*** (8/8) | **0.689 ± 0.167**, ns (4/15) | 0.855 ± 0.108, ns (4/5) |
| scProto (scPoli) | 0.535 ± 0.208, ns (6/8) | 0.599 ± 0.176, ns (8/15) | 0.854 ± 0.138, ns (2/5) |

**Rare homogeneity**

| Method | Pancreas | Lung | Immune |
|---|---|---|---|
| scProto Stage 2 | **0.601 ± 0.173** (ref) | 0.560 ± 0.257 (ref) | **0.859 ± 0.064** (ref) |
| SEACells (scVI) | 0.588 ± 0.180, ns (3/8) | **0.658 ± 0.180**, ns (2/15) | 0.819 ± 0.082, ns (4/5, p=.19) |
| Leiden (scVI) | 0.317 ± 0.242, **p=.012*** (8/8) | 0.635 ± 0.182, ns (4/15) | 0.729 ± 0.158, ns (5/5, p=.094) |
| scProto (scPoli) | 0.565 ± 0.155, ns (5/8) | 0.578 ± 0.221, ns (7/15) | 0.827 ± 0.102, ns (4/5, p=.19) |

**One dataset in our favour (Immune, highest on both), one level (Pancreas — highest
homogeneity, F1 within 0.025), one against (Lung).** Nothing significant except Leiden on
Pancreas. Closest near-miss in our favour: Immune homogeneity vs Leiden, 5/5, p=0.094.

Three things to know before anyone asks:

- **Scale of the change:** scVI's rare F1 went 0.18 → 0.64 (Pancreas), 0.35 → 0.68 (Lung)
  moving from d=8+Gaussian to its own settings. His objection was substantively correct.
- **Lung is not specific to our variant:** the baselines beat *the paper's own* scProto
  run there too (0.689 / 0.682 vs 0.599).
- **What we hold throughout is precision:** highest rare-metacell precision of any arm on
  Pancreas and Lung (0.919, 0.853) with the lowest coverage/recall there — purer rare
  metacells across fewer of them, which accounts for the F1 gaps on those two.

---

## 4. Weak spots — ours, internal only

1. **Cross-batch homogeneity is behind on 3/3**: 0.205 vs 0.411 (0/8 batches) Pancreas,
   0.466 vs 0.598 Lung, 0.303 vs 0.407 Immune. **This is a scProto property, not a
   consequence of the scVI swap** — the paper's own scPoli-based run shows the same shape
   (0.323 / 0.508 / 0.349), and batch entropy agrees (scPoli 0.399/0.538/0.229 vs
   Stage-2-on-scVI 0.301/0.559/0.223). An earlier draft of this file attributed it to
   stock scVI's encoder not being batch-conditioned; the numbers do not support that.
2. **Early stopping selects on modularity, and the objectives disagree.** Pancreas rare F1
   was **0.683 at the final epoch, 0.614 at the best-modularity checkpoint**. We report the
   latter because that is how every scProto run in the paper is scored, but part of the
   deficit is model selection, not capability. Cheap and untested: full 20-epoch budget,
   score both checkpoints. Lung and Immune both stopped at epoch 9 of 20.
3. **Stage 2 does not improve the latent's own rare-cell affinity purity** (0.473 vs 0.515,
   0.594 vs 0.655, 0.863 vs 0.868). The gain shows up at the assignment level, not in the
   embedding. Diagnostic only — not in any reviewer document.
4. **d=10, not the paper's d=8** — deliberate, so the comparison runs at the baseline's
   setting, but this arm is therefore not the paper's own configuration.

---

## 5. Where it is written

- `final-comment/rev1-F5RB-final.md` §1 (table, all three datasets) and §2 (Stage 1)
- `final-comment/ac-comment.md` §2
- `experiment-results/scvi_default_stage2_results.md` — full log, protocol, caveats

**Two traps:**

- **Superseded numbers — do not reuse:** an earlier draft cited modularity `0.603` and rare
  F1 `0.683±0.181` with "scProto leads on all three metrics". Those came from the final
  training epoch, not the best checkpoint. Corrected to `0.616` and `0.614`.
- **The notebook's own "scProto (scPoli Stage 1)" row is not the canonical run.** It
  resolved to folders giving 0.601 (Pancreas) and 0.664 (Lung) — `extract_model_key`
  strips `_cvae_e50`, so several sweep variants normalise to one key and `_resolve_run_dir`
  takes the last alphabetically. Cite 0.615 / 0.654 / 0.631 from the paper, as the
  documents do; pin the folder by exact name if that row is ever needed.

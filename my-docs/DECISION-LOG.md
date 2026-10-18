# Decision log — scProto

**One line per decision, newest last.** Each line is the ADR's own Summary,
copied verbatim. Detail lives in `adr/D##-*.md`.

This file is an INDEX. If a line needs to grow, that content belongs in its ADR.

```
cp adr/TEMPLATE.md adr/D56-short-kebab-title.md
# write the Summary (Y-statement), then copy it here as one row
```

Row format:

```
- **D56** `status` 2026-10-06 — In the context of <use case>, facing <concern>,
  we decided for <option> and neglected <alternatives>, to achieve <benefit>,
  accepting <downside>.
  -> adr/D56-short-kebab-title.md
```

Status: `proposed` | `running` | `works` | `fails` | `abandoned` | `superseded`.
`inconclusive-because-<X>` is temporary and must resolve.
IDs are never reused or renumbered. A closed ADR is never edited — a wrong
decision gets a new ADR, and both lines stay here.

---

## Entries

- **D46** `fails` 2026-09-25 — community + proto_recon at gradient-balanced weights.
  -> `archive/context/md/decisions.md` (pre-ADR, not in Y-statement form)
- **D55** `works` 2026-09-26 — lambda from gradient SHARE, not authority/span, all four terms.
  -> `archive/context/md/decisions.md` (pre-ADR, not in Y-statement form)

- **D56** `works` 2026-10-06 — In the context of choosing the four loss weights,
  facing that a single-minibatch gradnorm probe varies 2.0x on the same frozen
  checkpoint and value-balancing disagrees with gradient-balancing by 34x, we decided
  for a FROZEN full-pass gradnorm MEDIAN (`lambda_i = share_i / median_i`) over
  learnable GradNorm weights and value-based grid search, to achieve a reproducible,
  share-portable calibration, accepting ~5x per-step probe cost and a ~30% run-to-run
  spread on the cohesion terms, since traced to an unseeded prototype init (-> D57).
  -> `adr/D56-lambda-from-frozen-full-pass-gradnorm-median.md`

- **D58** `works` 2026-10-06 — In the context of a 4-term loss whose terms had
  never been individually tested, facing that we could not say whether community
  earns its place when cohesion is already there nor which cohesion kernel does the
  work, we decided for one dose curve per term around D56's share vector B — where
  dose=0 IS that term's ablation arm — over running ablation and sweep separately,
  to achieve both results from one set of runs inside a $5 cap, accepting one term
  varied at a time, a single dataset, and one seed per arm. Recon is most
  load-bearing and the only term with a generalization gap; community is weakest;
  aff buys batch mixing and gram buys rare-type recovery. Two pre-registered
  predictions falsified.
  -> `adr/D58-ablate-each-loss-term-around-share-vector-B.md`

- **D60** `proposed` 2026-10-18 — In the context of stage 2 measurably destroying
  what stage 1 provides (sigma collapsed 130,000x; cross-batch 0.963->0.856; test
  cohesion below the untrained init), we decided for a redesigned stage 2 — cohesion
  with prototypes DETACHED, responsibility-weighted GM-KL (sigma=1) as the prototypes'
  only teacher and the encoder's anti-collapse+margin term, recon kept as preservation
  anchor; 3 arms (init x community) against same-init epoch-0 bars — over proto_recon
  (redundant centroids), plain KL (the rare-merging pressure itself), and batch-mixing
  terms (forbidden), to achieve test cohesion above the untrained init WITH the init's
  cross-batch alignment, accepting one dataset, one seed per arm.
  -> `adr/D60-gmkl-recipe-detached-protos-three-arms.md`

<!-- new entries here, newest last -->

---

## Pre-ADR decisions (D01–D55) — frozen in the monoliths

Not being migrated. Frozen as written, which is what keeps the runs they
describe interpretable. Cite by ID; read in the archive.

```
archive/context/md/decisions.md       2,059 lines, 21 numbered sections
archive/context/md/EXPERIMENTS.md       785 lines, runs and outcomes
archive/context/md/LOSS_FORMULAS.md     595 lines, every term + its gradient
```

IDs referenced across the project:
`D7 D9 D10 D14 D16 D17 D18 D26 D29 D31 D32 D37 D39 D41 D45 D46 D48 D55`

Only D46 and D55 have their own headings in `decisions.md`; the rest are
referenced inline. To find one: `grep -rn "\[D29\]" my-docs/archive/ CLAUDE.md`

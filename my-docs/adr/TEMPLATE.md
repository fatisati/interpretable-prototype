---
id:             D56
title:
status:         proposed
date_opened:    2026-10-06
date_closed:
supersedes:
superseded_by:
tags:           []
datasets:       []
protocol:
comparable_to:
runs:           []
lit:            []
paper:
# optional, and only until a run manifest (run.json) exists -- see README
K:
code:
cost:
---

# D56 — title


## Summary

> Y-statement. All six clauses are required — if you cannot name a downside, it
> is a preference, not a decision; if you cannot name what you neglected, you
> did not choose. `because` is optional. Reword if the decision does not fit
> the shape, but keep all six parts.
>
> Then COPY THIS into `../DECISION-LOG.md` as this entry's row.

In the context of **use case / component**,
facing **concern / the quality at stake**,
we decided for **option**
and neglected **the alternatives**,
to achieve **benefit**,
accepting **downside**,
because **rationale** *(optional)*.

> Optional, one line: why this decision mattered more than its size suggests.

**Why it matters:**


## Context and Problem Statement

> What looked wrong, and the measured number that showed it.


## Decision Drivers

> What narrows the choice: a deadline, a cost, a frozen scoreboard, a
> constraint you refuse to break.

-


## Prior Art

> Per paper: what it claims / what we reuse / what it does not cover here.

-


## Considered Options

> One line each: why it might work ; what it costs. Include "do nothing".

-


## Decision Outcome

> The chosen option, the mechanism that makes it work, and the cost you
> explicitly accept.


### Confirmation

> Written BEFORE the run. Never edited afterwards.

- **Expect** —
- **Would falsify this** —
- **Scale/range check that must pass first** —


### Run

> Pointers only. The mechanics (K, seed, versions, cost) live in the run's own
> metrics.json and model-folder name — do not retype them here.

- **Runs** —
- **Deliberate departures from CLAUDE.md 9d, and why** —


### Consequences

> Every number with the path it was read from. Means with their std.

| metric | value | read from |
|---|---|---|
|        |       |           |

- **Good** —
- **Bad** —
- **Surprised me** —
- **Deviations from Confirmation** —


### What went wrong on the way

> Optional, and the most reusable section in the file. Every dead end, wrong
> assumption and aborted attempt, with the file:line or number that exposed it.
> Superseded numbers stay here -- replaced by re-measurement, not by argument.

1.


## Verdict

> works / fails / abandoned / superseded-by D##, then one line on what follows.
> If it fails, say what it rules out — that is the durable result.


## More Information

> Thread page, run dirs, result tables, notebook, related D##.

-


---

### Field reference

Frontmatter holds **conceptual** facts only — things that scope what the claim
means. Mechanical facts (K, seed, git sha, package versions, cost, script path)
belong to the run, not to the decision: they are recorded in the run's
`metrics.json` and model-folder name, and `runs:` is the pointer to them.

| field | meaning |
|---|---|
| `title` | same string as the row in `../DECISION-LOG.md` |
| `status` | proposed / running / works / fails / abandoned / superseded |
| `datasets` | which datasets the claim was tested on — its generalization scope |
| `protocol` | `NH-heldout` or `full-dataset`; defines which cells the metric means (CLAUDE.md 7a) |
| `comparable_to` | the frozen baseline set this is measured against — also fixes the K, since metrics move mechanically with it |
| `runs` | run ids / model-folder names: the pointer to every mechanical fact |
| `lit` | short refs, PDFs under `literature/` |
| `paper` | the table or figure this claim feeds |
| `K`, `code`, `cost` | optional; mechanical, so they belong to the run manifest. Fill them in only while no `run.json` exists, since today nothing else records the git sha. |

Full rules, examples and template provenance: `README.md`.

A minor or mechanical decision uses this same template — fill in Summary and
Verdict, leave the rest as a dash. The Y-statement is the irreducible minimum.

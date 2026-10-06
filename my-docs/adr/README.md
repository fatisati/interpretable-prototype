# ADR folder — how this works

One decision per file: `adr/D##-short-kebab-title.md`, plus one line in
`../DECISION-LOG.md`. **One template**, `TEMPLATE.md`, for every decision —
a minor one just leaves most sections as a dash.

Highest ID so far: **D58**. Next is **D59**. IDs are never reused or renumbered.

```
cp adr/TEMPLATE.md adr/D56-short-kebab-title.md
# write the Summary, then copy that line into ../DECISION-LOG.md
```

## Two rules that carry everything

**One decision per file, and it must be able to CLOSE.** The unit is something
that ends with `works` / `fails` / `abandoned`. If it cannot close, it is not an
ADR — it is a thread page (living summary, rewritten in place) or an open
question. One file per *topic* is what produced the old 2,059-line monolith.

**Append-only.** Never edit a closed entry's reasoning. A wrong decision gets a
NEW ADR; set the old one's `superseded_by` and the new one's `supersedes`. This
is what keeps old runs interpretable — the entry still describes the config that
produced its numbers.

## The Summary line — Y-statement

Every ADR opens with a one-line Summary, and that same line is its row in the
decision log. Write it once, copy it.

```
In the context of <use case / component>,
facing <concern / the quality at stake>,
we decided for <option>
and neglected <the alternatives>,
to achieve <benefit>,
accepting <downside>,
because <rationale>                      <- optional 7th clause
```

All six clauses are required; `because` is optional. **If you cannot name a
downside, it is a preference, not a decision. If you cannot name what you
neglected, you did not choose.** `and neglected` is the one-line echo of the
*Considered Options* section — it is what stops the log from reading as if each
decision were the only possibility. Reword the shape if a decision does not fit
it, but keep all six parts.

Origin: the WH(Y) template, Olaf Zimmermann (SATURN 2012), published as Zdun,
Zimmermann et al., *Sustainable Architectural Design Decisions*, IEEE Software
2013. Named for the Y in "why" — the structure makes the why unavoidable.

```
rules
  1. a CLAIM WITH A DIRECTION, not a topic — a topic never closes
  2. names the thing changed AND the direction of the change
  3. no outcome words — works/fails live in the `status` field
  4. no "investigate" / "explore" — those are open questions, not decisions
  5. readable with zero context, six months later
```

```
GOOD
  In the context of lambda choice across datasets, facing lambdas that are not
  portable between graphs, we decided for gradient-share calibration of all four
  terms to achieve comparable influence per term, accepting a single static probe
  snapshot rather than re-probing mid-run.

  In the context of the diagonal loss term, facing CPM's charge being unable to
  move a cell for similarity reasons, we decided for cohesion to replace nassoc
  to achieve similarity-driven assignment, accepting that gamma's derivation is
  abandoned with it.

BAD                                  why
  lambda calibration                 topic — never closes
  community loss experiments         no decision, no direction
  fixing the dead prototype problem  hides the choice
  D55 follow-up                      unreadable without opening D55
  gradient share WORKS                outcome belongs in `status`
```

If the line cannot be written this way, it is not a decision yet — put it in the
open-questions page and open the ADR once there is something to choose.

## Conceptual vs mechanical — what goes where

```
CONCEPTUAL  -> the ADR          the problem, the prior art, the options, the
                                choice, the prediction, what it rules out
MECHANICAL  -> the run manifest K, seed, git sha, versions, cost, script path,
                                the fully resolved config
```

An ADR must still make sense when the code is gone. `K=800` does not; "cohesion
replaces nassoc as the diagonal term, because CPM's charge cannot move a cell
for similarity reasons" does.

Mechanical facts are **machine-written, never hand-typed**, into the run's own
directory (`metrics.json` and a `run.json` beside it). `runs:` in the
frontmatter is the pointer. Note the model-folder name is **not** a sufficient
record of the config: per CLAUDE.md §9 it only includes values that differ from
the default, and truncates strings to 4 characters.

## The project-specific frontmatter fields

Not in MADR. Each is conceptual — it scopes what the claim *means*.

| field | why |
|---|---|
| `protocol:` NH-heldout / full-dataset | §7a: the scoreboard mixes two protocols, and nothing on disk says which one a claim was made under. Wrong mask → numbers *silently* comparable to nothing. |
| `comparable_to:` | §9d: K changed from 200 to the dataset default, so every saved SEACells baseline is K=200 and **not comparable**. Naming the baseline set freezes the scoreboard for that entry, and fixes the K with it. |
| `datasets:` | the claim's generalization scope — "tested on lung only" is a conceptual limit, not a config value. |
| `paper:` | links the decision to the claim it supports, so a reviewer question maps to an entry. |

Three more in the body:

| section | why |
|---|---|
| departures from §9d, and why | §9d is THE config. Several real bugs were "a default nobody chose" (`umap_steps_per_epoch=1000` = 24% of a pass; `coh_mem_hlife=500` = a measured no-op). A *deliberate* departure is a decision — record it. |
| scale/range check in Confirmation | §9b: two shipped bugs stayed invisible for weeks because nobody printed a term's value against its theoretical range. |
| every number with its read-from path | a number with no path is unverifiable in six weeks. Means always with std (§7). |

`K`, `code` (git sha) and `cost` are **optional** fields. They are mechanical and
belong to the run manifest — but no `run.json` exists yet, and nothing else in
this project records the git sha at all, so until it does they are the only
record and are worth filling in. Drop them from new entries once runs stamp
their own provenance. `seed` and package versions stay out: name them in the
*Run* section only when they deviate from the default.

## Template provenance

`TEMPLATE.md` is **MADR 4.0.0** (adr.github.io/madr, 2024-09-17), itself from
Nygard's original four sections. Verbatim MADR headings: *Context and Problem
Statement*, *Decision Drivers*, *Considered Options*, *Decision Outcome* →
*Consequences*, *Confirmation*, *More Information*.

- Dropped: MADR's optional *Pros and Cons of the Options* — folded into
  *Considered Options*, one line per option.
- Deviations: `date` split into `date_opened`/`date_closed` (an experiment
  pre-registers before and reports after); `decision-makers`/`consulted`/
  `informed` dropped (single researcher); MADR's *Confirmation* repurposed as
  the **pre-registered success criterion**.
- Added for research: *Summary* (Y-statement), *Prior Art*, *Run*, *What went
  wrong on the way*, *Verdict*.
- The Summary's Y-statement form is not merely MADR's companion — **MADR is
  derived from it**. MADR's chosen-alternative justification is the Y-statement's
  *to achieve*, and its consequences are the *accepting that*. Used here as the
  one-line summary rather than as a separate template.

## Related docs — different update rules, do not merge them

| doc | kind | update rule |
|---|---|---|
| `adr/D##-*.md` | decision record | append-only, never edited once closed |
| `../DECISION-LOG.md` | index | one line per ADR, copied from its Summary |
| `CLAUDE.md` §9d | living spec: what is true NOW | rewritten in place; states, never explains; cites D## |
| `archive/context/md/LOSS_FORMULAS.md` | living spec: formulas + gradients | rewritten in place |
| thread pages | living summary of ONE open question | rewritten in place; links down to D## |
| `metrics.json`, `run.json`, `*.csv` | run manifest | machine-written; never retyped into prose |

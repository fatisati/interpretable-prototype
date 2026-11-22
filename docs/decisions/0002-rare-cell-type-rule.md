# 0002 — Which rule decides that a cell type is "rare"

Date: 2026-11-22. Status: accepted (docs). Code default not yet switched — see "Follow-up".

## Context

Task 1 (`tasks/rare_cell_preservation.md`) scores every method separately on rare cell
types, so the result depends on which (cell type, batch) pairs count as rare. Rarity is
defined **per batch**: a type's *frequency* is its share of that batch's cells (30 of
1,000 cells = 3%). A rule then marks each (type, batch) pair as rare or not.

The task doc's original rule ("std") found almost no rare types on our first real
dataset, so we compared three rules. All three are already implemented in
`baselines/metrics.py` (`rare_group_table`, argument `rule`).

## Options considered

### 1. "std" — the original rule

Rare if frequency < batch mean − 1 × batch standard deviation (standard deviation = how
spread out the values are).

**Pros:** adapts to each batch; simple formula.
**Cons:** breaks on uneven batches, which are the normal case in single-cell data. One
or two huge types (e.g. 60% of the batch) make the spread so large that the cutoff
falls below 0%, so nothing can be rare.

### 2. "log_std" — same rule on log10(frequency)

Rare if log10(frequency) < batch mean − 1 × batch standard deviation, all on the log
scale (log scale = 1%, 10%, 100% are equal steps apart).

**Pros:** still adapts to each batch, but huge types no longer stretch the spread, so it
works on uneven batches.
**Cons:** a bit harder to explain in one sentence.

### 3. "fixed" — frequency below 1% of the batch

**Pros:** simplest to explain; identical across batches and datasets; familiar to
reviewers.
**Cons:** the 1% is an arbitrary choice, and it ignores context — 1% means something
different in a batch with 5 types than in one with 40.

## Evidence — Heart Cell Atlas (14 batches, 145 (type, batch) pairs)

From the saved `rare_groups.csv` tables of the scVI run
(`runs/rare_cell_preservation/heart_cell_atlas/scvi/metrics*/`):

| Rule | Rare pairs | Batches with no rare type | Largest rare frequency |
|---|---|---|---|
| std | 6 | 12 of 14 | 0.6% |
| log_std | 27 | 0 of 14 | 1.6% |
| fixed (1%) | 26 | 3 of 14 | 0.96% |

With "std", the per-batch rare metrics would be empty for 12 of 14 batches.

## Decision

**Default rule: "log_std"**, with threshold k = 1 standard deviation. The existing
fallback stays: a batch with fewer than 4 cell types uses the lowest quartile of
frequencies instead. "std" and "fixed" remain available as what-if checks (a re-run
with another rule, saved separately), not as the headline result.

## Considered and not adopted (for now): a minimum cell count

Some rare pairs have very few cells (down to 1), and a score from 1–3 cells is mostly
noise. We considered "a rare pair must have ≥ 10 cells". On the heart data with
log_std, that removes 16 of 27 rare pairs and leaves only 6 of 14 batches with any rare
type — it throws away most of the signal this task is about. Not adopted. If small
groups make results too noisy, a better option is to keep them rare but report the
cell count next to each per-pair score (already saved in `rare_groups.csv`), or to
weight per-pair averages by cell count.

## Follow-up (not done in this decision)

- Done 2026-11-22: `compute_metrics.OFFICIAL_RARE_RULE = "log_std"` -- log_std writes
  to `metrics/` and feeds `results/.../official/`; "std" and "fixed" write to
  `metrics__rare-<rule>/` (what-if copies only).
- Re-check this table on each new dataset (HLCA etc.) — the decision is based on one
  dataset.

# scProto

A method for building metacells that preserve rare cell states across
batches, guided by any cell-cell affinity graph (transcriptomic or spatial).

This branch (`clean-v2`) is a fresh start: an orphan branch with no shared
history, built to keep baselines, gap-checks, and the method itself
organized from day one.

## Folders

- `docs/` -- roadmap and decision records (ADRs).
- `tasks/` -- one file per research task: the gap, the metrics, why --
  written before running anything, never edited after results exist.
- `datasets/` -- how to get each dataset, preprocessing, checksums, splits.
- `baselines/` -- one file per baseline method, thin adapter code only
  (calls the real library, doesn't reimplement it).
- `scproto/` -- the method's own code.
- `runs/` -- raw output per execution: the exact config used + metrics.
  Write-once, never hand-edited. One subfolder per run.
- `results/` -- one file per task: baseline vs. scProto comparison,
  pulled from `runs/`, plus the written conclusion/verdict.
- `tests/` -- sanity checks that baselines and scProto both run correctly.

## Workflow

1. Write a task in `tasks/<task_name>.md`: the gap, the metrics that
   decide if it's real, and why -- before running anything.
2. Run baselines, save each to `runs/<task>/<dataset>/<baseline>/`.
3. Write the gap verdict in `results/<task>.md`.
4. If the gap is real, run scProto, save to `runs/<task>/<dataset>/scproto/`.
5. Update `results/<task>.md` with the comparison and final conclusion.

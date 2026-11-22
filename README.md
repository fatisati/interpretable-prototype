# scProto

A method for building metacells that preserve rare cell states across
batches, guided by any cell-cell affinity graph (transcriptomic or spatial).


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
2. Prepare the dataset (`datasets/prepare.py`), then the shared evaluation
   setup for that task -- train/test split, ground-truth affinity graphs,
   and cluster count (`baselines/prepare_eval.py`). Every baseline below
   reads these; none of them create their own copy -- see
   `docs/architecture.md` for the full pipeline design.
3. Run each baseline, save to `runs/<task>/<dataset>/<baseline>/`.
4. Compute metrics from each baseline's saved output
   (`baselines/compute_metrics.py`) -- rerunnable on its own, without
   retraining anything, if a new metric comes up later.
5. Build the comparison tables (`python baselines/build_results.py --task <task>`
   -> `results/<task>/official/tables.md`, generated, never hand-edited), then write
   the gap verdict by hand in `results/<task>.md`, citing those tables.
6. If the gap is real, run scProto, save to `runs/<task>/<dataset>/scproto/`.
7. Update `results/<task>.md` with the comparison and final conclusion.

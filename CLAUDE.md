# Working conventions for this repo

## Reproducibility: always record exact versions

Any time code depends on the environment (package versions, Python version) to produce a
result that matters — preprocessing a dataset, running a baseline, training a model —
record the exact versions used, next to the result. Package behavior changes across
versions (a different `scanpy` HVG default, a different `scvi-tools` training default)
silently changes output; "ran it with scvi-tools" is not reproducible, "ran it with
scvi-tools 1.5.0.post1, torch 2.13.0+cpu, python 3.13.11" is.

Where to put it: an "Environment" section in the relevant `.md` doc (e.g. a dataset's own
file in `datasets/`), not a single repo-wide version file — different datasets/baselines
pull in different libraries over time, and a global file goes stale the moment one changes.

## Preprocessing

Preprocess each dataset once, save the output, and have every baseline/trainer read the
saved file — never re-run normalization/HVG selection inside each baseline. See
`datasets/scripts/prepare.py`'s own docstring for why (fair comparison across methods requires
identical input).

## Commands and run records

- New or changed command for any stage -> update `docs/runbook.md` (the one place commands live;
  other docs link there instead of copying).
- Every stage script records what it ran: call `record_run(<its json sidecar>)` from
  `evaluation/run_record.py` after saving output. That writes the exact command, git commit, and
  times next to the result, so nobody has to log commands by hand.

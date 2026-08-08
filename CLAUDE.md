# CLAUDE.md

scProto (interpretable prototype / metacell learning). Experiments run on **Colab**
against Drive. Colab **will** disconnect mid-run, and we are time-constrained.
Every rule below follows from those two facts.

---

## 1. Notebook setup — copy the existing one, don't invent

Cell order for every new notebook:

1. `from google.colab import drive; drive.mount('/content/drive')`
2. pip install block (below) — **exact**
3. Markdown/comment: restart the runtime now (numpy/scipy/anndata are C-extension
   linked; an in-process upgrade won't take effect on imported modules)
4. Import-verification loop (copy `_checks` from
   `neurips_manuscript/rebuttle/notebooks/component_ablation.ipynb`)
5. `%run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py`
6. Extra imports only — whatever `nb_setup.py` doesn't already give you
7. Header markdown: purpose, scope, "read before running" warnings

```python
!pip install -q scarches faiss-gpu-cu12 scib-metrics
!pip install git+https://github.com/dpeerlab/SEACells.git --quiet --no-deps
!pip install numpy scipy --upgrade -q
!pip install -q palantir harmonypy
!pip install -q "numpy==1.26.4" "scipy==1.13.1"
!pip install --upgrade --force-reinstall numpy cupy-cuda12x
!pip install "numpy<2.3"
```

- Multi-pass **on purpose**. Collapsing to one pin, or merging lines, has been tried and
  fails. Leave it alone. No `-c constraints.txt`.
- SEACells always `--no-deps`, and installed **even with no SEACells baseline** —
  `prot_init='waypoint'` calls its diffusion-map MaxMin code.
- Extra packages (`leidenalg`, `igraph`, `umap-learn`, `bbknn`, `scvi-tools`) go on
  their own line **after** the block, unpinned.
- pip's log is not proof of success — resolver backtracking prints
  `Getting requirements to build wheel` even on clean installs. Only the import loop counts.
- `nb_setup.py` already does autoreload, `sys.path`, the `anndata.read` shim, all
  `*_DIR` env vars, `chdir`, dpi, and imports `get_trainer`, `run_mc_task`, all
  `LAMBDA_*`, `train_seacell`/`eval_seacell_task1/2/3`, `train_metaq`, `train_sure`,
  plus `paper_figures` + `result_tables`. **Don't re-import any of it.**
- Keep the jupytext `.ipynb` + `.py:percent` pairing.

## 2. Notebook best practice — thin, cheap, fast

- **Notebooks orchestrate; they don't implement.** Any function longer than a few lines
  goes in `interpretable_ssl/` and is imported. Logic in a notebook can't be reused,
  reloaded, or fixed once for all callers (see how the Harmony/ComBat notebooks share
  `evaluation/batch_correct_baselines.py`). Config cell + call cells + table cells only.
- **One AnnData in memory at a time.** After a dataset finishes: `del ad, mc_ad, t;
  gc.collect()` (and `torch.cuda.empty_cache()`). Never hold two datasets at once —
  that's what OOM-kills a Colab runtime mid-run.
- Don't `.copy()` an AnnData to add a column, don't densify a sparse matrix, don't
  re-read the same `.h5ad` in several cells — read once, reuse the variable.
- Keep cells small and single-purpose, so a disconnect loses one step, not an hour.
- Print progress (dataset, step, elapsed) so a stalled run is visible.
- Delete/`%reset` bulky intermediates once written to disk.
- Prefer `n_comps=50` PCA and the existing cached affinities over recomputing.

## 3. Notebook folder structure — one folder per notebook, typed subfolders

A notebook is **not** a loose `.ipynb` in a flat directory. Each one gets its **own
folder**, named for the experiment, and **every file type it produces gets its own
subfolder** inside it:

```
notebooks/<experiment_name>/
  <experiment_name>.ipynb     the notebook
  <experiment_name>.py        jupytext :percent pair (keep them paired)
  README.md                   what it answers, which dataset runs first, how to resume
  figures/                    .png / .pdf
  tables/                     .csv result tables
  notes/                      .md analysis / decisions
  tmp/                        scratch — never committed
```

- **Never add a new file to the repo root**, and never leave a new file loose in
  `notebooks/`. If a notebook makes a **new file type**, it gets a **new subfolder**
  with a descriptive name — never a flat dump alongside the notebook.
- Deeper substructure goes **by dataset, then by experiment**:
  `notebooks/<experiment>/tables/<dataset>/<what>.csv`.
- Filenames sort and self-describe: `<dataset>_<experiment>_<what>.<ext>`. No `tmp`,
  `new`, `final`, `test2`, `(1)`, or `.conflict` files committed.
- Notebook outputs stay **inside that notebook's own folder** (plus the run dirs under
  `MODEL_DIR`), so a rerun overwrites its own results and touches nothing else.

Repo-level folders these sit alongside:

```
interpretable_ssl/      library code — all reusable logic (see §2)
graphs/                 cached affinity .pkl (named by get_affinity_path)
files/                  cross-notebook shared artifacts — subfolder by type
results/<method>/       metrics output
neurips_manuscript/     paper, rebuttal, rebuttal notebooks
<MODEL_DIR>/<ds_id>/    (Drive) per-run checkpoints, metrics.json, per-MC CSVs
```

## 4. Load-if-exists is the DEFAULT, in every notebook

**Assume the notebook dies halfway.** The default behavior of every step, always:

> **check whether the output file exists → if yes, load it and report it → only compute
> when it's missing.** Recompute happens *only* when the user explicitly asks for a
> rerun.

- Put `FORCE_RERUN = False` in the config cell of every notebook. `False` (the default)
  means load-and-report; the user flips it to `True` to force recomputation. Never
  recompute just because a cell was re-run.
- This applies to **everything with an output file** — training, eval, tables, figures,
  affinities, baselines — not just training.
- Report the loaded result the same way a fresh one is reported, and print where it came
  from (`loaded from <path>` vs `computed`), so it's never ambiguous which you're seeing.

Use the mechanisms that already exist rather than inventing new ones:

- `find_metacells(load_pretrain=True)` — reuse `pretrain_checkpoint.pth`
- `load_umap=True` — skip training, load checkpoint
- `load_umap=True, skip_eval=True` — load model only; does **not** re-run eval or
  overwrite `clusters.npz` / `metacells.h5ad`
- SEACells: `load_seacell=True`; `metrics.json` is checked before recompute
- Affinities: check `get_affinity_path()`, load instead of regenerating

Never put an unguarded expensive step inside a dataset loop — write each dataset's
results before the next starts. Prefer many small resumable steps over one long cell.

## 5. Minimal epochs

- **Pretrain: `cvae_epochs=50`. scProto finetune: `train_epochs=20`.**
- Scale early stopping to fit 20 epochs: `eval_freq=2`, `patience=4`. (The old
  `eval_freq=3, patience=6` was for 50 epochs and effectively disables early stopping at 20.)
- Keep `batch_size=1024`, `umap_steps_per_epoch=500`.
- Don't raise epochs unasked. If something looks undertrained, **say so** rather than
  silently training longer.

## 6. One dataset first, fully

Run and **report one dataset completely** — train, eval, results table — before any
other dataset starts. If time or Colab runs out we still have one reportable result
instead of partial results everywhere. Say in the header which dataset is first and why.
Remaining datasets then run one at a time, each saving before the next.

## 7. Always report std

- No bare means. Every metric gets its std.
- **`metrics.json` is the source of truth for stds.** Eval writes every std scalar
  directly (`mc_metric_utils.py` for purity / niche purity / batch entropy /
  per-batch modularity; the four `eval_*_task3` functions for `std_ct_niche_rbo`).
  A std is a few bytes — always write it next to its mean, never make a table
  re-derive it.
- The per-MC CSVs (`purity_per_mc.csv`, `batch_entropy_per_mc.csv`,
  `niche_purity_per_mc.csv`, `ct_niche_rbo.csv`, `modularity_per_batch.csv`, `soft_*`)
  are now only a **backfill for runs saved before that change**. `result_tables.py`
  reads them **only** for keys `metrics.json` doesn't already have, and never overwrites
  a value that's already there.
- **A missing std is loud, not silent.** `show_table` prints a `WARNING: no std found
  for N column(s)` listing every `dataset/metric` shown as a bare mean. If you see it,
  re-run eval for those runs rather than reporting the bare mean.
- New metric → compute its std where the mean is computed, write both into
  `metrics.json`, and register the `(mean_col, std_col)` pair in `MEAN_STD_PAIRS` plus a
  display name in `result_tables.py`. Columns already named `*_mean`/`*_std` are paired
  automatically.
- If a value genuinely has no std, say so instead of printing a lone number.

## 8. What to save

**Save anything expensive to compute. Don't save what's cheap to recompute from what's
already saved.** Save immediately after computing, not at the end of a cell chain.

Always save: SEACells/archetypal results (slow — save as soon as they exist, before any
eval), `pretrain_checkpoint.pth` and the UMAP checkpoint, affinities via
`save_affinity()`, diffusion maps / eigendecompositions and other precomputed targets,
`metrics.json`, and the per-MC CSVs from §7.

Never save: dense copies of sparse matrices, PCA that recomputes in seconds, per-cell
embeddings that are one forward pass from a saved checkpoint, a full AnnData copy when
only one `.obs` column changed (save the column), anything re-derivable fast from a
saved checkpoint.

**Store content efficiently rather than storing less.** The SEACells path is the model:
a full per-tag `seacell_sc.h5ad` was hundreds of MB and mostly identical across tags, so
`save_seacell` now writes one shared `seacell_sc_base.h5ad` + hash manifest per dataset,
and each tag's `seacell_sc.h5ad` is a **small delta** of only the fields that actually
differ (`load_seacell` reconstructs it). Do the same for any new large artifact: dedupe
against a shared base, save a delta — don't drop the data.

Rule of thumb: expensive + small → always save. Cheap + huge → never. Expensive + huge →
dedupe it down first; if it's still huge, save it and say so in the notebook.

## 9. New config options — name every feature to be unique in 4 characters

**`model_name.py:44` truncates string values to their first 4 characters.** That makes
"4-char-distinct" a **naming requirement, decided when you invent the option** — not a
check done afterwards. Two values that share their first 4 chars produce the *same model
folder*, so the two runs silently overwrite each other's checkpoints.

```
string option:   <abbrev>-<str(value)[:4]>
numeric option:  <abbrev><value>          # not truncated, safe
```

- **Choose feature/value names that are already unique in their first 4 chars.**
  `'full'` vs `'diffusion'` → `full` / `diff`, fine. `'diffusion_a'` vs `'diffusion_b'`
  → both `diff`, **broken**; name them `'dfa'` / `'dfb'` instead.
- Never special-case the naming code to work around a bad name — rename the value.
- Numeric options are safe on this point (no truncation).

Then, for the option itself, all three must be present:

1. **Default** in `configs/defaults.py` (`get_defaults()`), with an inline comment. Must
   be the behavior-preserving value (usually `0`/`False`/off) so existing folder names
   don't change. Only values `!= defaults.get(key)` appear in the name at all.
2. **Abbreviation** in **`interpretable_ssl/constants.py`** (`ABBREVIATIONS`) — **every
   new feature goes in here, no exceptions**, however minor it seems. `model_name.py`
   only includes a key if `key in ABBREVIATIONS`; a key that isn't there is **silently
   dropped from the folder name**, so two runs differing only in that option get the
   same folder and clobber each other. Keep the abbreviation 3-6 chars and unique
   against the existing entries. Adding the default in `defaults.py` without adding the
   abbreviation here is the most common way this breaks — do both in the same edit.

   > **Edit `interpretable_ssl/constants.py`, NOT the `constants.py` at the repo root.**
   > The root file is a stale duplicate — `model_name.py` does
   > `from interpretable_ssl.constants import *`, so the root copy is never consulted
   > for model naming. It's already missing every recent feature (`lambda_sim_recon`,
   > `sim_recon_*`, `proto_usage_mode`, `covet_alpha`, ...). Adding an abbreviation
   > there looks correct and does nothing.
3. Print `t.get_model_name()` once and confirm the option appears with a distinguishable
   value.

## 10. Explanation style

Define every term and metric inline. Formulas as **plain-text code blocks with readable
variable names** — no LaTeX (read in a PowerShell terminal). Report what a result means,
but don't decide the claim or interpretation — ask first; that's the user's call.

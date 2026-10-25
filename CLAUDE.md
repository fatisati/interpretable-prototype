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
- Keep `batch_size=1024`.
- **`umap_steps_per_epoch=None` — an epoch is ONE PASS OVER THE EDGES, derived from the
  data, never a hardcoded step count.** See §5a.
- Don't raise epochs unasked. If something looks undertrained, **say so** rather than
  silently training longer.

## 5a. An epoch = one pass over the edges. Pass `umap_steps_per_epoch=None`.

**Always pass `umap_steps_per_epoch=None`.** `scproto.py` already implements it:

```python
n_samples = (min(len(edge_dataset), steps_per_epoch * batch_size)
             if steps_per_epoch else len(edge_dataset))   # None -> every edge, once
```

so `None` samples `len(edge_dataset)` edges, i.e. exactly `n_edges / batch_size` steps —
the only definition of "epoch" that means anything. **Do not pass a step count**: any
fixed number is an arbitrary fraction of the graph that silently changes meaning whenever
the dataset, the affinity or `batch_size` changes.

**What the old numbers actually were** (measured from the run logs, 2026-09-23):

```
umap_steps_per_epoch=1000  ->  1,024,000 edges/epoch  of  4,325,186 total   = 24% of the graph
one true pass (bk02)       ->  4,325,186 / 1024       =  4,224 steps        = 4.2x more
```

Every result in the [D39]/[D40]/[D41] sweep used the **code default of 1000**, i.e. a
quarter of a pass — not the 500 this file used to recommend, and not a real epoch. Those
runs are internally consistent (all used the same value) so their comparisons hold, but
the word "epoch" in them does not mean a pass over the data.

**Two consequences to keep straight:**

- Cost: a true epoch is ~4.2x the steps, so ~2.5-3 h per arm instead of ~35 min at
  `train_epochs=10`. Budget for it, or cut `train_epochs`, but do not go back to a
  hardcoded step count to save time.
- Anything defined in "epochs" has to be re-derived. `coh_mem_hlife` is the live example:
  it was set to 500 as "one epoch", which is wrong twice over — a real epoch is ~4,224
  steps, AND the memory's clock is the **cell revisit interval** (~8.65 steps), not the
  edge pass. At 500 the decay was a measured no-op (0.9881 at refresh). It is now derived
  from dataset and batch size, §9d.

`len(edge_dataset)` counts DIRECTED entries surviving the `max_weight / n_epochs`
threshold (4.33M on bk02 against 2.30M undirected edges), so the exact count shifts a
little with `train_epochs`. That is second-order; the point is that it comes from the
data, not from a constant.

## 6. One dataset first, fully

Run and **report one dataset completely** — train, eval, results table — before any
other dataset starts. If time or Colab runs out we still have one reportable result
instead of partial results everywhere. Say in the header which dataset is first and why.
Remaining datasets then run one at a time, each saving before the next.

## 6a. MEASURED run-to-run noise floor (pbmc-immune, K=300) — rare metrics are NOISE

Two `[D61]` arms turned out to be accidental REPLICATES (identical config, only
GPU nondeterminism between them). Their |difference| is the noise floor for a
single-run comparison on this dataset:

```
STABLE, trust differences above ~2x these:
  coh_aff_test_all   0.0012     within_test_all   0.0008
  coh_aff_train_all  0.0016     pooled_test_all   0.0058
  cross_train_all    0.0079     cross_test_all    0.0160
  AGREE% (per-proto batch vote)  ~3 points

NOISE, a single run says nothing below these:
  coh_aff_test_rare  0.0101     within_test_rare  0.0611
  pooled_test_rare   0.0564     rare_coverage     0.0667
  rare_homogeneity   0.0773     rare_f1           0.0822
```

Not a bug — small-sample statistics. ~1,034 rare cells, ~207 in the test half,
so 1 SD of a proportion near 0.6 is ~0.034 and 1 SD of the DIFFERENCE of two
runs is ~0.048.

**Consequence: never rank arms on a rare metric from one run each.** Either
report the difference against this floor, or run >=2 seeds per arm and report
mean±std across them. `[D60]`'s headline "rare within-batch purity +0.057" was
retracted on exactly this basis (floor 0.0611).

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

## 7a. Reproducing the NH scoreboard table (Coh/Entropy/Purity/Homog/F1/Coverage)

This is the recipe behind the recurring per-dataset table (`coh_aff`, `coh_gram`,
`entropy(test)`, `purity(test)` | `homog`, `f1`, `coverage`, one row per method).
It mixes TWO protocols in one table — get the masking wrong and the numbers are
silently not comparable to anything.

**Rows (methods), one per dataset:**
- **scProto** — that dataset's current best NH-trained arm (`clusters_full.npz`,
  covers all N cells: train assignments + held-out cells placed by a frozen-
  encoder forward pass). "Best" is a judgment call from held-out purity + rare
  metrics, not a fixed rule — see §9d-1 for how the arm was actually configured.
- **scPoli-NH** (`leiden_st1_nh_K<K>`) — Leiden on the Stage-1 CVAE latent,
  trained on the SAME 80% split, held-out cells placed via frozen-encoder + 15-NN
  majority. The fair, leak-free control.
- **SEACells** (`seacell`) — fit on the COMPLETE dataset (no held-out concept).
  Reported for context, not a matched comparison — say so in the caption.

**Columns 1-4 (`coh_aff`, `coh_gram`, `entropy(test)`, `purity(test)`): NH
protocol, held-out cells ONLY.**
```python
from modal_scripts.score_all_baselines import DATASETS, load_labels
from modal_scripts.score_nodeheldout import test_mask_for, purity_entropy_on_mask
from interpretable_ssl.evaluation.metacell_scoreboard import (
    cohesion_per_cell, load_affinity, read_obs)

n, h5, bk, lk, nk, K = DATASETS[ds]
mask = test_mask_for(ds, h5_path, n)          # boolean, True = held-out cell
A_full = load_affinity(f"graphs/affinity_{ds}{n}_ncomp50_kneighbors50_arbf.pkl")
# ^ the FULL graph, not the train-only graph the model trained on -- held-out
#   cells' true neighbours only exist in the full graph.
lab = load_labels(run_dir, n)                 # clusters_full.npz or clusters.npz
ca = cohesion_per_cell(lab, A_full, bcodes, sim='aff')[mask]   # mean/std OVER CELLS
cg = cohesion_per_cell(lab, A_full, bcodes, sim='gram')[mask]
pe = purity_entropy_on_mask(lab, cts, bts, mask)   # ct_purity_wmean/wstd,
                                                    # batch_entropy_wmean/wstd
                                                    # (size-weighted OVER METACELLS,
                                                    # test-members only)
```

**Columns 5-7 (`homog`, `f1`, `coverage`): full-dataset protocol, pools train +
held-out.** Same `lab` (already covers all N cells for the NH methods), but NO
mask — rare types are identified from FULL per-batch frequency, not held-out-only:
```python
from interpretable_ssl.evaluation.metacell_scoreboard import rare_metrics_paper
rare = rare_metrics_paper(lab, cts, bts)   # NO mask -- full cts/bts/lab
# rare_homogeneity_mean/std   (per-cell score, pooled over ALL rare cells)
# rare_f1_mean/std            (per-batch, then mean/std ACROSS BATCHES)
# rare_coverage_mean/std      (per-batch, then mean/std ACROSS BATCHES)
# rare_purity_mean/std exists too but is dropped from this table by request.
```

**Read the Drive-slowness warning in §11 before running any of this** — `A_full`
and the dataset's `.h5ad` are large files on the Drive-mounted path; copy them
to a local/scratch path first or `h5py`/`anndata` reads can take over a minute
each or hang outright.

Formatting: 2dp `mean±std`, bold = best in column (per dataset), underline =
second-best. `cov_s`/`rare_coverage_std` is 0 whenever every batch hit full
coverage (e.g. immune) — that's a real value, not a missing std.

## 7b. Compiling the ICLR manuscript — use `build.ps1`, never ad hoc `pdflatex`

`manuscripts/iclr/scproto_iclr2027/` has ONE correct build process, already
written: `.\build.ps1` (PowerShell, run from that folder). It runs the real
4-pass sequence (pdflatex → bibtex → pdflatex → pdflatex), keeps aux/log
clutter in `build/`, and copies the result to **`scproto_iclr2027.pdf`** in
that folder — that is THE canonical PDF, the only one that should exist there.
Read `manuscripts/iclr/scproto_iclr2027/README.md` before touching this
folder; it documents the full layout and known issues.

**Do not run bare `pdflatex main.tex` in that folder.** It was done this way
for most of a session (2026-09-26) before this note existed, producing a
second, wrongly-named `main.pdf` that silently skipped bibtex entirely — every
citation was unresolved the whole time, and nothing in pdflatex's own output
made that obviously wrong (it just prints "undefined citations" warnings that
are easy to dismiss as unrelated to whatever you're actually editing). Two
PDFs with different names for the same paper is itself a bug to avoid, not
just a citations issue — if you ever find more than one `*.pdf` in that
folder, that means an ad hoc compile happened somewhere; delete the wrong one
and rebuild with `build.ps1`.

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

**First decide: a new value for an existing key, or a brand-new key?** Check
`configs/defaults.py` (`get_defaults()`) and `interpretable_ssl/constants.py`
(`ABBREVIATIONS`) for the key *before* picking any names.

- **New variant of an existing key** — the key already has a default and an
  abbreviation (e.g. `nassoc_diag_norm` already exists; adding `'pair'` alongside its
  `'volume'` default is a variant, not a new key). The abbreviation is already there —
  **do not touch `constants.py`**. You only need to pick the new **value**'s name
  (4-char-distinct from every other value already used for that key) and, only if the
  key's own default is changing (usually it isn't), update `defaults.py`.
- **Completely new key** — nothing in either file mentions it yet (e.g. introducing
  `nassoc_offdiag_norm` where before there was no such option at all). This needs a
  **feature name** (the abbreviation itself) *and* a **value**, and both files below
  must be edited, in the same change — skipping either is how a new option silently
  fails to reach the model name.

Then, for the option itself, all three must be present (steps 1-2 apply in full only to
a completely new key; a variant of an existing key only needs step 3, plus step 1 if the
default is actually changing):

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
   value. This is the step that actually proves a saved run for this variant is
   findable by name later — do it even for a variant-only change.

## 9b. Every new loss term: check its SCALE and RANGE before trusting any result

A new loss term is not "working" because the run finished and the number went down. Two
real bugs shipped this way and both were invisible for weeks — the runs looked fine, the
metrics were plausible, and the loss decreased monotonically the whole time.

**Do these three checks the first time a new term runs, before reading any result:**

1. **Print the term's own value and compare it against its theoretical range.** If a
   quantity is a *density*, a *probability*, a *purity*, a *correlation* — it has a known
   range. Print it and check.
   > Real bug: the `nassoc` diagonal is an edge density, so it must lie in `[0, 1]`, and
   > the loss `(density - 1)^2` drives it to 1. The logged value was **371–510**.
   > `possible_pairs = size*(size-1)/2` clamps to exactly **0** for any prototype whose
   > *soft minibatch* size is below 1, so `density = edges/1e-8` exploded. Under
   > `(density-1)^2` an EMPTY prototype cost `1` while a merely SMALL one cost `~1e16` —
   > the loss was paying the model to kill prototypes. That was the entire cause of the
   > 15/200-survivor collapse, and fixing it moved DGE from 0.654 to 0.780.

2. **Print every term of the total loss side by side and check the ratios.** One term at
   1000x another is a bug until proven otherwise, no matter how sensible each looks alone.
   > Real bug: `nassoc=2874` vs `umap=2.5` — the regularizer was 99.5% of the total loss.
   > Cause: numerator computed at full-dataset scale (`nassoc_size_scale='full'`),
   > denominator at minibatch scale (summed weight of the *sampled* edges only), so the
   > term came out `n_edges_total/n_edges_batch` ~= 3900x too large.

3. **Check every denominator for a term that can vanish, and every minibatch estimate for
   a scale mismatch against its numerator.** Anything quadratic in a soft count
   (`possible_pairs`) is biased differently under subsampling than anything linear
   (`size`, `volume`), so a ratio of the two is NOT an unbiased estimate of the
   full-graph ratio. Rescale both sides to the same scale, or don't compare them.

Corollary for hyperparameters: a threshold parameter is only interpretable once the
quantity it is compared against is on a known scale. Derive it from measured properties
of the actual graph rather than sweeping blind, and re-derive it if the scaling of the
underlying quantity changes. (`cohesion`'s OLD span calibration used to be the example
here; it is superseded by the unified gradient-share method, §9d/§9d-1 — cohesion is no
longer calibrated differently from any other term.)

**`nassoc_gamma` — NO LONGER RELEVANT.** We are on cohesion. `nassoc` is **off** in every
current run (`lambda_nassoc = 0`), so gamma is not a parameter of anything we train and
must not be tuned, measured or reported.

Why it went away rather than being fixed: `nassoc` and `cohesion` are **both diagonal
terms, and cohesion replaces nassoc**. §9b's own argument is the reason — CPM's charge is
a function of the assignment COLUMN SUMS only,

```
d(CPM charge) / d assign[i][k] = gamma * ( size[k] - assign[i][k] )
```

so it cannot move a cell for SIMILARITY reasons at all; it is a size/capacity charge.
That is exactly the defect cohesion was built to fix. Running both puts two diagonal
terms in competition and drags in a threshold whose derivation was never settled, so any
cohesion result would be confounded by a gamma nobody chose on principle.

The old derivation (the CPM move rule, `measure_gamma_from_graph.py`, and the measured
values `s28nsc`/`bk02` → 0.026-0.030, `pancreas`/`arbf` → 0.11) is kept in
`decisions.md` §15 and `EXPERIMENTS.md` §11a **only** so runs that predate this change
stay interpretable. Do not carry it into new work.

Still live and unrelated to gamma: `epsilon` calibration **fails on every dataset tried**
and silently falls back to `umap_proto_effk=5.0`, a hard-coded default nothing chose,
which then drifts during training. **Still unfixed** — check the `[eps calibration]`
line in any run log.

## 9c. REMOVED — there is no nassoc base config any more

The nassoc/CPM base config that used to live here has been **deleted, not demoted**,
because leaving the code block visible is what caused a real mistake: it reads like the
config to start from, and it was copied into a cohesion experiment, silently adding a
second diagonal term and an unsettled `nassoc_gamma`.

**The config is §9d. There is no other.**

If you need to reproduce a pre-2026-09-24 nassoc run, the settings are in
`decisions.md` §15 and `EXPERIMENTS.md` §11a, and the run's own model-folder name
records them.

## 9d. THE loss: exactly FOUR active terms, ALL gradient-share calibrated.
Every other term is genuinely 0. (Unified 2026-09-26, `decisions.md` [D55].)

**Cohesion-gram + cohesion-aff + community [routed, prototypes detached] +
CVAE-recon.** These four, and only these four, are the design. Cohesion is no
longer a privileged, externally-pinned anchor — `authority/span` (kept below
for historical/interpretability reasons, and for reading any run that predates
2026-09-26) is SUPERSEDED as the mechanism that sets its lambda. All four terms
now get an explicit, USER-CHOSEN target share, and all four lambdas are solved
the same way, from measured gradient norms (§9d-1):

```
lambda_cohesion      > 0   gram, 2-hop   }  all four: user picks a target SHARE,
lambda_cohesion_aff  > 0   aff,  1-hop   }  lambda solved from measured gradient
lambda_umap          > 0   community [routed, umap_proto_sg=1]  }  norm, §9d-1.
lambda_recon         > 0   CVAE-recon                            }  ONE snapshot,
                                                                    no re-probing
                                                                    mid-run (explicit
                                                                    user decision).
--------------------------------------------------------------------
lambda_proto_recon   = 0                  OFF, genuinely -- not part of the design
lambda_nassoc        = 0   (cohesion REPLACES nassoc — §9b)
lambda_modularity    = 0                  OFF, genuinely -- not part of the design
lambda_proto_usage   = 0                  OFF, genuinely -- not part of the design
lambda_swav          = 0   lambda_kl = 0
proto_reinit_every   = 0
```

**Stage 2 (scProto fine-tuning, after CVAE pretraining) must freeze the batch
embedding: `freeze_batch_embedding=True`.** Stage 1 already achieved batch/bio
separation (z sheds batch info because the batch embedding + decoder carry it
instead); without freezing, stage 2 has nothing stopping the encoder from
drifting batch-specific structure back into z. This is what CVAE-recon's
anchor role actually depends on -- recon without a frozen batch embedding is
not testing the same mechanism as recon with one.

Nothing may outvote the term the model is built around. **Assert this in the run
script** — do not rely on having remembered to zero them, because
`LAMBDA_PROTO_UMAP_PRECON` turns several of them ON (see §10b) and starting from it and
setting only the cohesion keys leaves nassoc and proto_usage running.

**`cohesion_self = 1.0`, ALWAYS. Never 0.5.** It is the `defaults.py` default, so it is
also what you get by not passing it — but pass it explicitly and assert it, because old
scripts in `modal_scripts/` still carry `cohesion_self: 0.5` and get copied.

```python
lambda_config = dict(LAMBDA_PROTO_UMAP_PRECON) | {
    'lambda_umap':   <derive per §9d-1>,   # community -- NOT 0, NOT a guessed number.
                                           #   s28nsc is the one dataset where this
                                           #   derives to 0 (measured to fail there).
    'umap_proto_sg': 1,                    # <- REQUIRED whenever lambda_umap > 0.
                                           #   Detaches prototypes from community's
                                           #   similarity computation -- its gradient
                                           #   reaches the ENCODER ONLY, so cohesion
                                           #   keeps sole ownership of the prototype
                                           #   vectors ([D48]). Forgetting this silently
                                           #   reverts to UNROUTED community, which [D39]
                                           #   measured killing 125/200 prototypes.
    'lambda_recon':  <derive per §9d-1>,   # CVAE-recon, gradient-share calibrated --
                                           #   same caveat as lambda_umap above.
    'lambda_proto_recon': 0.0,             # genuinely OFF -- not one of the 4 active terms
    'lambda_nassoc': 0.0, 'lambda_modularity': 0.0, 'lambda_proto_usage': 0.0,
    'lambda_swav': 0.0, 'lambda_kl': 0.0,
    'lambda_cohesion':     <derive per §9d-1>, # } gradient-share calibrated, same as
    'lambda_cohesion_aff': <derive per §9d-1>, # } every other term -- NOT authority/span
                                   #   any more (superseded 2026-09-26, [D55]). The OLD
                                   #   authority/span numbers are kept here ONLY to
                                   #   interpret pre-2026-09-26 runs: authority=1.5403,
                                   #   lambda=authority/span, span=value(leiden at K)-
                                   #   value(random at K) on the training graph.
                                   #   s28nsc/bk02 gave 10.0/22.3; pancreas 4.04/11.99,
                                   #   lung 5.61/26.32, immune 8.74/33.65 (heldout-80%
                                   #   graphs). Do not use this formula for new runs.
    'cohesion_self':   1.0,   # <- ALWAYS 1.0. The self-pair counts like any other
                              #    member, sim(c,c)=1. This IS the plain formula:
                              #    "mean similarity to a prototype's members".
    'cohesion_diag':   1.0,   # sim(c,c) = 1 in both aff and gram modes
    'cohesion_global': 2,     # [D29] remembered full-graph profiles
    'coh_mem_init': 'full',   # [D32] memory from one inference pass, not uniform
    'coh_mem_hlife': c * N / uniq,  # [D31] DERIVE IT -- see the box below. NEVER 500.
    'proto_reinit_every': 0,  # [D39] no reinit
    # latent_dims: NOT passed -- use the defaults.py default of 8. The
    #   [D39]/[D40]/[D41] runs passed 16; that was an override, not the standard.
}
```

**The run-level arguments are part of THE config too.** They are not in `lambda_config`,
and four of them default to values that are **wrong here** — silently, so the run
finishes and reports numbers from a setup nobody chose:

```python
trainer, metrics, _ = run_mc_task(
    ds_id, affinity_type=<graph>,
    num_prototypes=None,         # <- USE THE DATASET DEFAULT. K is a property of the
                                 #   dataset, not of the experiment. None resolves via
                                 #   _resolve_num_prototypes to DATASETS[ds]
                                 #   ['num_prototypes']: s28nsc 800, crcx 670, lung 300,
                                 #   pbmc-immune 300, pancreas 220, fibnsc 204.
                                 #   (configs/defaults.py's 300 is only the fallback for
                                 #   a dataset with no entry.) Pass an int ONLY to
                                 #   reproduce an older run at its K.
    cvae_epochs=50,              # required, no default
    train_epochs=10,             # required, no default
    batch_size=1024,             # DEFAULT 256      <- wrong
    umap_steps_per_epoch=None,   # DEFAULT 1000     <- wrong: 24% of a pass, see §5a
    early_stop=False,            # DEFAULT True on 'modularity'  <- wrong: this arm has
                                 #   lambda_modularity = 0, so it would both stop AND
                                 #   pick its saved checkpoint by a metric it never
                                 #   optimizes. See §9e.
    eval_freq=2, patience=4,     # required positionally; unused when early_stop=False
    lambda_config=lambda_config,
    trainer_kwargs={'experiment_name': <name>},
    freeze_batch_embedding=True, # <- REQUIRED, default False <- wrong. Stage 1 already
                                 #   pushed batch info OUT of z (into the batch
                                 #   embedding + decoder); without freezing it in stage
                                 #   2, nothing stops the encoder drifting batch
                                 #   structure back into z, and CVAE-recon's anchor role
                                 #   depends on this staying frozen. See §9d.
)
```

**`coh_mem_hlife` IS DERIVED FROM DATASET SIZE AND BATCH SIZE — never a constant.**
The memory's clock is the **cell revisit interval**, not epochs: a remembered row is
*replaced* the moment its cell reappears in a batch, so the only timescale that matters
is how long that takes.

```
uniq  = unique cells per batch      # = batch_size * (1 + neg_rate) * 2, deduped.
                                    #   MEASURED, printed every run as
                                    #   "[D26] cohesion sample: <uniq> unique cells"
                                    #   s28nsc @ batch_size=1024, neg_rate=5 -> 6,757
R     = N_cells / uniq              # steps between a cell's visits: 58,423/6,757 = 8.65
coh_mem_hlife = c * R               # c = revisit cycles until a row is half-trusted
```

`c` is the only free choice and it is readable: `c=1` -> a row is worth 0.50 when it is
refreshed, `c=3` -> 0.79, `c=10` -> 0.93. Pick it from the logged `coh_mem_drift` (how
far an assignment actually moved between visits); if drift is ~0, decay is not needed at
all and `coh_mem_hlife=0` is the honest setting.

**MEASURED 2026-09-23 — the old `500` was a NO-OP.** At `half_life=500` a row has decayed
to **0.9881** by the time it is overwritten at step ~8.65. So [D31]'s decay did nothing in
any reported run; those results are effectively plain [D29] with the memory at full
weight. Nothing is invalidated — but do not describe those runs as having used age
decay, and do not copy `500`.

**K CHANGED — IT BREAKS THE SAVED BASELINES. Read before comparing anything.** Every
[D39]/[D40]/[D41] number was produced with `num_prototypes=200` hardcoded, which
overrode `s28nsc`'s dataset default of **800**. Runs that follow this file now use the
dataset default, i.e. **K=800 on s28nsc**, and therefore:

- every saved SEACells baseline is **K=200** (`seacell_X_bklam02_K200`,
  `seacell_X_bklam04_K200`, `seacell_bk08`, `seacell_arbf`, `seacell_K200`) — a K=800
  run is **not comparable to any of them**;
- `files/tables/s28nsc_lambda_sweep_vs_seacells.md` is a K=200 result and remains valid
  *as one*. Do not add a K=800 row to it;
- a head-to-head at the dataset default needs SEACells **refit at K=800** on each graph
  (four CPU fits) before the comparison means anything.

**Never compare across K.** The scoreboard metrics move mechanically with it: `n_dead` is
out of K; `coh_aff` carries a constant `K/N` floor (0.00342 at K=200 vs 0.0137 at K=800,
measured §9b-style on bk02); and purity / `n_covered` both rise as metacells get smaller
and more numerous.

**MULTI-BATCH IS NOT A KNOB.** There is no `coh_batch` option. The trainer counts bio
batches from the data: with >1 batch each cell is scored against `V[b]`, the node set of
its own bio-batch graph (all edges with at least one endpoint in `b`); with 1 batch
`V[b]` is every cell, so the denominator is the global size and the formula is
bit-identical to the batch-agnostic one (asserted to <1e-12 in
`tests/test_cohesion_memory.py`). Nothing is masked — cross-batch similarities keep
their real values. What is excluded are pairs whose zeros carry no information: a
cross-batch zero is mostly the batch effect, not measured dissimilarity, so it is absent
from the numerator and must be absent from the denominator too. Full formulas in
`my-tasks/context/md/LOSS_FORMULAS.md`.

**Two practical traps, both hit for real:**
- **Filename length.** Pass lambdas ROUNDED (2 dp). Full-precision floats render as
  `lcoh4.03750478377344` and push the run-folder name past the 255-byte limit — a run
  died with `OSError [Errno 36] File name too long` AFTER finishing training.
- **`affinity_type` truncates to 4 chars**, so `arbf` and `arbf_heldout20_seed0` both
  render `aff-arbf` and collide into ONE folder. Always pass a distinct
  `experiment_name` when training on a masked/held-out graph.

**Why 1.0 and not 0.5** — five reasons; the first is the mechanistic one:

0. **THE SELF TERM IS WHAT KEEPS PROTOTYPES ALIVE — without it they go empty.**
   A prototype's score is `within[k] / size[k]`. For an EMPTY slot that is 0. For a
   SINGLETON it is exactly `cohesion_self` (its only pair is itself, `sim(c,c)=1`):

   ```
   empty prototype      within = 0              size = 0   -> scores 0
   singleton, self=w    within = w * 1          size = 1   -> scores w
   ```

   So a live prototype beats an empty one by exactly `w`. At `w = 0` a singleton and an
   empty slot score **identically** — nothing in the objective prefers using a prototype
   at all, and they drain. `w = 1` gives the largest possible margin; `w = 0.5` halves it
   for no benefit. This is why the self term exists and why it should be at full weight.

1. **It is the plain formula.** At 1.0 the discount branch (`if cohesion_self != 1.0`)
   never executes, so cohesion reduces to exactly
   `sum over members of sim(c,j) / size[k]` with the self-loop at 1.
2. **It is SEACells-exact.** `defaults.py` [D17]: minimizing `||M - M B A||_F^2` equals
   maximizing `sum_k within_dot[k]/size[k]` with self-pairs INCLUDED, verified to 1.8e-12.
3. **SEACells itself does this.** `SEACells/build_graph.py` does `knn_graph.setdiag(1)`
   with the comment "Include self as neighbour", and its RBF gives `exp(0) = 1` on the
   diagonal — so their kernel has `M[i][i] = 1` too.
4. **The reason 0.5 existed is refuted.** [D30] introduced 0.5 to lower the founder gate
   (`sim > cohesion_self/2`) because small prototypes could not recruit. Measured
   2026-09-23: at the SAME gate of 0.5, free-logit optimisation gives **0 dead / 1
   fragment** while the trained encoder gives **5 dead / 54 fragments** — identical gate,
   opposite outcomes, so the gate was never the cause (the encoder was). The free-logit
   ceiling also barely moves between the two settings (coh_gram 0.0908 at 0.5 vs 0.0901
   at 1.0), so it is not a sensitive knob.

**Also settled with it ([D39], s28nsc/bk02/K=200, all with `cohesion_self=1.0`):**

- **gram + aff beats either alone.** joint_purity 0.4576 / dge_r 0.8105 / 0 dead, against
  aff-only 0.4472 / 0.8006 / 0 and gram-only 0.4558 / 0.6538 / 5.
- **No reinit needed.** With community off, `proto_reinit_every=0` gives 0 dead across
  every graph tested (lambda 0, 0.2, 0.4, 0.8).

## 9d-1. Every term's lambda, COHESION INCLUDED: decide a SHARE, derive lambda
from gradient norm. Never reuse a raw lambda value across datasets. (`decisions.md`
[D55], 2026-09-26)

`[D48]` routing (`umap_proto_sg=1`) detaches prototypes from community's similarity
computation, so its gradient reaches the encoder only — cohesion keeps sole
ownership of the prototypes. This is what makes community and CVAE-recon part
of the 4-term design (§9d) beyond s28nsc, rather than each dataset's own
one-off addition.

**The standard method (static, single-snapshot GradNorm — see `decisions.md`
[D55] for the full derivation; ONE snapshot, no re-probing mid-run, by explicit
user decision). All FOUR terms now go through this, symmetrically — cohesion
is no longer measured differently from the other three:**
```
1. Fix a target SHARE for EVERY term (cohesion-gram, cohesion-aff, community,
   recon), summing to 1 — ONE shared vector of shares, used for every dataset,
   not tuned per dataset.
2. Measure each term's raw encoder-gradient norm at a TEMPORARY lambda=1
   (grad_probe, one pass) -- encoder-only, matching [D48]'s routing (cohesion
   is the only term allowed to touch prototypes directly, but its ENCODER-side
   gradient is measured the same way as everything else now).
3. lambda_i = share_i / gnorm_i     -- for all four terms. NO total-budget
   constant needed. Training uses Adam/AdamW (`scproto.py`), which normalizes
   each parameter's update by a running estimate of its own gradient magnitude
   -- scaling every active term's gradient by the same overall constant leaves
   the actual parameter updates almost unchanged, because that constant cancels
   in Adam's update ratio. Only the RELATIVE proportions between terms (the
   shares) actually shape training; an absolute total is not a real degree of
   freedom worth tuning. (`authority=1.5403` used to quietly supply this role;
   it is RETIRED now that cohesion is gradient-share calibrated like every
   other term -- kept only to interpret runs that predate 2026-09-26.)
```
Still run the cheap 1-epoch dead-prototype check once before committing to a
full run -- not to tune a total budget, but because Adam's normalization runs
off a moving average that hasn't caught up yet at step 0, so an extreme share
choice can still misbehave for the first few steps.
Mechanics: set `grad_probe = N` (a positive int, e.g. 25) in `lambda_config` --
every N steps it backward-passes each active term separately and appends
`{lambda, value, gnorm_enc, gnorm_proto}` per term to `<dump>/grad_probe.jsonl`
(`configs/defaults.py`, `[D45]`). **Reading OLD (pre-2026-09-26) probe files has
one trap**: they measured cohesion at its REAL `authority/span`-deployed lambda,
not at a temporary lambda=1 like community/recon — so an old file's cohesion
`gnorm_enc` must be DIVIDED by its reported `lambda` to recover the per-unit
value before it's comparable to the others. A FRESH probe run under this
unified method should force cohesion's lambda to 1 too, for a direct reading
with no back-conversion. Conflating these is exactly how `s28nsc`'s `cvA` recon
lambda ended up ~11x too high (2026-09-26, this file).

A dataset's numeric lambda is NEVER portable to another dataset — gradient
scales differ per graph/dataset — but the SHARE vector is what should be portable.

**VERDICT: not yet applied consistently, and not yet applied to cohesion at
all.** Only s28nsc's community lambda was ever gradient-share derived (and
still failed on the scoreboard — see `decisions.md` [D39]/[D46]). Every dataset's
cohesion lambda still comes from the old `authority/span` formula (kept in the
code block above for now, marked superseded). Nothing currently deployed
should be read as validated by the unified method yet. See `decisions.md`
[D55] for the current per-dataset numbers.

## 9e. What the defaults actually are — check before assuming

Defaults live in **three** places and they disagree with each other. A value you did
not pass is NOT necessarily the one you think.

**1. `find_metacells` / `run_mc_task` signature** (`experiments/tasks.py`):

| arg | default | what our runs pass | note |
|---|---|---|---|
| `cvae_epochs` | **required** | 50 | no default — must be passed |
| `train_epochs` | **required** | 10-20 | no default |
| `batch_size` | **256** | 1024 | §5 wants 1024 — PASS IT |
| `umap_steps_per_epoch` | **1000** | *not passed* -> 1000 | **pass `None`** = one real pass, §5a |
| `load_pretrain` | **True** | (default) | reuses `pretrain_checkpoint.pth` if it exists |
| `load_umap` | False | (default) | True = skip training, load checkpoint |
| `early_stop` | **True** | False for cohesion arms | [D34] |
| `early_stop_metric` | **'modularity'** | n/a when early_stop=False | wrong metric for a cohesion-only arm |
| `min_delta` | 0.005 | (default) | |

**Pretraining: yes, always — but reused, not recomputed.** `load_pretrain=True` means the
CVAE encoder is loaded from `MODEL_DIR/<ds>/pretrain/pretrain_checkpoint.pth` when it
exists, and only trained (for `cvae_epochs`) when it does not. Pretraining does NOT depend
on the affinity graph, so one checkpoint covers every `affinity_type` / lambda sweep at a
given K — which is why a whole sweep costs one pretrain, not one per arm.

**2. `LAMBDA_PROTO_UMAP_PRECON`** (the usual starting dict) turns several terms ON:

```
lambda_umap=1          lambda_proto_recon=0.01     lambda_nassoc=1
lambda_proto_usage=0.1 proto_usage_mode='ema'      nassoc_agg='max'
prot_init='waypoint'   umap_similarity='proto'     calibrate_eps=1
```

So starting from it and only setting the cohesion keys leaves **nassoc and proto_usage
running**. A cohesion-only arm must explicitly zero `lambda_nassoc`, `lambda_modularity`,
`lambda_proto_usage`, `lambda_umap`, `lambda_proto_recon` — see the §9d block.

**3. `configs/defaults.py`** — the model-level defaults. The ones that bite:
`num_prototypes` 300 (only a fallback -- K comes from the dataset config, §9d),
`latent_dims` 8 (KEEP IT -- the D39/D40/D41 runs overrode it to 16), `affinity_type` `'uctx'`,
`prot_init` `'kmeans'` (PRECON overrides to `'waypoint'`), `umap_similarity` `'embedding'`
(PRECON overrides to `'proto'`), `epsilon` 0.05 (but `calibrate_eps=1` recomputes it at
runtime — measured 0.0664-0.0678 on s28nsc, so the 0.05 is never used), `seed` 31.

**On epochs, see §5a** — the [D39]/[D40]/[D41] sweep ran at the code default of 1000
steps, which is 24% of one pass over the edges, not an epoch. New runs pass
`umap_steps_per_epoch=None`.

**Early stopping: ON by default, on the WRONG metric for a cohesion arm.**
`early_stop=True, early_stop_metric='modularity', min_delta=0.005` with `eval_freq` /
`patience` required positionally. So unless you pass `early_stop=False`, training stops on
**Newman modularity** — a metric a cohesion-only arm has `lambda_modularity=0` for and
never optimizes. It also **selects the saved checkpoint**, so an arm can be reported at an
epoch chosen by a quantity it was not training on. `'homophily'` and `'modularity'` are the
only two metrics implemented; there is no `'cohesion'` option. Until there is:

- cohesion arms: pass **`early_stop=False`** ([D34] added the passthrough) and a fixed
  `train_epochs`. Every [D39]/[D40]/[D41] result used this.
- if you do want early stopping, say which metric and why — and remember eval runs only
  every `eval_freq` epochs, so `patience` is counted in evals, not epochs.

## 10. Explanation style

Define every term and metric inline. Formulas as **plain-text code blocks with readable
variable names** — no LaTeX (read in a PowerShell terminal). Report what a result means,
but don't decide the claim or interpretation — ask first; that's the user's call.

## 11. Running on Modal — GPU/CPU cloud alternative to Colab

Validated end-to-end (training + eval + SEACells + BANKSY) as of 2026-09. Use it when
Colab's install is too slow or a quick isolated experiment doesn't need a notebook at
all. Modal has its own local pip mirror (`pypi-mirror.modal.local`) — installs that crawl
on Colab's real-internet pip take seconds here, so most of §1's Colab pip pain doesn't
apply. Real money, real GPU-minutes: **check the actual cost/time before scaling up a
sweep**, same caution as any paid cloud resource.

**Windows shell gotcha, every single time.** The `modal` CLI's Rich-based progress output
emits a unicode checkmark that crashes with `'charmap' codec can't encode character`
under PowerShell/Git-Bash's default console codepage. Always prefix:
```
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run your_script.py
```

**Data path: a Modal Volume mirroring `MODEL_DIR`/`DATA_DIR`/`CODE_DIR`.** Create one
volume (`modal volume create <name>`), lay it out exactly like the real dirs, then set
env vars to match before importing anything:
```
/vol/data/spatial/<file>.h5ad          -> DATA_DIR
/vol/code/interpretable_ssl/...        -> CODE_DIR  (upload the package dir directly;
/vol/code/seacell_train.py                it's ~6MB, `modal volume put` handles dirs)
/vol/code/graphs/<affinity>.pkl        -> CODE_DIR/graphs  (cache -- see below)
/vol/code/data/<label_encoder>.pkl     -> CODE_DIR/data
/vol/models/<ds_id>/pretrain/...       -> MODEL_DIR  (reuse the existing pretrain
                                           checkpoint -- pretraining doesn't depend on
                                           affinity, so one checkpoint covers every
                                           affinity_type sweep at that K)
```
In the function: `os.environ['DATA_DIR']='/vol/data/'` etc., `sys.path.insert(0,
'/vol/code')`, `os.chdir('/vol/code')` (needed because `get_affinity_path`'s
`graph_dir='./graphs'` default is CWD-relative — same as `nb_setup.py`'s `chdir` on
Colab). `volume.commit()` at the end of every function that writes, or the run vanishes
when the container exits.

**Where `DATA_DIR`/`MODEL_DIR` actually live locally, for uploading to the volume.** On
Colab these point at the user's *personal* Google Drive, not this repo's own shared-drive
checkout — a plain search of this checkout will not find raw dataset `.h5ad` files or
model checkpoints. On a machine where the personal Drive is also synced, they show up as
separate Drive-shortcut mount points from the shared project folder:
```
DATA_DIR  -> G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data
MODEL_DIR -> G:\.shortcut-targets-by-id\1-0FnoSFlnEGxgKJQCAbspCuHc3osVs25\models
```
`modal volume put` can read directly from these paths once found.

**The package list that actually imports cleanly** (found by hitting every
`ModuleNotFoundError` once, not by guessing — reuse this instead of rediscovering it):
```
numpy scipy pandas anndata scanpy torch scarches wandb tqdm statsmodels diptest
palantir faiss-cpu scikit-learn matplotlib scib-metrics ipython
git+https://github.com/dpeerlab/SEACells.git   (separate .pip_install call, extra_options="--no-deps")
pybanksy   (only if computing a BANKSY embedding fresh)
```
`scib-metrics` and `ipython` are easy to miss: nothing in this project's own code hints
at needing them, but `interpretable_ssl.evaluation.metrics` hard-imports `scib_metrics`
at module level, and the real `SEACells` package hard-imports `IPython.display` at
*its* module level — both fire the moment you import almost anything from
`interpretable_ssl.trainers.scproto` or `.evaluation.metric_helpers`, training or not.
`modal.Image.pip_install("pkg", "--no-deps")` is invalid syntax — flags go in
`extra_options=`, not the package list.

**`Data_Handler.py` hard-crashes any import of the trainer chain outside Colab.** It
hardcodes `HOME_DIR = '/content/drive/MyDrive/codes/'` (unrelated to
`configs/paths.py`'s configurable one) and eagerly `open()`s
`{HOME_DIR}/Islander/meta/<dataset>/{cell2cat,batch2cat}.json` for eleven unrelated
datasets (`lung`, `lung_fetal_donor`, `lung_fetal_organoid`, `brain`, `breast`, `heart`,
`eye`, `gut_fetal`, `skin`, `COVID`, `pancreas`) at module import time, regardless of
which dataset you're actually using. Stub them in the image build (`{}` is enough,
they're dead code for anything outside those eleven datasets):
```
mkdir -p /content/drive/MyDrive/codes/Islander/meta/<name> && \
  echo '{}' > .../cell2cat.json && echo '{}' > .../batch2cat.json
```
for each of the eleven. One-time cost per image; don't try to fix the real bug (patching
`Data_Handler.py` itself) mid-experiment.

**The anndata-version gap.** Files written by an older anndata encode a Python `None`
(e.g. `uns['log1p']['base']`, `uns['pca']['params']['mask_var']) as an HDF5 dataset with
`encoding-type: null`, which current anndata (`0.11`-`0.12`+) has no reader registered
for at all — `IORegistryError`, not a version-mismatch warning. Don't patch the file;
register a reader once, before reading anything:
```python
from anndata._io.specs.registry import _REGISTRY, IOSpec
import h5py
@_REGISTRY.register_read(h5py.Dataset, IOSpec('null', '0.1.0'))
def _read_null(elem, _reader):
    return None
```

**`train_seacell` (and anything else built on `load_dataset`) re-reads the h5ad from
disk internally — it never sees an obsm key you only added to your in-memory `AnnData`.**
If you compute a new embedding (e.g. a BANKSY run at a lambda that isn't cached yet),
`ad.write_h5ad(path)` to persist it back to the *same file on the volume* before calling
`train_seacell`/`run_mc_task`, or it `KeyError`s looking for the obsm key on its own
fresh copy. This is the same reason `bk08`'s branch in `graph_generator.py` checks
`if 'X_bk08' in ad.obsm` against a file, not an object.

**The scArches source is checked out locally at `F:\codes\scarches` — READ IT
instead of guessing.** Modal installs `scarches` from PyPI and that resolves to
**0.6.1, the same version as this checkout** (verified 2026-10-25 by
`modal_scripts/check_scarches_version.py`, which prints the installed version
and the scPoli NB branch source — rerun it if you ever suspect a version skew).
`scarches` is NOT importable locally in this environment, so for anything
behavioural use the checkout to read code and a Modal smoke test to run it.

The three files worth knowing:
```
scarches/models/scpoli/scpoli.py          forward(): the mse / nb / zinb branches,
                                          the Decoder (its NB head is
                                          Linear -> Softmax OVER GENES, so the
                                          output is a COMPOSITION summing to 1),
                                          and theta = nn.Parameter(n_genes,
                                          n_conditions_combined)
scarches/models/trvae/losses.py           nb() -- the exact likelihood we mirror
scarches/dataset/scpoli/anndata.py        size_factors = adata.X.sum(1) and
                                          conditions_combined; our MultiCropsDataset
                                          INHERITS these, so read them off the
                                          dataset, never recompute them
```

**`modal volume put` from Git Bash SILENTLY UPLOADS NOWHERE if the remote path
starts with `/`.** MSYS path conversion rewrites `/code/interpretable_ssl/x.py`
into `C:/Program Files/Git/code/interpretable_ssl/x.py`, the CLI reports success
(printing the mangled path, which reads as normal truncated output), and the
volume keeps the OLD file. This cost a full experiment: `[D61]`'s six arms
(2026-10-19, ~$6 of GPU) trained the PREVIOUS `scproto.py` and silently tested
the wrong loss term; the whole ADR had to be invalidated. Always:

```
modal volume put <vol> <local> code/<path>      # RELATIVE remote path, no leading /
                                                # (or export MSYS_NO_PATHCONV=1)
modal volume get <vol> code/<path> /tmp/check   # then VERIFY BY CHECKSUM --
md5sum <local> /tmp/check                       # the CLI's success line is NOT proof
```

**And verify the feature reached the RUN, not just the trainer attribute.**
`init_attributes` setattr's any kwarg, so `assert getattr(t, 'my_new_flag') == 1`
passes even when the container is running library code that has never heard of
the flag. Two asserts that actually catch it, both cheap, both in-run:

```python
import inspect
from interpretable_ssl.trainers import scproto as _sp
assert "<a token from the new code>" in inspect.getsource(_sp)   # stale-copy check
assert "<abbrev>-<value>" in trainer.get_model_name()            # §9 step 3, asserted
```

The second is CLAUDE.md §9 step 3 turned into a guard instead of a printout.
`[D61]`'s probe DID print the folder name and nobody read it; the run folders
visibly lacked the new `prfm`/`prsg`/`prds` tokens for days.

**New `affinity_type` values for a lambda/parameter sweep don't need a code change.**
`load_or_build_affinity` checks the cache (`get_affinity_path`) before ever calling
`generate_affinity`. Precompute the embedding + kernel yourself (reuse
`compute_banksy_embedding` + the real `SEACells.build_graph.SEACellGraph.rbf()` — the
same recipe `bk08`'s branch uses, just with a different `lambda_param`), `save_affinity()`
it under a new tag (`'bk02'`, `'bk04'`, ... — already 4-char-distinct from `bk08`/`bk32`/
`bkmg`, see §9), and `run_mc_task(..., affinity_type='bk02')` finds it without touching
`graph_generator.py` at all. Building the affinity is CPU-only and cheap; only training
needs a GPU.

**GPU choice: T4 is enough.** A full scProto run at this project's usual scale (K=200,
~50-60k cells, `cvae_epochs=50` reusing an existing pretrain checkpoint, `train_epochs`
up to 20) took ~12-13 minutes wall time on a T4, comfortably inside a few dollars even
run several times. SEACells/BANKSY work needs no GPU at all — use a plain CPU function
(`cpu=4, memory=16384` was enough for `s28nsc`-scale SEACells fits), it's both cheaper
and skips needing a CUDA-enabled torch build for work that never touches the GPU.

**Reading Modal-volume output back through a Google-Drive-mounted mirror is flaky on
Windows — go through `modal volume get`/`ls` instead of the Drive-synced path when
you need certainty.** Two independent failure modes hit the same symptom (file
"missing" even though it's really there):
- Large files read with `h5py`/`anndata` directly off a Drive-streamed path can fail
  mid-read (`Invalid argument`, wrong byte offset) because Drive's virtual filesystem
  doesn't handle h5py's non-sequential seeks well. Copy locally first, then read.
- Auto-generated run-folder names (`proto_..._ds-<id>_NP<k>_..._v31`) plus a Drive
  shortcut path can exceed Windows' 260-char `MAX_PATH`. `os.listdir` on the *parent*
  still enumerates the file, but `open()`/`pandas.read_csv`/`shutil.copy`/PowerShell's
  `Copy-Item` all fail on the literal path — and the standard `\\?\` long-path prefix
  does **not** help on this Drive-mapped letter drive (tested, still fails). There is no
  per-call fix; either enable Windows long-path support system-wide (don't do this
  without asking — it's a machine-level registry change) or fetch the one file you need
  via `modal volume get <vol> <long/path/on/volume> <short/local/path>`, which reads
  Modal's storage directly and never touches the Drive-sync layer's path limit.


bfore applying new changes - commit old things - and log what was it and why we changed - so the older runs reproducble and we have a log of what ahppened and why we added things

when explain a formula - use understanble var names
try to answer all of my questions undetandble - correct and comact - do not hallusincate - do not say many unrelated things many noisy things

===
i have submitted my work for neurips - and i did alot in the rebuttle - the final decision (Sep 24 '26 (Anywhere on Earth)) would be out only a day before full submission edadline of iclr (Sep 25 '26 (Anywhere on Earth))

so i have to prepare new version of my paper - include new experiments - apply edits based on rebuttles feedbacks
and prepare paper for iclr submission so in case i got rejetced by neurips i will submit for iclr 

please dont make up hallucinated things from yourself - make sure whatever reasoning you do for me is real and solid - not hallusicianted - this is important

nassoc off diagonal correct dominator is min (voli, volj)
  [HISTORICAL as of 2026-09-24 — nassoc is OFF (lambda_nassoc = 0) in every current
   run; cohesion replaced it. True of nassoc if it is ever revived. See §9d.]
===
remind me these - if you thaught i am forgeting:
each run should have a decision before running - what do you expect? how would effect your decision
every run should end with verdict [works / fails / inconclusive-because- / abandoned, not worth the time.] - and incoclusive should be changed to some final state
freeze scoreboard [metrics / baselines]
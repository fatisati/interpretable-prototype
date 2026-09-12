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

Corollary for hyperparameters: a threshold parameter (`nassoc_gamma`) is only
interpretable once the quantity it is compared against is on a known scale. Derive it
from something real (`gamma = mean_weighted_degree * internal_fraction / (target_size -
0.5)` — measure `mean_weighted_degree` from the affinity, don't guess) rather than
sweeping blind, and re-derive it if the scaling of the underlying quantity changes.

## 9c. scProto base config — always start from this

The loss fixes in §9b are **not** the `defaults.py` defaults (defaults stay
behavior-preserving so existing model-folder names don't change). Pass them explicitly.
Validated on `s28nsc`/K=200: DGE `pearson_r` 0.654 -> 0.780, 0/200 dead prototypes, and it
also rescued the affinities where the old loss collapsed to 15/200 (0.342 -> 0.760).

```python
lambda_config = dict(LAMBDA_PROTO_UMAP_PRECON) | {
    'nassoc_diag_norm': 'pair',   # [D1] density, not containment: fixes the merge bias
    'nassoc_diag_loss': 'cpm',    # [D2] CPM reward, not an unreachable target of 1
    'nassoc_gamma':     0.15,     # derived from a 75-cell target, see §9b -- re-derive per graph
    'nassoc_size_scale': 'full',  # [D5] THE important one: without it density explodes (§9b)
    'nassoc_cpm_norm': 'gful',    # full-graph normalizer so gamma keeps a gradient AND the
                                  # term stays on the same scale as the other losses
}
```

`nassoc_cpm_norm='graf'` is the older minibatch-scaled normalizer, kept only so the runs
that used it stay reproducible — prefer `'gful'` with `nassoc_size_scale='full'`.

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


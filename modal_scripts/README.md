# modal_scripts

Runnable Modal jobs for this project. See `CLAUDE.md` §11 for the full recipe and
gotchas (Windows encoding, `Data_Handler.py` stubs, anndata `null`-reader patch, Modal
volume layout). This folder exists so that recipe survives across sessions — the first
working versions of these scripts were built in a session scratchpad (not committed) and
would otherwise have to be re-derived from scratch every time.

**Run with:**
```
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/<script>.py
```

## What's already on the `scproto-data` volume

Check with `modal volume ls scproto-data <path>` before recomputing anything — this
list is a snapshot, not a promise; verify before trusting it for a real report.

```
/vol/data/spatial/NSCLC_3D_section_28.h5ad     s28nsc raw data
/vol/data/spatial/crc_xenium/                  crcx raw data
/vol/code/interpretable_ssl/                   the package -- MUST be re-synced after
                                                any local edit (modal volume put), the
                                                image does not rebuild automatically
/vol/code/graphs/affinity_s28nsc..._bk02.pkl   cached affinities: bk02 (BANKSY λ=0.2),
/vol/code/graphs/affinity_s28nsc..._bk04.pkl   bk04 (λ=0.4), bk08 (λ=0.8, also on crcx)
/vol/code/graphs/affinity_s28nsc..._bk08.pkl
/vol/code/graphs/affinity_crcx..._bk08.pkl
/vol/models/s28nsc/pretrain/                   scPoli CVAE pretrain checkpoint (shared
                                                across every affinity_type sweep at K=200)
/vol/models/s28nsc/proto_bk08_np200_ema15_d1pair_modal_.../     D1-only (nassoc_diag_norm=
/vol/models/s28nsc/proto_bk02_np200_ema15_d1pair_modal_.../     'pair'), one per BANKSY λ,
/vol/models/s28nsc/proto_bk04_np200_ema15_d1pair_modal_.../     K=200, trained+evaluated
/vol/models/s28nsc/proto_bk08_npdef_ema15_d1pair_modal_.../NP800  same, K=800
/vol/models/s28nsc/seacell_X_bklam02_K200/     SEACells on the bk02/bk04 embeddings,
/vol/models/s28nsc/seacell_X_bklam04_K200/     K=200 -- the transcriptomics+BANKSY
                                                baseline, no scProto loss involved
/vol/models/crcx/                              crcx-side runs (bk08)
```

## Scripts

- `train_bk02_d1_offdiag_min.py` — scProto, s28nsc, K=200, `affinity_type='bk02'`
  (BANKSY λ=0.2), **both** nassoc fixes on: D1 (`nassoc_diag_norm='pair'`) + D4
  (`nassoc_offdiag_norm='min'`). One-variable-differs control is the existing
  `proto_bk02_np200_ema15_d1pair_modal` run (D1 only, off-diagonal left at the default
  `'sqrt'`) already on the volume — diff the two to attribute any change to D4.
  ~12-13 min on a T4 (a few dollars).
- `train_pancreas_nadn_pair.py` — scProto, **pancreas** (main-table, non-spatial), K=220,
  `affinity_type='arbf'`, D1 only (`nassoc_diag_norm='pair'`). First check of whether the
  collapse problem found on spatial data with the default loss also affects the
  main-table datasets. Reuses the existing pancreas pretrain checkpoint (uploaded to
  `models/pancreas/pretrain/pretrain_ds-pancreas_cvae_e50/`) — no fresh pretrain cost.
  One-variable-differs control is the canonical published run,
  `proto_umap_ds-panc_NP220_prtInit-wayp_aff-arbf_..._v31` (same base config,
  `nassoc_diag_norm` left at default `'volume'`).

## Before running: sync code

The image mounts `/vol/code` from the volume — it does **not** rebuild from your local
`interpretable_ssl/`. After any local edit to the package (e.g. a new loss term), push
it first:
```
modal volume put scproto-data interpretable_ssl code/interpretable_ssl --force
```
(from the repo root, using the volume-name-only form since these scripts all use
`modal.Volume.from_name("scproto-data")`).

## After running: pull results back

Training writes `metrics.json`, `cell_assignments.csv` etc. under
`/vol/models/s28nsc/<run>/` on the volume. Either:
- `modal volume get scproto-data models/s28nsc/<run> <local_dir>` for one run, or
- read them from Python in a follow-up Modal function (cheaper than round-tripping
  through local disk if the next step is also DGE eval on Modal).

Do **not** rely on the Drive-mounted mirror path for anything long or large — see
CLAUDE.md §11's Windows `MAX_PATH` / h5py-over-Drive notes.

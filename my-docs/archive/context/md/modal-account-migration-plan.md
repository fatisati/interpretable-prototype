# Modal account migration plan (`fatemehs-hashemig` → new account)

Written 2026-09-22, updated same day after actually staging the download half.
**STATUS: the download half is DONE.** Everything Modal-only is sitting at
`C:/mig_stage/` on local disk, and `C:/mig_stage/upload_to_new_account.sh` is a
ready-to-run script that does the entire upload half in one shot the moment you've
run `modal token new` for the new account and `modal volume create scproto-data`.
Just run that script — no need to re-derive anything below unless it fails.

Surveyed live from the `scproto-data` volume with `modal volume ls --json`.

**Two things the initial survey got wrong, corrected after actually downloading files:**

1. **`modal volume ls` reports a DIRECTORY's own size, not its recursive contents.**
   The "delta" files under `models/s28nsc/` (`seacell_X_bklam02_K200` etc.) looked like
   51–328 bytes in the listing — they are actually **3.9–16.4 MB each** (they are
   directories containing real per-tag files, per the CLAUDE.md section 8 dedup design).
   Same applies to every "size" shown for a `"type": "dir"` entry throughout this doc —
   treat those as unknown, not small, until actually downloaded.

2. **The local copy of `data/spatial/NSCLC_3D_section_28.h5ad` does NOT match the
   volume's copy** (245.8 MiB local vs 268.1 MiB on Modal — a real ~23 MB difference,
   not a rounding artifact). The Modal copy was almost certainly modified in-session
   (this codebase's notebooks write new `obsm` keys back to the h5ad on disk). **Do not
   substitute the local copy** — the correct one is now staged at
   `C:/mig_stage/data_spatial/NSCLC_3D_section_28.h5ad`, pulled directly from the volume.
   By contrast `data/pancreas_hvg.h5ad` (local 62,136,981 B vs Modal's 59.3 MiB) DOES
   match and was safely left on local disk / uploaded straight from there.

**A third thing worth recording: Windows `MAX_PATH` bit during staging, exactly as
CLAUDE.md section 11 warns.** `modal volume get` failed with `[WinError 3] The system
cannot find the path specified` when the destination was under this session's scratchpad
path (nested Drive-shortcut folder + a long session-id directory pushed the full path
over 260 chars). Fixed by using a short destination (`C:/mig_stage/...`) instead — not a
per-call flag, a genuinely different (short) path. Keep this in mind for the upload half
too: `C:/mig_stage/...` source paths are already short, so `upload_to_new_account.sh`
should not hit it, but if a future `get`/`put` mysteriously "can't find" a path that
provably exists, this is the first thing to suspect.

## The core principle

**Move source data and code. Do not move computed results.** Every run this session and
before it can be regenerated from `interpretable_ssl/` + the affinity graphs + the raw
h5ads by re-running the corresponding script — CLAUDE.md's own "load-if-exists" rule
means the results only exist to avoid recomputation, not because they're irreplaceable.
Moving them just relocates a cache. The one exception is **SEACells outputs**, which are
slow to (re)compute (dozens of minutes, not seconds) and worth carrying over.

## Inventory, with move / don't-move and why

| Path | Size | Move? | Why |
|---|---|---|---|
| `code/interpretable_ssl/` | 5.8 MB | **YES** | The library. Everything depends on it. |
| `code/seacell_train.py` | 23.4 KB | **YES** | Entry point scripts import from it. |
| `code/data/*_label_encoder.pkl` (3 files) | ~1.4 KB total | **YES** | Tiny, and `load_dataset` needs them at import time — missing them breaks everything, not just SEACells. |
| `code/graphs/affinity_s28nsc..._bk02.pkl` | 53.4 MB | **YES** | The BANKSY λ=0.2 affinity for s28nsc — used in nearly every run this session. Recomputable but slow (full diffusion-map + kernel build); cheaper to move than rebuild. |
| `code/graphs/affinity_s28nsc..._bk04.pkl` | 50.4 MB | **YES** | Same, λ=0.4. Only skip if you've decided bk04 is dead for future work. |
| `code/graphs/affinity_s28nsc..._bk08.pkl` | 46.7 MB | **YES** | Same, λ=0.8. |
| `code/graphs/affinity_pancreas..._arbf.pkl` | 13.5 MB | **YES** | Pancreas's only affinity graph in use. Small, no reason not to. |
| `code/graphs/affinity_crcx..._bk08.pkl` | 41.3 MB | **MAYBE** | Only 2 run dirs ever used it (`models/crcx`, both essentially untouched this session). Skip unless CRC/Xenium work resumes. |
| `data/spatial/NSCLC_3D_section_28.h5ad` | 268.1 MB | **YES** | The raw s28nsc data. Needed for: any new training run, `eval_dge_headtohead.py` (DGE ground truth), any obs/label lookup. This is the single largest required file — budget real upload time for it. |
| `data/pancreas_hvg.h5ad` | 59.3 MB | **YES** | Same role for pancreas. |
| `data/spatial/crc_xenium/crc_xenium_prepped.h5ad` | 42.9 MB | **MAYBE** | Same call as the bk08 CRC graph above — only needed if CRC/Xenium work resumes. |
| `code/files/celltype_niches_full_s28nsc.csv` | 13.0 MB | **YES** | The DGE ground-truth table (`nd.get_ground_truth`). Without it `eval_dge_headtohead.py` cannot run at all on s28nsc. |
| `code/files/tables/`, `code/files/figures/` | small (< 1 MB combined, mostly subfolders from earlier eval scripts) | **NO** | Regenerated by whichever eval script writes them; not read as input by anything. |
| `models/*/pretrain/*/pretrain_checkpoint.pth` | 265 KB each (s28nsc LD16 one confirmed; expect similarly small for other K/LD combos) | **YES, all of them** | This is the single best-value item to move. Pretraining is `cvae_epochs=50` and is the slow, expensive step every proto run depends on (CLAUDE.md §11: "reuse the existing pretrain checkpoint — pretraining doesn't depend on affinity, so one checkpoint covers every affinity_type sweep at that K"). Tiny to move, expensive to regenerate. Find every `pretrain_checkpoint.pth` under `models/*/pretrain/` and move all of them — there were at least 3 for s28nsc alone at different K/latent-dim combos. |
| `models/s28nsc/seacell_sc_base.h5ad` + `.manifest.json` | 147.3 MB + 1.9 KB | **YES** | The shared SEACells base (CLAUDE.md §8's dedup design). A full SEACells fit takes real wall-clock time; this is the one non-checkpoint result worth the bandwidth. |
| `models/s28nsc/seacell_X_bklam02_K200`, `..._bklam04_K200`, `seacell_K200` (delta files) | tiny (51–328 B each — these are small delta files per §8, not full copies) | **YES** | Worthless without the base above, but they ARE the reference SEACells numbers every comparison in this session was measured against (`seac` in every table). Move alongside the base. |
| **All other `models/s28nsc/*` run dirs** (~87 of the 90 listed) | ~10–15 MB each (typically: `cell_assignments.csv` ~4 MB, `umap_cells.csv` ~4 MB, `metacells.h5ad` ~0.8 MB, `umap_checkpoint.pth` ~0.8 MB, `modularity.json` ~0.5 MB, remaining CSVs/JSON small) → **~1.0–1.3 GB total** | **NO** | These are today's experiment outputs: `pub_*`, `mch_*`, `rt2_*`, `cod_*`, `nao_*`, `coh_*`, `modu_*`, `fin_*`, `wrm_*`, `abl_*`, etc. Every number quoted in this session's `decisions.md`/git log is already **committed as text** (the metrics, not the raw files). Re-running any one of them from the pretrain checkpoint + affinity graph above costs ~13 min / ~$0.13 on a T4 — far cheaper than moving ~1.2 GB, and picking the 2–3 you actually want to keep training from is more useful than a blanket copy. |
| `models/pancreas/*` (10 run dirs) | similar per-dir size, smaller totals | **NO**, same reasoning | Except: check for a pancreas SEACells base/delta the same way as s28nsc — none was found in this survey (`models/pancreas/pretrain` is the only match), so there is nothing SEACells-shaped to carry over there yet. |
| `models/crcx/*` (2 run dirs) | small | **NO** | Essentially untouched this session; regenerate if CRC work resumes. |
| `code/__pycache__/` | negligible | **NO** | Compiled bytecode, regenerated automatically on first import. |
| `code/notebooks/s28nsc_dge_lite` | ~6 B (looks like an empty/placeholder dir) | **NO** | Nothing of substance found here in this survey — verify before assuming, but don't budget time for it. |
| `foldit-assets`, `polyp-datasets`, `organ-seg-models`, `organai-seg-models`, `polyp-models` (other volumes) | unknown, not surveyed | **OUT OF SCOPE** | Unrelated projects sharing the same Modal account. Not part of this migration unless you say otherwise. |

## Total to move, and why this order

```
tier 1 -- code + label encoders            ~5.8 MB     seconds
tier 2 -- affinity graphs (4, skip crcx)    ~164 MB     ~1-2 min
tier 3 -- raw h5ads (2, skip crc_xenium)    ~327 MB     ~4-6 min   <- the long pole
tier 4 -- DGE ground truth CSV               13 MB      seconds
tier 5 -- ALL pretrain checkpoints          ~1-2 MB     seconds   <- huge value/size ratio
tier 6 -- SEACells base + deltas           ~147 MB      ~1-2 min
                                    -----------------
                                    ~660 MB total, ~10-15 min of actual transfer
```

Do tiers 1–2 first (needed to even construct the Modal image / affinity path), then 5–6
(cheap, irreplaceable-in-practice), then 3–4 only once you know which dataset(s) you're
resuming — if it's s28nsc-only work, skip the pancreas h5ad and vice versa, cutting the
long pole roughly in half.

## What's already staged, right now, on local disk

```
C:/mig_stage/
  s28nsc/pretrain_NP200.pth              256 KB   pretrain ckpt, K=200 default latent
  s28nsc/pretrain_LD16_NP200.pth         265 KB   pretrain ckpt, K=200 latent=16 (today's runs)
  pancreas/pretrain_e50.pth              2.0 MB   pretrain ckpt, pancreas default latent
  pancreas/pretrain_LD20.pth             2.0 MB   pretrain ckpt, pancreas latent=20
  code_graphs/affinity_..._bk04.pkl       50 MB   the one graph not on local disk at all
  code_data/pancras_label_encoder.pkl    397 B
  code_data/crcx_label_encoder.pkl       430 B
  code_data/NSCLC_3D_section_28.pkl      571 B
  seacell/seacell_sc_base.h5ad           147 MB   the shared SEACells base
  seacell/seacell_sc_base.manifest.json  1.9 KB
  seacell/seacell_X_bklam02_K200         3.9 MB   delta -- reference numbers used ALL session
  seacell/seacell_X_bklam04_K200         14 MB    delta
  seacell/seacell_K200                   16 MB    delta
  data_spatial/NSCLC_3D_section_28.h5ad  268 MB   the MISMATCHED file -- use this, not local
  upload_to_new_account.sh               the one-shot upload script, ready to run
────────────────────────────────────────
  ~504 MB staged locally, verified present
```

Everything else needed (code, bk02/bk08/pancreas-arbf graphs, the DGE ground-truth CSV,
the pancreas h5ad) is already correct on local disk at its normal repo/Drive path and
`upload_to_new_account.sh` reads it from there directly — no need to stage those too.

## Step-by-step, once you're logged into the new account

```bash
modal token new                        # log into the NEW account (interactive, browser)
modal volume create scproto-data       # on the new account
bash C:/mig_stage/upload_to_new_account.sh
```

That's it — the script does all 8 steps (code, label encoders, all 4 affinity graphs,
DGE ground truth, all 4 pretrain checkpoints, the SEACells base + 3 deltas, both raw
h5ads, the 11 Islander stub pairs) and prints a verification listing at the end. It
sources everything either from the local repo (for files verified byte-identical to the
volume) or from `C:/mig_stage/` (for files that were Modal-only or didn't match). See
"What's already staged" above for exactly what's in each location and why.

If you need to redo the download half for any reason (e.g. new files appear on the old
account before you switch), the individual `modal volume get` commands used to build
`C:/mig_stage/` are in this session's history — the pattern for each was:
```bash
modal volume get --force scproto-data "<volume path>" "C:/mig_stage/<short_dest>"
```
using a **short destination path** (not the scratchpad path) to avoid the `MAX_PATH`
failure documented above.

## What you get for free, with nothing moved

Once tiers 1–2 (code + affinity graphs) are on the new account, **any of today's ~30
Modal scripts in `modal_scripts/` runs immediately** — they all read `LAMBDA_PROTO_UMAP_PRECON`
and the graph path from the volume, not from any of the run-output directories being
skipped. The only thing lost by skipping tier-1 result dirs is having to re-train rather
than re-load a specific comparison arm — and per CLAUDE.md's own load-if-exists rule,
that was always the fallback path, not a special case.

## What this plan deliberately leaves out

- **CRC/Xenium** (graph, h5ad, 2 run dirs) — clearly cold this session; move on request.
- **Old s28nsc single-batch pretrain variants** (`pretrain_ds-s28nsc_cvae_e50_NP200`,
  `pretrain_ds-s28nsc_cvae_e50` without `_LD16`) — kept in the "move all pretrain
  checkpoints" step since they're tiny, but flagged: if you know only the `_LD16_NP200`
  one is still relevant, moving just that one shaves a negligible amount of time and
  isn't worth the judgment call under time pressure.
- **Every experiment run directory from today** (`pub_*/mch_*/rt2_*/cod_*/...`) — by
  design. The numbers are in the git-committed decision log; the files are a cache.

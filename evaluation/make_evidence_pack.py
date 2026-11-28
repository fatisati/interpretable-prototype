"""Copy one dataset's run outputs into a small folder that can be committed,
so the other machine can compare against the real files (not only hashes).

Writes repro/evidence/<machine>/<dataset>/, laid out like the run folder so
evaluation/compare_runs.py reads either one:

    hvg_genes.txt                 genes in the processed file, in matrix order
    split.csv, split.json         (from datasets/frozen/<dataset>/ when the run folder has none)
    cluster_config.json, affinity_config.json
    affinity_full.npz             (skipped with --no-affinity; ~4 MB for heart)
    <baseline>/embedding.csv.gz   or assignments.csv.gz
    <baseline>/config.json
    <baseline>/metrics*/          every metrics folder, as is
    pip_freeze.txt                exact installed packages

The processed .h5ad itself is NOT copied (58 MB for heart): its content is
covered by the fingerprint's hashes and by hvg_genes.txt.

Usage:
    python evaluation/make_evidence_pack.py \\
        --run-dir runs/rare_cell_preservation/heart_cell_atlas \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --machine mac-studio
"""
import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen  # noqa: E402  committed gene list + split


def gzip_copy(src, dst):
    # mtime=0 -> the same input always gives the same .gz bytes
    with open(src, "rb") as fin, open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as fout:
        shutil.copyfileobj(fin, fout)


def run(run_dir, h5ad_path, machine, include_affinity=True):
    import anndata as ad

    run_dir = Path(run_dir)
    out = REPO_ROOT / "repro" / "evidence" / machine / run_dir.name
    out.mkdir(parents=True, exist_ok=True)

    genes = [str(g) for g in ad.read_h5ad(h5ad_path, backed="r").var_names]
    with open(out / "hvg_genes.txt", "w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(genes) + "\n")

    for name in ("split.csv", "split.json", "cluster_config.json", "affinity_config.json"):
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, out / name)
    # Since the freeze the split is committed in datasets/frozen/, not kept in the run folder.
    dataset = frozen.dataset_id(h5ad_path)
    if not (out / "split.csv").exists() and frozen.split_path(dataset).exists():
        split = pd.read_csv(frozen.split_path(dataset))
        split.to_csv(out / "split.csv", index=False, lineterminator="\n")
        shutil.copyfile(frozen.split_meta_path(dataset), out / "split.json")
    if include_affinity and (run_dir / "affinity_full.npz").exists():
        shutil.copyfile(run_dir / "affinity_full.npz", out / "affinity_full.npz")

    for folder in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        tables = [folder / n for n in ("embedding.csv", "assignments.csv") if (folder / n).exists()]
        if not tables:
            continue
        target = out / folder.name
        target.mkdir(exist_ok=True)
        for table in tables:
            gzip_copy(table, target / (table.name + ".gz"))
        if (folder / "config.json").exists():
            shutil.copyfile(folder / "config.json", target / "config.json")
        for metrics_dir in sorted(folder.glob("metrics*")):
            shutil.copytree(metrics_dir, target / metrics_dir.name, dirs_exist_ok=True)

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(out.parent / "pip_freeze.txt", "w", newline="\n", encoding="utf-8") as handle:
        handle.write(freeze.stdout.replace("\r\n", "\n"))

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"[evidence] wrote {out}  ({size / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--no-affinity", action="store_true")
    a = p.parse_args()
    run(a.run_dir, a.h5ad, a.machine, include_affinity=not a.no_affinity)

"""Frozen per-dataset decisions: the gene list and the train/test split.

Rule: code proposes, a committed file decides. Gene selection (HVG) and the
split are computed ONCE, saved as small text files under
`datasets/frozen/<dataset>/`, committed to git, and from then on every run
reads those files instead of recomputing. Why: recomputing can silently give
a different answer on another machine or package version. Example: the heart
atlas HVG step picked 2 different genes (of 2000) on Windows vs a Mac, from
the same raw file and the same code, and that alone can move rare-cell metrics.

    datasets/frozen/<dataset>/
        genes.txt    kept genes, one per line, in matrix order
        genes.json   how they were chosen + sha256 of genes.txt + run record
        split.csv    cell_id,split for every kept cell (also pins the cell IDs);
                     split.csv.gz for big datasets (> BIG_SPLIT_CELLS cells)
        split.json   split parameters + counts + sha256 + run record

<dataset> is the processed file's name without `_hvg.h5ad`, e.g.
`datasets/processed/heart_cell_atlas_hvg.h5ad` -> `heart_cell_atlas`.

Who writes what:
  - stage 1 (datasets/scripts/prepare*.py) writes genes.txt the first time,
    or when asked with --refresh-frozen; otherwise it reads it.
  - stage 2 (evaluation/prepare_eval.py) does the same for the split
    (--refresh-split).
  - everything later only reads, and stops if its data doesn't match.
A deliberate refresh shows up as a git diff, so a changed gene list or split
is a reviewed decision, never a side effect.
"""
import hashlib
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from run_record import REPO_ROOT, record_run

# Overridable so a remote machine (e.g. Modal) can point at its own copy.
FROZEN_ROOT = Path(os.environ.get("SCPROTO_FROZEN_DIR", REPO_ROOT / "datasets" / "frozen"))
BIG_SPLIT_CELLS = 100_000  # above this, store split.csv.gz to keep the repo small


def dataset_id(h5ad_path):
    """`datasets/processed/heart_cell_atlas_hvg.h5ad` -> `heart_cell_atlas`."""
    return Path(h5ad_path).stem.removesuffix("_hvg")


def frozen_dir(dataset):
    return FROZEN_ROOT / dataset


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def package_versions(packages):
    out = {"python": platform.python_version()}
    for pkg in packages:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


# ---------------------------------------------------------------- genes

def genes_path(dataset):
    return frozen_dir(dataset) / "genes.txt"


def read_genes(dataset):
    path = genes_path(dataset)
    if not path.exists():
        return None
    return [g for g in path.read_text(encoding="utf-8").splitlines() if g]


def write_genes(dataset, genes, method):
    """Write genes.txt + genes.json. `method` describes how they were chosen."""
    path = genes_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "\n" line endings on every OS, so the sha256 is the same everywhere.
    path.write_bytes(("\n".join(genes) + "\n").encode("utf-8"))
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "dataset": dataset,
        "n_genes": len(genes),
        "sha256": file_sha256(path),
        "method": method,
        "environment": package_versions(["scanpy", "scikit-misc", "anndata", "numpy", "scipy", "pandas"]),
    }, indent=2))
    record_run(sidecar)
    return path


def select_genes(adata, dataset, propose, method, refresh=False):
    """Return `adata` subset to the dataset's frozen gene list.

    propose(adata) -> gene names to keep, in matrix order. Only called when
    no frozen list exists yet or refresh=True; it may add columns to
    adata.var (e.g. scanpy's HVG stats), which are dropped again so the
    output looks the same whichever path ran.
    """
    frozen = read_genes(dataset)
    if frozen is not None and not refresh:
        missing = sorted(set(frozen) - set(adata.var_names))
        if missing:
            raise SystemExit(
                f"[frozen] {len(missing)} genes in {genes_path(dataset)} are not in this "
                f"data (first few: {missing[:5]}). Wrong raw file? Check its sha256 "
                f"against datasets/{dataset}.md."
            )
        print(f"[frozen] using committed gene list {genes_path(dataset)} "
              f"({len(frozen):,} genes) -- gene selection skipped")
        return adata[:, frozen].copy()

    var_cols = list(adata.var.columns)
    genes = list(propose(adata))
    adata.var = adata.var[var_cols]
    if frozen is not None:
        added, removed = sorted(set(genes) - set(frozen)), sorted(set(frozen) - set(genes))
        print(f"[frozen] --refresh-frozen: {len(added)} genes added {added[:10]}, "
              f"{len(removed)} removed {removed[:10]}")
    path = write_genes(dataset, genes, method)
    print(f"[frozen] wrote {path} ({len(genes):,} genes) -- commit it")
    return adata[:, genes].copy()


def check_genes(adata, dataset):
    """Stop if `adata`'s genes are not exactly the frozen list, in order."""
    frozen = read_genes(dataset)
    if frozen is None:
        raise FileNotFoundError(
            f"{genes_path(dataset)} does not exist -- run stage 1 (see docs/runbook.md) "
            f"for this dataset first."
        )
    if list(adata.var_names) != frozen:
        diff = sorted(set(adata.var_names) ^ set(frozen))
        raise ValueError(
            f"genes in this data do not match {genes_path(dataset)} "
            f"({len(diff)} differ, e.g. {diff[:5]}; or same genes, other order). "
            f"The processed file is out of date: re-run stage 1."
        )


# ---------------------------------------------------------------- split

def split_path(dataset):
    """The dataset's split file: split.csv, or split.csv.gz for big datasets."""
    d = frozen_dir(dataset)
    gz = d / "split.csv.gz"
    return gz if gz.exists() else d / "split.csv"


def split_meta_path(dataset):
    return frozen_dir(dataset) / "split.json"


def write_split(dataset, cell_ids, split, params):
    """Write split.csv(.gz) + split.json. `params` = what decided the split
    (seed, test_size, label_key, batch_key); load_split checks against it."""
    import pandas as pd

    d = frozen_dir(dataset)
    d.mkdir(parents=True, exist_ok=True)
    name = "split.csv.gz" if len(cell_ids) > BIG_SPLIT_CELLS else "split.csv"
    for old in (d / "split.csv", d / "split.csv.gz"):
        old.unlink(missing_ok=True)
    path = d / name
    df = pd.DataFrame({"cell_id": cell_ids, "split": split})
    if name.endswith(".gz"):
        # mtime=0: the same content gives the same bytes, so the sha256 is stable.
        import gzip
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    else:
        path.write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    meta = {
        **params,
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "file": name,
        "sha256": file_sha256(path),
        "environment": package_versions(["scikit-learn", "numpy", "pandas"]),
    }
    split_meta_path(dataset).write_text(json.dumps(meta, indent=2))
    record_run(split_meta_path(dataset))
    print(f"[frozen] wrote {path} ({meta['n_train']:,} train / {meta['n_test']:,} test) -- commit it")
    return path


def fingerprint(dataset):
    """sha256 of the frozen files, for a result's config.json: says exactly
    which gene list and split that result used."""
    out = {"dataset": dataset}
    for key, path in (("genes_sha256", genes_path(dataset)), ("split_sha256", split_path(dataset))):
        out[key] = file_sha256(path) if path.exists() else None
    return out

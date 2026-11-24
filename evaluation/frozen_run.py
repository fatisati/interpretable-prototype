"""Freeze one baseline's output (embedding + clusters) as a committed file, so
every machine scores exactly the same cells in exactly the same clusters.

Why: the gene list and split are already frozen (evaluation/frozen.py), but a
baseline's embedding is still recomputed per machine, and it differs in the
last digits between machines (PCA on heart, Windows vs Mac, same genes: up to
0.002). That is enough to change a few neighbour edges, and Leiden then returns
a different clustering (ARI 0.77) although its seed and version are the same.
Leiden is repeatable only when its input is identical, so the input is what
gets frozen. Same rule as frozen.py: code proposes, a committed file decides.

    datasets/frozen/<dataset>/baselines/<baseline>/
        embedding.csv.gz     the baseline's embedding.csv (or assignments.csv.gz
                             for SEACells), cluster columns included
        source_config.json   the run's config.json when it was frozen
        result.json          sha256 + content summary + run record

Three commands, all take the baseline's run folder:

    freeze    copy the run's output into datasets/frozen/ (then commit it).
              Done ONCE, on the machine whose results are the official ones.
    check     compare this machine's own run with the frozen one: coordinates,
              nearest neighbours, clusters. Says how close a recompute gets.
    restore   put the frozen output into the run folder, so stage 4
              (compute_metrics.py) scores it. The machine's own output is kept
              next to it as embedding.local.csv.

Usage (from the repo root):
    python evaluation/frozen_run.py freeze  --run-dir runs/rare_cell_preservation/heart_cell_atlas/pca
    python evaluation/frozen_run.py check   --run-dir runs/rare_cell_preservation/heart_cell_atlas/pca
    python evaluation/frozen_run.py restore --run-dir runs/rare_cell_preservation/heart_cell_atlas/pca
The dataset is the run folder's parent name; the baseline is the run folder's name.
"""
import argparse
import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_record import record_run  # first local import: records the script's start time
import frozen

TABLES = ("embedding.csv", "assignments.csv")
CLUSTER_COLS = ("cluster_train_run", "cluster_full_run")


def baseline_dir(dataset, baseline):
    return frozen.frozen_dir(dataset) / "baselines" / baseline


def frozen_table_path(dataset, baseline):
    """The frozen output file for this baseline, or None if nothing is frozen."""
    for name in TABLES:
        path = baseline_dir(dataset, baseline) / (name + ".gz")
        if path.exists():
            return path
    return None


def local_table_path(run_dir):
    for name in TABLES:
        if (run_dir / name).exists():
            return run_dir / name
    raise SystemExit(f"[frozen_run] {run_dir} has no embedding.csv or assignments.csv -- run the baseline first.")


def names(run_dir):
    run_dir = Path(run_dir)
    return run_dir, run_dir.parent.name, run_dir.name


def read_frozen_bytes(path):
    with gzip.open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------- freeze

def freeze(run_dir, replace=False):
    from fingerprint import fingerprint_baseline

    run_dir, dataset, baseline = names(run_dir)
    src = local_table_path(run_dir)
    out = baseline_dir(dataset, baseline)
    existing = frozen_table_path(dataset, baseline)
    if existing and not replace:
        raise SystemExit(
            f"[frozen_run] {existing} already exists. A frozen result is replaced only on "
            f"purpose: add --replace, then commit the diff."
        )
    out.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        (out / (name + ".gz")).unlink(missing_ok=True)

    # "\n" line endings and gzip mtime=0: the same content gives the same bytes
    # (and the same sha256) on every OS.
    content = src.read_bytes().replace(b"\r\n", b"\n")
    dst = out / (src.name + ".gz")
    with open(dst, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        f.write(content)

    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    (out / "source_config.json").write_bytes((json.dumps(config, indent=2) + "\n").encode("utf-8"))

    summary = fingerprint_baseline(run_dir) or {}
    summary.pop("config", None)
    summary.pop("metrics", None)
    sidecar = out / "result.json"
    sidecar.write_text(json.dumps({
        "dataset": dataset,
        "baseline": baseline,
        "file": dst.name,
        "sha256": frozen.file_sha256(dst),
        "frozen_inputs": config.get("frozen") or frozen.fingerprint(dataset),
        "content": summary,
        "environment": config.get("environment"),
    }, indent=2))
    record_run(sidecar)
    print(f"[frozen_run] wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB) -- commit {out}")
    return dst


# ---------------------------------------------------------------- check

def same_neighbours_share(a, b, k=15):
    """Share of cells whose k nearest neighbours are the same set in both."""
    from sklearn.neighbors import NearestNeighbors

    def neighbours(x):
        idx = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(x).kneighbors(x, return_distance=False)
        return np.sort(idx[:, 1:], axis=1)

    return float((neighbours(a) == neighbours(b)).all(axis=1).mean())


def check(run_dir, tol=0.01, min_same_neighbours=0.99):
    from sklearn.metrics import adjusted_rand_score

    run_dir, dataset, baseline = names(run_dir)
    frozen_path = frozen_table_path(dataset, baseline)
    if frozen_path is None:
        raise SystemExit(f"[frozen_run] nothing frozen for {dataset}/{baseline} -- run `freeze` on the official machine.")
    local_path = run_dir / "embedding.local.csv"
    if not local_path.exists():
        local_path = local_table_path(run_dir)
    a = pd.read_csv(frozen_path)
    b = pd.read_csv(local_path)
    print(f"[frozen_run] frozen: {frozen_path}\n[frozen_run] local:  {local_path}")

    if set(a["cell_id"].astype(str)) != set(b["cell_id"].astype(str)):
        print("   cells: DIFFERENT sets of cell ids -- not the same dataset or split")
        return "DIFFERENT"
    b = b.set_index(b["cell_id"].astype(str)).loc[a["cell_id"].astype(str)].reset_index(drop=True)
    identical, close = True, True

    if "split" in a and "split" in b:
        same_split = bool((a["split"] == b["split"]).all())
        print(f"   split column: {'same' if same_split else 'DIFFERENT'}")
        identical &= same_split
        close &= same_split

    latent = [c for c in a.columns if c.startswith("latent_")]
    if latent:
        xa, xb = a[latent].to_numpy(np.float64), b[latent].to_numpy(np.float64)
        diff = float(np.abs(xa - xb).max())
        # A component may come out with the opposite sign on another machine
        # (PCA): that changes no distance, so it is reported separately.
        signs = np.where((xa * xb).sum(axis=0) < 0, -1.0, 1.0)
        diff_aligned = float(np.abs(xa - xb * signs).max())
        share = same_neighbours_share(xa, xb * signs)
        print(f"   coordinates: max difference {diff:.3g} "
              f"({int((signs < 0).sum())} of {len(latent)} columns sign-flipped; after aligning: {diff_aligned:.3g})")
        print(f"   15 nearest neighbours: identical for {share:.4%} of cells")
        identical &= diff == 0.0
        close &= diff_aligned <= tol and share >= min_same_neighbours

    for col in CLUSTER_COLS:
        if col in a and col in b:
            keep = (a[col].notna() & b[col].notna()).to_numpy()
            la, lb = a.loc[keep, col].astype(str), b.loc[keep, col].astype(str)
            ari = float(adjusted_rand_score(la, lb))
            print(f"   {col}: {la.nunique()} vs {lb.nunique()} clusters, adjusted Rand index {ari:.4f} (1 = same grouping)")
            identical &= ari == 1.0
    if "metacell" in a and "metacell" in b:  # SEACells assignments
        ari = float(adjusted_rand_score(a["metacell"].astype(str), b["metacell"].astype(str)))
        print(f"   metacell: adjusted Rand index {ari:.4f}")
        identical &= ari == 1.0

    verdict = "IDENTICAL" if identical else ("WITHIN TOLERANCE" if close else "DIFFERENT")
    print(f"[frozen_run] {verdict}"
          + ("" if identical else " -- score the frozen file (`restore`) so both machines report the same numbers"))
    return verdict


# ---------------------------------------------------------------- restore

def restore(run_dir):
    run_dir, dataset, baseline = names(run_dir)
    frozen_path = frozen_table_path(dataset, baseline)
    if frozen_path is None:
        raise SystemExit(f"[frozen_run] nothing frozen for {dataset}/{baseline} -- run `freeze` on the official machine.")
    meta = json.loads((baseline_dir(dataset, baseline) / "result.json").read_text())
    sha = frozen.file_sha256(frozen_path)
    if sha != meta["sha256"]:
        raise SystemExit(f"[frozen_run] {frozen_path} does not match the sha256 in result.json -- the file was changed.")
    inputs_now = frozen.fingerprint(dataset)
    if meta.get("frozen_inputs") and meta["frozen_inputs"] != inputs_now:
        raise SystemExit(
            f"[frozen_run] {frozen_path} was made with gene list / split {meta['frozen_inputs']}, "
            f"but datasets/frozen/{dataset}/ now holds {inputs_now}. Freeze the baseline again."
        )

    content = read_frozen_bytes(frozen_path)
    target = run_dir / frozen_path.name.removesuffix(".gz")
    run_dir.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes().replace(b"\r\n", b"\n") != content:
        local_copy = target.with_name(target.stem + ".local.csv")
        if not local_copy.exists():  # keep this machine's own output, once
            target.replace(local_copy)
            print(f"[frozen_run] kept this machine's own output as {local_copy}")
    target.write_bytes(content)

    config_path = run_dir / "config.json"
    source = json.loads((baseline_dir(dataset, baseline) / "source_config.json").read_text())
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    # The clustering fields must describe the file that is now in the folder.
    for key in ("k_train_target", "k_full_target", "resolution_train", "resolution_full",
                "n_clusters_train", "n_clusters_full"):
        if key in source:
            config[key] = source[key]
    config["frozen_run"] = {
        "file": frozen_path.relative_to(frozen.FROZEN_ROOT).as_posix(),
        "sha256": sha,
        "made_with": source.get("environment"),
    }
    config_path.write_text(json.dumps(config, indent=2))
    record_run(config_path)
    print(f"[frozen_run] wrote {target} from {frozen_path}\n"
          f"[frozen_run] metrics/ for this run is now stale: re-run evaluation/compute_metrics.py")
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["freeze", "check", "restore"])
    p.add_argument("--run-dir", required=True, help="baseline run folder, runs/<task>/<dataset>/<baseline>")
    p.add_argument("--replace", action="store_true", help="freeze: overwrite an existing frozen result")
    p.add_argument("--tol", type=float, default=0.01,
                   help="check: largest coordinate difference still called WITHIN TOLERANCE")
    a = p.parse_args()
    if a.command == "freeze":
        freeze(a.run_dir, a.replace)
    elif a.command == "restore":
        restore(a.run_dir)
    else:
        sys.exit(0 if check(a.run_dir, a.tol) != "DIFFERENT" else 1)

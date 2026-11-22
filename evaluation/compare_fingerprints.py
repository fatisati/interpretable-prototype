"""Compare two fingerprint files (evaluation/fingerprint.py) stage by stage.

Prints one line per value, in pipeline order:
    SAME    identical
    CLOSE   numbers that differ by less than --tol
    DIFF    different (both values shown)
    ONLY-A / ONLY-B   present on one side only

Each line carries a level:
    L0  inputs that must be exactly equal (cells, genes, counts, split, K)
    L1  deterministic numerics (PCA, affinity graph, the PCA baseline)
    L2  trained / stochastic methods (scVI, SEACells) -- single runs are
        expected to differ; compare these through several seeds instead
    env package versions (information only)

The FIRST L0 DIFF is the place to start: everything after it differs as a
consequence. Exit code 1 if any L0 value differs, else 0.

Usage:
    python evaluation/compare_fingerprints.py \\
        repro/fingerprints/heart_cell_atlas/win-main.json \\
        repro/fingerprints/heart_cell_atlas/mac-studio.json
    add --all to also print SAME lines and values present on one side only
    (the summary always counts them).
"""
import argparse
import json
import sys
from pathlib import Path

STAGE_ORDER = ["processed", "split_params", "split", "cluster_config", "pca",
               "affinity_params", "affinity_full", "affinity_train", "baselines", "environment"]
L0_STAGES = {"processed", "split_params", "split", "cluster_config", "affinity_params"}
L1_STAGES = {"pca", "affinity_full", "affinity_train"}
DETERMINISTIC_BASELINES = {"pca"}
IGNORED_KEYS = {"machine", "run_dir", "dataset"}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def level_of(key):
    stage = key.split(".")[0]
    if stage in L0_STAGES:
        return "L0"
    if stage in L1_STAGES:
        return "L1"
    if stage == "environment":
        return "env"
    if stage == "baselines":
        parts = key.split(".")
        if len(parts) > 2 and parts[2] == "cells_hash":
            return "L0"
        return "L1" if parts[1] in DETERMINISTIC_BASELINES else "L2"
    return "--"


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare_values(a, b, tol):
    if a == b:
        return "SAME", ""
    if numeric(a) and numeric(b):
        diff = abs(a - b)
        return ("CLOSE" if diff <= tol else "DIFF"), f"|diff| = {diff:.3g}"
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and all(map(numeric, a + b)):
        diff = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
        return ("CLOSE" if diff <= tol else "DIFF"), f"max |diff| = {diff:.3g}"
    return "DIFF", ""


def brief(value, width=44):
    text = json.dumps(value, default=str)
    return text if len(text) <= width else text[: width - 3] + "..."


def stage_rank(key):
    stage = key.split(".")[0]
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tol", type=float, default=1e-6, help="numbers closer than this count as CLOSE")
    p.add_argument("--all", action="store_true", help="also print SAME and ONLY-A / ONLY-B lines")
    args = p.parse_args()

    fa = json.loads(Path(args.a).read_text(encoding="utf-8"))
    fb = json.loads(Path(args.b).read_text(encoding="utf-8"))
    name_a, name_b = fa.get("machine", "A"), fb.get("machine", "B")
    da = {k: v for k, v in flatten(fa) if k.split(".")[0] not in IGNORED_KEYS}
    db = {k: v for k, v in flatten(fb) if k.split(".")[0] not in IGNORED_KEYS}
    keys = sorted(set(da) | set(db), key=lambda k: (stage_rank(k), k))

    print(f"A = {name_a}   ({args.a})\nB = {name_b}   ({args.b})\n")
    counts, first_l0 = {}, None
    for key in keys:
        level = level_of(key)
        if key not in da:
            status, note = "ONLY-B", brief(db[key])
        elif key not in db:
            status, note = "ONLY-A", brief(da[key])
        else:
            status, extra = compare_values(da[key], db[key], args.tol)
            note = "" if status == "SAME" else f"A={brief(da[key])}  B={brief(db[key])}  {extra}"
        counts[(level, status)] = counts.get((level, status), 0) + 1
        if level == "L0" and status != "SAME" and first_l0 is None:
            first_l0 = key
        if status in ("DIFF", "CLOSE") or args.all:
            print(f"{level:3s} {status:6s} {key}  {note}")

    print("\nSummary (level: status = count)")
    for level in ("L0", "L1", "L2", "env"):
        parts = [f"{status} = {n}" for (lv, status), n in sorted(counts.items()) if lv == level]
        print(f"  {level:3s} {', '.join(parts) if parts else 'nothing compared'}")
    if first_l0:
        print(f"\nFIRST INPUT DIFFERENCE (L0): {first_l0}\n"
              "Start here. Later differences are expected consequences of this one.")
        sys.exit(1)
    print("\nAll L0 inputs are identical. Remaining differences are numerics (L1) or training runs (L2).")


if __name__ == "__main__":
    main()

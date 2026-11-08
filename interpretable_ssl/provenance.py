"""Provenance record for a training run -- everything that has to agree before two runs on
different machines (Modal CUDA, local CPU, local MPS) may be compared on the scoreboard.

The loss recipe and lambdas are already recorded in RECIPE.md / arm_summary.json and in the
model folder name. What was NOT recorded, and does move results, is the set of INPUT
artefacts and the software that built them:

    data h5ad          HVG selection + normalisation (prepare_pbmc_immune.py documents that the
                       original recipe is unrecorded; a checksum at least makes drift visible)
    split npz          train/test ids -- must be identical across machines
    affinity pickle    SEACells adaptive-RBF kernel; depends on scanpy's approximate kNN
    pretrain ckpt      stage-1 CVAE every stage-2 arm inherits; one per machine
    packages           torch / scanpy / anndata / numpy / scipy / scarches / leidenalg / igraph
    code               git commit + dirty flag
    device             cuda / mps / cpu

Usage (local script or Modal function alike):

    from interpretable_ssl.provenance import write_provenance
    write_provenance(run_dir, files={'data_h5ad': path, 'split_npz': path,
                                     'affinity_pkl': path, 'pretrain_ckpt': path},
                     affinity=aff_csr, device=str(trainer.device), extra={...})

Checksums are md5 over the full file (a 190 MB h5ad takes ~0.3 s). `compare_provenance`
tells two records apart by artefact so the scoreboard can say WHICH input differs.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

PACKAGES = ("torch", "numpy", "scipy", "scanpy", "anndata", "scarches", "sklearn",
            "leidenalg", "igraph", "SEACells", "faiss")


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def file_record(path):
    if path is None or not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    return {"path": os.path.abspath(path), "exists": True, "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
            "md5": md5_file(path)}


def package_versions():
    out = {"python": sys.version.split()[0]}
    from importlib import metadata
    dist = {"sklearn": "scikit-learn", "igraph": "igraph", "SEACells": "SEACells", "faiss": "faiss-cpu"}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(dist.get(name, name))
        except Exception:  # noqa: BLE001 -- absent package is itself the record
            try:
                out[name] = getattr(__import__(name), "__version__", "present")
            except Exception:  # noqa: BLE001
                out[name] = None
    return out


def git_state(repo_dir=None):
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        run = lambda *a: subprocess.check_output(["git", *a], cwd=repo_dir, text=True,
                                                 stderr=subprocess.DEVNULL).strip()
        return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}
    except Exception:  # noqa: BLE001
        return {"commit": None, "branch": None, "dirty": None}


def affinity_stats(A, batch_codes=None):
    """Summary of a (symmetric) affinity: enough to tell two kernels apart without the file."""
    import numpy as np
    import scipy.sparse as sp
    A = sp.csr_matrix(A)
    n = A.shape[0]
    co = sp.triu(A, k=1).tocoo()
    deg = np.asarray(A.sum(1)).ravel()
    out = {"n_cells": int(n), "nnz": int(A.nnz), "n_undirected_edges": int(co.nnz),
           "weight_sum": float(A.sum()), "deg_mean": float(deg.mean()),
           "deg_median": float(np.median(deg)), "deg_min": float(deg.min()),
           "deg_max": float(deg.max()), "diag_sum": float(A.diagonal().sum())}
    if batch_codes is not None:
        b = np.asarray(batch_codes)
        out["within_batch_edge_frac"] = float((b[co.row] == b[co.col]).mean())
    return out


def device_record(device=None):
    rec = {"device": str(device) if device is not None else None, "platform": platform.platform(),
           "machine": platform.machine(), "hostname": platform.node()}
    try:
        import torch
        rec["torch_cuda"] = torch.cuda.is_available()
        rec["torch_mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return rec


def build_provenance(files=None, affinity=None, batch_codes=None, device=None, extra=None):
    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "files": {k: file_record(v) for k, v in (files or {}).items()},
           "packages": package_versions(), "git": git_state(), "device": device_record(device)}
    if affinity is not None:
        rec["affinity_stats"] = affinity_stats(affinity, batch_codes)
    if extra:
        rec["extra"] = extra
    return rec


def write_provenance(run_dir, files=None, affinity=None, batch_codes=None, device=None,
                     extra=None, name="provenance.json"):
    os.makedirs(run_dir, exist_ok=True)
    rec = build_provenance(files, affinity, batch_codes, device, extra)
    path = os.path.join(run_dir, name)
    with open(path, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    return path


def compare_provenance(a, b, keys=("data_h5ad", "split_npz", "affinity_pkl", "pretrain_ckpt")):
    """Return {artefact: 'same' | 'differs' | 'missing'} for two provenance records (dicts or paths)."""
    load = lambda x: json.load(open(x)) if isinstance(x, str) else x
    a, b = load(a), load(b)
    out = {}
    for k in keys:
        fa, fb = a.get("files", {}).get(k), b.get("files", {}).get(k)
        if not fa or not fb or not fa.get("exists") or not fb.get("exists"):
            out[k] = "missing"
        else:
            out[k] = "same" if fa["md5"] == fb["md5"] else "differs"
    pa, pb = a.get("packages", {}), b.get("packages", {})
    out["packages_differ"] = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
    return out

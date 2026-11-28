# Reference container for the clustering check (repro/container_check.py).
#
# Linux, python 3.13.11, and the Windows machine's package versions: the pip freeze in
# repro/evidence/windows/pip_freeze.txt is used as a constraints file, so every package
# installed here, dependencies included, gets exactly the version listed there. Lines
# for packages that are not installed (Windows-only ones, torch, conda's own) do nothing.
#
# Build from the repo root; the build context is the folder that holds the pip freeze:
#   docker build -f repro/docker/Dockerfile -t clean-v2-ref repro/evidence/windows
#
# Holds what the PCA and scVI baselines, the neighbour graph and Leiden need. torch is the
# CPU build, as on Windows (2.13.0+cpu), from PyTorch's own package index. No SEACells yet.
FROM python:3.13.11-slim

# One thread everywhere: a sum taken in a different order can change the last digit.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    PYTHONHASHSEED=0

COPY pip_freeze.txt /tmp/constraints.txt
# Wheels only, so nothing is compiled here; docrep (pure Python, needed by scvi-tools) has no wheel.
RUN pip install --no-cache-dir --only-binary=:all: --no-binary=docrep -c /tmp/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    scanpy anndata pynndescent umap-learn leidenalg igraph numpy scipy scikit-learn pandas numba llvmlite \
    torch scvi-tools

WORKDIR /repo

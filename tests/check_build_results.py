"""Sanity check for stage 4 -> stage 5 (compute_metrics.py -> build_results.py).

Builds a small synthetic dataset in a temp folder, scores it with the REAL
compute_metrics.run (so the metrics files have exactly the real format),
then runs build_results.run on them and checks:
  1. the expected files are written,
  2. the expectation checks come out as designed (a baseline that scatters
     rare cells must FAIL to beat SEACells... i.e. E3 = PASS, etc.),
  3. a second run gives byte-identical output (deterministic).

Plain python, no pytest needed:
    python tests/check_build_results.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import build_results  # noqa: E402
import compute_metrics  # noqa: E402

RNG = np.random.default_rng(0)
TASK = "rare_cell_preservation"
DATASET = "toy"

# Per-batch type counts: A-F common-ish, G and H rare. (With fewer, more
# spread-out types the "mean - 1 std" rule flags nothing as rare.)
TYPE_COUNTS = {"A": 60, "B": 55, "C": 50, "D": 45, "E": 40, "F": 35, "G": 8, "H": 4}
BATCHES = ["b1", "b2", "b3"]


def make_cells():
    rows = []
    for b in BATCHES:
        for t, n in TYPE_COUNTS.items():
            for i in range(n):
                rows.append({"cell_id": f"{b}_{t}_{i}", "donor": b, "cell_type": t})
    df = pd.DataFrame(rows)
    # Stratified 80/20 split on (type, batch), like prepare_eval.py.
    df["split"] = "train"
    for _, idx in df.groupby(["donor", "cell_type"]).groups.items():
        idx = list(idx)
        n_test = max(1, len(idx) // 5)
        df.loc[RNG.choice(idx, n_test, replace=False), "split"] = "test"
    return df.reset_index(drop=True)


def affinity_for(df):
    """Same type = similar (0.8), different type = 0. Zero diagonal, like build_affinity."""
    t = df["cell_type"].to_numpy()
    dense = (t[:, None] == t[None, :]).astype(float) * 0.8
    np.fill_diagonal(dense, 0.0)
    return csr_matrix(dense)


def clusters(df, scatter_prob):
    """Cluster = cell type, except each cell is moved to one of a few mixed
    "leftover" clusters with a probability that grows as its type gets rarer.
    scatter_prob=0 -> perfect clustering (our fake SEACells).

    (Scattering into the big common-type clusters instead makes Spearman go
    negative while the slope stays positive: thousands of common cells each
    lose a little, a few rare cells lose a lot, and Spearman counts cells.)"""
    out = []
    for t in df["cell_type"]:
        rarity = 1 - TYPE_COUNTS[t] / max(TYPE_COUNTS.values())
        p = scatter_prob * rarity
        out.append(f"leftover_{RNG.integers(3)}" if RNG.random() < p else f"c_{t}")
    return out


def full_job_adapter(column):
    """Adapter for a transductive baseline: one 'full' job (all cells)."""
    def adapter(embedding_path, label_key, batch_key, affinity_dir):
        df = pd.read_csv(embedding_path).rename(columns={label_key: "true_type", batch_key: "batch"})
        df["cluster"] = df[column]
        return {"full": {"df": df, "affinity": compute_metrics.load_npz(Path(affinity_dir) / "affinity_full.npz"),
                         "score_mask": np.ones(len(df), dtype=bool)}}
    return adapter


def seacells_adapter(embedding_path, label_key, batch_key, affinity_dir):
    df = pd.read_csv(embedding_path).rename(columns={label_key: "true_type", batch_key: "batch"})
    train = df[df["split"] == "train"].reset_index(drop=True).copy()
    train["cluster"] = train["SEACell_train"]
    full = df.copy()
    full["cluster"] = full["SEACell_full"]
    load = compute_metrics.load_npz
    return {
        "train": {"df": train, "affinity": load(Path(affinity_dir) / "affinity_train.npz"),
                  "score_mask": np.ones(len(train), dtype=bool)},
        "full": {"df": full, "affinity": load(Path(affinity_dir) / "affinity_full.npz"),
                 "score_mask": np.ones(len(full), dtype=bool)},
    }


def build_fake_runs(runs_root):
    ds_dir = Path(runs_root) / TASK / DATASET
    ds_dir.mkdir(parents=True)
    cells = make_cells()
    train = cells[cells["split"] == "train"].reset_index(drop=True)
    save_npz(ds_dir / "affinity_full.npz", affinity_for(cells))
    save_npz(ds_dir / "affinity_train.npz", affinity_for(train))

    # scVI: train run mildly scatters rare cells, full run (test side) scatters more.
    scvi = cells.copy()
    scvi.loc[scvi["split"] == "train", "cluster_train_run"] = clusters(train, 0.4)
    scvi["cluster_full_run"] = clusters(cells, 0.7)
    (ds_dir / "scvi").mkdir()
    scvi.to_csv(ds_dir / "scvi" / "embedding.csv", index=False)

    harmony = cells.copy()
    harmony["cluster"] = clusters(cells, 0.6)
    (ds_dir / "harmony").mkdir()
    harmony.to_csv(ds_dir / "harmony" / "embedding.csv", index=False)

    sea = cells.copy()
    sea.loc[sea["split"] == "train", "SEACell_train"] = clusters(train, 0.0)
    sea["SEACell_full"] = clusters(cells, 0.0)
    (ds_dir / "seacells").mkdir()
    sea.to_csv(ds_dir / "seacells" / "assignments.csv", index=False)

    compute_metrics.ADAPTERS["harmony"] = full_job_adapter("cluster")
    compute_metrics.ADAPTERS["seacells"] = seacells_adapter
    for baseline, fname in [("scvi", "embedding.csv"), ("harmony", "embedding.csv"),
                            ("seacells", "assignments.csv")]:
        compute_metrics.run(baseline, ds_dir / baseline / fname, "cell_type", "donor", ds_dir)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        runs_root, results_dir = Path(tmp) / "runs", Path(tmp) / "results"
        build_fake_runs(runs_root)

        tables, summary = build_results.run(TASK, runs_root, results_dir)
        text1, csv1 = tables.read_bytes(), summary.read_bytes()
        build_results.run(TASK, runs_root, results_dir)
        assert tables.read_bytes() == text1 and summary.read_bytes() == csv1, "output not deterministic"

        headline, freq, bins, _ = build_results.load_all(runs_root, TASK)
        assert not bins.empty, "frequency_bins.csv not written/read"

        e2 = build_results.check_e2(freq, [DATASET])
        e3 = build_results.check_vs_reference(headline, [("cohesion", "rare")])
        status = {(r["baseline"], r["job"]): r["status"] for r in e3}
        assert status[("scvi", "train")] == "PASS", e3
        assert status[("scvi", "test")] == "N/A", e3
        assert status[("harmony", "full")] == "PASS", e3
        assert e2[0]["status"] in ("PASS", "FAIL") and "slope" in e2[0]["detail"], e2

        # Every baseline that scatters rare cells should show a positive slope
        # (metric falls as frequency falls). Not asserted on Spearman: in this
        # toy, cohesion's self-similarity term (sim(i,i)=1) gives tiny clusters
        # a tiny bonus, and Spearman (rank-only) picks that up as a consistent
        # negative trend even though the size of the effect is ~0.01.
        e1 = build_results.check_e1(freq)
        scattered = [r for r in e1 if r["baseline"] != "seacells"]
        assert all(r["slope"] > 0 for r in scattered), scattered
        # ...so E1 must PASS them all (slope decides), and flag the cohesion rows
        # where Spearman disagrees instead of silently failing them.
        assert all(r["status"] == "PASS" for r in scattered), scattered
        flagged = [r for r in scattered if "Spearman disagrees" in r["reason"]]
        assert any(r["metric"] == "cohesion" for r in flagged), scattered
        assert e2[0]["status"] == "PASS", e2

        # Regression: fully-missed rare types must count as F1 = 0, not be
        # dropped from the average (scVI test scatters most rare cells).
        f1 = build_results.lookup(headline, DATASET, "scvi", "test", "rare_f1", "rare")
        cov = build_results.lookup(headline, DATASET, "scvi", "test", "coverage", "rare")
        assert f1 < 0.5 and cov < 0.5, (f1, cov)

        print("\n" + "=" * 70 + "\nGENERATED TABLES (first 60 lines)\n" + "=" * 70)
        print("\n".join(tables.read_text(encoding="utf-8").splitlines()[:60]))
        print("\nE2:", e2)
        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

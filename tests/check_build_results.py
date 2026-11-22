"""Sanity check for stage 4 -> stage 5 (compute_metrics.py -> build_results.py).

Builds a small synthetic dataset in a temp folder, scores it with the REAL
compute_metrics.run (so the metrics files have exactly the real format),
then runs build_results.run on them and checks:
  1. the expected files are written, with every baseline and no PASS/FAIL,
  2. stage 4 numbers come out as designed (a fully-missed rare type counts
     as F1 = 0; baselines that scatter rare cells have a positive slope),
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
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
    """Same type = similar (0.8), different type = 0. Zero diagonal, like prepare_eval.py."""
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
    scvi["cluster_full_run"] = clusters(cells, 0.7)
    # Embedding: one direction per type, rarer types blurred more (noisier).
    types = sorted(TYPE_COUNTS)
    onehot = np.eye(len(types))[[types.index(t) for t in cells["cell_type"]]]
    noise = np.array([0.2 + 2 * (1 - TYPE_COUNTS[t] / max(TYPE_COUNTS.values())) for t in cells["cell_type"]])
    latent = onehot + RNG.normal(size=onehot.shape) * noise[:, None]
    for d in range(latent.shape[1]):
        scvi[f"latent_{d}"] = latent[:, d]
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

        headline, freq, groups, _, rare, f1 = build_results.load_all(runs_root, TASK)
        assert not groups.empty, "frequency_groups.csv not written/read"

        text = tables.read_text(encoding="utf-8")
        for needed in ("scvi", "harmony", "seacells", "cohesion", "Appendix",
                       "not scored yet:** bbknn, combat"):
            assert needed in text, f"missing from tables: {needed}"
        assert "PASS" not in text and "FAIL" not in text, "results file must not judge (no PASS/FAIL)"

        def value(baseline, job, metric, population):
            hit = headline[(headline["baseline"] == baseline) & (headline["job"] == job)
                           & (headline["metric"] == metric) & (headline["population"] == population)]
            return float(hit["mean"].iloc[0])

        # Every baseline that scatters rare cells must show "metric falls as
        # frequency falls" -- Spearman AND slope positive, across groups. This
        # is the case the old per-cell Spearman got wrong for cohesion (it
        # came out strongly negative: thousands of common cells with a tiny
        # small-cluster bonus outvoted the few scattered rare cells).
        scattered = freq[freq["baseline"] != "seacells"]
        assert (scattered["spearman_corr"] > 0).all(), scattered
        assert (scattered["slope_per_log10_freq"] > 0).all(), scattered
        # One group = one cell type in one batch: 8 types x 3 batches.
        assert (freq["n_groups"] <= 24).all() and (freq["n_groups"] >= 20).all(), freq
        assert set(groups.columns) >= {"batch", "true_type", "freq", "n_cells", "mean", "std", "is_rare"}
        out = results_dir / TASK / "official"
        assert tables == out / "tables.md" and (out / "data" / "groups.csv").exists()
        assert len(list((out / "figures").glob("*.png"))) == 8, "one figure per metric expected (4 cluster + 4 neighbor)"

        # Rare rules: fixed cutoff flags exactly the groups under it; every
        # rule's cutoff is a real frequency (> 0 for the log rule, which can't
        # go negative -- the problem with the task file's std rule).
        import metrics
        cells = make_cells().rename(columns={"cell_type": "true_type", "donor": "batch"})
        fixed = metrics.rare_group_table(cells, "fixed", fixed_cutoff=0.02)
        assert set(fixed.loc[fixed["is_rare"], "true_type"]) == {"H"}, fixed
        log = metrics.rare_group_table(cells, "log_std")
        assert (log["threshold"] > 0).all() and log["is_rare"].any(), log
        assert set(rare.columns) >= {"batch", "true_type", "n_cells", "freq", "threshold", "is_rare"}

        # Regression: fully-missed rare types must count as F1 = 0, not be
        # dropped from the average (scVI test scatters most rare cells).
        f1 = value("scvi", "test", "rare_f1", "rare")
        cov = value("scvi", "test", "coverage", "rare")
        assert f1 < 0.9 and cov < 0.9, (f1, cov)  # missed types pull F1 down (was 1.0 before the fix)

        print("\n" + "=" * 70 + "\nGENERATED TABLES (first 60 lines)\n" + "=" * 70)
        print("\n".join(text.splitlines()[:60]))
        print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

"""One-off: train scVI with another training seed on the SAME (seed-0) split."""
import sys
sys.path.insert(0, "/repo/baselines"); sys.path.insert(0, "/repo/evaluation")
import scvi_baseline as sb
orig = sb.load_split
sb.load_split = lambda adata, label, batch, test_size, seed, path: orig(adata, label, batch, test_size, 0, path)
sb.run("datasets/processed/heart_cell_atlas_hvg.h5ad", "donor", "cell_type",
       "runs/rare_cell_preservation/heart_cell_atlas/scvi_ref_arm64_seed1", layer="counts", seed=1, accelerator="cpu")

"""[D58] Pull finished ablation arms off the Modal volume and make them scorable.

For each d58 arm:
  1. find its run dir on the volume (the auto-generated long name)
  2. download clusters_full.npz (+ d58_arm_summary.json, grad_probe.jsonl)
     into files/volume_runs/pbmc-immune/d58_<arm>/
  3. ALSO write clusters.npz holding the SAME full-length labels, because
     score_all_baselines.load_labels() only ever looks for "clusters.npz" --
     a run that has only clusters_full.npz is SILENTLY skipped by the scorer
     ("no usable per-cell labels"). That is why d59_v2 / d59_v3 never reached
     the immune scoreboard despite holding correct 33,506-length labels.

Then score with:
    DATA_DIR=<dir> GRAPHS_DIR=<dir> python modal_scripts/score_nodeheldout.py pbmc-immune

usage:  python modal_scripts/collect_d58_arms.py [arm ...]
        (no args = every arm named in train_d58_ablation.ARMS)
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

VOL = "scproto-data"
DS = "pbmc-immune"
MODEL_SUBDIR = "models/pbmc-immune_arbf_nodeheldout20_seed0_train"
OUT_ROOT = os.path.join(REPO, "files", "volume_runs", DS)
N_CELLS = 33506

ARM_NAMES = [
    "base_s31", "base_s32", "base_s33",
    "gram0", "aff0", "coh0", "comm0", "recon0",
    "comm2x", "recon2x", "gram2x", "aff2x",
]


def _run(cmd):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)


def list_volume_dirs():
    r = _run(f'modal volume ls {VOL} {MODEL_SUBDIR}')
    if r.returncode != 0:
        print("modal volume ls failed:\n", r.stdout, r.stderr)
        return []
    return [ln.strip() for ln in r.stdout.splitlines()
            if ln.strip().startswith(MODEL_SUBDIR + "/")]


def main(arms):
    dirs = list_volume_dirs()
    if not dirs:
        print("no run dirs found on the volume -- is anything finished yet?")
        return
    ok, missing = [], []
    for arm in arms:
        # the run dir starts with the experiment_name we set: d58_<arm>_ds-...
        pref = f"{MODEL_SUBDIR}/d58_{arm}_"
        cand = [d for d in dirs if d.startswith(pref)]
        if not cand:
            missing.append(arm)
            continue
        src = cand[0]
        dst = os.path.join(OUT_ROOT, f"d58_{arm}")
        os.makedirs(dst, exist_ok=True)
        got = False
        for fn in ("clusters_full.npz", "d58_arm_summary.json", "grad_probe.jsonl"):
            lp = os.path.join(dst, fn)
            if os.path.exists(lp):
                got = got or fn == "clusters_full.npz"
                continue
            r = _run(f'modal volume get {VOL} "{src}/{fn}" "{lp}"')
            if r.returncode == 0 and os.path.exists(lp):
                got = got or fn == "clusters_full.npz"
            elif fn == "clusters_full.npz":
                print(f"  {arm}: clusters_full.npz not on the volume yet")
        if not got:
            missing.append(arm)
            continue
        # make it scorable: load_labels() reads ONLY clusters.npz
        z = np.load(os.path.join(dst, "clusters_full.npz"), allow_pickle=True)
        lab = np.asarray(z["labels"]).ravel()
        assert lab.shape[0] == N_CELLS, f"{arm}: labels {lab.shape} != {N_CELLS}"
        np.savez(os.path.join(dst, "clusters.npz"), labels=lab)
        n_mc = len(np.unique(lab))
        extra = ""
        sp = os.path.join(dst, "d58_arm_summary.json")
        if os.path.exists(sp):
            with open(sp) as f:
                s = json.load(f)
            extra = (f"  train={s.get('train_seconds')}s gpu={s.get('gpu')} "
                     f"n_dead={s.get('metrics', {}).get('n_dead')}")
        print(f"  OK  d58_{arm:<10} {n_mc} metacells used{extra}")
        ok.append(arm)

    print(f"\ncollected {len(ok)}/{len(arms)}: {ok}")
    if missing:
        print(f"not ready / missing: {missing}")
    print(f"\nnow score with:\n"
          f"  DATA_DIR=<scratch>/data/ GRAPHS_DIR=<scratch>/graphs \\\n"
          f"    python modal_scripts/score_nodeheldout.py {DS}")


if __name__ == "__main__":
    main(sys.argv[1:] or ARM_NAMES)

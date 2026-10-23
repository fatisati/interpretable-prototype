import os, sys
from collections import Counter
import numpy as np, pandas as pd
REPO = r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes\interpretable-prototype"
sys.path.insert(0, REPO); os.chdir(REPO)
from modal_scripts.score_all_baselines import DATASETS, load_labels
from interpretable_ssl.evaluation.metacell_scoreboard import read_obs
DS="pbmc-immune"; RUNS=os.path.join(REPO,"files","volume_runs",DS)
n,h5,bk,lk,nk,K = DATASETS[DS]
obs = read_obs(os.path.join(os.environ["DATA_DIR"],h5),[bk,lk])
cts=np.asarray(obs[lk]).astype(str); bts=np.asarray(obs[bk]).astype(str)
freq=pd.Series(cts).value_counts()
rare=list(freq[freq<=np.quantile(freq.values,0.25)].index)
for arm in ["d60_epoch0_leid","d60_leid_noc","d61_pr3_lv","d61_pr3_lvdec_nr"]:
    lab=load_labels(os.path.join(RUNS,arm),n)
    if lab is None: continue
    print(f"=== {arm} ===")
    for t in rare:
        mt = cts==t
        # protos holding >=5 cells of this rare type
        rows=[]
        for mc in np.unique(lab[mt]):
            s = mt & (lab==mc)
            if s.sum()<5: continue
            prot = lab==mc
            own = Counter(cts[prot]).most_common(1)[0][0]
            bcomp = Counter(bts[s]).most_common(2)
            rows.append((int(mc), int(s.sum()), int(prot.sum()), own,
                         "+".join(f"{b}:{c}" for b,c in bcomp)))
        owned = sum(1 for r in rows if r[3]==t)
        print(f"  {t:<30} n={mt.sum():<5} in {len(rows)} protos(>=5 cells), "
              f"{owned} where it is the MAJORITY")
        for mc,c,tot,own,bc in sorted(rows,key=lambda r:-r[1])[:4]:
            star = " <-OWNS" if own==t else ""
            print(f"      mc{mc:<4} {c:>4} of {tot:<5} cells | proto majority={own:<28} {bc}{star}")
    print()

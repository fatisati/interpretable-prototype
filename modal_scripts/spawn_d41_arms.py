"""[D41] Fire the 16 arms as SPAWNED calls on a DEPLOYED app.

WHY NOT `modal run --detach`: with a local entrypoint, detach only keeps the LAST
triggered function alive once the parent process dies -- Modal prints exactly
that warning. Two launches were lost to it here: the first when a shell pipe
(`| tail`) closed stdout and killed the parent, the second when the local machine
briefly lost DNS ([Errno 11001] getaddrinfo failed). Both times every container
received a cancellation signal mid-training.

A DEPLOYED app plus Function.spawn() has no such dependency: the calls are
queued server-side and keep running whatever happens to this laptop.

Usage:
    modal deploy modal_scripts/train_d41_multibatch_any.py
    python modal_scripts/spawn_d41_arms.py pancreas pbmc-immune lung crcx
"""
import sys

import modal

sys.path.insert(0, ".")
from modal_scripts.train_d41_multibatch_any import arms_for  # noqa: E402

targets = sys.argv[1:] or ["pancreas", "pbmc-immune", "lung", "crcx"]
fn = modal.Function.from_name("scproto-d41-any", "run")

jobs = []
for ds in targets:
    jobs.extend(arms_for(ds))

calls = []
for j in jobs:
    c = fn.spawn(*j)
    calls.append((f"{j[0]}/{j[1]}", c.object_id))
    print(f"spawned {j[0]}/{j[1]:<8} -> {c.object_id}")

with open("modal_scripts/.d41_spawned_calls.txt", "w") as f:
    for name, oid in calls:
        f.write(f"{name}\t{oid}\n")
print(f"\n{len(calls)} calls spawned; ids saved to "
      f"modal_scripts/.d41_spawned_calls.txt")
print("They run server-side -- this machine can disconnect safely.")

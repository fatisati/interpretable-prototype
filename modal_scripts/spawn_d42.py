"""Spawn the [D42] runs on the DEPLOYED app -- survives client disconnects.

`modal run --detach` with a local entrypoint only keeps the LAST triggered
function alive once the parent dies; two earlier launches were lost that way (a
shell pipe closing stdout, and a transient local DNS failure). A deployed app
plus Function.spawn() has no such dependency.

  modal deploy modal_scripts/train_d42_heldout.py
  python modal_scripts/spawn_d42.py [datasets...]
"""
import sys
import modal
sys.path.insert(0, ".")
from modal_scripts.train_d42_heldout import jobs_for

targets = sys.argv[1:] or ["pancreas", "lung", "pbmc-immune"]
fn = modal.Function.from_name("scproto-d42", "run")
calls = []
for j in jobs_for(targets):
    c = fn.spawn(*j)
    calls.append((j[0], c.object_id))
    print(f"spawned {j[0]:<14} K={j[1]:<4} gram={j[2]:.2f} aff={j[3]:.2f} "
          f"gamma={j[4]} -> {c.object_id}")
with open("modal_scripts/.d42_calls.txt", "w") as f:
    for n, o in calls:
        f.write(f"{n}\t{o}\n")
print(f"\n{len(calls)} spawned; ids in modal_scripts/.d42_calls.txt")

"""Write down what was actually run, next to the output it produced.

Every stage script calls `record_run(<its json sidecar>)` right after it saves
its output. That appends one entry to a `run_history` list inside the JSON:

    script            which script ran (repo-relative path)
    command           the exact command line, copy-pasteable
    git_commit        the code version (commit hash) checked out at the time
    git_dirty         true if tracked code had uncommitted edits -> the commit
                      alone does NOT fully describe the code that ran.
                      Prose *.md docs don't count.
    git_dirty_files   the uncommitted tracked files behind git_dirty
    script_committed  false if the script itself isn't in git at all
    started_at, finished_at

Why a list and not one field: recluster.py and compute_metrics.py re-run on
an existing folder, so replacing the entry would lose the original command.
A script that writes a fresh JSON (e.g. a new training run) starts a new list,
which is correct -- the output is new.

Importing this module also prints a warning at the start of the run if the code has
uncommitted edits (soft gate; build_results.py refuses such runs -- the hard gate).

How to run each stage (the recipe, written by hand) lives in docs/runbook.md;
this module is the record of what really happened. Never crashes a run: if git
is unavailable, the git fields are null.

No heavy imports on purpose -- datasets/scripts/ import this too.
"""
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Imported at the top of every stage script, so this is the script's start time.
STARTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _script_path():
    try:
        return Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return sys.argv[0]


def uncommitted():
    """(changed tracked files, script_untracked) for this process; ([], False) if git is unavailable.

    Untracked files other than the script don't count: new data/notes don't change the code
    that ran. An untracked *script* does -- nobody else can get it back."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is None:
        return [], False
    # split, not line[3:]: _git strips the output, which eats the first line's leading space.
    changed = [line.split(None, 1)[1] for line in status.splitlines() if line.strip()]
    # Prose docs (*.md) don't change what a run computes; an edit to one (e.g. the runbook,
    # by a parallel session) must not mark every run in that window dirty. Data files that
    # do change results (genes.txt, split.csv, *.json) still count.
    changed = [f for f in changed if not f.strip('"').endswith(".md")]
    script = _script_path()
    script_untracked = (script.endswith(".py") and not Path(script).is_absolute()
                        and _git("ls-files", "--error-unmatch", "--", script) is None)
    return changed, script_untracked


def _warn_if_uncommitted():
    """Soft gate: say so loudly at the START of a run, so a long run can be stopped and the
    code committed first. Never blocks (quick tests stay quick); the hard gate is in
    build_results.py. ASCII only: a Windows console can't always print symbols."""
    changed, script_untracked = uncommitted()
    if not changed and not script_untracked:
        return
    lines = ["", "WARNING: uncommitted code -- this run will NOT be reproducible "
                 "(run record gets git_dirty=true). Commit first if you will keep the output."]
    lines += [f"  changed: {f}" for f in changed[:5]]
    if len(changed) > 5:
        lines.append(f"  ... and {len(changed) - 5} more (git status)")
    if script_untracked:
        lines.append(f"  not in git at all: {_script_path()}")
    print("\n".join(lines) + "\n", file=sys.stderr, flush=True)


try:  # never crash a run over a warning
    _warn_if_uncommitted()
except Exception:
    pass


def current_run():
    """One run_history entry describing this process."""
    script = _script_path()
    available = _git("rev-parse", "HEAD") is not None
    changed, _ = uncommitted()
    return {
        "script": script,
        "command": "python " + shlex.join([script, *sys.argv[1:]]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(changed) if available else None,
        "git_dirty_files": changed,
        "script_committed": _git("ls-files", "--error-unmatch", "--", script) is not None,
        "started_at": STARTED_AT,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def record_run(json_path):
    """Append this run to `run_history` in json_path (created if missing)."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text()) if json_path.exists() else {}
    data.setdefault("run_history", []).append(current_run())
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, default=str))
    return data

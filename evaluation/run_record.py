"""Write down what was actually run, next to the output it produced.

Every stage script calls `record_run(<its json sidecar>)` right after it saves
its output. That appends one entry to a `run_history` list inside the JSON:

    script            which script ran (repo-relative path)
    command           the exact command line, copy-pasteable
    git_commit        the code version (commit hash) checked out at the time
    git_dirty         true if tracked code had uncommitted edits -> the commit
                      alone does NOT fully describe the code that ran
    script_committed  false if the script itself isn't in git at all
    started_at, finished_at

Why a list and not one field: recluster.py and compute_metrics.py re-run on
an existing folder, so replacing the entry would lose the original command.
A script that writes a fresh JSON (e.g. a new training run) starts a new list,
which is correct -- the output is new.

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


def current_run():
    """One run_history entry describing this process."""
    script = _script_path()
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "script": script,
        "command": "python " + shlex.join([script, *sys.argv[1:]]),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
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

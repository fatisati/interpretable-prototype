"""Live progress of every stage script, in MLflow (local only, nothing leaves the laptop).

See every run in one browser page:
    python evaluation/dashboard.py        -> http://localhost:5000

What a stage script does (three calls, all optional after start):
    tracking.start(out_dir, kind="baseline", params={...}, stages=["load", "training", ...])
    tracking.stage("training")            # entering a new step
    tracking.progress(done, total)        # inside a long step (epochs, cluster-search tries)
    tracking.log_summary(nested_dict)     # final numbers (compute_metrics.py)
The run is closed automatically when the process exits: FINISHED on a normal exit,
FAILED on an exception (the error is kept in the `error` tag), KILLED on Ctrl+C.

One MLflow run = one script call on one folder, e.g. "neurips_bmmc / scvi". Its
state is MLflow's own (RUNNING / FINISHED / FAILED / KILLED). A process killed from
outside (task manager, machine sleeps or shuts down) can't report, so it would stay
RUNNING forever; each run stores its process id, and the next start() or the
dashboard marks such runs FAILED with error "process died without reporting".

Time left (`eta_min` metric, and the readable `progress` tag):
  current step:  average time per unit so far x units left (epochs, tries)
  later steps:   their duration in the last FINISHED run of the same script on the
                 same folder, if there is one; otherwise unknown, and the tag says so.

Never crashes a run: any tracking error prints one warning and turns tracking off.
Turn it off on purpose with SCPROTO_TRACKING=0. The run record JSON next to each
output (run_record.py) stays the source of truth for what ran; this is the live view.
"""
import atexit
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from run_record import REPO_ROOT, current_run

TRACKING_URI = os.environ.get("SCPROTO_MLFLOW_URI", f"sqlite:///{(REPO_ROOT / 'mlruns.db').as_posix()}")
ARTIFACT_ROOT = REPO_ROOT / "mlartifacts"
MIN_WRITE_INTERVAL = 5.0  # seconds between progress writes (stage changes always write)

_s = {"on": os.environ.get("SCPROTO_TRACKING", "1") != "0", "run_id": None}


def _warn_off(err):
    if _s["on"]:
        print(f"[tracking] WARNING: turned off for this run ({type(err).__name__}: {err}). "
              f"The run itself continues.", file=sys.stderr)
    _s["on"] = False


def _safe(fn):
    def wrapper(*a, **kw):
        if not _s["on"]:
            return None
        try:
            return fn(*a, **kw)
        except Exception as err:  # tracking must never take a run down
            _warn_off(err)
            return None
    wrapper.__doc__ = fn.__doc__
    return wrapper


def _client():
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow, mlflow.tracking.MlflowClient()


def _names_from_dir(out_dir):
    """runs/<task>/<dataset>/<method> -> (task, dataset, method)."""
    p = Path(out_dir).resolve()
    try:
        parts = p.relative_to(REPO_ROOT / "runs").parts
    except ValueError:
        parts = ()
    if len(parts) >= 3:
        return parts[0], parts[1], "/".join(parts[2:])
    if len(parts) == 2:  # dataset-level folder, e.g. prepare_eval's out-dir
        return parts[0], parts[1], "-"
    return "misc", p.parent.name, p.name


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        return True  # can't tell -> don't touch it


def mark_dead_runs():
    """RUNNING runs from this machine whose process is gone -> FAILED. Returns how many."""
    if not _s["on"]:
        return 0
    try:
        _, client = _client()
        host, n = socket.gethostname(), 0
        for exp in client.search_experiments():
            for r in client.search_runs([exp.experiment_id], "attributes.status = 'RUNNING'"):
                t = r.data.tags
                if t.get("host") == host and t.get("pid") and r.info.run_id != _s["run_id"] \
                        and not _pid_alive(t["pid"]):
                    client.set_tag(r.info.run_id, "error", "process died without reporting "
                                   "(killed from outside, crashed hard, or the machine slept/shut down)")
                    client.set_terminated(r.info.run_id, "FAILED")
                    n += 1
        return n
    except Exception as err:
        _warn_off(err)
        return 0


@_safe
def start(out_dir, kind, params=None, stages=None):
    """Open a run for this process. kind: what the script does, e.g. "eval setup",
    "baseline", "recluster", "metrics". stages: the planned steps, in order (used for
    the time-left estimate of later steps)."""
    mlflow, client = _client()
    task, dataset, method = _names_from_dir(out_dir)
    exp = client.get_experiment_by_name(task)
    exp_id = exp.experiment_id if exp else client.create_experiment(task, ARTIFACT_ROOT.as_uri())
    mark_dead_runs()

    # Typical step durations from the last finished run of the same kind on the same folder.
    typical = {}
    prev = client.search_runs(
        [exp_id], f"tags.dataset = '{dataset}' and tags.method = '{method}' and tags.kind = '{kind}' "
                  f"and attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"], max_results=1)
    if prev:
        typical = {k.split("/", 1)[1]: v for k, v in prev[0].data.metrics.items()
                   if k.startswith("duration_min/")}

    rec = current_run()
    name = f"{dataset} / {method}" if kind == "baseline" else f"{dataset} / {method} / {kind}"
    run = client.create_run(exp_id, run_name=name, tags={
        "dataset": dataset, "method": method, "kind": kind,
        "command": rec["command"], "git_commit": str(rec["git_commit"]),
        "git_dirty": str(rec["git_dirty"]), "out_dir": str(out_dir),
        "host": socket.gethostname(), "pid": str(os.getpid()),
        "mlflow.source.name": rec["script"], "mlflow.source.git.commit": str(rec["git_commit"]),
    })
    _s.update(run_id=run.info.run_id, stages=list(stages or []), typical=typical, stage=None,
              t_run=time.time(), t_stage=time.time(), last_write=0.0, durations={}, error=None,
              exit="FINISHED")
    for k, v in (params or {}).items():
        client.log_param(run.info.run_id, k, str(v)[:6000])
    atexit.register(_end)
    _prev_hook = sys.excepthook

    def hook(exc_type, exc, tb):
        _s["exit"] = "KILLED" if issubclass(exc_type, KeyboardInterrupt) else "FAILED"
        _s["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
        _s["traceback"] = "".join(traceback.format_exception(exc_type, exc, tb))
        _prev_hook(exc_type, exc, tb)
    sys.excepthook = hook
    print(f"[tracking] live progress: python evaluation/dashboard.py -> http://localhost:5000  "
          f"(run '{name}')")
    _write(force=True)


def _eta_later_steps():
    stages, cur = _s["stages"], _s["stage"]
    later = stages[stages.index(cur) + 1:] if cur in stages else []
    if not later:
        return 0.0, True
    known = [_s["typical"].get(s) for s in later]
    return sum(k for k in known if k is not None), all(k is not None for k in known)


def _write(done=None, total=None, force=False):
    now = time.time()
    if not force and now - _s["last_write"] < MIN_WRITE_INTERVAL:
        return
    _s["last_write"] = now
    _, client = _client()
    rid, cur = _s["run_id"], _s["stage"] or "starting"
    stage_min = (now - _s["t_stage"]) / 60
    metrics = {"elapsed_min": (now - _s["t_run"]) / 60}

    eta_stage = None
    if done and total:
        metrics["step_pct"] = 100.0 * done / total
        eta_stage = stage_min / done * max(total - done, 0)
    elif _s["typical"].get(cur) is not None:  # no inner progress: use last run's duration
        eta_stage = max(_s["typical"][cur] - stage_min, 0)
    later, later_known = _eta_later_steps()

    parts = [cur]
    if _s["stages"] and cur in _s["stages"]:
        parts[0] = f"{cur} (step {_s['stages'].index(cur) + 1}/{len(_s['stages'])})"
    if done is not None and total:
        parts.append(f"{done}/{total} ({100 * done / total:.0f}%)")
    if eta_stage is not None:
        eta = eta_stage + later
        metrics["eta_min"] = eta
        parts.append(f"~{eta:.0f} min left" + ("" if later_known else " + later steps (unknown, first run)"))
    else:
        parts.append(f"{stage_min:.0f} min in this step, time left unknown (first run)")
    for k, v in metrics.items():
        client.log_metric(rid, k, float(v))
    client.set_tag(rid, "stage", cur)
    client.set_tag(rid, "progress", ", ".join(parts))
    client.set_tag(rid, "last_update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def _close_stage():
    if _s.get("stage"):
        mins = (time.time() - _s["t_stage"]) / 60
        _s["durations"][_s["stage"]] = mins
        _, client = _client()
        client.log_metric(_s["run_id"], f"duration_min/{_s['stage']}", mins)


@_safe
def stage(name):
    """Mark the start of a new step (closes the previous one and records its duration)."""
    if _s["run_id"] is None:
        return
    _close_stage()
    _s.update(stage=name, t_stage=time.time())
    if name not in _s["stages"]:
        _s["stages"].append(name)
    _write(force=True)


@_safe
def progress(done, total):
    """Inside a long step: `done` of `total` units finished (throttled to one write per 5 s)."""
    if _s["run_id"] is not None:
        _write(done, total)


@_safe
def log_metrics(metrics, step=None):
    """Numbers to chart over time, e.g. losses per epoch (step = epoch)."""
    if _s["run_id"] is None:
        return
    _, client = _client()
    for k, v in metrics.items():
        try:
            client.log_metric(_s["run_id"], k, float(v), step=step or 0)
        except (TypeError, ValueError):
            pass


@_safe
def log_summary(summary, prefix=""):
    """Final numbers from a nested dict: {"test": {"rare_f1": {"mean": 0.8}}} -> test/rare_f1/mean."""
    flat = {}

    def walk(d, path):
        for k, v in d.items():
            key = f"{path}/{k}" if path else str(k)
            if isinstance(v, dict):
                walk(v, key)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[key] = v
    walk(summary, prefix)
    log_metrics(flat)


def _end():
    if not _s["on"] or _s["run_id"] is None:
        return
    try:
        _close_stage()
        _, client = _client()
        rid = _s["run_id"]
        if _s["error"]:
            client.set_tag(rid, "error", _s["error"][:5000])
            client.log_text(rid, _s.get("traceback", ""), "traceback.txt")
        client.set_tag(rid, "stage", "done" if _s["exit"] == "FINISHED" else _s["stage"] or "starting")
        client.set_tag(rid, "progress", {"FINISHED": "finished", "FAILED": f"failed in {_s['stage']}",
                                         "KILLED": f"stopped (Ctrl+C) in {_s['stage']}"}[_s["exit"]]
                       + f" after {(time.time() - _s['t_run']) / 60:.0f} min")
        client.set_tag(rid, "last_update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        client.log_metric(rid, "eta_min", 0.0)
        client.set_terminated(rid, _s["exit"])
    except Exception as err:
        _warn_off(err)

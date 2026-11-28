"""Live run dashboard (local only): python evaluation/dashboard.py

Opens two pages, both reading the same local MLflow database (mlruns.db):
  http://localhost:5001   STATUS: one table of every run -- state, step, progress,
                          time left, last update. Refreshes itself every 10 s.
                          Each run name links to its MLflow page.
  http://localhost:5000   MLflow itself: loss charts per run, parameters, final
                          metrics, side-by-side comparison of runs.
Leave this terminal open while you use them; Ctrl+C stops the pages, not your runs.

Runs whose process is gone (killed, machine slept) are marked FAILED on every
refresh, so nothing shows RUNNING forever. What each run logs: evaluation/tracking.py.
"""
import argparse
import html
import subprocess
import sys
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import os  # noqa: E402

os.environ["SCPROTO_NO_DIRTY_WARNING"] = "1"  # viewer, not a run: skip the uncommitted-code warning
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking  # noqa: E402

STATE_STYLE = {"RUNNING": "#1f6feb", "FINISHED": "#1a7f37", "FAILED": "#cf222e",
               "KILLED": "#9a6700", "SCHEDULED": "#57606a"}
MAX_ROWS = 60


def _ago(ms):
    s = max(time.time() - ms / 1000, 0)
    return f"{s / 60:.0f} min ago" if s < 3600 else (f"{s / 3600:.1f} h ago" if s < 86400 else
                                                     datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d"))


def _duration(r):
    end = r.info.end_time or time.time() * 1000
    m = (end - r.info.start_time) / 60000
    return f"{m:.0f} min" if m < 120 else f"{m / 60:.1f} h"


def status_page(mlflow_port):
    tracking.mark_dead_runs()
    _, client = tracking._client()
    rows = []
    for exp in client.search_experiments():
        for r in client.search_runs([exp.experiment_id], max_results=MAX_ROWS,
                                    order_by=["attributes.start_time DESC"]):
            rows.append((exp, r))
    # running first, then newest first
    rows.sort(key=lambda er: (er[1].info.status != "RUNNING", -er[1].info.start_time))
    body = []
    for exp, r in rows[:MAX_ROWS]:
        t, st = r.data.tags, r.info.status
        url = f"http://localhost:{mlflow_port}/#/experiments/{exp.experiment_id}/runs/{r.info.run_id}"
        eta = r.data.metrics.get("eta_min")
        detail = t.get("error", "") if st == "FAILED" else ""
        body.append(
            "<tr>"
            f"<td><span class=st style='background:{STATE_STYLE.get(st, '#57606a')}'>{st}</span></td>"
            f"<td><a href='{url}' target=_blank>{html.escape(r.info.run_name or r.info.run_id)}</a>"
            f"<div class=sub>{html.escape(exp.name)}</div></td>"
            f"<td>{html.escape(t.get('progress', ''))}"
            + (f"<div class=err>{html.escape(detail[:300])}</div>" if detail else "") + "</td>"
            f"<td class=num>{'' if eta is None or st != 'RUNNING' else f'{eta:.0f} min'}</td>"
            f"<td class=num>{_duration(r)}</td>"
            f"<td class=num>{html.escape(t.get('last_update', '')[11:16])}"
            f"<div class=sub>started {_ago(r.info.start_time)}</div></td>"
            "</tr>")
    n_running = sum(1 for _, r in rows if r.info.status == "RUNNING")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta http-equiv=refresh content=10><title>({n_running}) runs</title>
<style>
 body{{font:14px system-ui,sans-serif;margin:24px;color:#1f2328}}
 h1{{font-size:18px;margin:0 0 4px}} .top{{color:#57606a;margin-bottom:16px}}
 table{{border-collapse:collapse;width:100%}} th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #d0d7de;vertical-align:top}}
 th{{font-weight:600;color:#57606a;font-size:12px;text-transform:uppercase}}
 .st{{color:#fff;border-radius:10px;padding:2px 8px;font-size:12px;font-weight:600}}
 .sub{{color:#57606a;font-size:12px}} .err{{color:#cf222e;font-size:12px;margin-top:4px}}
 .num{{white-space:nowrap}} a{{color:#0969da;text-decoration:none}}
</style></head><body>
<h1>Runs: {n_running} running</h1>
<div class=top>Refreshes every 10 s. Click a run for its charts in MLflow. Updated {datetime.now():%H:%M:%S}.</div>
<table><tr><th>State</th><th>Run</th><th>Progress</th><th>Time left</th><th>Duration</th><th>Last update</th></tr>
{''.join(body) or '<tr><td colspan=6>No runs yet. Start any stage script (docs/runbook.md).</td></tr>'}
</table></body></html>"""


def serve_status(port, mlflow_port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                page, code = status_page(mlflow_port).encode("utf-8"), 200
            except Exception as err:
                page, code = f"<pre>status page error: {html.escape(repr(err))}</pre>".encode(), 500
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a):  # keep the terminal quiet
            pass

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5001, help="status page")
    p.add_argument("--mlflow-port", type=int, default=5000, help="MLflow page (charts)")
    args = p.parse_args()
    tracking._client()  # creates the database on first use
    mlflow_ui = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking.TRACKING_URI,
         "--default-artifact-root", tracking.ARTIFACT_ROOT.as_uri(),
         "--host", "127.0.0.1", "--port", str(args.mlflow_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[dashboard] status:  http://localhost:{args.port}\n"
          f"[dashboard] charts:  http://localhost:{args.mlflow_port}   (MLflow)\n"
          f"[dashboard] Ctrl+C stops both pages; your runs keep going.")
    try:
        serve_status(args.port, args.mlflow_port)
    except KeyboardInterrupt:
        pass
    finally:
        mlflow_ui.terminate()

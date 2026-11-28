"""Open the live run dashboard (MLflow UI, local only): python evaluation/dashboard.py

Then open http://localhost:5000 -> experiment (e.g. rare_cell_preservation).
Leave this terminal open while you use the page; Ctrl+C stops the page, not your runs.

First time only, in the runs table: "Columns" -> tick the tags `progress`, `stage`,
`last_update` and the metric `eta_min`. The browser remembers the choice.

Before starting, runs whose process is gone (killed, machine slept) are marked FAILED,
so nothing shows RUNNING forever. See evaluation/tracking.py for what each run logs.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=5000)
    args = p.parse_args()
    n = tracking.mark_dead_runs()
    if n:
        print(f"[dashboard] marked {n} run(s) FAILED: their process is gone")
    print(f"[dashboard] http://localhost:{args.port}   (Ctrl+C to stop the page; runs keep going)")
    subprocess.call([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", tracking.TRACKING_URI,
                     "--default-artifact-root", tracking.ARTIFACT_ROOT.as_uri(),
                     "--host", "127.0.0.1", "--port", str(args.port)])

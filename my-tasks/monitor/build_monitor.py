"""Local run monitor -- one self-contained HTML page showing the state of scProto runs.

WHAT IT SHOWS
-------------
  LIVE     epoch traces parsed from local Modal-run logs (unused_proto, reinit counts,
           loss, q+/q-), for runs still training. This is the only place the per-epoch
           prototype-death curve is visible.
  DONE     metrics.json + K_live (rows of purity_per_mc.csv) pulled from the Modal
           volume for finished runs.
  REFS     the reference numbers every new run is judged against, so the comparison is
           on the same screen instead of in a different file.

USAGE
    python my-tasks/monitor/build_monitor.py             # build once
    python my-tasks/monitor/build_monitor.py --watch     # rebuild every 60s

Then open  my-tasks/monitor/index.html  in a browser. The page carries a 30s meta
refresh, so with --watch running it updates itself.

NOTE ON COST: `modal volume get` on metrics.json / purity_per_mc.csv is a few KB per
run and costs nothing meaningful, but it is a network round trip each -- so the pulled
set is restricted to RUN_PREFIXES below. Add a prefix to track a new experiment.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "index.html")
VOL = "scproto-data"

# Only runs whose folder name starts with one of these is pulled from the volume.
RUN_PREFIXES = ("nao_", "coh_", "cod_", "rat_", "rt2_", "fin_", "modu_", "modupb_",
                "cpmk_", "repro_paper_")
DATASETS = ("s28nsc", "pancreas")

# Where to look for local Modal run logs (epoch traces for in-flight runs).
LOG_GLOBS = [
    os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "**", "*.log"),
    os.path.join(os.environ.get("TEMP", "/tmp"), "claude", "**", "tasks", "*.output"),
]

# name -> (w_purity, w_nichePur, modularity, coverage, note)
REFERENCES = {
    "s28nsc": [
        ("SEACells bk02 K=200",      None,   None,   None,   None, "DGE 0.849  int_frac 0.4465"),
        ("Leiden-modularity K=200",  None,   None,   None,   None, "DGE 0.856  int_frac 0.5147"),
        ("Leiden-CPM K=200",         None,   None,   None,   None, "DGE 0.630  int_frac 0.5509"),
        ("scProto best (CPM)",       None,   None,   None,   None, "DGE 0.842  int_frac 0.3338"),
        ("modularity lambda=1",    0.8473, 0.4833, 0.3893, None, "dead 42   DGE 0.672"),
        ("modularity lambda=10",   0.8686, 0.4939, 0.3087, None, "dead 0    DGE 0.822"),
        ("modularity lambda=100",  0.8508, 0.4802, 0.2750, None, "dead 0    DGE 0.769"),
    ],
    "pancreas": [
        ("published loss (repro)", None, None, 0.6924, 1.0000, "batch_ent 0.4220  K_live 220"),
        ("modularity GLOBAL l=10", None, None, 0.5070, 0.7857, "batch_ent 0.2471  K_live 220"),
        ("CPM g=0.042 l=50",       None, None, 0.4652, 0.9286, "batch_ent 1.0907"),
        ("modularity pbch l=0.2",  None, None, 0.7195, 0.7857, "VOID -- K_live 32, confounded"),
    ],
}

EPOCH_RE = re.compile(
    r">>> Epoch (\d+)/~\d+ \| loss=([\d.]+).*?unused_proto=(\d+)")
REINIT_RE = re.compile(r"\[reinit\] revived (\d+) dead prototypes")
COH_RE = re.compile(r"\[D14\] cohesion epoch-1 value = ([\d.]+)")
NASSOC_RE = re.compile(r"nassoc=([\d.]+) \[diag=([\d.]+) offdiag=([\d.]+)\]")


def sh(cmd, timeout=120):
    env = dict(os.environ, MSYS_NO_PATHCONV="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        # encoding/errors are explicit: modal's Rich output carries unicode that the
        # default Windows console codepage cannot decode (CLAUDE.md section 11).
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", errors="replace",
                           timeout=timeout, env=env, cwd=REPO)
        return r.stdout
    except Exception:
        return ""


def list_runs():
    runs = []
    for ds in DATASETS:
        out = sh(f"modal volume ls {VOL} /models/{ds}")
        for line in out.splitlines():
            name = line.strip().split("/")[-1]
            if name.startswith(RUN_PREFIXES):
                runs.append((ds, name))
    return runs


def pull(ds, name, fname):
    """Fetch one small artifact, cached on disk so repeated builds are cheap."""
    # Cache under a HASH of the run name, not the name itself: the auto-generated
    # folder names plus this repo's Drive-shortcut path blow past Windows' 260-char
    # MAX_PATH, and the \?\ prefix does not help on a Drive-mapped letter drive
    # (CLAUDE.md section 11). Hashing keeps every cache path short.
    key = hashlib.sha1(f"{ds}/{name}".encode()).hexdigest()[:16]
    local = os.path.join(CACHE, f"{key}__{fname}")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        # Only trust the cache for a FINISHED run. A run still training writes
        # metrics.json incrementally (modularity lands before the task-1 keys), so
        # caching it permanently froze every in-flight row on the page.
        if fname == "metrics.json":
            try:
                if "weighted_mean_cell_type_purity" in open(
                        local, encoding="utf-8", errors="replace").read():
                    return local
            except Exception:
                pass
        elif time.time() - os.path.getmtime(local) < 900:
            return local
        else:
            return local if os.path.getsize(local) > 0 else None
    os.makedirs(os.path.dirname(local), exist_ok=True)
    sh(f'modal volume get --force {VOL} "/models/{ds}/{name}/{fname}" "{local}"')
    return local if os.path.exists(local) else None


def short_cfg(name):
    """Pull the human-meaningful config tokens out of the auto-generated folder name."""
    keep = []
    for tok in name.split("_"):
        if re.match(r"^(NP\d+|LD\d+|aff-\w+|lna[\d.]+|naa[\d.]+|naon-\w+|lmod[\d.]+"
                    r"|mres[\d.]+|mrsp[\d.]+|magg-\w+|lcoh[\d.]+|lpu[\d.]+|prie\d+"
                    r"|nagm[\d.]+|nadl-\w+|nadn-\w+)$", tok):
            keep.append(tok)
    return " ".join(keep)


def parse_logs():
    """Epoch traces for in-flight runs, keyed by whichever arm name the log mentions."""
    traces = {}
    for g in LOG_GLOBS:
        for path in glob.glob(g, recursive=True):
            try:
                txt = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            if ">>> Epoch" not in txt:
                continue
            names = re.findall(r"=== (\w[\w\d_]+)\n", txt)
            key = names[0] if names else os.path.basename(path)
            ep = [{"epoch": int(a), "loss": float(b), "unused": int(c)}
                  for a, b, c in EPOCH_RE.findall(txt)]
            if not ep:
                continue
            t = traces.setdefault(key, {"arms": names, "epochs": [], "reinit": [],
                                        "cohesion": None, "nassoc": None,
                                        "log": os.path.basename(path)})
            t["epochs"] = ep
            t["reinit"] = [int(x) for x in REINIT_RE.findall(txt)]
            m = COH_RE.search(txt)
            if m:
                t["cohesion"] = float(m.group(1))
            m = NASSOC_RE.findall(txt)
            if m:
                t["nassoc"] = {"total": float(m[-1][0]), "diag": float(m[-1][1]),
                               "offdiag": float(m[-1][2])}
    return traces


def collect():
    data = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": [], "traces": parse_logs()}
    for ds, name in list_runs():
        mpath = pull(ds, name, "metrics.json")
        m = {}
        if mpath:
            try:
                m = json.load(open(mpath, encoding="utf-8", errors="replace"))
            except Exception:
                m = {}
        cpath = pull(ds, name, "purity_per_mc.csv")
        k_live = None
        if cpath and os.path.exists(cpath):
            try:
                k_live = sum(1 for _ in open(cpath, encoding="utf-8",
                                             errors="replace")) - 1
            except Exception:
                pass
        data["runs"].append({
            "dataset": ds, "name": name.split("_ds-")[0], "cfg": short_cfg(name),
            "w_purity": m.get("weighted_mean_cell_type_purity"),
            "w_niche": m.get("weighted_mean_niche_purity"),
            "b_entropy": m.get("weighted_mean_batch_entropy"),
            "modularity": m.get("modularity"),
            "coverage": m.get("coverage"),
            "dead": m.get("n_unused_protos"),
            "k_live": k_live,
            "complete": bool(m.get("weighted_mean_cell_type_purity") is not None),
        })
    data["runs"].sort(key=lambda r: (r["dataset"], not r["complete"], r["name"]))
    return data


PAGE = """<meta http-equiv="refresh" content="15">
<title>scProto run monitor</title>
<style>
 :root{--bg:#fff;--fg:#16181d;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb;
       --ok:#047857;--warn:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
 @media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;
   --line:#30363d;--card:#161b22;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}}
 :root[data-theme=dark]{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;
   --card:#161b22;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--accent:#58a6ff}
 :root[data-theme=light]{--bg:#fff;--fg:#16181d;--mut:#6b7280;--line:#e5e7eb;
   --card:#f9fafb;--ok:#047857;--warn:#b45309;--bad:#b91c1c;--accent:#1d4ed8}
 body{background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,system-ui,sans-serif;
      margin:0;padding:24px;max-width:1400px}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:28px 0 8px;color:var(--mut);
    text-transform:uppercase;letter-spacing:.06em;font-weight:600}
 .sub{color:var(--mut);font-size:12px;margin-bottom:20px}
 .wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
 table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
 th,td{padding:7px 11px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
 th{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
 td:first-child,th:first-child{text-align:left} td.cfg{text-align:left;color:var(--mut);
   font:11px ui-monospace,monospace} tr:last-child td{border-bottom:none}
 tr.ref td{color:var(--mut);font-style:italic;background:transparent}
 tr.live td:first-child::before{content:"● ";color:var(--warn)}
 .note{text-align:left;color:var(--mut);font-size:12px}
 .bad{color:var(--bad);font-weight:600} .ok{color:var(--ok);font-weight:600}
 .spark{display:inline-block;vertical-align:middle}
 .card{border:1px solid var(--line);border-radius:8px;background:var(--card);
       padding:14px 16px;margin-bottom:10px}
 .card h3{margin:0 0 8px;font-size:14px} .kv{color:var(--mut);font-size:12px}
 code{font:12px ui-monospace,monospace;background:var(--card);padding:1px 5px;
      border-radius:4px;border:1px solid var(--line)}
</style>
<h1>scProto run monitor</h1>
<div class="sub">generated __TS__ &middot; page self-refreshes every 15s, data rebuilt every 20s &middot;
 run <code>python my-tasks/monitor/build_monitor.py --watch</code> to keep the data fresh</div>
__BODY__
"""


def fmt(v, d=4, bad=None, good=None):
    if v is None:
        return '<td style="color:var(--mut)">--</td>'
    if isinstance(v, float):
        cls = ""
        if bad is not None and v >= bad:
            cls = ' class="bad"'
        elif good is not None and v >= good:
            cls = ' class="ok"'
        return f"<td{cls}>{v:.{d}f}</td>"
    cls = ""
    if bad is not None and v >= bad:
        cls = ' class="bad"'
    return f"<td{cls}>{v}</td>"


def sparkline(vals, w=110, h=22):
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1
    pts = " ".join(f"{i*w/(len(vals)-1):.1f},{h - (v-lo)/rng*(h-3) - 1.5:.1f}"
                   for i, v in enumerate(vals))
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{pts}" fill="none" stroke="currentColor" '
            f'stroke-width="1.5" stroke-linejoin="round"/></svg>')


def build(data):
    B = []
    tr = data["traces"]
    if tr:
        B.append("<h2>In flight &mdash; per-epoch traces from local logs</h2>")
        for key, t in sorted(tr.items()):
            eps = t["epochs"]
            unused = [e["unused"] for e in eps]
            loss = [e["loss"] for e in eps]
            last = eps[-1]
            extra = []
            if t["cohesion"] is not None:
                extra.append(f'cohesion epoch-1 <b>{t["cohesion"]:.4f}</b> '
                             f'(range [0,1]; Leiden reference 0.179)')
            if t["nassoc"]:
                n = t["nassoc"]
                extra.append(f'nassoc {n["total"]:.4f} '
                             f'[diag <b>{n["diag"]:.3f}</b> offdiag <b>{n["offdiag"]:.3f}</b>]')
            if t["reinit"]:
                extra.append(f'reinit fired {len(t["reinit"])}x, revived '
                             f'{", ".join(str(x) for x in t["reinit"])}')
            arms = ", ".join(t["arms"]) if t["arms"] else key
            B.append(
                f'<div class="card"><h3>{arms}</h3>'
                f'<div class="kv">epoch <b>{last["epoch"]}</b>/~20 &middot; '
                f'loss {last["loss"]:.4f} {sparkline(loss)} &middot; '
                f'unused_proto <b>{last["unused"]}</b>/200 {sparkline(unused)}</div>'
                + ("".join(f'<div class="kv">{e}</div>' for e in extra))
                + f'<div class="kv" style="opacity:.6">log: {t["log"]}</div></div>')

    for ds in DATASETS:
        runs = [r for r in data["runs"] if r["dataset"] == ds]
        if not runs:
            continue
        B.append(f"<h2>{ds}</h2><div class='wrap'><table><tr>"
                 "<th>run</th><th>config</th><th>purity</th><th>niche pur</th>"
                 "<th>batch ent</th><th>modularity</th><th>coverage</th>"
                 "<th>dead</th><th>K live</th><th class='note'>note</th></tr>")
        for r in runs:
            cls = "" if r["complete"] else ' class="live"'
            B.append(
                f"<tr{cls}><td>{r['name']}</td><td class='cfg'>{r['cfg']}</td>"
                + fmt(r["w_purity"]) + fmt(r["w_niche"]) + fmt(r["b_entropy"])
                + fmt(r["modularity"]) + fmt(r["coverage"], 3)
                + fmt(r["dead"], 0, bad=20) + fmt(r["k_live"], 0)
                + f"<td class='note'>{'' if r['complete'] else 'running / eval in progress'}</td></tr>")
        for name, wp, wn, mod, cov, note in REFERENCES.get(ds, []):
            B.append(f"<tr class='ref'><td>{name}</td><td class='cfg'>reference</td>"
                     + fmt(wp) + fmt(wn) + "<td>--</td>" + fmt(mod) + fmt(cov, 3)
                     + "<td>--</td><td>--</td>" + f"<td class='note'>{note}</td></tr>")
        B.append("</table></div>")
    return "\n".join(B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="rebuild every 20s")
    a = ap.parse_args()
    while True:
        d = collect()
        html = PAGE.replace("__TS__", d["generated"]).replace("__BODY__", build(d))
        open(OUT, "w", encoding="utf-8").write(html)
        json.dump(d, open(os.path.join(HERE, "data.json"), "w"), indent=1)
        print(f"[{d['generated']}] wrote {OUT}  "
              f"({len(d['runs'])} runs, {len(d['traces'])} live logs)")
        if not a.watch:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()

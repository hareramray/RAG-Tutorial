"""Run every notebook in a fresh kernel, without API calls or model downloads.

Writes executed copies and a report into ignored artifacts/, never source notebooks.
Usage: python scripts/verify_notebooks.py [00 04 ...]
"""
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["RAG_OFFLINE"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
OUTPUT = ROOT / "artifacts" / "offline_notebooks"
OUTPUT.mkdir(parents=True, exist_ok=True)
prefixes = sys.argv[1:]
paths = [p for p in sorted((ROOT / "notebooks").glob("*.ipynb"))
         if not prefixes or any(p.name.startswith(prefix) for prefix in prefixes)]
if not paths:
    raise SystemExit("No notebooks matched.")
report = []
for path in paths:
    start = time.perf_counter()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    print(f"RUN {path.name}", flush=True)
    try:
        # python3 resolves to this environment's kernel. The authoring kernel
        # name is also installed by setup.ps1 for interactive use.
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
        status, error = "passed", None
        nbformat.write(notebook, OUTPUT / path.name)
    except Exception as exc:
        status, error = "failed", str(exc)
        print(error, flush=True)
    record = {"notebook": path.name, "status": status,
              "seconds": round(time.perf_counter() - start, 2), "error": error}
    report.append(record)
    print(f"{status.upper()} {path.name} ({record['seconds']}s)", flush=True)
(OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"Offline verification: {sum(r['status'] == 'passed' for r in report)}/{len(report)} passed.")
print("This checks local execution only; it does not validate Claude answers or semantic model quality.")
raise SystemExit(1 if any(r["status"] == "failed" for r in report) else 0)

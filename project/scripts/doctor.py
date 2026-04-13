from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

print(f"python_executable={sys.executable}")
print(f"python_version={sys.version}")
print(f"platform={platform.platform()}")
print(f"cwd={Path.cwd()}")
print(f"project_root={ROOT}")

for tool in ["python", "pip", "pytest"]:
    print(f"which_{tool}={shutil.which(tool)}")

metrics_path = ROOT / "project/artifacts/metrics_summary.json"
print(f"metrics_exists={metrics_path.exists()}")
if metrics_path.exists():
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("best_run=")
    print(json.dumps(payload.get("best_run", {}), indent=2, ensure_ascii=False))

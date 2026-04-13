from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_end_to_end_smoke() -> None:
    subprocess.run([sys.executable, str(ROOT / "project/scripts/make_demo_dataset.py")], check=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "project/scripts/run_experiment.py"),
            "--config",
            str(ROOT / "project/configs/base.yaml"),
        ],
        check=True,
    )

    metrics_path = ROOT / "project/artifacts/metrics_summary.json"
    tasks_path = ROOT / "project/artifacts/label_studio_tasks.json"
    assert metrics_path.exists()
    assert tasks_path.exists()

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert len(payload["all_runs"]) == 3
    assert "best_run" in payload

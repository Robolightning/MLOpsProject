from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "project" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mlops_project.labelstudio import export_tasks
from mlops_project.train import run_experiments
from mlops_project.utils import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    summary = run_experiments(cfg)
    export_tasks(cfg)
    print(summary)

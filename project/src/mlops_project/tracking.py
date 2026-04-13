from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
import uuid

import joblib


@dataclass
class _FallbackRun:
    base_dir: Path
    run_name: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.params: dict = {}
        self.metrics: dict = {}


class TrackingClient:
    def __init__(self) -> None:
        self._mlflow = None
        self._active_run = None
        self._fallback_base_dir = Path("project/mlruns_fallback")
        try:
            import mlflow  # type: ignore
            import mlflow.sklearn  # type: ignore
            self._mlflow = mlflow
        except Exception:
            self._mlflow = None

    @property
    def backend_name(self) -> str:
        return "mlflow" if self._mlflow is not None else "fallback"

    def set_tracking_uri(self, uri: str) -> None:
        if self._mlflow is not None:
            self._mlflow.set_tracking_uri(uri)

    def set_experiment(self, name: str) -> None:
        if self._mlflow is not None:
            self._mlflow.set_experiment(name)
        else:
            (self._fallback_base_dir / name).mkdir(parents=True, exist_ok=True)
            self._fallback_base_dir = self._fallback_base_dir / name

    @contextmanager
    def start_run(self, run_name: str):
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name):
                yield
        else:
            self._active_run = _FallbackRun(self._fallback_base_dir, run_name)
            try:
                yield
            finally:
                payload = {
                    "run_name": run_name,
                    "params": self._active_run.params,
                    "metrics": self._active_run.metrics,
                }
                with (self._active_run.run_dir / "run.json").open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                self._active_run = None

    def log_param(self, key: str, value) -> None:
        if self._mlflow is not None:
            self._mlflow.log_param(key, value)
        else:
            self._active_run.params[key] = value

    def log_params(self, params: dict) -> None:
        for key, value in params.items():
            self.log_param(key, value)

    def log_metrics(self, metrics: dict) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics)
        else:
            self._active_run.metrics.update(metrics)

    def log_artifact(self, path: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifact(path)
        else:
            dst = self._active_run.run_dir / Path(path).name
            shutil.copy2(path, dst)

    def log_model(self, model, name: str, input_example=None) -> None:
        if self._mlflow is not None:
            import mlflow.sklearn  # type: ignore

            kwargs = {"name": name}
            try:
                from mlflow.models import ModelSignature  # type: ignore
                from mlflow.types import ColSpec, Schema  # type: ignore

                kwargs["signature"] = ModelSignature(
                    inputs=Schema([ColSpec(type="string")]),
                    outputs=Schema([ColSpec(type="string")]),
                )
            except Exception:
                pass

            mlflow.sklearn.log_model(model, **kwargs)
        else:
            joblib.dump(model, self._active_run.run_dir / f"{name}.joblib")

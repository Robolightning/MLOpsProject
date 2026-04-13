from __future__ import annotations

from pathlib import Path
import shutil

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from mlops_project.data import build_modeling_frame, read_dataset, split_frame
from mlops_project.tracking import TrackingClient
from mlops_project.utils import ensure_dir, set_global_seed, write_json


def build_pipeline(max_features: int, ngram_max: int, c: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> dict:
    train_pred = model.predict(train_df["text"])
    test_pred = model.predict(test_df["text"])
    return {
        "train_accuracy": round(float(accuracy_score(train_df[target_column], train_pred)), 6),
        "test_accuracy": round(float(accuracy_score(test_df[target_column], test_pred)), 6),
        "train_f1_macro": round(float(f1_score(train_df[target_column], train_pred, average="macro")), 6),
        "test_f1_macro": round(float(f1_score(test_df[target_column], test_pred, average="macro")), 6),
    }


def run_experiments(config: dict) -> dict:
    random_state = int(config["random_state"])
    set_global_seed(random_state)

    artifacts_dir = ensure_dir(config["artifacts_dir"])
    tracker = TrackingClient()
    tracker.set_tracking_uri(config["tracking_uri"])
    tracker.set_experiment(config["experiment_name"])

    dataset = read_dataset(config["input"]["dataset_path"])
    frame = build_modeling_frame(
        dataset,
        text_columns=config["input"]["text_columns"],
        target_column=config["input"]["target_column"],
    )

    train_df, test_df = split_frame(
        frame,
        target_column=config["input"]["target_column"],
        test_size=float(config["input"]["test_size"]),
        random_state=random_state,
    )

    label_counts = train_df[config["input"]["target_column"]].value_counts().sort_index().to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}

    best_summary: dict | None = None
    best_model = None
    experiment_summaries = []

    for run_cfg in config["experiment_grid"]:
        with tracker.start_run(run_name=run_cfg["run_name"]):
            tracker.log_params(run_cfg)
            tracker.log_param("dataset_path", config["input"]["dataset_path"])
            tracker.log_param("text_columns", ",".join(config["input"]["text_columns"]))
            tracker.log_param("target_column", config["input"]["target_column"])
            tracker.log_param("random_state", random_state)
            tracker.log_param("tracking_backend", tracker.backend_name)

            model = build_pipeline(
                max_features=int(run_cfg["max_features"]),
                ngram_max=int(run_cfg["ngram_max"]),
                c=float(run_cfg["c"]),
                random_state=random_state,
            )
            model.fit(train_df["text"], train_df[config["input"]["target_column"]])
            metrics = evaluate_model(model, train_df, test_df, config["input"]["target_column"])
            tracker.log_metrics(metrics)

            metrics_path = artifacts_dir / f"{run_cfg['run_name']}_classification_metrics.json"
            label_counts_path = artifacts_dir / f"{run_cfg['run_name']}_dataset_label_counts.json"
            write_json(metrics, metrics_path)
            write_json(label_counts, label_counts_path)
            tracker.log_artifact(str(metrics_path))
            tracker.log_artifact(str(label_counts_path))
            input_example = train_df["text"].head(3).tolist()
            tracker.log_model(model, name="model", input_example=input_example)

            summary = {
                "run_name": run_cfg["run_name"],
                **run_cfg,
                **metrics,
            }
            experiment_summaries.append(summary)
            if best_summary is None or metrics["test_f1_macro"] > best_summary["test_f1_macro"]:
                best_summary = summary
                best_model = model

    if best_summary is None or best_model is None:
        raise RuntimeError("No experiments were executed")

    best_model_dir = artifacts_dir / "best_model"
    if best_model_dir.exists():
        shutil.rmtree(best_model_dir)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_dir / "model.joblib")

    summary_payload = {
        "best_run": best_summary,
        "all_runs": experiment_summaries,
    }
    write_json(summary_payload, artifacts_dir / "metrics_summary.json")
    return summary_payload

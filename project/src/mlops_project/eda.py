from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from mlops_project.data import build_modeling_frame, read_dataset
from mlops_project.utils import ensure_dir, write_json


def run_eda(config: dict) -> dict:
    dataset = read_dataset(config["input"]["dataset_path"])
    frame = build_modeling_frame(
        dataset,
        text_columns=config["input"]["text_columns"],
        target_column=config["input"]["target_column"],
    )

    figures_dir = ensure_dir(Path("project/reports/figures"))

    label_counts = frame[config["input"]["target_column"]].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    label_counts.plot(kind="bar")
    plt.title("Category distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "category_distribution.png")
    plt.close()

    text_lengths = frame["text"].str.len()
    plt.figure(figsize=(8, 4))
    text_lengths.hist(bins=20)
    plt.title("Text length distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "text_length_distribution.png")
    plt.close()

    missing_share = dataset.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    missing_share.plot(kind="bar")
    plt.title("Missing share by column")
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_share.png")
    plt.close()

    report = {
        "rows_total": int(len(dataset)),
        "rows_after_text_filter": int(len(frame)),
        "n_classes": int(frame[config["input"]["target_column"]].nunique()),
        "max_text_length": int(text_lengths.max()),
        "mean_text_length": float(round(text_lengths.mean(), 2)),
    }
    write_json(report, Path("project/reports/eda_summary.json"))
    return report

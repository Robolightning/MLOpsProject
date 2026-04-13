from __future__ import annotations

from pathlib import Path

from mlops_project.data import build_modeling_frame, read_dataset
from mlops_project.utils import write_json


def export_tasks(config: dict) -> list[dict]:
    dataset = read_dataset(config["input"]["dataset_path"])
    frame = build_modeling_frame(
        dataset,
        text_columns=config["input"]["text_columns"],
        target_column=config["input"]["target_column"],
    )

    sample = frame.head(int(config["label_studio"]["max_tasks"])).copy()
    tasks = []
    for idx, row in sample.iterrows():
        tasks.append(
            {
                "id": int(idx),
                "data": {
                    "title": row.get("title", ""),
                    "vendor": row.get("vendor", ""),
                    "description": row.get("description", ""),
                    "text": row["text"],
                    "category_ind": int(row[config["input"]["target_column"]]),
                },
            }
        )
    write_json(tasks, Path(config["label_studio"]["export_path"]))
    return tasks

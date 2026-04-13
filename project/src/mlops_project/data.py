from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".snappy"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def make_text_feature(df: pd.DataFrame, text_columns: Iterable[str]) -> pd.Series:
    cols = [col for col in text_columns if col in df.columns]
    if not cols:
        raise ValueError("No text columns were found in dataset")
    prepared = df[cols].fillna("").astype(str)
    return prepared.apply(lambda row: " ".join(v.strip() for v in row if v and v.strip()), axis=1)


def build_modeling_frame(df: pd.DataFrame, text_columns: list[str], target_column: str) -> pd.DataFrame:
    required = [target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = df.copy()
    frame["text"] = make_text_feature(frame, text_columns)
    frame = frame.loc[frame["text"].str.len() > 0].copy()
    frame[target_column] = frame[target_column].astype(int)
    return frame


def split_frame(frame: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    return train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target_column],
    )

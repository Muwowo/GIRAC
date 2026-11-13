"""Process data loading, chronological splits, and source-based scaling."""

import csv
from dataclasses import dataclass

import numpy as np


def read_series(path, dimensions):
    names = [
        f"{prefix}{index}"
        for prefix, count in zip(("u", "o", "y"), dimensions)
        for index in range(1, count + 1)
    ]
    rows = []
    previous_time = -float("inf")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing = set(names).difference(headers)
        if missing or len(headers) != len(set(headers)):
            detail = ", ".join(sorted(missing)) or "duplicate column names"
            raise ValueError(f"{path}: invalid CSV header ({detail}).")
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[name]) for name in names]
                if "time" in headers:
                    time = float(row["time"])
                    if not np.isfinite(time) or time <= previous_time:
                        raise ValueError("time must be finite and strictly increasing")
                    previous_time = time
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"{path}:{line}: nonfinite process value.")
            rows.append(values)
    if not rows:
        raise ValueError(f"{path}: no process samples.")
    return np.asarray(rows, dtype=np.float64)


def split_series(series, window, protocol):
    if protocol == "reported":
        if len(series) <= window + 1:
            raise ValueError(f"At least {window + 2} rows are required.")
        return series, None, series
    train_end = int(0.60 * len(series))
    validation_end = int(0.75 * len(series))
    blocks = series[:train_end], series[train_end:validation_end], series[validation_end:]
    if min(map(len, blocks)) <= window:
        raise ValueError("Each 60/15/25 time block must contain more rows than the window.")
    return blocks


def series_parts(series, dimensions):
    inputs, observations, _ = dimensions
    return series[..., :inputs], series[..., inputs:inputs + observations], series[..., inputs + observations:]


@dataclass(frozen=True)
class Scale:
    offset: np.ndarray
    span: np.ndarray

    @classmethod
    def from_source(cls, train, mode):
        if mode == "identity":
            return cls(np.zeros(train.shape[1]), np.ones(train.shape[1]))
        minimum = train.min(axis=0)
        return cls(minimum, np.maximum(train.max(axis=0) - minimum, 1e-8))

    def apply(self, series):
        return (series - self.offset) / self.span

    def to_dict(self):
        return {"offset": self.offset.tolist(), "span": self.span.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(np.asarray(data["offset"]), np.asarray(data["span"]))

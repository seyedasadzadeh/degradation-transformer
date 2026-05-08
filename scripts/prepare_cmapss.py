#!/usr/bin/env python3
"""
Prepare C-MAPSS turbofan run-to-failure data for the degradation corpus.

Expected output schema:
    engine_id,cycle,health_index

For training files, each engine trajectory runs to failure. This script derives
a simple health index from remaining useful life:

    health_index = (max_cycle - cycle) / (max_cycle - 1)

So health starts near 1 and reaches 0 at failure. The source adapter later flips
this decreasing health signal into increasing degradation.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path("data/processed/cmapss_fd001_health.csv")


def load_cmapss_train(path):
    data = np.genfromtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least unit and cycle columns.")
    return data


def extract_health_rows(path, dataset_id="FD001"):
    data = load_cmapss_train(path)
    engine_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)

    rows = []
    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        engine_cycles = cycles[mask]
        max_cycle = int(engine_cycles.max())
        denom = max(max_cycle - 1, 1)
        for cycle in sorted(engine_cycles):
            health = (max_cycle - int(cycle)) / denom
            rows.append(
                {
                    "engine_id": int(engine_id),
                    "cycle": int(cycle),
                    "health_index": float(np.clip(health, 0.0, 1.0)),
                    "dataset_id": dataset_id,
                    "max_cycle": max_cycle,
                }
            )
    return rows


def write_health_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["engine_id"], row["cycle"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["engine_id", "cycle", "health_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "engine_id": row["engine_id"],
                    "cycle": row["cycle"],
                    "health_index": row["health_index"],
                }
            )
    return output_path


def prepare_cmapss_health(
    train_path,
    output_path=DEFAULT_OUTPUT,
    dataset_id="FD001",
):
    rows = extract_health_rows(train_path, dataset_id=dataset_id)
    if not rows:
        raise ValueError("No C-MAPSS health rows were extracted.")
    return write_health_csv(rows, output_path), rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True, help="Path to train_FD001.txt or another C-MAPSS train file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--dataset-id", default="FD001", help="C-MAPSS subset id, e.g. FD001.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path, rows = prepare_cmapss_health(
        train_path=args.train_path,
        output_path=args.output,
        dataset_id=args.dataset_id,
    )
    engine_count = len({row["engine_id"] for row in rows})
    print(f"Wrote {len(rows)} health rows from {engine_count} engines to {output_path}")


if __name__ == "__main__":
    main()

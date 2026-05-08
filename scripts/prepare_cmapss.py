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

The same raw file can also be converted into a sensor-composite degradation
index. This is not a label supplied by C-MAPSS; it is a richer proxy built from
normalized sensor drift within each engine run.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path("data/processed/cmapss_fd001_health.csv")
DEFAULT_SENSOR_OUTPUT = Path("data/processed/cmapss_fd001_sensor_degradation.csv")
DEFAULT_SENSOR_NUMBERS = (2, 3, 4, 7, 11, 12, 15, 17, 20, 21)


def load_cmapss_train(path):
    data = np.genfromtxt(path, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least unit and cycle columns.")
    return data


def _sensor_column(sensor_number):
    if sensor_number < 1:
        raise ValueError("sensor numbers are 1-based and must be positive.")
    return 4 + int(sensor_number)


def _moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    window = int(window)
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


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


def extract_sensor_degradation_rows(
    path,
    dataset_id="FD001",
    sensor_numbers=DEFAULT_SENSOR_NUMBERS,
    smooth_window=5,
):
    data = load_cmapss_train(path)
    max_column = data.shape[1] - 1
    sensor_columns = [_sensor_column(sensor_number) for sensor_number in sensor_numbers]
    missing = [
        sensor_number
        for sensor_number, column in zip(sensor_numbers, sensor_columns)
        if column > max_column
    ]
    if missing:
        raise ValueError(f"C-MAPSS file does not contain sensor columns: {missing}")

    engine_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)

    rows = []
    for engine_id in np.unique(engine_ids):
        mask = engine_ids == engine_id
        order = np.argsort(cycles[mask])
        engine_cycles = cycles[mask][order]
        sensor_values = data[mask][:, sensor_columns][order]

        normalized_parts = []
        for sensor_idx in range(sensor_values.shape[1]):
            signal = sensor_values[:, sensor_idx]
            span = np.nanmax(signal) - np.nanmin(signal)
            if not np.isfinite(span) or span <= 1e-12:
                continue

            if signal[-1] >= signal[0]:
                degradation = (signal - np.nanmin(signal)) / span
            else:
                degradation = (np.nanmax(signal) - signal) / span
            normalized_parts.append(degradation)

        if not normalized_parts:
            continue

        degradation_index = np.nanmean(np.stack(normalized_parts, axis=0), axis=0)
        degradation_index = _moving_average(degradation_index, smooth_window)
        degradation_index = np.maximum.accumulate(degradation_index)
        degradation_index = np.clip(degradation_index, 0.0, 1.0)

        for cycle, degradation in zip(engine_cycles, degradation_index):
            rows.append(
                {
                    "engine_id": int(engine_id),
                    "cycle": int(cycle),
                    "sensor_degradation": float(degradation),
                    "dataset_id": dataset_id,
                    "sensor_numbers": tuple(int(sensor) for sensor in sensor_numbers),
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


def write_sensor_degradation_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["engine_id"], row["cycle"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["engine_id", "cycle", "sensor_degradation"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "engine_id": row["engine_id"],
                    "cycle": row["cycle"],
                    "sensor_degradation": row["sensor_degradation"],
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


def prepare_cmapss_sensor_degradation(
    train_path,
    output_path=DEFAULT_SENSOR_OUTPUT,
    dataset_id="FD001",
    sensor_numbers=DEFAULT_SENSOR_NUMBERS,
    smooth_window=5,
):
    rows = extract_sensor_degradation_rows(
        train_path,
        dataset_id=dataset_id,
        sensor_numbers=sensor_numbers,
        smooth_window=smooth_window,
    )
    if not rows:
        raise ValueError("No C-MAPSS sensor degradation rows were extracted.")
    return write_sensor_degradation_csv(rows, output_path), rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, required=True, help="Path to train_FD001.txt or another C-MAPSS train file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--dataset-id", default="FD001", help="C-MAPSS subset id, e.g. FD001.")
    parser.add_argument("--signal", choices=["health", "sensor_degradation"], default="health")
    parser.add_argument("--sensor", type=int, action="append", dest="sensors", help="1-based sensor number for sensor_degradation mode. Can be repeated.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Moving-average window for sensor_degradation mode.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.signal == "health":
        output_path, rows = prepare_cmapss_health(
            train_path=args.train_path,
            output_path=args.output,
            dataset_id=args.dataset_id,
        )
        signal_name = "health"
    else:
        output_path, rows = prepare_cmapss_sensor_degradation(
            train_path=args.train_path,
            output_path=args.output,
            dataset_id=args.dataset_id,
            sensor_numbers=tuple(args.sensors or DEFAULT_SENSOR_NUMBERS),
            smooth_window=args.smooth_window,
        )
        signal_name = "sensor degradation"
    engine_count = len({row["engine_id"] for row in rows})
    print(f"Wrote {len(rows)} {signal_name} rows from {engine_count} engines to {output_path}")


if __name__ == "__main__":
    main()

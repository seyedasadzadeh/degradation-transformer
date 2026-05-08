#!/usr/bin/env python3
"""
Prepare NASA battery capacity data for the degradation corpus.

Expected output schema:
    cell_id,cycle,capacity

Supported inputs:
    - Processed or semi-processed CSV files with capacity rows.
    - NASA PCoE MATLAB .mat battery files, when scipy is installed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_OUTPUT = Path("data/processed/nasa_battery_capacity.csv")


def _scalar(value):
    value = np.asarray(value)
    if value.size == 0:
        return None
    return value.reshape(-1)[0].item()


def _field(obj, name):
    if obj is None:
        return None
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    if getattr(obj, "dtype", None) is not None and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return None


def _cell_id_from_path(path):
    return Path(path).stem


def extract_rows_from_csv(
    path,
    cell_id_column="cell_id",
    cycle_column="cycle",
    capacity_column="capacity",
):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header.")
        fieldnames = set(reader.fieldnames)
        for required in [cycle_column, capacity_column]:
            if required not in fieldnames:
                raise ValueError(f"Column {required!r} was not found in {path}.")

        for row in reader:
            cell_id = row[cell_id_column] if cell_id_column in fieldnames else _cell_id_from_path(path)
            rows.append(
                {
                    "cell_id": str(cell_id),
                    "cycle": int(float(row[cycle_column])),
                    "capacity": float(row[capacity_column]),
                    "source_file": Path(path).name,
                }
            )
    return rows


def extract_rows_from_mat(path):
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "Reading NASA .mat files requires scipy. Install it with `pip install scipy`, "
            "or first convert the raw files to CSV."
        ) from exc

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    public_keys = [key for key in mat if not key.startswith("__")]
    if not public_keys:
        raise ValueError(f"No public MATLAB variables found in {path}.")

    stem = _cell_id_from_path(path)
    key = stem if stem in mat else public_keys[0]
    battery = mat[key]
    cycles = np.atleast_1d(_field(battery, "cycle"))

    rows = []
    discharge_cycle = 0
    for raw_cycle_idx, cycle in enumerate(cycles):
        cycle_type = _scalar(_field(cycle, "type"))
        if str(cycle_type).lower() != "discharge":
            continue

        data = _field(cycle, "data")
        capacity = _scalar(_field(data, "Capacity"))
        if capacity is None or not np.isfinite(float(capacity)):
            continue

        rows.append(
            {
                "cell_id": stem,
                "cycle": discharge_cycle,
                "capacity": float(capacity),
                "source_file": Path(path).name,
                "raw_cycle_index": int(raw_cycle_idx),
            }
        )
        discharge_cycle += 1
    return rows


def discover_input_files(raw_dir, inputs=None):
    files = []
    if inputs:
        files.extend(Path(path) for path in inputs)
    if raw_dir is not None:
        raw_dir = Path(raw_dir)
        files.extend(sorted(raw_dir.glob("*.mat")))
        files.extend(sorted(raw_dir.glob("*.csv")))

    unique = []
    seen = set()
    for path in files:
        path = Path(path)
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def extract_rows(
    files,
    cell_id_column="cell_id",
    cycle_column="cycle",
    capacity_column="capacity",
):
    rows = []
    for path in files:
        suffix = Path(path).suffix.lower()
        if suffix == ".mat":
            rows.extend(extract_rows_from_mat(path))
        elif suffix == ".csv":
            rows.extend(
                extract_rows_from_csv(
                    path,
                    cell_id_column=cell_id_column,
                    cycle_column=cycle_column,
                    capacity_column=capacity_column,
                )
            )
        else:
            raise ValueError(f"Unsupported file type for {path}. Expected .mat or .csv.")
    return rows


def write_capacity_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: (row["cell_id"], row["cycle"]))

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "cycle", "capacity"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "cell_id": row["cell_id"],
                    "cycle": row["cycle"],
                    "capacity": row["capacity"],
                }
            )
    return output_path


def prepare_nasa_battery_capacity(
    raw_dir=None,
    inputs=None,
    output_path=DEFAULT_OUTPUT,
    cell_id_column="cell_id",
    cycle_column="cycle",
    capacity_column="capacity",
):
    files = discover_input_files(raw_dir=raw_dir, inputs=inputs)
    if not files:
        raise ValueError("No input files found. Provide --raw-dir and/or one or more --input files.")

    rows = extract_rows(
        files,
        cell_id_column=cell_id_column,
        cycle_column=cycle_column,
        capacity_column=capacity_column,
    )
    if not rows:
        raise ValueError("No discharge capacity rows were extracted.")
    return write_capacity_csv(rows, output_path), rows


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory containing raw NASA .mat files or CSV files.")
    parser.add_argument("--input", type=Path, nargs="*", default=None, help="Specific .mat or .csv files to process.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path.")
    parser.add_argument("--cell-id-column", default="cell_id", help="CSV cell id column. Defaults to filename if missing.")
    parser.add_argument("--cycle-column", default="cycle", help="CSV cycle column.")
    parser.add_argument("--capacity-column", default="capacity", help="CSV capacity column.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path, rows = prepare_nasa_battery_capacity(
        raw_dir=args.raw_dir,
        inputs=args.input,
        output_path=args.output,
        cell_id_column=args.cell_id_column,
        cycle_column=args.cycle_column,
        capacity_column=args.capacity_column,
    )
    cell_count = len({row["cell_id"] for row in rows})
    print(f"Wrote {len(rows)} capacity rows from {cell_count} cells to {output_path}")


if __name__ == "__main__":
    main()

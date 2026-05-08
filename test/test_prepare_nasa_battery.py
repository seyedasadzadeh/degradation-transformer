import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_nasa_battery.py"
spec = importlib.util.spec_from_file_location("prepare_nasa_battery", SCRIPT_PATH)
prepare_nasa_battery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_nasa_battery)


def test_prepare_nasa_battery_capacity_from_processed_csv(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "cells.csv").write_text(
        "cell_id,cycle,capacity\n"
        "B0006,1,1.82\n"
        "B0005,1,1.84\n"
        "B0005,0,1.86\n"
        "B0006,0,1.88\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed" / "nasa_battery_capacity.csv"

    output_path, rows = prepare_nasa_battery.prepare_nasa_battery_capacity(
        raw_dir=raw_dir,
        output_path=output,
    )

    assert output_path == output
    assert len(rows) == 4
    assert output.read_text(encoding="utf-8") == (
        "cell_id,cycle,capacity\n"
        "B0005,0,1.86\n"
        "B0005,1,1.84\n"
        "B0006,0,1.88\n"
        "B0006,1,1.82\n"
    )


def test_prepare_nasa_battery_capacity_with_custom_csv_schema(tmp_path):
    path = tmp_path / "B0005.csv"
    path.write_text(
        "cycle_index,cap_ah\n"
        "0,1.86\n"
        "1,1.84\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"

    _, rows = prepare_nasa_battery.prepare_nasa_battery_capacity(
        inputs=[path],
        output_path=output,
        cycle_column="cycle_index",
        capacity_column="cap_ah",
    )

    assert [row["cell_id"] for row in rows] == ["B0005", "B0005"]
    assert output.read_text(encoding="utf-8") == (
        "cell_id,cycle,capacity\n"
        "B0005,0,1.86\n"
        "B0005,1,1.84\n"
    )


def test_prepare_nasa_battery_capacity_rejects_empty_input(tmp_path):
    try:
        prepare_nasa_battery.prepare_nasa_battery_capacity(raw_dir=tmp_path)
    except ValueError as e:
        assert "No input files found" in str(e)
    else:
        raise AssertionError("Expected a ValueError for an empty raw directory.")

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_cmapss.py"
spec = importlib.util.spec_from_file_location("prepare_cmapss", SCRIPT_PATH)
prepare_cmapss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_cmapss)


def test_prepare_cmapss_health_from_train_file(tmp_path):
    train_path = tmp_path / "train_FD001.txt"
    train_path.write_text(
        "1 1 0 0 100 1\n"
        "1 2 0 0 100 1\n"
        "1 3 0 0 100 1\n"
        "2 1 0 0 100 1\n"
        "2 2 0 0 100 1\n",
        encoding="utf-8",
    )
    output = tmp_path / "cmapss_fd001_health.csv"

    output_path, rows = prepare_cmapss.prepare_cmapss_health(
        train_path=train_path,
        output_path=output,
        dataset_id="FD001",
    )

    assert output_path == output
    assert len(rows) == 5
    assert output.read_text(encoding="utf-8") == (
        "engine_id,cycle,health_index\n"
        "1,1,1.0\n"
        "1,2,0.5\n"
        "1,3,0.0\n"
        "2,1,1.0\n"
        "2,2,0.0\n"
    )


def test_prepare_cmapss_health_rejects_invalid_file(tmp_path):
    train_path = tmp_path / "bad.txt"
    train_path.write_text("1\n2\n3\n", encoding="utf-8")

    try:
        prepare_cmapss.prepare_cmapss_health(train_path=train_path)
    except ValueError as e:
        assert "at least unit and cycle columns" in str(e)
    else:
        raise AssertionError("Expected a ValueError for invalid C-MAPSS data.")


def test_prepare_cmapss_sensor_degradation_from_train_file(tmp_path):
    train_path = tmp_path / "train_FD001.txt"
    rows = []
    for engine_id in [1, 2]:
        for cycle in [1, 2, 3, 4]:
            sensor_2 = 100 + cycle * engine_id
            sensor_3 = 90 - cycle * engine_id
            values = [engine_id, cycle, 0.0, 0.0, 0.0]
            sensors = [sensor_2, sensor_3] + [1.0] * 19
            rows.append(" ".join(str(value) for value in values + sensors))
    train_path.write_text("\n".join(rows), encoding="utf-8")

    output = tmp_path / "cmapss_fd001_sensor_degradation.csv"

    output_path, extracted = prepare_cmapss.prepare_cmapss_sensor_degradation(
        train_path=train_path,
        output_path=output,
        sensor_numbers=(2, 3),
        smooth_window=1,
    )

    assert output_path == output
    assert len(extracted) == 8
    assert output.read_text(encoding="utf-8").splitlines()[0] == "engine_id,cycle,sensor_degradation"
    first_engine = [row for row in extracted if row["engine_id"] == 1]
    assert first_engine[0]["sensor_degradation"] < first_engine[-1]["sensor_degradation"]

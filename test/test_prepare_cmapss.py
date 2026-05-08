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

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_training_corpus.py"
spec = importlib.util.spec_from_file_location("build_training_corpus", SCRIPT_PATH)
build_training_corpus_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_training_corpus_module)


def test_build_training_corpus_synthetic_only(tmp_path):
    output = tmp_path / "episodes.npy"
    metadata_output = tmp_path / "metadata.json"
    diagnostics_output = tmp_path / "diagnostics.json"

    episodes, metadata, diagnostics = build_training_corpus_module.build_training_corpus(
        n_synthetic=12,
        episode_length=40,
        seed=5,
        output_path=output,
        metadata_path=metadata_output,
        diagnostics_path=diagnostics_output,
        context_window=10,
        future_window=5,
    )

    assert episodes.shape == (12, 40)
    assert len(metadata) == 12
    assert diagnostics["included_sources"] == ["synthetic"]
    assert diagnostics["window_budget"]["total_windows"] == 312
    assert np.load(output).shape == (12, 40)
    assert len(json.loads(metadata_output.read_text(encoding="utf-8"))) == 12
    assert json.loads(diagnostics_output.read_text(encoding="utf-8"))["overall_shape"]["n_episodes"] == 12


def test_build_training_corpus_includes_nasa_battery_source(tmp_path):
    nasa_path = tmp_path / "nasa.csv"
    nasa_path.write_text(
        "cell_id,cycle,capacity\n"
        "B0005,0,1.86\n"
        "B0005,1,1.82\n"
        "B0005,2,1.74\n",
        encoding="utf-8",
    )
    output = tmp_path / "episodes.npy"
    metadata_output = tmp_path / "metadata.json"
    diagnostics_output = tmp_path / "diagnostics.json"

    episodes, metadata, diagnostics = build_training_corpus_module.build_training_corpus(
        n_synthetic=8,
        episode_length=30,
        seed=7,
        include_nasa_battery=True,
        nasa_path=nasa_path,
        output_path=output,
        metadata_path=metadata_output,
        diagnostics_path=diagnostics_output,
    )

    assert episodes.shape == (9, 30)
    assert len(metadata) == 9
    assert diagnostics["included_sources"] == ["synthetic", "nasa_battery_capacity"]
    assert any(item["source_type"] == "real" for item in metadata)
    assert any(item["mechanism"] == "nasa_battery_capacity" for item in metadata)


def test_build_training_corpus_includes_cmapss_fd001_source(tmp_path):
    cmapss_path = tmp_path / "cmapss.csv"
    cmapss_path.write_text(
        "engine_id,cycle,health_index\n"
        "1,1,1.0\n"
        "1,2,0.5\n"
        "1,3,0.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "episodes.npy"
    metadata_output = tmp_path / "metadata.json"
    diagnostics_output = tmp_path / "diagnostics.json"

    episodes, metadata, diagnostics = build_training_corpus_module.build_training_corpus(
        n_synthetic=8,
        episode_length=30,
        seed=7,
        include_cmapss_fd001=True,
        cmapss_path=cmapss_path,
        output_path=output,
        metadata_path=metadata_output,
        diagnostics_path=diagnostics_output,
    )

    assert episodes.shape == (9, 30)
    assert len(metadata) == 9
    assert diagnostics["included_sources"] == ["synthetic", "cmapss_fd001_health"]
    assert any(item["source_type"] == "real_simulated" for item in metadata)
    assert any(item["mechanism"] == "cmapss_fd001_health" for item in metadata)


def test_parse_source_weights():
    assert build_training_corpus_module._parse_source_weights(["battery=0.5", "wear=0.2"]) == {
        "battery": 0.5,
        "wear": 0.2,
    }

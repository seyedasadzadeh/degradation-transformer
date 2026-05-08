#!/usr/bin/env python3
"""
Build the training degradation corpus from synthetic and optional real sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generation import CorpusConfig
from src.generation import corpus_diagnostics_report
from src.generation import generate_degradation_corpus_from_config
from src.ingestion import combine_degradation_corpora
from src.real_datasets import CMAPSSTurbofanHealthSource
from src.real_datasets import CMAPSSTurbofanSensorSource
from src.real_datasets import NASABatteryCapacitySource


DEFAULT_NASA_PATH = Path("data/processed/nasa_battery_capacity.csv")
DEFAULT_CMAPSS_PATH = Path("data/processed/cmapss_fd001_health.csv")
DEFAULT_CMAPSS_SENSOR_PATH = Path("data/processed/cmapss_fd001_sensor_degradation.csv")


def _parse_source_weights(items):
    if not items:
        return None
    weights = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid source weight {item!r}. Expected key=value.")
        key, value = item.split("=", 1)
        weights[key] = float(value)
    return weights


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, sort_keys=True)


def build_training_corpus(
    n_synthetic=5000,
    episode_length=100,
    seed=42,
    max_value=15,
    source_weights=None,
    include_nasa_battery=False,
    nasa_path=DEFAULT_NASA_PATH,
    include_cmapss_fd001=False,
    cmapss_path=DEFAULT_CMAPSS_PATH,
    include_cmapss_fd001_sensors=False,
    cmapss_sensor_path=DEFAULT_CMAPSS_SENSOR_PATH,
    output_path=Path("degradation_episodes.npy"),
    metadata_path=Path("artifacts/corpus_metadata.json"),
    diagnostics_path=Path("artifacts/corpus_diagnostics.json"),
    context_window=None,
    future_window=1,
    stride=1,
):
    synthetic_config = CorpusConfig(
        episode_length=episode_length,
        n_episodes=n_synthetic,
        seed=seed,
        max_value=max_value,
        source_weights=source_weights,
        return_metadata=True,
    )
    synthetic = generate_degradation_corpus_from_config(synthetic_config)
    corpora = [synthetic]
    included_sources = ["synthetic"]

    nasa_path = Path(nasa_path)
    if include_nasa_battery:
        if not nasa_path.exists():
            raise FileNotFoundError(
                f"NASA battery processed CSV was not found at {nasa_path}. "
                "Run scripts/prepare_nasa_battery.py first."
            )
        nasa_source = NASABatteryCapacitySource(
            path=nasa_path,
            episode_length=episode_length,
            max_value=max_value,
        )
        corpora.append(nasa_source.load())
        included_sources.append("nasa_battery_capacity")

    cmapss_path = Path(cmapss_path)
    if include_cmapss_fd001:
        if not cmapss_path.exists():
            raise FileNotFoundError(
                f"C-MAPSS FD001 processed CSV was not found at {cmapss_path}. "
                "Run scripts/prepare_cmapss.py first."
            )
        cmapss_source = CMAPSSTurbofanHealthSource(
            path=cmapss_path,
            episode_length=episode_length,
            max_value=max_value,
        )
        corpora.append(cmapss_source.load())
        included_sources.append("cmapss_fd001_health")

    cmapss_sensor_path = Path(cmapss_sensor_path)
    if include_cmapss_fd001_sensors:
        if not cmapss_sensor_path.exists():
            raise FileNotFoundError(
                f"C-MAPSS FD001 sensor degradation CSV was not found at {cmapss_sensor_path}. "
                "Run scripts/prepare_cmapss.py --signal sensor_degradation first."
            )
        cmapss_sensor_source = CMAPSSTurbofanSensorSource(
            path=cmapss_sensor_path,
            episode_length=episode_length,
            max_value=max_value,
        )
        corpora.append(cmapss_sensor_source.load())
        included_sources.append("cmapss_fd001_sensor_degradation")

    episodes, metadata = combine_degradation_corpora(*corpora)
    diagnostics = corpus_diagnostics_report(
        episodes,
        metadata,
        context_window=context_window,
        future_window=future_window,
        stride=stride,
    )
    diagnostics["included_sources"] = included_sources
    diagnostics["build_config"] = {
        "n_synthetic": int(n_synthetic),
        "episode_length": int(episode_length),
        "seed": None if seed is None else int(seed),
        "max_value": float(max_value),
        "source_weights": source_weights,
        "include_nasa_battery": bool(include_nasa_battery),
        "nasa_path": str(nasa_path),
        "include_cmapss_fd001": bool(include_cmapss_fd001),
        "cmapss_path": str(cmapss_path),
        "include_cmapss_fd001_sensors": bool(include_cmapss_fd001_sensors),
        "cmapss_sensor_path": str(cmapss_sensor_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "diagnostics_path": str(diagnostics_path),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, episodes.astype(np.float32))
    write_json(metadata_path, metadata)
    write_json(diagnostics_path, diagnostics)
    return episodes, metadata, diagnostics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-synthetic", type=int, default=5000)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-value", type=float, default=15)
    parser.add_argument("--source-weight", action="append", default=None, help="Synthetic source weight as key=value.")
    parser.add_argument("--include-nasa-battery", action="store_true")
    parser.add_argument("--nasa-path", type=Path, default=DEFAULT_NASA_PATH)
    parser.add_argument("--include-cmapss-fd001", action="store_true")
    parser.add_argument("--cmapss-path", type=Path, default=DEFAULT_CMAPSS_PATH)
    parser.add_argument("--include-cmapss-fd001-sensors", action="store_true")
    parser.add_argument("--cmapss-sensor-path", type=Path, default=DEFAULT_CMAPSS_SENSOR_PATH)
    parser.add_argument("--output", type=Path, default=Path("degradation_episodes.npy"))
    parser.add_argument("--metadata-output", type=Path, default=Path("artifacts/corpus_metadata.json"))
    parser.add_argument("--diagnostics-output", type=Path, default=Path("artifacts/corpus_diagnostics.json"))
    parser.add_argument("--context-window", type=int, default=None)
    parser.add_argument("--future-window", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    episodes, metadata, diagnostics = build_training_corpus(
        n_synthetic=args.n_synthetic,
        episode_length=args.episode_length,
        seed=args.seed,
        max_value=args.max_value,
        source_weights=_parse_source_weights(args.source_weight),
        include_nasa_battery=args.include_nasa_battery,
        nasa_path=args.nasa_path,
        include_cmapss_fd001=args.include_cmapss_fd001,
        cmapss_path=args.cmapss_path,
        include_cmapss_fd001_sensors=args.include_cmapss_fd001_sensors,
        cmapss_sensor_path=args.cmapss_sensor_path,
        output_path=args.output,
        metadata_path=args.metadata_output,
        diagnostics_path=args.diagnostics_output,
        context_window=args.context_window,
        future_window=args.future_window,
        stride=args.stride,
    )
    print(
        f"Wrote {episodes.shape[0]} episodes x {episodes.shape[1]} steps "
        f"to {args.output} from {diagnostics['included_sources']}"
    )
    print(f"Wrote metadata to {args.metadata_output}")
    print(f"Wrote diagnostics to {args.diagnostics_output}")


if __name__ == "__main__":
    main()

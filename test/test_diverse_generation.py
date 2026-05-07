import numpy as np

from src.generation import CorpusConfig
from src.generation import corpus_composition_table
from src.generation import corpus_diagnostics_report
from src.generation import corpus_metadata_summary
from src.generation import corpus_window_budget_summary
from src.generation import default_degradation_mechanism_registry
from src.generation import degradation_shape_diagnostics
from src.generation import generate_degradation_corpus
from src.generation import generate_degradation_corpus_from_config
from src.generation import generate_diverse_degradation_episodes


def test_generate_diverse_degradation_episodes_shape_and_bounds():
    episodes = generate_diverse_degradation_episodes(
        episode_length=80,
        n_episodes=128,
        seed=42,
    )

    assert episodes.shape == (128, 80)
    assert episodes.dtype == np.float32
    assert np.isfinite(episodes).all()
    assert episodes.min() >= 0
    assert episodes.max() < 15


def test_diverse_generator_adds_normalized_shape_variety():
    episodes = generate_diverse_degradation_episodes(
        episode_length=80,
        n_episodes=128,
        seed=7,
    )
    diagnostics = degradation_shape_diagnostics(episodes, signature_decimals=2)

    t = np.linspace(0, 1, 80)
    linear_only = np.stack([(i + 1) * t for i in range(128)], axis=0)
    linear_diagnostics = degradation_shape_diagnostics(linear_only, signature_decimals=2)

    assert diagnostics["unique_signature_fraction"] > 0.80
    assert diagnostics["unique_normalized_signatures"] > linear_diagnostics["unique_normalized_signatures"]
    assert diagnostics["mean_curvature_sign_changes"] > linear_diagnostics["mean_curvature_sign_changes"]


def test_degradation_corpus_returns_structured_metadata():
    episodes, metadata = generate_degradation_corpus(
        episode_length=72,
        n_episodes=96,
        seed=11,
        return_metadata=True,
    )

    assert episodes.shape == (96, 72)
    assert len(metadata) == 96

    required_keys = {
        "mechanism",
        "family",
        "domain",
        "source_type",
        "observed_variable",
        "mechanism_family",
        "parameters",
        "covariates",
        "monotonic_expected",
        "registry_mechanism",
        "registry_domain",
        "episode_length",
        "max_value",
        "observation_effects",
    }
    assert required_keys.issubset(metadata[0])

    summary = corpus_metadata_summary(metadata)
    assert summary["n_episodes"] == 96
    assert len(summary["mechanisms"]) >= 2
    assert len(summary["domains"]) >= 2
    assert sum(summary["mechanisms"].values()) == 96


def test_degradation_corpus_source_weights_can_force_a_mechanism():
    registry = default_degradation_mechanism_registry()
    source_weights = {mechanism.name: 0.0 for mechanism in registry}
    source_weights["battery_capacity_fade"] = 1.0

    episodes, metadata = generate_degradation_corpus(
        episode_length=64,
        n_episodes=24,
        seed=3,
        source_weights=source_weights,
        apply_observation_effects=False,
        return_metadata=True,
    )

    assert episodes.shape == (24, 64)
    assert {item["mechanism"] for item in metadata} == {"battery_capacity_fade"}
    assert {item["domain"] for item in metadata} == {"battery"}
    assert all(item["observed_variable"] == "capacity_loss" for item in metadata)


def test_corpus_config_generates_corpus():
    config = CorpusConfig(
        episode_length=50,
        n_episodes=16,
        seed=13,
        source_weights={"wear_transition": 1.0, "generic": 0.0},
        apply_observation_effects=False,
    )

    episodes, metadata = generate_degradation_corpus_from_config(config)

    assert episodes.shape == (16, 50)
    assert len(metadata) == 16
    assert all(item["episode_length"] == 50 for item in metadata)


def test_corpus_diagnostics_report_includes_composition_and_window_budget():
    episodes, metadata = generate_degradation_corpus(
        episode_length=60,
        n_episodes=48,
        seed=21,
        return_metadata=True,
    )

    report = corpus_diagnostics_report(
        episodes,
        metadata,
        context_window=20,
        future_window=10,
        stride=2,
    )

    assert report["overall_shape"]["n_episodes"] == 48
    assert report["window_budget"]["windows_per_episode"] == 16
    assert report["window_budget"]["total_windows"] == 768
    assert sum(row["count"] for row in report["mechanism_table"]) == 48
    assert sum(row["count"] for row in report["domain_table"]) == 48
    assert set(report["shape_by_group"]).issubset({item["mechanism"] for item in metadata})


def test_corpus_composition_table_and_window_budget_helpers():
    metadata = [
        {"mechanism": "a", "domain": "x"},
        {"mechanism": "a", "domain": "x"},
        {"mechanism": "b", "domain": "y"},
    ]

    table = corpus_composition_table(metadata, key="mechanism")
    budget = corpus_window_budget_summary(
        episode_length=100,
        context_window=60,
        future_window=20,
        stride=5,
        n_episodes=3,
    )

    assert table == [
        {"value": "a", "count": 2, "fraction": 2 / 3},
        {"value": "b", "count": 1, "fraction": 1 / 3},
    ]
    assert budget["windows_per_episode"] == 5
    assert budget["total_windows"] == 15

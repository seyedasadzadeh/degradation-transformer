import numpy as np

from src.generation import degradation_shape_diagnostics
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

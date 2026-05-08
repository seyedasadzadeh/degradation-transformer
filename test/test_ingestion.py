import numpy as np

from src.ingestion import ArrayDatasetSource
from src.ingestion import CSVDatasetSource
from src.ingestion import NPYDatasetSource
from src.ingestion import combine_degradation_corpora
from src.ingestion import load_degradation_sources
from src.ingestion import prepare_degradation_episodes


def test_prepare_degradation_episodes_orients_decreasing_capacity():
    capacity = np.array([
        [1.0, 0.9, 0.7, 0.4],
        [0.8, 0.75, 0.6, 0.2],
    ])

    episodes = prepare_degradation_episodes(
        capacity,
        direction="decreasing",
        max_value=15,
    )

    assert episodes.shape == (2, 4)
    assert episodes.dtype == np.float32
    assert np.allclose(episodes[:, 0], 0)
    assert np.all(episodes[:, -1] > episodes[:, 1])
    assert episodes.max() < 15


def test_array_dataset_source_returns_corpus_metadata():
    source = ArrayDatasetSource(
        name="fake_battery_capacity",
        domain="battery",
        observed_variable="capacity",
        mechanism_family="capacity_fade_lab_cycles",
        parent_mechanism="battery_capacity_fade",
        direction="decreasing",
        episode_length=6,
        episodes=np.array([[1.0, 0.95, 0.85, 0.7], [1.0, 0.9, 0.8, 0.5]]),
        covariates={"temperature_c": 25},
    )

    episodes, metadata = source.load()

    assert episodes.shape == (2, 6)
    assert len(metadata) == 2
    assert metadata[0]["source_type"] == "real"
    assert metadata[0]["domain"] == "battery"
    assert metadata[0]["parent_mechanism"] == "battery_capacity_fade"
    assert metadata[0]["covariates"]["temperature_c"] == 25
    assert metadata[0]["episode_length"] == 6


def test_npy_dataset_source_loads_local_array(tmp_path):
    path = tmp_path / "episodes.npy"
    np.save(path, np.array([[0.0, 1.0, 2.0], [0.0, 0.5, 2.0]], dtype=np.float32))

    source = NPYDatasetSource(
        name="fake_wear_npy",
        domain="wear",
        observed_variable="wear_depth",
        mechanism_family="bench_wear",
        path=path,
        episode_length=5,
    )

    episodes, metadata = source.load()

    assert episodes.shape == (2, 5)
    assert {item["mechanism"] for item in metadata} == {"fake_wear_npy"}


def test_csv_dataset_source_loads_wide_matrix(tmp_path):
    path = tmp_path / "wide.csv"
    path.write_text("0,1,2,3\n1,2,3,5\n", encoding="utf-8")

    source = CSVDatasetSource(
        name="fake_corrosion_wide",
        domain="corrosion",
        observed_variable="thickness_loss",
        mechanism_family="coupon_corrosion",
        path=path,
    )

    episodes, metadata = source.load()

    assert episodes.shape == (2, 4)
    assert metadata[0]["source_name"] == "fake_corrosion_wide"


def test_csv_dataset_source_loads_long_table(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text(
        "unit,time,value\n"
        "b,1,0.9\n"
        "a,1,1.0\n"
        "b,0,1.0\n"
        "a,0,1.1\n"
        "a,2,0.7\n"
        "b,2,0.6\n",
        encoding="utf-8",
    )

    source = CSVDatasetSource(
        name="fake_long_battery",
        domain="battery",
        observed_variable="capacity",
        mechanism_family="capacity_fade_lab_cycles",
        parent_mechanism="battery_capacity_fade",
        path=path,
        value_column="value",
        id_column="unit",
        time_column="time",
        direction="decreasing",
    )

    episodes, metadata = source.load()

    assert episodes.shape == (2, 3)
    assert {item["episode_id"] for item in metadata} == {"a", "b"}
    assert np.allclose(episodes[:, 0], 0)


def test_load_degradation_sources_combines_sources_with_matching_length():
    source_a = ArrayDatasetSource(
        name="a",
        domain="battery",
        observed_variable="capacity",
        mechanism_family="capacity_fade",
        direction="decreasing",
        episode_length=4,
        episodes=np.array([[1.0, 0.8, 0.7]]),
    )
    source_b = ArrayDatasetSource(
        name="b",
        domain="wear",
        observed_variable="wear_depth",
        mechanism_family="wear_depth",
        episode_length=4,
        episodes=np.array([[0.0, 0.2, 0.5, 1.0], [0.0, 0.1, 0.4, 0.8]]),
    )

    episodes, metadata = load_degradation_sources([source_a, source_b])

    assert episodes.shape == (3, 4)
    assert len(metadata) == 3
    assert [item["source_name"] for item in metadata] == ["a", "b", "b"]


def test_combine_degradation_corpora_concatenates_synthetic_and_real_style_corpora():
    synthetic = (
        np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
        [{"source_type": "synthetic_mechanistic", "mechanism": "shape_grammar"}],
    )
    real = (
        np.array([[0.0, 0.5, 1.0], [0.0, 0.6, 1.2]], dtype=np.float32),
        [
            {"source_type": "real", "mechanism": "fake_real"},
            {"source_type": "real", "mechanism": "fake_real"},
        ],
    )

    episodes, metadata = combine_degradation_corpora(synthetic, real)

    assert episodes.shape == (3, 3)
    assert [item["source_type"] for item in metadata] == ["synthetic_mechanistic", "real", "real"]

import numpy as np

from src.real_datasets import CMAPSSTurbofanHealthSource
from src.real_datasets import NASABatteryCapacitySource


def test_nasa_battery_capacity_source_loads_processed_local_csv(tmp_path):
    path = tmp_path / "nasa_battery_capacity.csv"
    path.write_text(
        "cell_id,cycle,capacity\n"
        "B0005,0,1.86\n"
        "B0006,0,1.88\n"
        "B0005,1,1.82\n"
        "B0006,1,1.83\n"
        "B0005,2,1.74\n"
        "B0006,2,1.72\n",
        encoding="utf-8",
    )

    source = NASABatteryCapacitySource(path=path, episode_length=5)
    episodes, metadata = source.load()

    assert episodes.shape == (2, 5)
    assert episodes.dtype == np.float32
    assert np.allclose(episodes[:, 0], 0)
    assert np.all(episodes[:, -1] > episodes[:, 1])
    assert {item["episode_id"] for item in metadata} == {"B0005", "B0006"}
    assert {item["domain"] for item in metadata} == {"battery"}
    assert {item["source_name"] for item in metadata} == {"nasa_battery_capacity"}
    assert all(item["parent_mechanism"] == "battery_capacity_fade" for item in metadata)
    assert all(item["raw_schema"]["value_column"] == "capacity" for item in metadata)
    assert all("NASA Ames" in item["citation"] for item in metadata)


def test_nasa_battery_capacity_source_supports_custom_processed_schema(tmp_path):
    path = tmp_path / "custom_nasa.csv"
    path.write_text(
        "battery,cycle_index,cap_ah\n"
        "cell_a,0,2.0\n"
        "cell_a,1,1.8\n"
        "cell_a,2,1.4\n",
        encoding="utf-8",
    )

    source = NASABatteryCapacitySource(
        path=path,
        episode_length=3,
        id_column="battery",
        time_column="cycle_index",
        value_column="cap_ah",
    )
    episodes, metadata = source.load()

    assert episodes.shape == (1, 3)
    assert metadata[0]["episode_id"] == "cell_a"
    assert metadata[0]["raw_schema"] == {
        "id_column": "battery",
        "time_column": "cycle_index",
        "value_column": "cap_ah",
    }


def test_cmapss_turbofan_health_source_loads_processed_local_csv(tmp_path):
    path = tmp_path / "cmapss_fd001_health.csv"
    path.write_text(
        "engine_id,cycle,health_index\n"
        "1,1,1.0\n"
        "1,2,0.5\n"
        "1,3,0.0\n"
        "2,1,1.0\n"
        "2,2,0.0\n",
        encoding="utf-8",
    )

    source = CMAPSSTurbofanHealthSource(path=path, episode_length=4)
    episodes, metadata = source.load()

    assert episodes.shape == (2, 4)
    assert np.allclose(episodes[:, 0], 0)
    assert np.all(episodes[:, -1] > episodes[:, 1])
    assert {item["domain"] for item in metadata} == {"turbofan"}
    assert {item["source_type"] for item in metadata} == {"real_simulated"}
    assert {item["parent_mechanism"] for item in metadata} == {"turbofan_degradation"}
    assert all("C-MAPSS" in item["citation"] for item in metadata)

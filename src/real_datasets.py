from dataclasses import dataclass
from pathlib import Path

from .ingestion import CSVDatasetSource


@dataclass(frozen=True)
class NASABatteryCapacitySource(CSVDatasetSource):
    path: str | Path = ""
    name: str = "nasa_battery_capacity"
    domain: str = "battery"
    observed_variable: str = "capacity"
    mechanism_family: str = "capacity_fade_lab_cycles"
    parent_mechanism: str | None = "battery_capacity_fade"
    source_type: str = "real"
    direction: str = "decreasing"
    value_column: str | None = "capacity"
    id_column: str | None = "cell_id"
    time_column: str | None = "cycle"
    episode_length: int | None = 100
    dataset_url: str = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
    citation: str = "NASA Ames Prognostics Center of Excellence Battery Data Set"

    def load(self):
        episodes, metadata = super().load()
        for item in metadata:
            item["dataset_url"] = self.dataset_url
            item["citation"] = self.citation
            item["raw_schema"] = {
                "id_column": self.id_column,
                "time_column": self.time_column,
                "value_column": self.value_column,
            }
        return episodes, metadata


@dataclass(frozen=True)
class CMAPSSTurbofanHealthSource(CSVDatasetSource):
    path: str | Path = ""
    name: str = "cmapss_fd001_health"
    domain: str = "turbofan"
    observed_variable: str = "health_index"
    mechanism_family: str = "rul_derived_turbofan_health"
    parent_mechanism: str | None = "turbofan_degradation"
    source_type: str = "real_simulated"
    direction: str = "decreasing"
    value_column: str | None = "health_index"
    id_column: str | None = "engine_id"
    time_column: str | None = "cycle"
    episode_length: int | None = 100
    dataset_url: str = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
    citation: str = "NASA C-MAPSS Jet Engine Simulated Data"

    def load(self):
        episodes, metadata = super().load()
        for item in metadata:
            item["dataset_url"] = self.dataset_url
            item["citation"] = self.citation
            item["raw_schema"] = {
                "id_column": self.id_column,
                "time_column": self.time_column,
                "value_column": self.value_column,
            }
        return episodes, metadata


@dataclass(frozen=True)
class CMAPSSTurbofanSensorSource(CSVDatasetSource):
    path: str | Path = ""
    name: str = "cmapss_fd001_sensor_degradation"
    domain: str = "turbofan"
    observed_variable: str = "sensor_degradation"
    mechanism_family: str = "sensor_composite_turbofan_degradation"
    parent_mechanism: str | None = "turbofan_degradation"
    source_type: str = "real_simulated"
    direction: str = "increasing"
    value_column: str | None = "sensor_degradation"
    id_column: str | None = "engine_id"
    time_column: str | None = "cycle"
    episode_length: int | None = 100
    dataset_url: str = "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data"
    citation: str = "NASA C-MAPSS Jet Engine Simulated Data"

    def load(self):
        episodes, metadata = super().load()
        for item in metadata:
            item["dataset_url"] = self.dataset_url
            item["citation"] = self.citation
            item["raw_schema"] = {
                "id_column": self.id_column,
                "time_column": self.time_column,
                "value_column": self.value_column,
            }
            item["monotonic_expected"] = False
        return episodes, metadata

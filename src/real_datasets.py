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

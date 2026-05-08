from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _as_2d_episodes(values):
    episodes = np.asarray(values, dtype=np.float64)
    if episodes.ndim == 1:
        episodes = episodes[None, :]
    if episodes.ndim != 2:
        raise ValueError("episodes must be a 1D or 2D numeric array.")
    return episodes


def _orient_as_degradation(episodes, direction="auto"):
    if direction not in {"auto", "increasing", "decreasing"}:
        raise ValueError("direction must be one of: auto, increasing, decreasing.")
    episodes = np.asarray(episodes, dtype=np.float64)
    if direction == "decreasing":
        return -episodes
    if direction == "increasing":
        return episodes

    delta = np.nanmedian(episodes[:, -1] - episodes[:, 0])
    return -episodes if delta < 0 else episodes


def _resample_episodes(episodes, episode_length=None):
    episodes = np.asarray(episodes, dtype=np.float64)
    if episode_length is None or episodes.shape[1] == int(episode_length):
        return episodes
    episode_length = int(episode_length)
    if episode_length < 2:
        raise ValueError("episode_length must be at least 2 when resampling.")

    old_x = np.linspace(0, 1, episodes.shape[1])
    new_x = np.linspace(0, 1, episode_length)
    return np.stack([np.interp(new_x, old_x, row) for row in episodes], axis=0)


def _normalize_episodes(episodes, max_value=15):
    episodes = np.asarray(episodes, dtype=np.float64)
    mins = np.nanmin(episodes, axis=1, keepdims=True)
    spans = np.nanmax(episodes, axis=1, keepdims=True) - mins
    valid = np.isfinite(spans[:, 0]) & (spans[:, 0] > 1e-12)
    normalized = (episodes[valid] - mins[valid]) / spans[valid]
    return np.clip(normalized * (float(max_value) - 1e-5), 0, float(max_value) - 1e-5)


def prepare_degradation_episodes(
    episodes,
    episode_length=None,
    max_value=15,
    direction="auto",
    normalize=True,
):
    episodes = _as_2d_episodes(episodes)
    episodes = episodes[np.isfinite(episodes).all(axis=1)]
    if episodes.size == 0:
        raise ValueError("No finite episodes were available after filtering.")

    episodes = _orient_as_degradation(episodes, direction=direction)
    episodes = _resample_episodes(episodes, episode_length=episode_length)
    if normalize:
        episodes = _normalize_episodes(episodes, max_value=max_value)
    return episodes.astype(np.float32)


def real_source_metadata(
    source_name,
    domain,
    observed_variable,
    mechanism_family,
    n_episodes,
    source_type="real",
    parent_mechanism=None,
    parameters=None,
    covariates=None,
):
    metadata = []
    for episode_idx in range(int(n_episodes)):
        metadata.append(
            {
                "mechanism": source_name,
                "family": source_name,
                "domain": domain,
                "source_type": source_type,
                "observed_variable": observed_variable,
                "mechanism_family": mechanism_family,
                "parent_mechanism": parent_mechanism or domain,
                "parameters": dict(parameters or {}),
                "covariates": dict(covariates or {}),
                "monotonic_expected": True,
                "source_name": source_name,
                "episode_id": episode_idx,
            }
        )
    return metadata


@dataclass(frozen=True)
class DegradationDatasetSource:
    name: str
    domain: str
    observed_variable: str
    mechanism_family: str
    source_type: str = "real"
    parent_mechanism: str | None = None
    direction: str = "auto"
    episode_length: int | None = None
    max_value: float = 15
    normalize: bool = True

    def load(self):
        raise NotImplementedError

    def _prepare(self, episodes, parameters=None, covariates=None):
        prepared = prepare_degradation_episodes(
            episodes,
            episode_length=self.episode_length,
            max_value=self.max_value,
            direction=self.direction,
            normalize=self.normalize,
        )
        metadata = real_source_metadata(
            source_name=self.name,
            domain=self.domain,
            observed_variable=self.observed_variable,
            mechanism_family=self.mechanism_family,
            n_episodes=prepared.shape[0],
            source_type=self.source_type,
            parent_mechanism=self.parent_mechanism,
            parameters=parameters,
            covariates=covariates,
        )
        for item in metadata:
            item["episode_length"] = int(prepared.shape[1])
            item["max_value"] = float(self.max_value)
        return prepared, metadata


@dataclass(frozen=True)
class ArrayDatasetSource(DegradationDatasetSource):
    episodes: object = None
    parameters: dict | None = None
    covariates: dict | None = None

    def load(self):
        if self.episodes is None:
            raise ValueError("ArrayDatasetSource requires episodes.")
        return self._prepare(self.episodes, parameters=self.parameters, covariates=self.covariates)


@dataclass(frozen=True)
class NPYDatasetSource(DegradationDatasetSource):
    path: str | Path = ""
    parameters: dict | None = None
    covariates: dict | None = None

    def load(self):
        episodes = np.load(self.path, allow_pickle=False)
        return self._prepare(episodes, parameters=self.parameters, covariates=self.covariates)


@dataclass(frozen=True)
class CSVDatasetSource(DegradationDatasetSource):
    path: str | Path = ""
    value_column: str | None = None
    id_column: str | None = None
    time_column: str | None = None
    delimiter: str = ","
    parameters: dict | None = None
    covariates: dict | None = None

    def load(self):
        if self.value_column is None:
            episodes = np.genfromtxt(self.path, delimiter=self.delimiter, dtype=np.float64)
            if episodes.ndim == 2 and np.isnan(episodes[0]).any():
                episodes = episodes[1:]
            return self._prepare(episodes, parameters=self.parameters, covariates=self.covariates)

        table = np.genfromtxt(self.path, delimiter=self.delimiter, names=True, dtype=None, encoding="utf-8")
        names = table.dtype.names or ()
        for column in [self.value_column, self.id_column]:
            if column is not None and column not in names:
                raise ValueError(f"Column {column!r} was not found in {self.path}.")
        if self.time_column is not None and self.time_column not in names:
            raise ValueError(f"Column {self.time_column!r} was not found in {self.path}.")

        ids = table[self.id_column] if self.id_column is not None else np.zeros(table.shape[0], dtype=int)
        rows = []
        for episode_id in np.unique(ids):
            mask = ids == episode_id
            group = table[mask]
            if self.time_column is not None:
                order = np.argsort(group[self.time_column])
                group = group[order]
            rows.append(np.asarray(group[self.value_column], dtype=np.float64))

        min_length = min(len(row) for row in rows)
        episodes = np.stack([row[:min_length] for row in rows], axis=0)
        prepared, metadata = self._prepare(episodes, parameters=self.parameters, covariates=self.covariates)
        for item, episode_id in zip(metadata, np.unique(ids)):
            item["episode_id"] = str(episode_id)
        return prepared, metadata


def load_degradation_sources(sources):
    episodes_parts = []
    metadata = []
    for source in sources:
        source_episodes, source_metadata = source.load()
        episodes_parts.append(source_episodes)
        metadata.extend(source_metadata)

    if not episodes_parts:
        raise ValueError("At least one source is required.")

    lengths = {episodes.shape[1] for episodes in episodes_parts}
    if len(lengths) != 1:
        raise ValueError("All sources must produce the same episode length. Set episode_length on each source.")

    episodes = np.concatenate(episodes_parts, axis=0).astype(np.float32)
    return episodes, metadata


def combine_degradation_corpora(*corpora):
    episodes_parts = []
    metadata = []
    for episodes, corpus_metadata in corpora:
        episodes = _as_2d_episodes(episodes).astype(np.float32)
        episodes_parts.append(episodes)
        metadata.extend(list(corpus_metadata))

    if not episodes_parts:
        raise ValueError("At least one corpus is required.")

    lengths = {episodes.shape[1] for episodes in episodes_parts}
    if len(lengths) != 1:
        raise ValueError("All corpora must have the same episode length before combining.")

    return np.concatenate(episodes_parts, axis=0).astype(np.float32), metadata

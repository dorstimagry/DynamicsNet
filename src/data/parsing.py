"""Utilities for parsing raw trip folders into synchronized tensors.

The legacy :mod:`parse_trips_data.py` script performed heavy filtering and
stored preprocessed tensors on disk.  The new pipeline focuses on time
alignment and light validation; optional filters are applied later inside the
dataset.  This keeps the saved data close to the raw measurements while still
being convenient to consume from PyTorch.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

LOGGER = logging.getLogger(__name__)


SensorSpec = Mapping[str, str]


DEFAULT_SENSORS: Dict[str, tuple[str, str]] = {
    "rear_left_speed": ("rear_left_wheel_speed.csv", "data_value"),
    "rear_right_speed": ("rear_right_wheel_speed.csv", "data_value"),
    "throttle": ("throttle.csv", "data_value"),
    "brake": ("brake.csv", "data_value"),
    "angle": ("imu.csv", "pitch"),
    "driving_mode": ("driving_mode.csv", "data_value"),
}


@dataclass(slots=True)
class TripParserConfig:
    root_folder: Path = Path("/opt/imagry/trips")
    car_model: str = "ECentro"
    vehicle_id: Optional[str] = "ECENTRO_HA_03"
    dt: float = 0.02
    out_dir: Path = Path("processed_data/ECentro/ECENTRO_HA_03")
    out_file: str = "all_trips_data.pt"
    sensors: Mapping[str, tuple[str, str]] = field(default_factory=lambda: dict(DEFAULT_SENSORS))
    save_metadata: bool = True
    require_driving_mode: bool = True

    def with_defaults(self) -> "TripParserConfig":
        cfg = TripParserConfig(
            root_folder=self.root_folder,
            car_model=self.car_model,
            vehicle_id=self.vehicle_id,
            dt=self.dt,
            out_dir=self.out_dir,
            out_file=self.out_file,
            sensors=dict(self.sensors),
            save_metadata=self.save_metadata,
            require_driving_mode=self.require_driving_mode,
        )
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        return cfg


class TripDatasetParser:
    """Parse raw trip folders into synchronized numpy/tensor datasets."""

    def __init__(self, config: TripParserConfig) -> None:
        self.config = config.with_defaults()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self) -> dict[str, dict[str, np.ndarray]]:
        """Parse trips and return a dictionary keyed by segment id."""

        cfg = self.config
        segments: dict[str, dict[str, np.ndarray]] = {}
        metadata: MutableMapping[str, object] = {
            "desired_car_model": cfg.car_model,
            "dt": cfg.dt,
            "vehicle_id": cfg.vehicle_id,
            "num_valid_trips": 0,
            "valid_trip_ids": [],
        }

        for trip_dir in self._iter_trip_dirs():
            LOGGER.info("Processing %s", trip_dir.name)
            try:
                entries = self._load_trip(trip_dir)
            except ValueError as exc:
                LOGGER.warning("Skipping %s: %s", trip_dir.name, exc)
                continue

            for idx, entry in enumerate(entries, start=1):
                seg_id = f"{trip_dir.name}_seg{idx}"
                segments[seg_id] = entry
                metadata["valid_trip_ids"].append(seg_id)

        metadata["num_valid_trips"] = len(metadata["valid_trip_ids"])
        segments["metadata"] = dict(metadata)
        return segments

    def save(self) -> Path:
        """Parse and persist the dataset to ``out_dir/out_file``."""

        cfg = self.config
        dataset = self.parse()
        output_path = cfg.out_dir / cfg.out_file
        torch.save(dataset, output_path)

        if cfg.save_metadata:
            metadata = dataset.get("metadata", {})
            (cfg.out_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2)
            )

        LOGGER.info("Saved %d segments to %s", len(dataset) - 1, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Trip processing helpers
    # ------------------------------------------------------------------
    def _iter_trip_dirs(self) -> Iterator[Path]:
        cfg = self.config
        if not cfg.root_folder.exists():
            raise FileNotFoundError(cfg.root_folder)

        for trip_dir in sorted(cfg.root_folder.iterdir()):
            if not trip_dir.is_dir():
                continue
            if not self._matches_filters(trip_dir):
                continue
            yield trip_dir

    def _matches_filters(self, trip_dir: Path) -> bool:
        cfg = self.config
        car_info = trip_dir / "car_info.json"
        if not car_info.exists():
            LOGGER.debug("Missing car_info.json in %s", trip_dir)
            return False
        with car_info.open() as fh:
            if json.load(fh).get("car_type") != cfg.car_model:
                return False

        if cfg.vehicle_id:
            aidriver = trip_dir / "aidriver_info.json"
            if not aidriver.exists():
                LOGGER.debug("Missing aidriver_info.json in %s", trip_dir)
                return False
            with aidriver.open() as fh:
                if json.load(fh).get("vehicle_id") != cfg.vehicle_id:
                    return False

        return True

    def _load_trip(self, trip_dir: Path) -> List[dict[str, np.ndarray]]:
        sensors = self.config.sensors
        raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for sensor_name, (filename, column) in sensors.items():
            series = self._load_sensor(trip_dir / filename, column)
            if series is None:
                LOGGER.debug("Trip %s missing %s", trip_dir.name, filename)
                continue
            raw[sensor_name] = series

        required = {"rear_left_speed", "rear_right_speed", "throttle", "brake"}
        if not required.issubset(raw.keys()):
            raise ValueError("missing essential sensor files")

        # Normalise timestamps to seconds relative to the earliest sensor sample
        base_time = min(ts[0] for ts, _ in raw.values())
        median_steps: List[float] = []
        for ts, _ in raw.values():
            diffs = np.diff(ts)
            diffs = diffs[diffs > 0]
            if diffs.size > 0:
                median_steps.append(float(np.median(diffs)))

        scale_factor = 1.0
        if median_steps:
            median_step = float(np.median(median_steps))
            if self.config.dt > 0:
                ratio = median_step / self.config.dt
                if ratio > 100.0:
                    scale_factor = ratio

        adjusted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (ts, xs) in raw.items():
            ts_normalised = ((ts - base_time) / scale_factor).astype(np.float64)
            adjusted[name] = (ts_normalised, xs)
        raw = adjusted

        t_axis = self._build_timeline(raw.values())
        data = {name: self._interp_to_axis(t_axis, *series) for name, series in raw.items()}

        # Derived signals -------------------------------------------------
        speed = (data["rear_left_speed"] + data["rear_right_speed"]) / 2.0 / 3.6
        angle = data.get("angle")
        if angle is not None:
            angle = np.deg2rad(angle)

        driving_mode = data.get("driving_mode")
        segments = [np.arange(len(t_axis))]
        if self.config.require_driving_mode and driving_mode is not None:
            mask = driving_mode == 7
            if mask.any():
                idx = np.where(mask)[0]
                breaks = np.where(np.diff(idx) > 1)[0]
                segments = [segment for segment in np.split(idx, breaks + 1) if len(segment) > 1]

        results: List[dict[str, np.ndarray]] = []
        for seg in segments:
            window = slice(seg[0], seg[-1] + 1)
            segment_data = {
                "time": t_axis[window] - t_axis[window][0],
                "speed": speed[window],
                "throttle": data["throttle"][window],
                "brake": data["brake"][window],
                "angle": angle[window] if angle is not None else np.zeros_like(t_axis[window]),
            }
            if driving_mode is not None:
                segment_data["driving_mode"] = driving_mode[window]
            results.append(segment_data)

        return results

    # ------------------------------------------------------------------
    # Signal utilities
    # ------------------------------------------------------------------
    def _load_sensor(self, path: Path, column: str) -> tuple[np.ndarray, np.ndarray] | None:
        if not path.exists():
            return None
        df = pd.read_csv(path)
        if column not in df or "time_stamp" not in df:
            return None
        ts = df["time_stamp"].to_numpy()
        values = df[column].to_numpy()
        mask = np.isfinite(ts) & np.isfinite(values)
        ts = ts[mask]
        values = values[mask]
        if len(ts) < 2:
            return None
        order = np.argsort(ts)
        ts = ts[order]
        values = values[order]
        unique, idx = np.unique(ts, return_index=True)
        ts = unique
        values = values[idx]
        if np.any(np.diff(ts) <= 0):
            return None
        return ts, values

    def _build_timeline(self, series: Iterable[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        series_list = list(series)
        if not series_list:
            raise ValueError("no sensor data provided")
        times = [ts for ts, _ in series_list]
        t0 = min(ts[0] for ts in times)
        t1 = max(ts[-1] for ts in times)
        dt = self.config.dt
        num = int(np.floor((t1 - t0) / dt)) + 1
        if num <= 1:
            raise ValueError("insufficient duration")
        if num > 5_000_000:
            raise ValueError(f"timeline too long ({num} steps)")
        return np.linspace(t0, t0 + dt * (num - 1), num=num)

    def _interp_to_axis(self, axis: np.ndarray, ts: np.ndarray, xs: np.ndarray) -> np.ndarray:
        interp = interp1d(ts, xs, kind="linear", bounds_error=False, fill_value="extrapolate")
        return interp(axis)


__all__ = ["TripParserConfig", "TripDatasetParser"]



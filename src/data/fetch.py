"""Utilities for downloading trip data from S3.

This module refactors the legacy ``fetch_trips.py`` script into a reusable
component that can be imported by scripts as well as unit tests.  The public
surface area exposes a small dataclass configuration object plus a
``TripFetcher`` helper that hides the boto3 specifics and provides structured
logging and dry-run support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

LOGGER = logging.getLogger(__name__)


DEFAULT_FILES: Sequence[str] = (
    "car_info.json",
    "driving_mode.csv",
    "rear_left_wheel_speed.csv",
    "rear_right_wheel_speed.csv",
    "throttle.csv",
    "brake.csv",
    "imu.csv",
)


def _ensure_date(value: dt.date | dt.datetime | str) -> dt.date:
    """Normalize a variety of date-like values into a ``datetime.date``.

    Accepts ``YYYY-MM-DD`` strings, ``datetime.date`` and ``datetime.datetime``
    values.  Raises ``ValueError`` on invalid inputs.
    """

    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return dt.datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError as exc:  # pragma: no cover - defensive branch
            raise ValueError(f"Invalid date string: {value!r}") from exc
    raise TypeError(f"Unsupported date value type: {type(value)!r}")


def daterange(start: dt.date, end: dt.date) -> Iterator[dt.date]:
    """Yield all days in the inclusive date interval ``[start, end]``."""

    day = start
    while day <= end:
        yield day
        day += dt.timedelta(days=1)


@dataclass(slots=True)
class FetchTripsConfig:
    """Configuration for downloading time-synced trip data from S3."""

    car: str
    start: dt.date | dt.datetime | str
    end: dt.date | dt.datetime | str
    dest: Path = Path("data/raw/trips")
    files: Sequence[str] | None = None
    vehicle_id: Optional[str] = None
    max_gb: Optional[float] = None
    overwrite: bool = False
    dry_run: bool = False
    bucket: str = "trips-backup"
    root_prefix: str = "trips_metadata"

    def normalized(self) -> "FetchTripsConfig":
        """Return a copy with normalized dates and resolved defaults."""

        start_date = _ensure_date(self.start)
        end_date = _ensure_date(self.end)
        if end_date < start_date:
            raise ValueError("End date cannot be earlier than start date")

        if self.files is None:
            files = list(DEFAULT_FILES)
        else:
            files = list(self.files)
        if self.vehicle_id and "aidriver_info.json" not in files:
            files.append("aidriver_info.json")

        return FetchTripsConfig(
            car=self.car,
            start=start_date,
            end=end_date,
            dest=self.dest,
            files=files,
            vehicle_id=self.vehicle_id,
            max_gb=self.max_gb,
            overwrite=self.overwrite,
            dry_run=self.dry_run,
            bucket=self.bucket,
            root_prefix=self.root_prefix,
        )


class TripFetcher:
    """Download trip folders from S3 with optional filtering and dry-run."""

    def __init__(
        self,
        config: FetchTripsConfig,
        s3_client: Optional[BaseClient] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config.normalized()
        self.dest = self.config.dest
        self.dest.mkdir(parents=True, exist_ok=True)
        self.logger = logger or LOGGER
        self.s3: BaseClient = s3_client or boto3.client("s3")

        self._cap_bytes = (
            int(self.config.max_gb * 1024**3)
            if self.config.max_gb is not None
            else None
        )

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the download procedure according to the configuration."""

        cfg = self.config
        self.logger.info(
            "Fetching trips for car=%s vehicle_id=%s %s→%s",
            cfg.car,
            cfg.vehicle_id or "*",
            cfg.start,
            cfg.end,
        )
        if cfg.files:
            self.logger.info("Required files: %s", ", ".join(cfg.files))
        if cfg.max_gb:
            self.logger.info("Download size cap: %.2f GB", cfg.max_gb)

        downloaded_bytes = 0
        for day in daterange(cfg.start, cfg.end):
            for trip_prefix in self._list_trip_prefixes(day):
                if not self._is_target_car(trip_prefix):
                    continue

                if cfg.vehicle_id and not self._has_vehicle_id(trip_prefix):
                    continue

                trip_id = Path(trip_prefix.rstrip("/")).name
                dest_dir = self.dest / trip_id
                if dest_dir.exists() and not cfg.overwrite:
                    self.logger.info("Skip %s (already exists)", trip_id)
                    continue

                size = (
                    self._required_files_size(trip_prefix, cfg.files)
                    if cfg.files
                    else self._full_folder_size(trip_prefix)
                )
                if size is None:
                    self.logger.info("Skip %s (missing required files)", trip_id)
                    continue

                if self._cap_bytes and downloaded_bytes + size > self._cap_bytes:
                    self.logger.info(
                        "Stopping download after %.2f GB (cap %.2f GB)",
                        downloaded_bytes / 1e9,
                        cfg.max_gb,
                    )
                    return

                self._sync_trip(trip_prefix, dest_dir, cfg.files)
                if not cfg.dry_run:
                    downloaded_bytes += size

        if cfg.dry_run:
            self.logger.info("Dry-run completed (no downloads performed)")
        else:
            self.logger.info("Finished downloading %.2f GB", downloaded_bytes / 1e9)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _list_trip_prefixes(self, day: dt.date) -> Iterable[str]:
        prefix = f"{self.config.root_prefix}/{day:%Y/%m/%d}/"
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.config.bucket, Prefix=prefix, Delimiter="/"
        ):
            for common in page.get("CommonPrefixes", []):
                yield common["Prefix"]

    def _is_target_car(self, prefix: str) -> bool:
        try:
            obj = self.s3.get_object(
                Bucket=self.config.bucket, Key=f"{prefix}car_info.json"
            )
        except ClientError as exc:  # pragma: no cover - network interaction
            if exc.response["Error"].get("Code") in {"404", "NoSuchKey"}:
                return False
            raise

        try:
            payload = obj["Body"].read()
            car_type = json.loads(payload).get("car_type")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.logger.info("Skip %s (invalid car_info.json)", prefix)
            return False
        finally:
            obj["Body"].close()

        return car_type == self.config.car

    def _has_vehicle_id(self, prefix: str) -> bool:
        try:
            obj = self.s3.get_object(
                Bucket=self.config.bucket, Key=f"{prefix}aidriver_info.json"
            )
        except ClientError as exc:  # pragma: no cover - network interaction
            if exc.response["Error"].get("Code") in {"404", "NoSuchKey"}:
                self.logger.info("Skip %s (missing aidriver_info.json)", prefix)
                return False
            raise
        try:
            payload = obj["Body"].read()
            vehicle = json.loads(payload).get("vehicle_id")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.logger.info("Skip %s (invalid aidriver_info.json)", prefix)
            return False
        finally:
            obj["Body"].close()
        return vehicle == self.config.vehicle_id

    def _required_files_size(self, prefix: str, files: Sequence[str]) -> Optional[int]:
        total = 0
        for filename in files:
            key = f"{prefix}{filename}"
            try:
                head = self.s3.head_object(Bucket=self.config.bucket, Key=key)
            except ClientError as exc:
                if exc.response["Error"].get("Code") in {"404", "NoSuchKey"}:
                    return None
                raise
            total += head["ContentLength"]
        return total

    def _full_folder_size(self, prefix: str) -> int:
        total = 0
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.config.bucket,
            Prefix=prefix,
        ):
            for obj in page.get("Contents", []):
                total += obj["Size"]
        return total

    def _sync_trip(self, prefix: str, dest_dir: Path, files: Sequence[str]) -> None:
        command = [
            "aws",
            "s3",
            "sync",
            f"s3://{self.config.bucket}/{prefix}",
            str(dest_dir),
            "--only-show-errors",
        ]
        if files:
            command += ["--exclude", "*"]
            for filename in files:
                command += ["--include", filename]

        if self.config.dry_run:
            self.logger.info("DRY-RUN: %s", " ".join(command))
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading %s", dest_dir.name)
        # Use subprocess.run instead of check_call so that logs are visible.
        import subprocess

        subprocess.run(command, check=True)


__all__ = ["FetchTripsConfig", "TripFetcher"]



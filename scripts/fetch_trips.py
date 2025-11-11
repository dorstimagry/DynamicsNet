#!/usr/bin/env python3
"""CLI wrapper around :mod:`src.data.fetch` for downloading trip data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.data.fetch import FetchTripsConfig, TripFetcher


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download trip data from S3")
    parser.add_argument("--car", required=True, help="Car type to filter")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/trips"),
        help="Destination directory for downloaded trips",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Required files per trip (missing files skip the trip)",
    )
    parser.add_argument(
        "--vehicle-id",
        help="Optional vehicle/platform identifier (aidriver_info.json)",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        help="Optional total download size limit in gigabytes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trip directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading",
    )
    parser.add_argument(
        "--bucket",
        default="trips-backup",
        help="S3 bucket name (default: trips-backup)",
    )
    parser.add_argument(
        "--root-prefix",
        default="trips_metadata",
        help="S3 prefix containing trip folders",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = FetchTripsConfig(
        car=args.car,
        start=args.start,
        end=args.end,
        dest=args.dest,
        files=args.files if args.files else None,
        vehicle_id=args.vehicle_id,
        max_gb=args.max_gb,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        bucket=args.bucket,
        root_prefix=args.root_prefix,
    )

    fetcher = TripFetcher(config=config)
    fetcher.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



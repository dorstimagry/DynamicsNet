#!/usr/bin/env python3
"""
Selective Trip Downloader for S3
================================

This script downloads trip data from an S3 bucket, with the following features:
- Skips an entire trip if **any** file requested with `--files` is missing.
- Supports filtering by car type and date range.
- Allows specifying a maximum total download size (in GB).
- Supports dry-run mode to preview actions without downloading.
- Can overwrite existing downloads if requested.

Usage
-----

Run the script from the command line:

    python fetch_trips.py --car <CAR_TYPE> --start <YYYY-MM-DD> --end <YYYY-MM-DD> [options]

Required Arguments:
-------------------
--car        The car type to filter trips by (e.g., "hyundai_ioniq").
--start      Start date (inclusive) in YYYY-MM-DD format.
--end        End date (inclusive) in YYYY-MM-DD format.

Optional Arguments:
-------------------
--dest       Destination directory for downloaded trips (default: /opt/imagry/trips).
--files      List of files to download for each trip. If any are missing, the trip is skipped.
             Defaults to:
                 car_info.json
                 driving_mode.csv
                 rear_left_wheel_speed.csv
                 rear_right_wheel_speed.csv
                 throttle.csv
                 brake.csv
                 imu.csv
--vehicle-id    Specific platform ID (e.g., "NIRO_SJ_03").
--max-gb     Maximum total download size in gigabytes. Stops when cap is reached.
--overwrite  Overwrite existing trip folders in the destination directory.
-n, --dry-run
             Perform a dry run: print actions without downloading files.

Examples
--------

1. Download all trips for car type "hyundai_ioniq" from June 1 to June 3, 2024:

    python fetch_trips.py --car hyundai_ioniq --start 2024-06-01 --end 2024-06-03

2. Download only specific files for each trip, skipping trips where any are missing:

    python fetch_trips.py --car hyundai_ioniq --start 2024-06-01 --end 2024-06-03 \
        --files car_info.json throttle.csv brake.csv

3. Limit total download to 10 GB and overwrite existing trips:

    python fetch_trips.py --car hyundai_ioniq --start 2024-06-01 --end 2024-06-03 \
        --max-gb 10 --overwrite

4. Preview what would be downloaded without actually downloading (dry run):

    python fetch_trips.py --car hyundai_ioniq --start 2024-06-01 --end 2024-06-03 --dry-run

Notes
-----
- Requires AWS credentials configured for S3 access.
- Uses the "trips-backup" bucket and "trips_metadata" prefix.
- If `--files` is specified, trips missing any of those files are skipped.
- If `--max-gb` is specified, downloading stops when the cap is reached.
- If the destination folder for a trip exists and `--overwrite` is not set, the trip is skipped.
"""

from __future__ import annotations
import argparse, datetime as dt, json, subprocess, sys
from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.exceptions import ClientError

# --------------------------------------------------------------------- #
# Quick helpers                                                         #
# --------------------------------------------------------------------- #
def parse_date(s: str) -> dt.date:
    return dt.datetime.strptime(s, "%Y-%m-%d").date()

def daterange(a: dt.date, b: dt.date) -> Iterator[dt.date]:
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)

# --------------------------------------------------------------------- #
# S3 helpers                                                            #
# --------------------------------------------------------------------- #
BUCKET      = "trips-backup"
ROOT_PREFIX = "trips_metadata"
s3          = boto3.client("s3")

def list_trip_prefixes(day: dt.date) -> Iterator[str]:
    prefix = f"{ROOT_PREFIX}/{day:%Y/%m/%d}/"
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in pg.get("CommonPrefixes", []):
            yield cp["Prefix"]            # ends with '/'

def get_car_type(prefix: str) -> Optional[str]:
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}car_info.json")
        return json.load(obj["Body"]).get("car_type")
    except Exception:
        return None

def required_files_size(prefix: str, files: list[str]) -> Optional[int]:
    """
    Returns cumulative size *only if every requested file exists*.
    If any is missing → None.
    """
    total = 0
    for f in files:
        key = f"{prefix}{f}"
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
            total += head["ContentLength"]
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return None
            raise
    return total

def full_folder_size(prefix: str) -> int:
    total = 0
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix):
        for obj in pg.get("Contents", []):
            total += obj["Size"]
    return total

def sync_trip(prefix: str, dest_root: Path, files: list[str] | None,
              dry: bool) -> str:
    trip_id = Path(prefix.rstrip("/")).name
    dest    = dest_root / trip_id
    cmd = ["aws","s3","sync",f"s3://{BUCKET}/{prefix}",str(dest),
           "--only-show-errors"]
    if files:
        cmd += ["--exclude","*"] + sum([["--include",f] for f in files], [])
    if dry: return "DRY-RUN   "+" ".join(cmd)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(cmd)
    return f"✓ downloaded {trip_id}"

# --------------------------------------------------------------------- #
# Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", required=True)
    ap.add_argument("--start", type=parse_date, required=True)
    ap.add_argument("--end",   type=parse_date, required=True)
    ap.add_argument("--dest",  type=Path, default=Path("/opt/imagry/trips"))
    ap.add_argument("--files", nargs="*", metavar="FILE",
                    default=["car_info.json",
                             "driving_mode.csv",
                             "rear_left_wheel_speed.csv",
                             "rear_right_wheel_speed.csv",
                             "throttle.csv",
                             "brake.csv",
                             "imu.csv"],
                    help="download only these files; if any is missing, trip is skipped")
    ap.add_argument("--vehicle-id", type=str, help="Filter trips by vehicle_id in aidriver_info.json")
    ap.add_argument("--max-gb", type=float, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("-n","--dry-run", action="store_true")
    args = ap.parse_args()

    # Always include aidriver_info.json if vehicle_id filtering is requested
    if args.vehicle_id and "aidriver_info.json" not in args.files:
        args.files.append("aidriver_info.json")

    if args.end < args.start:
        sys.exit("end date before start date")

    args.dest.mkdir(parents=True, exist_ok=True)
    cap_bytes = None if args.max_gb is None else int(args.max_gb*1024**3)
    want = args.files or None

    print(f"→ Trips {args.start} … {args.end}  car_type == {args.car}  vehicle_id == {args.vehicle_id}")
    if want:   print("→ Required files:", ", ".join(want))
    if cap_bytes: print(f"→ Size cap ≈ {args.max_gb:g} GB")

    downloaded = 0
    for day in daterange(args.start, args.end):
        for prefix in list_trip_prefixes(day):
            if get_car_type(prefix) != args.car:
                continue

            # If vehicle_id filter is specified, download and check aidriver_info.json
            if args.vehicle_id:
                try:
                    obj = s3.get_object(Bucket=BUCKET, Key=f"{prefix}aidriver_info.json")
                    aidriver_info = json.load(obj["Body"])
                    if aidriver_info.get("vehicle_id") != args.vehicle_id:
                        continue
                except ClientError as e:
                    if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                        print("SKIP (no aidriver_info.json)", Path(prefix).name)
                        continue
                    raise

            trip_id = Path(prefix.rstrip("/")).name
            dest = args.dest / trip_id
            if dest.exists() and not args.overwrite:
                print("SKIP (exists)", trip_id)
                continue

            if want:
                sz = required_files_size(prefix, want)
                if sz is None:
                    print("SKIP (missing files)", trip_id)
                    continue
            else:
                sz = full_folder_size(prefix)

            if cap_bytes and downloaded + sz > cap_bytes:
                print(f"STOP size cap – downloaded {downloaded/1e9:.2f} GB")
                return

            msg = sync_trip(prefix, args.dest, want, args.dry_run)
            print(f"{msg:80s} ({sz/1e6:8.1f} MB)")
            if not args.dry_run: downloaded += sz

    print("\nDONE (dry-run)" if args.dry_run else
          f"\nDONE – downloaded {downloaded/1e9:.2f} GB")

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
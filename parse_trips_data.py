#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trip-parser that keeps only *continuous full-control* (driving_mode == 7) parts
and drops idle / faulty data.

What’s new (compared with previous version)
-------------------------------------------
➤ Segments are rejected if **any throttle or brake sample < 0**
➤ Everything else (variance filter, low-pass fallback, NaN/dup cleanup,
  per-segment splitting) is unchanged
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

# ------------------------------------------------------------------ #
# Configuration                                                       #
# ------------------------------------------------------------------ #
ROOT_FOLDER       = "/opt/imagry/trips"
DESIRED_CAR_MODEL = "ECentro"
DESIRED_VEHICLE_ID = "ECENTRO_HA_03"  # e.g., "NIRO_SJ_03" – set to None to disable filtering
DT                = 0.02
OUT_DIR           = Path(f"processed_data/{DESIRED_CAR_MODEL}/{DESIRED_VEHICLE_ID}")
OUT_FILE          = "all_trips_data.pt"


# variance thresholds
MIN_VAR_THR = 20.0   # (%²)
MIN_VAR_BR  = 20.0

# Filtering options
FILTER_BY_DRIVING_MODE = False
FILTER_STOPPED_WITH_BRAKE = False
FILTER_STOPPED_WITH_THROTTLE = False

# sensors: filename, column
SENSOR_FILES = {
    "rear_left_speed":  ("rear_left_wheel_speed.csv",  "data_value"),
    "rear_right_speed": ("rear_right_wheel_speed.csv", "data_value"),
    "throttle":         ("throttle.csv",               "data_value"),
    "brake":            ("brake.csv",                  "data_value"),
    "angle":            ("imu.csv",                    "pitch"),
}
DRIVE_MODE_FILE = ("driving_mode.csv", "data_value")   # mode == 7 → keep

# ------------------------------------------------------------------ #
# Generic helpers                                                    #
# ------------------------------------------------------------------ #
def butter_low(x, fc, fs, order=3):
    if len(x) < 2*order+1: return x
    b,a = butter(order, fc/(0.5*fs), "low")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*filtfilt.*")
        try:               return filtfilt(b,a,x)
        except ValueError: return x

def accel(v, dt):                       # low-pass + gradient
    v_f = butter_low(v, 1.0, 1/dt)
    return np.gradient(v_f, dt, edge_order=2) if v_f.size>=3 else np.zeros_like(v_f)

def clean(ts, xs):
    m = np.isfinite(ts)&np.isfinite(xs)
    ts,xs = ts[m], xs[m]
    if ts.size<2: return None
    o = np.argsort(ts); ts,xs = ts[o], xs[o]
    uniq,idx = np.unique(ts, return_index=True)
    ts,xs = ts[idx], xs[idx]
    if ts.size<2 or np.any(np.diff(ts)<=0): return None
    return ts,xs

def load_csv(path: Path, col: str):
    try: df = pd.read_csv(path)
    except FileNotFoundError: return None
    if col not in df or "time_stamp" not in df: return None
    return clean(df["time_stamp"].to_numpy(), df[col].to_numpy())

# ------------------------------------------------------------------ #
# Trip → list of full-control segments                               #
# ------------------------------------------------------------------ #
def segments_from_trip(
    folder: Path,
    dt=DT,
    filter_by_driving_mode: bool = FILTER_BY_DRIVING_MODE,
    filter_stopped_with_brake: bool = FILTER_STOPPED_WITH_BRAKE,
    filter_stopped_with_throttle: bool = FILTER_STOPPED_WITH_THROTTLE,
) -> list[pd.DataFrame]:
    raw={}
    for k,(fname,col) in SENSOR_FILES.items():
        ser = load_csv(folder/fname, col)
        if ser is None:
            print(f"  ⚠ missing {fname}"); continue
        t,x = ser
        if k=="throttle": x=np.where(x>100,x-108.47458,x)
        raw[k]=(t,x)

    mode_ser = load_csv(folder/DRIVE_MODE_FILE[0], DRIVE_MODE_FILE[1])
    if {"rear_left_speed","rear_right_speed"}-raw.keys():
        return []

    # common timeline
    all_ts = np.concatenate([v[0] for v in raw.values()])
    t0,t1  = all_ts.min(), all_ts.max()
    t_axis = np.arange(0, t1-t0+dt, dt)

    def interp(t,x,kind="linear"):
        f = interp1d(t-t0,x,kind=kind,bounds_error=False,fill_value="extrapolate")
        return f(t_axis)

    data={k:interp(*v) for k,v in raw.items()}

    # Convert wheel speeds from km/h to m/s (1 km/h = 1/3.6 m/s)
    if "rear_left_speed" in data:
        data["rear_left_speed"] = data["rear_left_speed"] / 3.6
    if "rear_right_speed" in data:
        data["rear_right_speed"] = data["rear_right_speed"] / 3.6

    data["speed"]=(data["rear_left_speed"]+data["rear_right_speed"])/2
    if "angle" in data: data["angle"]=np.deg2rad(data["angle"])
    data["acceleration"]=accel(data["speed"], dt)

    # Driving mode filtering
    if filter_by_driving_mode:
        if mode_ser is None:
            return []
        mode_vec = interp(*mode_ser, kind="nearest")
        mask = mode_vec == 7.0
        if not mask.any():
            return []
        idx = np.where(mask)[0]
        seg_bounds = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    else:
        # No driving mode filtering: treat the whole timeline as one segment
        seg_bounds = [np.arange(len(t_axis))]

    if any(~np.isfinite(a).all() for a in data.values()):
        print("  ⚠ NaNs/inf – skip"); return []

    good = []
    for seg in seg_bounds:
        if seg.size < 2:
            continue
        df = pd.DataFrame({k: v[seg] for k, v in data.items()}, index=t_axis[seg] - t_axis[seg][0])

        # filters -------------------------------------------------------
        if np.var(df.get("throttle", 0)) < MIN_VAR_THR and np.var(df.get("brake", 0)) < MIN_VAR_BR:
            continue
        if ("throttle" in df and (df["throttle"] < 0).any()) or ("brake" in df and (df["brake"] < 0).any()):
            continue

        # Optionally filter out zero speed with active brake
        if filter_stopped_with_brake and "speed" in df and "brake" in df:
            is_stopped_with_brake = (df["speed"] < 0.1) & (df["brake"] > 0.02)
            if is_stopped_with_brake.any():
                valid_mask = ~is_stopped_with_brake
                if valid_mask.sum() < 10:
                    continue
                valid_indices = np.where(valid_mask)[0]
                breaks = np.where(np.diff(valid_indices) > 1)[0]
                subsegments = np.split(valid_indices, breaks + 1)
                valid_subsegments = [sub for sub in subsegments if len(sub) >= 10]
                if not valid_subsegments:
                    continue
                for subseg in valid_subsegments:
                    sub_df = df.iloc[subseg].copy()
                    sub_df.index = np.arange(0, len(sub_df) * DT, DT)[:len(sub_df)]
                    good.append(sub_df)
                continue

        # Optionally filter out zero speed with active throttle
        if filter_stopped_with_throttle and "speed" in df and "throttle" in df:
            is_stopped_with_throttle = (df["speed"] < 0.1) & (df["throttle"] > 0.02)
            if is_stopped_with_throttle.any():
                valid_mask = ~is_stopped_with_throttle
                if valid_mask.sum() < 10:
                    continue
                valid_indices = np.where(valid_mask)[0]
                breaks = np.where(np.diff(valid_indices) > 1)[0]
                subsegments = np.split(valid_indices, breaks + 1)
                valid_subsegments = [sub for sub in subsegments if len(sub) >= 10]
                if not valid_subsegments:
                    continue
                for subseg in valid_subsegments:
                    sub_df = df.iloc[subseg].copy()
                    sub_df.index = np.arange(0, len(sub_df) * DT, DT)[:len(sub_df)]
                    good.append(sub_df)
                continue

        good.append(df)
    return good

# ------------------------------------------------------------------ #
# Build dataset                                                      #
# ------------------------------------------------------------------ #

def build_dataset(
    root=Path(ROOT_FOLDER),
    car=DESIRED_CAR_MODEL,
    out_dir=OUT_DIR,
    out_file=OUT_FILE,
    filter_by_driving_mode: bool = FILTER_BY_DRIVING_MODE,
    filter_stopped_with_brake: bool = FILTER_STOPPED_WITH_BRAKE,
    filter_stopped_with_throttle: bool = FILTER_STOPPED_WITH_THROTTLE,
):

    out_dir.mkdir(parents=True, exist_ok=True)
    ds, ids = {},[]

    for trip in sorted(p for p in root.iterdir() if p.is_dir()):
        info_path = trip / "car_info.json"
        aidriver_path = trip / "aidriver_info.json"

        # Filter by car type
        if not info_path.exists(): return
        try:
            car_type = json.load(open(info_path)).get("car_type")
            if car_type != car:
                continue
        except Exception:
            continue

        # Filter by vehicle_id (if desired)
        if DESIRED_VEHICLE_ID:
            if not aidriver_path.exists(): return
            try:
                aidriver_info = json.load(open(aidriver_path))
                if aidriver_info.get("vehicle_id") != DESIRED_VEHICLE_ID:
                    continue
            except Exception:
                continue

        print(f"▶ {trip.name}")
        segs = segments_from_trip(
            trip,
            filter_by_driving_mode=filter_by_driving_mode,
            filter_stopped_with_brake=filter_stopped_with_brake,
            filter_stopped_with_throttle=filter_stopped_with_throttle,
        )
        if not segs:
            print("   no valid segments"); continue
        for i, df in enumerate(segs, 1):
            tid = f"{trip.name}_seg{i}"
            ds[tid] = {c: df[c].astype(np.float32).to_numpy() for c in df}
            ds[tid]["time"] = df.index.astype(np.float32).to_numpy()
            ids.append(tid)
            print(f"   ✓ seg{i}  len={len(df)}")

    if not ids:
        print("No valid segments found.")
        return
        
    lengths=defaultdict(int)
    for tid in ids:
        for k,v in ds[tid].items(): lengths[k]+=len(v)
    meta=dict(desired_car_model=car, dt=DT, num_valid_trips=len(ids),
              valid_trip_ids=ids,
              measurement_lengths={k:int(v) for k,v in lengths.items()},
              measurement_shapes={k:[v] for k,v in lengths.items()})
    ds["metadata"]=meta
    print(f"\nTotal segments: {len(ids)}")
    torch.save(ds, out_dir/out_file)
    (out_dir/"metadata.json").write_text(json.dumps(meta,indent=4))
    print(f"\nSaved {len(ids)} segments → {out_dir/out_file}")

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    build_dataset()

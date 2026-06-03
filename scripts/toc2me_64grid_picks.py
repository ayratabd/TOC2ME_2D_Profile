#!/usr/bin/env python3
"""Build 64x64 velocity grid and derive 64-point traveltime curves for 3 events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_EVENTS = DEFAULT_OUTPUTS / "selected_events.csv"
DEFAULT_STATIONS = DEFAULT_OUTPUTS / "selected_stations.csv"
DEFAULT_PICKS = DEFAULT_OUTPUTS / "selected_picks_pp.csv"
DEFAULT_STATION_IDS = DEFAULT_OUTPUTS / "selected_event_picks_9stations.csv"
DEFAULT_VELOCITY_CSV = DEFAULT_OUTPUTS / "velocity_model_vp.csv"
DEFAULT_VELOCITY_MAT = REPO_ROOT / "data" / "ToC2MEVelModel.mat"


def load_velocity_model(velocity_csv: Path, velocity_mat: Path) -> Tuple[np.ndarray, np.ndarray]:
    if velocity_csv.exists():
        df = pd.read_csv(velocity_csv)
        return df["depth_m"].to_numpy(dtype=float), df["vp_m_s"].to_numpy(dtype=float)
    if velocity_mat.exists():
        mat = sio.loadmat(velocity_mat)
        z = mat.get("z")
        vp = mat.get("vp")
        if z is None or vp is None:
            raise ValueError("Velocity model MAT missing 'z' or 'vp' arrays")
        return z.squeeze().astype(float), vp.squeeze().astype(float)
    raise FileNotFoundError("Velocity model not found")


def filter_events(events: pd.DataFrame, depth_max: float, exclude_deepest: int) -> pd.DataFrame:
    filtered = events.copy()
    filtered = filtered[filtered["depth_m"].notna()]
    filtered = filtered.sort_values("depth_m", ascending=False).iloc[exclude_deepest:]
    filtered = filtered[filtered["depth_m"] <= depth_max]
    return filtered


def load_station_ids(path: Path) -> list[int]:
    if not path.exists():
        raise FileNotFoundError(f"Station IDs file not found: {path}")
    df = pd.read_csv(path)
    if "station_id" not in df.columns:
        raise ValueError("Station IDs file must contain a station_id column")
    return sorted({int(v) for v in df["station_id"].dropna().unique().tolist()})


def apply_station_subset(
    stations: pd.DataFrame,
    picks: pd.DataFrame,
    subset: str,
    station_ids_path: Path,
    east_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    if subset == "all":
        return stations, picks, []

    station_ids = load_station_ids(station_ids_path)
    stations = stations[stations["station_id"].isin(station_ids)].copy()
    if stations.empty:
        raise ValueError("No stations remain after applying station_ids filter")

    if subset == "east5":
        stations = stations.sort_values("x_m", ascending=False).head(east_count).copy()
        station_ids = stations["station_id"].astype(int).tolist()

    picks = picks[picks["station_id"].isin(station_ids)].copy()
    return stations, picks, station_ids


def choose_square_window(events: pd.DataFrame, stations: pd.DataFrame, width_m: float, step_m: float) -> Tuple[float, float]:
    min_event = float(events["along_line_m"].min())
    max_event = float(events["along_line_m"].max())

    start_min = max_event - width_m
    start_max = min_event
    if start_min > start_max:
        center = 0.5 * (min_event + max_event)
        start_min = start_max = center - 0.5 * width_m

    candidates = np.arange(start_min, start_max + step_m, step_m)
    station_pos = stations["along_line_m"].to_numpy(dtype=float)
    event_center = 0.5 * (min_event + max_event)

    best_start = candidates[0]
    best_count = -1
    best_center_delta = float("inf")
    for start in candidates:
        end = start + width_m
        count = int(((station_pos >= start) & (station_pos <= end)).sum())
        center_delta = abs((start + end) * 0.5 - event_center)
        if count > best_count or (count == best_count and center_delta < best_center_delta):
            best_start = start
            best_count = count
            best_center_delta = center_delta

    return float(best_start), float(best_start + width_m)


def choose_three_events(events: pd.DataFrame) -> pd.DataFrame:
    along = events["along_line_m"].to_numpy(dtype=float)
    min_along = float(np.min(along))
    max_along = float(np.max(along))
    span = max_along - min_along
    left_max = min_along + span / 3.0
    mid_max = min_along + 2.0 * span / 3.0

    def pick_group(mask: np.ndarray, center: float) -> pd.Series:
        subset = events[mask]
        if subset.empty:
            idx = (events["along_line_m"] - center).abs().idxmin()
            return events.loc[idx]
        idx = subset["dist_to_line_m"].idxmin()
        return subset.loc[idx]

    left = pick_group(along <= left_max, min_along)
    middle = pick_group((along > left_max) & (along <= mid_max), 0.5 * (left_max + mid_max))
    right = pick_group(along > mid_max, max_along)

    chosen = pd.DataFrame([left, middle, right]).drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    return chosen


def grid_indices(value: float, step: float, count: int) -> int:
    idx = int(round(value / step))
    return int(max(0, min(count - 1, idx)))


def build_traveltime_curve(
    grid_x: np.ndarray,
    station_x: np.ndarray,
    station_times: np.ndarray,
) -> np.ndarray:
    assigned = {}
    for x, t in zip(station_x, station_times):
        idx = int(np.argmin(np.abs(grid_x - x)))
        assigned.setdefault(idx, []).append(t)

    idxs = sorted(assigned.keys())
    if len(idxs) < 2:
        return np.full_like(grid_x, np.nan, dtype=float)

    times = np.array([np.mean(assigned[i]) for i in idxs], dtype=float)
    x_known = grid_x[idxs]
    return np.interp(grid_x, x_known, times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 64x64 velocity grid and traveltimes.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--station-ids", type=Path, default=DEFAULT_STATION_IDS)
    parser.add_argument("--station-subset", choices=["all", "east5"], default="all")
    parser.add_argument("--east-count", type=int, default=5)
    parser.add_argument("--velocity-csv", type=Path, default=DEFAULT_VELOCITY_CSV)
    parser.add_argument("--velocity-mat", type=Path, default=DEFAULT_VELOCITY_MAT)
    parser.add_argument("--depth-max", type=float, default=3500.0)
    parser.add_argument("--exclude-deepest", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    stations = pd.read_csv(args.stations)
    picks = pd.read_csv(args.picks)

    stations, picks, station_subset_ids = apply_station_subset(
        stations,
        picks,
        args.station_subset,
        args.station_ids,
        args.east_count,
    )

    filtered = filter_events(events, args.depth_max, args.exclude_deepest)
    if filtered.empty:
        raise ValueError("No events remain after filtering")

    dist_min, dist_max = choose_square_window(filtered, stations, args.depth_max, 10.0)
    in_window = (filtered["along_line_m"] >= dist_min) & (filtered["along_line_m"] <= dist_max)
    filtered = filtered[in_window]
    if filtered.empty:
        raise ValueError("No events inside the square window")

    chosen = choose_three_events(filtered)

    station_mask = (stations["along_line_m"] >= dist_min) & (stations["along_line_m"] <= dist_max)
    stations_win = stations[station_mask].copy()

    grid_n = args.grid_size
    width = args.depth_max
    dx = width / (grid_n - 1)
    dz = width / (grid_n - 1)
    grid_x = np.linspace(dist_min, dist_max, grid_n)
    grid_depth = np.linspace(0.0, args.depth_max, grid_n)

    depth_m, vp_m_s = load_velocity_model(args.velocity_csv, args.velocity_mat)
    vp_depth = np.interp(grid_depth, depth_m, vp_m_s)
    vp_grid = np.tile(vp_depth[:, None], (1, grid_n))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "velocity_model_64x64.npy", vp_grid)

    event_rows = []
    for _, row in chosen.iterrows():
        x_rel = row["along_line_m"] - dist_min
        z_rel = row["depth_m"]
        x_idx = grid_indices(x_rel, dx, grid_n)
        z_idx = grid_indices(z_rel, dz, grid_n)
        event_rows.append(
            {
                "event_id": int(row["event_id"]),
                "along_line_m": row["along_line_m"],
                "depth_m": row["depth_m"],
                "x_rel_m": x_rel,
                "z_rel_m": z_rel,
                "x_idx": x_idx,
                "z_idx": z_idx,
                "dist_to_line_m": row["dist_to_line_m"],
            }
        )

    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(args.out_dir / "event_locations_64.csv", index=False)

    pick_rows = []
    curve_rows = []
    fig_data = []

    for _, event in event_df.iterrows():
        event_id = int(event["event_id"])
        event_picks = picks[picks["event_id"] == event_id]
        event_picks = event_picks.merge(
            stations_win[["station_id", "along_line_m"]],
            on="station_id",
            how="inner",
        )
        if event_picks.empty:
            continue

        station_x = event_picks["along_line_m"].to_numpy(dtype=float)
        station_t = event_picks["pick_time_s"].to_numpy(dtype=float)

        traveltime_curve = build_traveltime_curve(grid_x, station_x, station_t)
        fig_data.append((event_id, traveltime_curve))

        for station_id, pick_time in zip(event_picks["station_id"], event_picks["pick_time_s"]):
            pick_rows.append(
                {
                    "event_id": event_id,
                    "station_id": int(station_id),
                    "pick_time_s": float(pick_time),
                }
            )

        for idx, (x, t) in enumerate(zip(grid_x, traveltime_curve)):
            curve_rows.append(
                {
                    "event_id": event_id,
                    "grid_index": idx,
                    "along_line_m": x,
                    "x_rel_m": x - dist_min,
                    "travel_time_s": t,
                }
            )

    pd.DataFrame(pick_rows).to_csv(args.out_dir / "selected_event_picks_9stations.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(args.out_dir / "traveltime_curves_64.csv", index=False)

    summary = {
        "grid_size": grid_n,
        "depth_max_m": args.depth_max,
        "dist_min_m": dist_min,
        "dist_max_m": dist_max,
        "dx_m": dx,
        "dz_m": dz,
        "event_ids": event_df["event_id"].tolist(),
        "station_count": int(len(stations_win)),
        "station_subset": args.station_subset,
        "station_subset_ids": station_subset_ids,
    }
    (args.out_dir / "grid64_summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for event_id, curve in fig_data:
        ax.plot(grid_x - dist_min, curve, label=f"Event {event_id}")

    ax.set_xlabel("Distance from top-left corner (m)")
    ax.set_ylabel("Travel time (s)")
    ax.set_title("Observed P-Arrival Traveltimes (64 points)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "traveltime_curves_64.png", dpi=200)


if __name__ == "__main__":
    main()

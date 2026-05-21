#!/usr/bin/env python3
"""Select a dense 2D profile and extract first-arrival picks.

Defaults are aligned to the Rodriguez-Pradilla demo files in this repo.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.io as sio

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATIONS = REPO_ROOT / "Rodriguez-Pradilla" / "LinEpiLoc_ToC2ME_Demo" / "ToC2ME_Demo_recloc.txt"
DEFAULT_SLOC = REPO_ROOT / "Rodriguez-Pradilla" / "LinEpiLoc_ToC2ME_Demo" / "ToC2ME_Demo_sloc.txt"
DEFAULT_PP = REPO_ROOT / "Rodriguez-Pradilla" / "LinEpiLoc_ToC2ME_Demo" / "ToC2ME_Demo_PP_ReP.txt"
DEFAULT_SP = REPO_ROOT / "Rodriguez-Pradilla" / "LinEpiLoc_ToC2ME_Demo" / "ToC2ME_Demo_SP_ReP.txt"
DEFAULT_CATALOG = REPO_ROOT / "Rodriguez-Pradilla" / "Catalog_Rodriguez-Pradilla2019_PhDThesis.csv"
DEFAULT_VELOCITY_MODEL = REPO_ROOT / "data" / "ToC2MEVelModel.mat"

SEG2_DT_RE = re.compile(r"\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})")


@dataclass(frozen=True)
class Line2D:
    azimuth_deg: float
    normal_offset: float
    line_point: np.ndarray  # shape (2,)
    direction: np.ndarray   # unit vector shape (2,)
    normal: np.ndarray      # unit vector shape (2,)


def parse_seg2_datetime(seg2_name: str) -> Optional[datetime]:
    match = SEG2_DT_RE.search(seg2_name)
    if not match:
        return None
    yy, mm, dd, hh, mi, ss = (int(g) for g in match.groups())
    return datetime(2000 + yy, mm, dd, hh, mi, ss)


def load_stations(path: Path) -> pd.DataFrame:
    stations = pd.read_csv(path)
    stations = stations.rename(
        columns={
            "StationName": "station_id",
            "NAD83_X_m": "x_m",
            "NAD83_Y_m": "y_m",
            "z_m": "z_m",
            "DrillDepth_m": "drill_depth_m",
            "LoadedDepth_m": "loaded_depth_m",
        }
    )
    stations["station_id"] = stations["station_id"].astype(int)
    return stations


def load_sloc(path: Path) -> pd.DataFrame:
    sloc = pd.read_csv(path)
    sloc["event_time"] = sloc["Seg2File"].apply(parse_seg2_datetime)
    sloc["event_id"] = np.arange(len(sloc))
    return sloc


def load_picks(path: Path) -> pd.DataFrame:
    picks = pd.read_csv(path, header=None)
    picks = picks.dropna(axis=1, how="all")
    return picks


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(path)
    catalog["event_time"] = pd.to_datetime(
        catalog["Date"].astype(str) + " " + catalog["Time"].astype(str),
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce",
    )
    return catalog


def merge_events_with_catalog(sloc: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    events = sloc.merge(catalog, on="event_time", how="left")
    cat_cols = [
        "Local East (m)",
        "Local North (m)",
        "Elevation (m.a.s.l.)",
        "Mw",
        "Average P arrival (s)",
        "Average S arrival (s)",
    ]
    missing = events["Local East (m)"].isna()
    if missing.any():
        catalog_sorted = catalog.sort_values("event_time")
        to_fill = events.loc[missing, ["event_time"]].sort_values("event_time")
        filled = pd.merge_asof(
            to_fill,
            catalog_sorted,
            on="event_time",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=1),
        )
        for col in cat_cols:
            if col in filled.columns:
                events.loc[missing, col] = filled[col].values
    return events


def add_event_depth(events: pd.DataFrame, reference_elev_m: float) -> pd.DataFrame:
    depth = None
    if "Depth (m)" in events.columns:
        depth = events["Depth (m)"].astype(float)
    elif "TVD (m)" in events.columns:
        depth = events["TVD (m)"].astype(float)
    elif "Elevation (m.a.s.l.)" in events.columns:
        depth = reference_elev_m - events["Elevation (m.a.s.l.)"].astype(float)
    if depth is None:
        events["depth_m"] = np.nan
    else:
        events["depth_m"] = depth
    return events


def load_velocity_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(path)
    z = mat.get("z")
    vp = mat.get("vp")
    if z is None or vp is None:
        raise ValueError("Velocity model missing 'z' or 'vp' arrays")
    return z.squeeze().astype(float), vp.squeeze().astype(float)


def unit_vectors(azimuth_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(azimuth_deg)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    return direction, normal


def best_line_for_azimuth(points: np.ndarray, azimuth_deg: float, half_width_m: float) -> tuple[Line2D, int]:
    direction, normal = unit_vectors(azimuth_deg)
    perp = points @ normal
    perp_sorted = np.sort(perp)

    left = 0
    best_count = 0
    best_center = perp_sorted[0]
    for right in range(len(perp_sorted)):
        while perp_sorted[right] - perp_sorted[left] > 2 * half_width_m:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_center = 0.5 * (perp_sorted[right] + perp_sorted[left])

    line_point = best_center * normal
    line = Line2D(
        azimuth_deg=azimuth_deg,
        normal_offset=best_center,
        line_point=line_point,
        direction=direction,
        normal=normal,
    )
    return line, best_count


def distance_to_line(points: np.ndarray, line: Line2D) -> np.ndarray:
    return np.abs(points @ line.normal - line.normal_offset)


def along_line_coordinate(points: np.ndarray, line: Line2D) -> np.ndarray:
    return points @ line.direction


def select_best_line(
    stations_xy: np.ndarray,
    events_xy: Optional[np.ndarray],
    half_width_m: float,
    azimuth_step_deg: float,
) -> tuple[Line2D, pd.DataFrame]:
    azimuths = np.arange(0.0, 180.0 + 0.5 * azimuth_step_deg, azimuth_step_deg)
    rows = []
    best = None
    best_count = -1
    for az in azimuths:
        line, station_count = best_line_for_azimuth(stations_xy, az, half_width_m)
        event_count = 0
        if events_xy is not None:
            event_count = int((distance_to_line(events_xy, line) <= half_width_m).sum())
        rows.append(
            {
                "azimuth_deg": az,
                "station_count": station_count,
                "event_count": event_count,
                "normal_offset": line.normal_offset,
            }
        )
        if station_count > best_count:
            best = line
            best_count = station_count
    summary = pd.DataFrame(rows).sort_values(["station_count", "event_count"], ascending=False)
    return best, summary


def estimate_velocity(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    event_ids: Iterable[int],
    station_ids: Iterable[int],
    phase: str,
) -> pd.DataFrame:
    results = []
    station_ids = list(station_ids)
    col_to_pos = {sid - 1: pos for pos, sid in enumerate(station_ids)}
    station_xy = stations.set_index("station_id").loc[station_ids, ["x_m", "y_m"]].to_numpy(dtype=float)

    for event_id in event_ids:
        event_row = events.loc[events["event_id"] == event_id]
        if event_row.empty:
            continue
        event_xy = event_row[["Local East (m)", "Local North (m)"]].iloc[0].to_numpy(dtype=float)
        if np.isnan(event_xy).any():
            continue

        row = picks.loc[event_id, [sid - 1 for sid in station_ids]]
        picks_series = row.replace(0, np.nan).dropna()
        if len(picks_series) < 2:
            continue

        station_idx = [col_to_pos[int(idx)] for idx in picks_series.index]
        offsets = np.linalg.norm(station_xy[station_idx] - event_xy, axis=1)
        t2 = np.square(picks_series.to_numpy(dtype=float))
        r2 = np.square(offsets)
        slope, intercept = np.polyfit(r2, t2, 1)
        if slope <= 0:
            continue

        velocity = 1.0 / math.sqrt(slope)
        results.append(
            {
                "event_id": event_id,
                "phase": phase,
                "velocity_m_s": velocity,
                "n_picks": len(picks_series),
                "t0_s": math.sqrt(max(intercept, 0.0)),
            }
        )

    return pd.DataFrame(results)


def build_pick_table(
    picks: pd.DataFrame,
    events: pd.DataFrame,
    stations: pd.DataFrame,
    event_ids: Iterable[int],
    station_ids: Iterable[int],
    phase: str,
) -> pd.DataFrame:
    station_ids = list(station_ids)
    station_idx = np.array([sid - 1 for sid in station_ids], dtype=int)
    event_subset = picks.loc[event_ids, station_idx]
    event_subset.index.name = "event_id"
    event_subset.columns = station_ids
    event_subset = event_subset.replace(0, np.nan)

    rows = []
    for event_id, row in event_subset.iterrows():
        event_time = events.loc[events["event_id"] == event_id, "event_time"].iloc[0]
        for station_id, pick_time in row.dropna().items():
            rows.append(
                {
                    "event_id": event_id,
                    "event_time": event_time,
                    "station_id": station_id,
                    "phase": phase,
                    "pick_time_s": pick_time,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a dense profile and extract picks.")
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--sloc", type=Path, default=DEFAULT_SLOC)
    parser.add_argument("--pp", type=Path, default=DEFAULT_PP)
    parser.add_argument("--sp", type=Path, default=DEFAULT_SP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--velocity-mat", type=Path, default=DEFAULT_VELOCITY_MODEL)
    parser.add_argument("--ref-elev", type=float, default=898.5, help="Reference elevation (m a.s.l.)")
    parser.add_argument("--corridor", type=float, default=500.0, help="Half-width in meters")
    parser.add_argument("--az-step", type=float, default=5.0, help="Azimuth step in degrees")
    parser.add_argument("--min-picks", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=30)
    parser.add_argument("--phase", choices=["P", "both"], default="P")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    stations = load_stations(args.stations)
    sloc = load_sloc(args.sloc)
    pp = load_picks(args.pp)
    sp = load_picks(args.sp)
    catalog = load_catalog(args.catalog)
    events = merge_events_with_catalog(sloc, catalog)
    events = add_event_depth(events, args.ref_elev)

    if pp.shape != sp.shape:
        raise ValueError("PP and SP picks have different shapes")
    if pp.shape[1] != len(stations):
        raise ValueError("Pick columns do not match number of stations")

    stations_xy = stations[["x_m", "y_m"]].to_numpy(dtype=float)
    events_xy = events[["Local East (m)", "Local North (m)"]].to_numpy(dtype=float)
    events_xy = events_xy[~np.isnan(events_xy).any(axis=1)] if len(events) else None

    best_line, line_table = select_best_line(stations_xy, events_xy, args.corridor, args.az_step)

    station_dist = distance_to_line(stations_xy, best_line)
    stations["dist_to_line_m"] = station_dist
    stations["along_line_m"] = along_line_coordinate(stations_xy, best_line)
    selected_stations = stations[stations["dist_to_line_m"] <= args.corridor]

    event_dist = distance_to_line(events[["Local East (m)", "Local North (m)"]].to_numpy(), best_line)
    events["dist_to_line_m"] = event_dist
    events["along_line_m"] = along_line_coordinate(
        events[["Local East (m)", "Local North (m)"]].to_numpy(), best_line
    )
    events_near = events[(events["dist_to_line_m"] <= args.corridor) & events["event_time"].notna()]

    station_ids = selected_stations["station_id"].tolist()
    station_idx = [sid - 1 for sid in station_ids]
    pp_near = pp.loc[events_near["event_id"], station_idx]
    pick_counts = (pp_near > 0).sum(axis=1)
    events_near = events_near.assign(pick_count=pick_counts.values)
    events_near = events_near[events_near["pick_count"] >= args.min_picks]
    events_near = events_near.sort_values("pick_count", ascending=False).head(args.max_events)

    selected_event_ids = events_near["event_id"].tolist()

    picks_pp = build_pick_table(pp, events, stations, selected_event_ids, station_ids, "P")
    picks_sp = pd.DataFrame()
    if args.phase == "both":
        picks_sp = build_pick_table(sp, events, stations, selected_event_ids, station_ids, "S")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    line_table.to_csv(args.out_dir / "profile_candidates.csv", index=False)
    selected_stations.to_csv(args.out_dir / "selected_stations.csv", index=False)
    events_near.to_csv(args.out_dir / "selected_events.csv", index=False)
    picks_pp.to_csv(args.out_dir / "selected_picks_pp.csv", index=False)
    if args.phase == "both" and not picks_sp.empty:
        picks_sp.to_csv(args.out_dir / "selected_picks_sp.csv", index=False)

    summary = {
        "azimuth_deg": best_line.azimuth_deg,
        "normal_offset": best_line.normal_offset,
        "corridor_half_width_m": args.corridor,
        "selected_station_count": int(len(selected_stations)),
        "selected_event_count": int(len(events_near)),
        "phase": args.phase,
        "profile_along_min_m": float(selected_stations["along_line_m"].min()),
        "profile_along_max_m": float(selected_stations["along_line_m"].max()),
    }
    (args.out_dir / "profile_summary.json").write_text(json.dumps(summary, indent=2))

    velocity_p = estimate_velocity(pp, events, stations, selected_event_ids, station_ids, "P")
    velocity_all = velocity_p
    if args.phase == "both":
        velocity_s = estimate_velocity(sp, events, stations, selected_event_ids, station_ids, "S")
        velocity_all = pd.concat([velocity_p, velocity_s], ignore_index=True)
    if not velocity_all.empty:
        velocity_all.to_csv(args.out_dir / "velocity_estimates.csv", index=False)
        summary_vel = velocity_all.groupby("phase")["velocity_m_s"].median().to_dict()
        (args.out_dir / "velocity_summary.json").write_text(json.dumps(summary_vel, indent=2))

    if args.velocity_mat.exists():
        z, vp = load_velocity_model(args.velocity_mat)
        model_df = pd.DataFrame({"depth_m": z, "vp_m_s": vp})
        model_df.to_csv(args.out_dir / "velocity_model_vp.csv", index=False)

        event_depths = events_near[["event_id", "depth_m"]].dropna()
        if not event_depths.empty:
            vp_samples = np.interp(event_depths["depth_m"], z, vp, left=np.nan, right=np.nan)
            sample_df = event_depths.copy()
            sample_df["vp_m_s"] = vp_samples
            sample_df.to_csv(args.out_dir / "event_vp_samples.csv", index=False)

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(stations["x_m"], stations["y_m"], s=18, c="#999999", label="stations")
        ax.scatter(
            selected_stations["x_m"],
            selected_stations["y_m"],
            s=24,
            c="#2b8cbe",
            label="selected stations",
        )
        ax.scatter(
            events_near["Local East (m)"],
            events_near["Local North (m)"],
            s=14,
            c="#e34a33",
            label="selected events",
        )
        line_dir = best_line.direction
        line_point = best_line.line_point
        proj = along_line_coordinate(stations_xy, best_line)
        t_min, t_max = proj.min(), proj.max()
        base_t = float(line_point @ line_dir)
        line_pts = np.vstack(
            [
                line_point + (t_min - base_t) * line_dir,
                line_point + (t_max - base_t) * line_dir,
            ]
        )
        ax.plot(line_pts[:, 0], line_pts[:, 1], color="#222222", linewidth=1.5, label="profile")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(args.out_dir / "profile_map.png", dpi=200)


if __name__ == "__main__":
    main()

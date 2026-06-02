#!/usr/bin/env python3
"""Plot top view of profile with stations and events."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_STATIONS = DEFAULT_OUTPUTS / "selected_stations.csv"
DEFAULT_EVENTS = DEFAULT_OUTPUTS / "selected_events.csv"
DEFAULT_PICKS = DEFAULT_OUTPUTS / "selected_event_picks_9stations.csv"
DEFAULT_INFERENCE = DEFAULT_OUTPUTS / "event_locations_64.csv"
DEFAULT_PROFILE = DEFAULT_OUTPUTS / "profile_summary.json"
DEFAULT_FIG = DEFAULT_OUTPUTS / "profile_top_view.png"
DEFAULT_EAST_FIG = DEFAULT_OUTPUTS / "profile_top_view_east5.png"


def unit_vectors(azimuth_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(azimuth_deg)
    direction = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    return direction, normal


def build_profile_line(
    azimuth_deg: float,
    normal_offset: float,
    points_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direction, normal = unit_vectors(azimuth_deg)
    line_point = normal_offset * normal
    proj = points_xy @ direction
    base_t = float(line_point @ direction)
    t_min = float(np.min(proj))
    t_max = float(np.max(proj))
    line_pts = np.vstack(
        [
            line_point + (t_min - base_t) * direction,
            line_point + (t_max - base_t) * direction,
        ]
    )
    return line_pts, direction, normal, line_point


def build_line_segment(
    center: np.ndarray,
    direction: np.ndarray,
    length_m: float,
) -> np.ndarray:
    half = 0.5 * length_m
    return np.vstack([center - half * direction, center + half * direction])


def fit_line_pca(
    points: np.ndarray,
    fallback_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    if len(points) < 2:
        direction = fallback_direction
    else:
        centered = points - center
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, np.argmax(eigvals)]
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    return center, direction, normal


def select_east_stations(
    stations: pd.DataFrame,
    normal: np.ndarray,
    normal_offset: float,
    count: int,
) -> Tuple[pd.DataFrame, np.ndarray]:
    if stations.empty:
        return stations.copy(), np.array([], dtype=float)
    points = stations[["x_m", "y_m"]].to_numpy(dtype=float)
    signed = points @ normal - normal_offset
    pos_mask = signed > 0.0
    neg_mask = signed < 0.0
    if pos_mask.any() and neg_mask.any():
        mean_pos = float(stations.loc[pos_mask, "x_m"].mean())
        mean_neg = float(stations.loc[neg_mask, "x_m"].mean())
        east_is_pos = mean_pos >= mean_neg
    elif pos_mask.any():
        east_is_pos = True
    elif neg_mask.any():
        east_is_pos = False
    else:
        return stations.copy(), signed

    mask = pos_mask if east_is_pos else neg_mask
    if not mask.any():
        candidate = np.arange(len(stations))
        dist = np.abs(signed)
    else:
        candidate = np.where(mask)[0]
        dist = signed if east_is_pos else -signed
    order = np.argsort(dist[candidate])[::-1]
    chosen = candidate[order[:count]]
    return stations.iloc[chosen].copy(), signed[chosen]


def distance_to_line(points: np.ndarray, line_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return np.abs((points - line_point) @ normal)


def choose_highlight_events(
    events: pd.DataFrame,
    inference_events: pd.DataFrame,
    east_stations: pd.DataFrame,
    line_point: np.ndarray,
    normal: np.ndarray,
    station_radius_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return (
            pd.DataFrame(columns=events.columns),
            pd.DataFrame(columns=events.columns),
            pd.DataFrame(columns=events.columns),
        )

    north_infer = pd.DataFrame(columns=events.columns)
    if not inference_events.empty:
        idx = inference_events["Local North (m)"].idxmax()
        north_infer = inference_events.loc[[idx]]

    if east_stations.empty:
        return north_infer, pd.DataFrame(columns=events.columns), pd.DataFrame(columns=events.columns)

    stations_sorted = east_stations.sort_values("along_line_m")
    station_ref = stations_sorted.iloc[len(stations_sorted) // 2]

    station_xy = station_ref[["x_m", "y_m"]].to_numpy(dtype=float)
    event_xy = events[["Local East (m)", "Local North (m)"]].to_numpy(dtype=float)
    dists = np.linalg.norm(event_xy - station_xy, axis=1)
    near_mask = dists <= station_radius_m
    near_events = events[near_mask].copy()
    if near_events.empty:
        near_events = events.copy()

    north_y = None
    if not north_infer.empty:
        north_y = float(north_infer.iloc[0]["Local North (m)"])
    elif not near_events.empty:
        idx = near_events["Local North (m)"].idxmax()
        north_infer = near_events.loc[[idx]]
        north_y = float(north_infer.iloc[0]["Local North (m)"])

    south_next = pd.DataFrame(columns=events.columns)
    if north_y is not None:
        south_candidates = near_events[near_events["Local North (m)"] < north_y]
        if not south_candidates.empty:
            idx = south_candidates["Local North (m)"].idxmax()
            south_next = south_candidates.loc[[idx]]

    south_cloud = near_events[near_events["Local North (m)"] < station_xy[1]].copy()
    if south_cloud.empty:
        south_cloud = near_events.copy()

    south_xy = south_cloud[["Local East (m)", "Local North (m)"]].to_numpy(dtype=float)
    dist_to_profile = distance_to_line(south_xy, line_point, normal)
    south_cloud = south_cloud.assign(dist_to_profile=dist_to_profile)

    taken_ids = set()
    if not north_infer.empty:
        taken_ids.update(north_infer["event_id"].tolist())
    if not south_next.empty:
        taken_ids.update(south_next["event_id"].tolist())

    south_cloud = south_cloud[~south_cloud["event_id"].isin(taken_ids)]
    south_near = pd.DataFrame(columns=events.columns)
    if not south_cloud.empty:
        idx = south_cloud["dist_to_profile"].idxmin()
        south_near = south_cloud.loc[[idx]].drop(columns=["dist_to_profile"])

    return north_infer, south_next, south_near


def get_points(events: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    pts = []
    if not events.empty:
        pts.append(events[["Local East (m)", "Local North (m)"]].to_numpy(dtype=float))
    if not stations.empty:
        pts.append(stations[["x_m", "y_m"]].to_numpy(dtype=float))
    if not pts:
        return np.zeros((0, 2), dtype=float)
    return np.vstack(pts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top view with stations and events.")
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--inference-events", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--profile-summary", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--profile-mode",
        choices=["summary", "right5", "east5"],
        default="summary",
        help="Use the profile summary or fit a line through east-side stations.",
    )
    parser.add_argument("--right-count", type=int, default=5)
    parser.add_argument("--line-length", type=float, default=3500.0)
    parser.add_argument("--corridor-half", type=float, default=None)
    parser.add_argument("--zoom-pad", type=float, default=400.0)
    parser.add_argument("--station-radius", type=float, default=800.0)
    parser.add_argument("--out-fig", type=Path, default=None)
    args = parser.parse_args()

    stations = pd.read_csv(args.stations)
    events = pd.read_csv(args.events)

    stations_9 = stations.copy()
    if args.picks.exists():
        picks = pd.read_csv(args.picks)
        station_ids = sorted(picks["station_id"].unique().tolist())
        stations_9 = stations[stations["station_id"].isin(station_ids)]

    inference_ids = []
    inference_events: Optional[pd.DataFrame] = None
    if args.inference_events.exists():
        infer = pd.read_csv(args.inference_events)
        if "event_id" in infer.columns:
            inference_ids = [int(v) for v in infer["event_id"].unique().tolist()]
    if inference_ids and "event_id" in events.columns:
        inference_events = events[events["event_id"].isin(inference_ids)]
    else:
        inference_events = pd.DataFrame(columns=events.columns)

    points_xy = get_points(events, stations)

    line_pts = None
    corridor_pts = None
    east_stations = pd.DataFrame(columns=stations.columns)
    line_point = None
    direction = None
    normal = None
    corridor_half = None
    if args.profile_summary.exists() and len(points_xy):
        summary = json.loads(args.profile_summary.read_text())
        azimuth = float(summary["azimuth_deg"])
        offset = float(summary["normal_offset"])
        corridor_half = float(summary["corridor_half_width_m"])
        line_pts, direction, normal, line_point = build_profile_line(azimuth, offset, points_xy)
        if args.profile_mode in ("right5", "east5"):
            east_stations, signed = select_east_stations(stations_9, normal, offset, args.right_count)
            if not east_stations.empty:
                east_points = east_stations[["x_m", "y_m"]].to_numpy(dtype=float)
                center, direction, normal = fit_line_pca(east_points, direction)
                line_pts = build_line_segment(center, direction, args.line_length)
                line_point = center

        if args.corridor_half is not None:
            corridor_half = args.corridor_half
        elif args.profile_mode in ("right5", "east5"):
            corridor_half = 200.0

        if line_pts is not None and normal is not None and corridor_half is not None:
            corridor_pts = (
                line_pts + corridor_half * normal,
                line_pts - corridor_half * normal,
            )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    if not events.empty:
        ax.scatter(
            events["Local East (m)"],
            events["Local North (m)"],
            s=12,
            c="#bdbdbd",
            label="selected events",
        )
    highlight_north = pd.DataFrame(columns=events.columns)
    highlight_next_south = pd.DataFrame(columns=events.columns)
    highlight_south = pd.DataFrame(columns=events.columns)
    if args.profile_mode == "east5" and line_point is not None and normal is not None:
        highlight_north, highlight_next_south, highlight_south = choose_highlight_events(
            events,
            inference_events,
            east_stations,
            line_point,
            normal,
            args.station_radius,
        )
    elif inference_events is not None and not inference_events.empty:
        highlight_north = inference_events.copy()
    if not stations.empty:
        ax.scatter(
            stations["x_m"],
            stations["y_m"],
            s=20,
            c="#9e9e9e",
            label="stations (selected)",
        )
    if not stations_9.empty:
        ax.scatter(
            stations_9["x_m"],
            stations_9["y_m"],
            s=40,
            c="#2b8cbe",
            marker="s",
            label="stations (9)",
            edgecolors="#0b3c5d",
            linewidths=0.5,
        )
    if not east_stations.empty:
        ax.scatter(
            east_stations["x_m"],
            east_stations["y_m"],
            s=70,
            c="#fdae61",
            marker="D",
            label="stations (east 5)",
            edgecolors="#7f3b08",
            linewidths=0.6,
        )

    if not highlight_north.empty:
        ax.scatter(
            highlight_north["Local East (m)"],
            highlight_north["Local North (m)"],
            s=110,
            c="#d7301f",
            marker="*",
            label="north inference event",
            edgecolors="#222222",
            linewidths=0.6,
        )
    if not highlight_next_south.empty:
        ax.scatter(
            highlight_next_south["Local East (m)"],
            highlight_next_south["Local North (m)"],
            s=110,
            c="#fdae61",
            marker="*",
            label="next south event",
            edgecolors="#222222",
            linewidths=0.6,
        )
    if not highlight_south.empty:
        ax.scatter(
            highlight_south["Local East (m)"],
            highlight_south["Local North (m)"],
            s=110,
            c="#2b8cbe",
            marker="*",
            label="south near profile",
            edgecolors="#222222",
            linewidths=0.6,
        )

    if line_pts is not None:
        ax.plot(line_pts[:, 0], line_pts[:, 1], color="#111111", linewidth=1.5, label="profile")
    if corridor_pts is not None:
        for pts in corridor_pts:
            ax.plot(pts[:, 0], pts[:, 1], color="#444444", linestyle="--", linewidth=1.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Top View: Profile, Stations, and Events")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if args.profile_mode == "east5" and line_pts is not None:
        x_min = float(np.min(line_pts[:, 0])) - args.zoom_pad
        x_max = float(np.max(line_pts[:, 0])) + args.zoom_pad
        y_min = float(np.min(line_pts[:, 1])) - args.zoom_pad
        y_max = float(np.max(line_pts[:, 1])) + args.zoom_pad
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    out_fig = args.out_fig
    if out_fig is None:
        if args.profile_mode in ("right5", "east5"):
            out_fig = DEFAULT_EAST_FIG
        else:
            out_fig = DEFAULT_FIG
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=200)


if __name__ == "__main__":
    main()

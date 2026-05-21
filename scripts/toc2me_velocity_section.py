#!/usr/bin/env python3
"""Generate a 2D P-velocity section (10 m spacing) and overlay stations/events."""
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
DEFAULT_PROFILE_SUMMARY = DEFAULT_OUTPUTS / "profile_summary.json"
DEFAULT_STATIONS = DEFAULT_OUTPUTS / "selected_stations.csv"
DEFAULT_EVENTS = DEFAULT_OUTPUTS / "selected_events.csv"
DEFAULT_VELOCITY_CSV = DEFAULT_OUTPUTS / "velocity_model_vp.csv"
DEFAULT_VELOCITY_MAT = REPO_ROOT / "data" / "ToC2MEVelModel.mat"
DEFAULT_FIG = DEFAULT_OUTPUTS / "velocity_section_2d.png"
DEFAULT_FIG_ZOOM = DEFAULT_OUTPUTS / "velocity_section_2d_zoomed.png"


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


def build_velocity_grid(
    depth_m: np.ndarray,
    vp_m_s: np.ndarray,
    depth_step: float,
    dist_min: float,
    dist_max: float,
    dist_step: float,
    max_depth: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_depth is None:
        max_depth = float(depth_m.max())
    depths = np.arange(0.0, max_depth + depth_step, depth_step)
    vp_depth = np.interp(depths, depth_m, vp_m_s)

    distances = np.arange(dist_min, dist_max + dist_step, dist_step)
    vp_grid = np.tile(vp_depth[:, None], (1, len(distances)))
    return distances, depths, vp_grid


def filter_events_for_zoom(events: pd.DataFrame, depth_max: float, exclude_deepest: int) -> pd.DataFrame:
    if "depth_m" not in events.columns:
        return events.copy()

    filtered = events.copy()
    filtered = filtered[filtered["depth_m"].notna()]
    if exclude_deepest > 0:
        filtered = filtered.sort_values("depth_m", ascending=False).iloc[exclude_deepest:]
    filtered = filtered[filtered["depth_m"] <= depth_max]
    return filtered


def choose_square_window(
    events: pd.DataFrame,
    stations: pd.DataFrame,
    width_m: float,
    dist_step: float,
) -> tuple[float, float]:
    min_event = float(events["along_line_m"].min())
    max_event = float(events["along_line_m"].max())

    start_min = max_event - width_m
    start_max = min_event
    if start_min > start_max:
        center = 0.5 * (min_event + max_event)
        start_min = start_max = center - 0.5 * width_m

    candidates = np.arange(start_min, start_max + dist_step, dist_step)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2D P-velocity section with overlays.")
    parser.add_argument("--profile-summary", type=Path, default=DEFAULT_PROFILE_SUMMARY)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--velocity-csv", type=Path, default=DEFAULT_VELOCITY_CSV)
    parser.add_argument("--velocity-mat", type=Path, default=DEFAULT_VELOCITY_MAT)
    parser.add_argument("--dist-step", type=float, default=10.0)
    parser.add_argument("--depth-step", type=float, default=10.0)
    parser.add_argument("--zoom", action="store_true")
    parser.add_argument("--depth-max", type=float, default=3500.0)
    parser.add_argument("--exclude-deepest", type=int, default=3)
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()

    if not args.stations.exists() or not args.events.exists():
        raise FileNotFoundError("Run toc2me_profile.py first to generate selected stations/events.")

    stations = pd.read_csv(args.stations)
    events = pd.read_csv(args.events)

    dist_min = None
    dist_max = None
    if args.profile_summary.exists():
        summary = json.loads(args.profile_summary.read_text())
        dist_min = float(summary.get("profile_along_min_m", np.nan))
        dist_max = float(summary.get("profile_along_max_m", np.nan))

    if dist_min is None or np.isnan(dist_min) or dist_max is None or np.isnan(dist_max):
        dist_min = float(min(stations["along_line_m"].min(), events["along_line_m"].min()))
        dist_max = float(max(stations["along_line_m"].max(), events["along_line_m"].max()))

    dist_min = np.floor(dist_min / args.dist_step) * args.dist_step
    dist_max = np.ceil(dist_max / args.dist_step) * args.dist_step

    plot_events = events
    if args.zoom:
        plot_events = filter_events_for_zoom(events, args.depth_max, args.exclude_deepest)
        dist_min, dist_max = choose_square_window(plot_events, stations, args.depth_max, args.dist_step)
        if args.out_fig == DEFAULT_FIG:
            args.out_fig = DEFAULT_FIG_ZOOM

    depth_m, vp_m_s = load_velocity_model(args.velocity_csv, args.velocity_mat)
    distances, depths, vp_grid = build_velocity_grid(
        depth_m,
        vp_m_s,
        args.depth_step,
        dist_min,
        dist_max,
        args.dist_step,
        max_depth=args.depth_max if args.zoom else None,
    )

    import matplotlib.pyplot as plt

    fig_size = (7, 7) if args.zoom else (10, 5)
    fig, ax = plt.subplots(figsize=fig_size)
    mesh = ax.pcolormesh(distances, depths, vp_grid, shading="auto", cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("Vp (m/s)")

    station_mask = (stations["along_line_m"] >= dist_min) & (stations["along_line_m"] <= dist_max)
    ax.scatter(
        stations.loc[station_mask, "along_line_m"],
        np.zeros(int(station_mask.sum())),
        s=20,
        c="#ffffff",
        edgecolors="#111111",
        label="Stations",
        zorder=3,
    )

    if "depth_m" in plot_events.columns:
        event_depths = plot_events["depth_m"].to_numpy(dtype=float)
        mask = ~np.isnan(event_depths)
        ax.scatter(
            plot_events.loc[mask, "along_line_m"],
            event_depths[mask],
            s=18,
            c="#e34a33",
            label="Events",
            zorder=4,
        )

    ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_ylim(depths.max(), 0)
    ax.set_xlim(dist_min, dist_max)
    if args.zoom:
        ax.set_aspect("equal", adjustable="box")
    ax.set_title("ToC2ME P-Velocity Section (10 m spacing)")
    ax.legend(loc="upper right")
    fig.tight_layout()

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=200)


if __name__ == "__main__":
    main()

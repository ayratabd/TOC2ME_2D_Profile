#!/usr/bin/env python3
"""Compare observed pick curves vs skfmm-generated surface curves."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_EVENTS = DEFAULT_OUTPUTS / "event_locations_64.csv"
DEFAULT_CURVES = DEFAULT_OUTPUTS / "traveltime_curves_64.csv"
DEFAULT_VELOCITY = DEFAULT_OUTPUTS / "velocity_model_64x64.npy"
DEFAULT_SUMMARY = DEFAULT_OUTPUTS / "grid64_summary.json"
DEFAULT_PICKS = DEFAULT_OUTPUTS / "selected_event_picks_9stations.csv"
DEFAULT_STATIONS = DEFAULT_OUTPUTS / "selected_stations.csv"
DEFAULT_FIG = DEFAULT_OUTPUTS / "compare_traveltime_curves.png"


def load_real_curves(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    curves = pd.read_csv(path)
    curves = curves.sort_values(["event_id", "grid_index"]).reset_index(drop=True)
    grid_x = curves["x_rel_m"].unique()
    real = {}
    for event_id, group in curves.groupby("event_id"):
        real[int(event_id)] = group["travel_time_s"].to_numpy(dtype=float)
    return grid_x, real


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot observed vs skfmm curves.")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--velocity", type=Path, default=DEFAULT_VELOCITY)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--picks", type=Path, default=DEFAULT_PICKS)
    parser.add_argument("--stations", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument(
        "--residual-spline",
        choices=["none", "cubic", "pchip"],
        default="pchip",
        help="Fit a residual spline to picks on top of skfmm curves.",
    )
    parser.add_argument(
        "--out-curves",
        type=Path,
        default=None,
        help="Write selected curves to CSV.",
    )
    parser.add_argument(
        "--out-curves-source",
        choices=["residual", "skfmm", "picks"],
        default="residual",
        help="Source curves to write when --out-curves is set.",
    )
    parser.add_argument("--out-fig", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()

    events = pd.read_csv(args.events)
    grid_x, real_curves = load_real_curves(args.curves)
    velocity = np.load(args.velocity)
    summary = json.loads(args.summary.read_text())
    dx_m = float(summary["dx_m"])
    dist_min = float(summary["dist_min_m"])

    try:
        import skfmm
    except Exception as exc:
        raise RuntimeError("skfmm is required for this comparison") from exc

    skfmm_curves = {}
    skfmm_curves_raw = {}
    for _, row in events.iterrows():
        event_id = int(row["event_id"])
        x_idx = int(row["x_idx"])
        z_idx = int(row["z_idx"])
        mask = np.ones_like(velocity, dtype=float)
        mask[z_idx, x_idx] = -1.0
        tt = skfmm.travel_time(mask, speed=velocity, dx=dx_m)
        surface_raw = tt[0, :].astype(float)
        surface = surface_raw - np.min(surface_raw)
        skfmm_curves[event_id] = surface
        skfmm_curves_raw[event_id] = surface_raw

    residual_curves = {}
    residual_curves_raw = {}
    if args.residual_spline != "none":
        picks = pd.read_csv(args.picks)
        stations = pd.read_csv(args.stations)[["station_id", "along_line_m"]]
        picks = picks.merge(stations, on="station_id", how="left")
        picks["x_rel_m"] = picks["along_line_m"] - dist_min

        if args.residual_spline == "cubic":
            from scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import PchipInterpolator

        # Fit residuals to force exact pick matches while following skfmm shape.
        for event_id in events["event_id"].tolist():
            if event_id not in skfmm_curves_raw:
                continue
            event_picks = picks[picks["event_id"] == event_id].dropna(subset=["x_rel_m", "pick_time_s"])
            if event_picks.empty:
                continue
            x_pick = event_picks["x_rel_m"].to_numpy(dtype=float)
            t_pick = event_picks["pick_time_s"].to_numpy(dtype=float)
            t_fmm = np.interp(x_pick, grid_x, skfmm_curves_raw[event_id])
            residual = t_pick - t_fmm

            residual_df = pd.DataFrame({"x": x_pick, "residual": residual})
            residual_df = residual_df.groupby("x", as_index=False).mean().sort_values("x")
            if len(residual_df) < 2:
                continue
            x_res = residual_df["x"].to_numpy(dtype=float)
            r_res = residual_df["residual"].to_numpy(dtype=float)

            if args.residual_spline == "cubic":
                spline = CubicSpline(x_res, r_res, bc_type="natural")
                r_curve = spline(grid_x)
            else:
                r_curve = PchipInterpolator(x_res, r_res)(grid_x)

            corrected = skfmm_curves_raw[event_id] + r_curve
            residual_curves_raw[event_id] = corrected
            residual_curves[event_id] = corrected - np.min(corrected)

    if args.out_curves is not None:
        if args.out_curves_source == "residual":
            if args.residual_spline == "none" or not residual_curves_raw:
                raise ValueError("Residual spline must be enabled to write residual curves")
            curve_map = residual_curves_raw
        elif args.out_curves_source == "picks":
            curve_map = real_curves
        else:
            curve_map = skfmm_curves_raw

        out_rows = []
        for event_id in events["event_id"].tolist():
            event_id = int(event_id)
            if event_id not in curve_map:
                continue
            curve = curve_map[event_id]

            for idx, (x_rel, t) in enumerate(zip(grid_x, curve)):
                out_rows.append(
                    {
                        "event_id": event_id,
                        "grid_index": idx,
                        "along_line_m": float(x_rel + dist_min),
                        "x_rel_m": float(x_rel),
                        "travel_time_s": float(t),
                    }
                )

        args.out_curves.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(args.out_curves, index=False)

    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, event_id in enumerate(events["event_id"].tolist()):
        color = colors[idx % len(colors)]
        if event_id in real_curves:
            real = real_curves[event_id].astype(float)
            real = real - np.min(real)
            ax.plot(grid_x, real, color=color, linestyle="--", label=f"Event {event_id} picks")
        if event_id in skfmm_curves:
            ax.plot(grid_x, skfmm_curves[event_id], color=color, linestyle="-", label=f"Event {event_id} skfmm")
        if event_id in residual_curves:
            ax.plot(
                grid_x,
                residual_curves[event_id],
                color=color,
                linestyle=":",
                label=f"Event {event_id} residual spline",
            )

    ax.set_xlabel("Distance from top-left corner (m)")
    ax.set_ylabel("Travel time (s), min-subtracted")
    ax.set_title("Observed vs skfmm Surface Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=200)


if __name__ == "__main__":
    main()

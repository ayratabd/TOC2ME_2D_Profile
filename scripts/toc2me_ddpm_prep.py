#!/usr/bin/env python3
"""Prepare 64x64 traveltime grids and velocity channel for DDPM input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = REPO_ROOT / "outputs"
DEFAULT_CURVES = DEFAULT_OUTPUTS / "traveltime_curves_64.csv"
DEFAULT_VELOCITY = DEFAULT_OUTPUTS / "velocity_model_64x64.npy"


def load_curves(path: Path) -> Tuple[List[int], np.ndarray, np.ndarray]:
    curves = pd.read_csv(path)
    curves = curves.sort_values(["event_id", "grid_index"]).reset_index(drop=True)
    event_ids = curves["event_id"].unique().tolist()
    grid_x = curves["x_rel_m"].unique()
    if len(grid_x) != 64:
        raise ValueError("Expected 64 grid points in curves")

    rows = []
    for event_id in event_ids:
        subset = curves[curves["event_id"] == event_id]
        if len(subset) != 64:
            raise ValueError(f"Event {event_id} does not have 64 points")
        rows.append(subset["travel_time_s"].to_numpy(dtype=float))

    return event_ids, grid_x, np.vstack(rows)


def normalize_min0(curves: np.ndarray) -> np.ndarray:
    mins = curves.min(axis=1, keepdims=True)
    return curves - mins


def make_grids(curves: np.ndarray, depth_samples: int) -> np.ndarray:
    grids = []
    for curve in curves:
        grids.append(np.tile(curve[None, :], (depth_samples, 1)))
    return np.stack(grids, axis=0)


def normalize_tt(curves: np.ndarray, tt_min: float, tt_max: float) -> np.ndarray:
    curves = np.clip(curves, tt_min, tt_max)
    return 2.0 * (curves - tt_min) / (tt_max - tt_min) - 1.0


def normalize_velocity(
    velocity: np.ndarray,
    method: str,
    train_min: float,
    train_max: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if method == "train-clip":
        clipped = np.clip(velocity, train_min, train_max)
        norm = 2.0 * (clipped - train_min) / (train_max - train_min) - 1.0
        info = {"v_min": train_min, "v_max": train_max, "method": method}
        return norm, info

    if method == "data-minmax":
        v_min = float(np.min(velocity))
        v_max = float(np.max(velocity))
        norm = 2.0 * (velocity - v_min) / (v_max - v_min) - 1.0
        info = {"v_min": v_min, "v_max": v_max, "method": method}
        return norm, info

    raise ValueError("Unknown velocity normalization method")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DDPM inputs from 64-point curves.")
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--velocity", type=Path, default=DEFAULT_VELOCITY)
    parser.add_argument("--tt-min", type=float, default=0.0)
    parser.add_argument("--tt-max", type=float, default=0.66)
    parser.add_argument("--vel-norm", choices=["train-clip", "data-minmax"], default="train-clip")
    parser.add_argument("--train-vmin", type=float, default=1500.0)
    parser.add_argument("--train-vmax", type=float, default=4500.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    event_ids, grid_x, curves = load_curves(args.curves)
    curves_min0 = normalize_min0(curves)

    depth_samples = 64
    tt_grids = make_grids(curves_min0, depth_samples)

    velocity = np.load(args.velocity)
    if velocity.shape != (64, 64):
        raise ValueError("Velocity grid must be 64x64")

    physical_stack = np.concatenate([tt_grids, velocity[None, :, :]], axis=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "stack_physical_4x64x64.npy", physical_stack)

    tt_norm = normalize_tt(curves_min0, args.tt_min, args.tt_max)
    tt_norm_grids = make_grids(tt_norm, depth_samples)

    vel_norm, vel_info = normalize_velocity(velocity, args.vel_norm, args.train_vmin, args.train_vmax)
    norm_stack = np.concatenate([tt_norm_grids, vel_norm[None, :, :]], axis=0)
    np.save(args.out_dir / "stack_normalized_4x64x64.npy", norm_stack)

    meta = {
        "event_ids": event_ids,
        "tt_min": args.tt_min,
        "tt_max": args.tt_max,
        "velocity_norm": vel_info,
        "grid_x_min": float(np.min(grid_x)),
        "grid_x_max": float(np.max(grid_x)),
    }
    (args.out_dir / "ddpm_stack_metadata.json").write_text(json.dumps(meta, indent=2))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for event_id, curve in zip(event_ids, curves_min0):
        ax.plot(grid_x, curve, label=f"Event {event_id}")

    ax.set_xlabel("Distance from top-left corner (m)")
    ax.set_ylabel("Travel time (s), min-subtracted")
    ax.set_title("Min-Subtracted P-Arrival Curves (64 points)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "traveltime_curves_64_min0.png", dpi=200)


if __name__ == "__main__":
    main()

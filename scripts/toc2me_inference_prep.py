#!/usr/bin/env python3
"""Prepare inference tensors using observed traveltime curves instead of skfmm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


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


def normalize_minmax(arr: np.ndarray, vmin: float, vmax: float, clip: bool = True) -> np.ndarray:
    if clip:
        arr = np.clip(arr, vmin, vmax)
    return 2.0 * (arr - vmin) / (vmax - vmin) - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inference tensors from observed traveltimes.")
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--velocity", type=Path, default=DEFAULT_VELOCITY)
    parser.add_argument("--tt-min", type=float, default=0.0)
    parser.add_argument("--tt-max", type=float, default=0.66)
    parser.add_argument("--train-vmin", type=float, default=1500.0)
    parser.add_argument("--train-vmax", type=float, default=4500.0)
    parser.add_argument("--vel-norm", choices=["train-clip", "data-minmax"], default="train-clip")
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUTS)
    args = parser.parse_args()

    event_ids, grid_x, curves = load_curves(args.curves)
    curves_min0 = curves - curves.min(axis=1, keepdims=True)

    depth_samples = 64
    obs_tt_maps = np.tile(curves_min0[:, None, :], (1, depth_samples, 1))
    gt_tt_maps = obs_tt_maps.copy()

    velocity = np.load(args.velocity)
    if velocity.shape != (64, 64):
        raise ValueError("Velocity grid must be 64x64")
    smooth_velocity = gaussian_filter(velocity, sigma=args.smooth_sigma)

    norm_gt_tt = normalize_minmax(gt_tt_maps, args.tt_min, args.tt_max, clip=True)
    norm_obs_tt = normalize_minmax(obs_tt_maps, args.tt_min, args.tt_max, clip=True)

    if args.vel_norm == "train-clip":
        norm_vel = normalize_minmax(velocity, args.train_vmin, args.train_vmax, clip=True)
        norm_smooth = normalize_minmax(smooth_velocity, args.train_vmin, args.train_vmax, clip=True)
        vel_info = {"method": "train-clip", "vmin": args.train_vmin, "vmax": args.train_vmax}
    else:
        vmin = float(velocity.min())
        vmax = float(velocity.max())
        norm_vel = normalize_minmax(velocity, vmin, vmax, clip=False)
        norm_smooth = normalize_minmax(smooth_velocity, vmin, vmax, clip=False)
        vel_info = {"method": "data-minmax", "vmin": vmin, "vmax": vmax}

    c0 = np.concatenate([norm_gt_tt, norm_vel[None, :, :]], axis=0)
    c1 = np.concatenate([norm_obs_tt, norm_smooth[None, :, :]], axis=0)
    cond = np.concatenate([norm_obs_tt, norm_smooth[None, :, :]], axis=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "inference_c0.npy", c0)
    np.save(args.out_dir / "inference_c1.npy", c1)
    np.save(args.out_dir / "inference_cond.npy", cond)

    physical_stack = np.concatenate([gt_tt_maps, velocity[None, :, :]], axis=0)
    np.save(args.out_dir / "inference_stack_physical_4x64x64.npy", physical_stack)

    normalized_stack = np.concatenate([norm_gt_tt, norm_vel[None, :, :]], axis=0)
    np.save(args.out_dir / "inference_stack_normalized_4x64x64.npy", normalized_stack)

    meta = {
        "event_ids": event_ids,
        "tt_min": args.tt_min,
        "tt_max": args.tt_max,
        "velocity_norm": vel_info,
        "smooth_sigma": args.smooth_sigma,
    }
    (args.out_dir / "inference_metadata.json").write_text(json.dumps(meta, indent=2))

    try:
        import torch

        sample = {
            "c0": torch.from_numpy(c0).float(),
            "c1": torch.from_numpy(c1).float(),
            "cond": torch.from_numpy(cond).float(),
            "event_ids": event_ids,
        }
        torch.save(sample, args.out_dir / "inference_sample.pt")
    except Exception:
        pass

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for event_id, curve in zip(event_ids, curves_min0):
        ax.plot(grid_x, curve, label=f"Event {event_id}")
    ax.set_xlabel("Distance from top-left corner (m)")
    ax.set_ylabel("Travel time (s), min-subtracted")
    ax.set_title("Observed P-Arrival Curves (min-subtracted)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out_dir / "inference_tt_curves_min0.png", dpi=200)


if __name__ == "__main__":
    main()

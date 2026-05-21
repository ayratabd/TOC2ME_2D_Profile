# import os
import torch
import numpy as np
import skfmm
import json
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

# Import the key-value map from openfwi
from .openfwi import FWIDataset, build_lmdb_dataset, MinMaxNormalize, _DATASET_KV_MAP

from pathlib import Path
from scipy.interpolate import interp1d



class MicroseismicDataset(FWIDataset):
    
    def __init__(self, sample_list, lmdb_data, opt, return_raw_for_debug=False):
        
        super().__init__(sample_list, lmdb_data)
        
        self.image_size = opt.image_size
        self._return_raw_for_debug = return_raw_for_debug

        # Fixed physical normalizations (same as before)
        self.VELOCITY_MIN = 1500.0  # m/s
        self.VELOCITY_MAX = 4500.0  # m/s
        self.model_normalizer = MinMaxNormalize(self.VELOCITY_MIN, self.VELOCITY_MAX)

        self.TT_MIN = 0.0
#         self.TT_MAX = 1.0  # keep margin to accommodate Δt shift
        self.TT_MAX = 0.66 # = 700 * sqrt(2) / 1500
        self.tt_normalizer = MinMaxNormalize(self.TT_MIN, self.TT_MAX)

        # Geometry
        self.dx_m = getattr(opt, "dx_meters", 10.9375)

        # Event realism knobs
        self.num_events = getattr(opt, "num_events", 3)
        
        # Per-event origin-time shift (seconds): tweak range if needed
        self.origin_time_shift_range = getattr(opt, "origin_time_shift_range", (0.05, 0.5))

        # Optional extra realism: add mild bandlimit to observed TT
        self.blur_sigma_px = getattr(opt, "obs_tt_blur_sigma", 0.0)  # 0 disables
        
        # For deterministic per-sample randomness
        self.base_seed = getattr(opt, "microseis_seed", 12345)

    def __getitem__(self, idx):
        
        # ---- Load & shape velocity -------------------------------------------------
        velocity_model = super().__getitem__(idx)
        if isinstance(velocity_model, torch.Tensor):
            velocity_model = velocity_model.numpy()
        if velocity_model.ndim == 3:
            velocity_model = np.squeeze(velocity_model, axis=0)  # (H,W)
        H, W = velocity_model.shape 
             
        # ---- Build 3 random sources & true TT maps --------------------------------
#         rng = np.random.default_rng()
        
        # Seed RNG with a deterministic value for this idx
        rng = np.random.default_rng(self.base_seed + int(idx))
        
        # If your models are always square image_size x image_size, this is fine:
        # sources = rng.integers(0, self.image_size, size=(self.num_events, 2))

        # Slightly safer: separate ranges for row (H) and col (W)
#         src_rows = rng.integers(0, H, size=(self.num_events,))
#         src_cols = rng.integers(0, W, size=(self.num_events,))
#         src_rows = rng.integers(1, H-1, size=(self.num_events,))
        src_rows = rng.integers(H-H//3, H-1, size=(self.num_events,))
        src_cols = rng.integers(1, W-1, size=(self.num_events,))
        sources = np.stack([src_rows, src_cols], axis=1)   # (E, 2)        
                
        gt_tt_maps = []       
        for i in range(self.num_events):
            mask = np.ones_like(velocity_model, dtype=float)
            mask[sources[i, 0], sources[i, 1]] = -1.0
            tt = skfmm.travel_time(mask, speed=velocity_model, dx=self.dx_m)
            gt_tt_maps.append(tt)
        gt_tt_maps = np.stack(gt_tt_maps, axis=0)  # (E,H,W)

        if self._return_raw_for_debug:
            return velocity_model, gt_tt_maps
        
        
        
        # ---- Observed TT: pure surface picks (Normalized) ------------------------
        obs_tt_maps = []
        for e in range(self.num_events):
            surface_row = gt_tt_maps[e, 0, :]                 # (W,)
            
            # Normalize: subtract the minimum value of this specific source
            surface_row = surface_row - np.min(surface_row)
            
            observed = np.tile(surface_row, (H, 1))           # (H,W)
            
#             if self.blur_sigma_px and self.blur_sigma_px > 0:
#                 observed = gaussian_filter(observed, sigma=self.blur_sigma_px)
            
            obs_tt_maps.append(observed)
            
        obs_tt_maps = np.stack(obs_tt_maps, axis=0)  # (E,H,W)
        
#         # ---- Observed TT: pure right VSP picks (NO t0 shift) ------------------------
#         obs_tt_maps = []
#         for e in range(self.num_events):
#             right_col = gt_tt_maps[e, :, -1]                 # (W,)
#             observed = np.tile(right_col[:, None], (1, W))           # (H,W)
# #             if self.blur_sigma_px and self.blur_sigma_px > 0:
# #                 observed = gaussian_filter(observed, sigma=self.blur_sigma_px)
#             obs_tt_maps.append(observed)
#         obs_tt_maps = np.stack(obs_tt_maps, axis=0)  # (E,H,W)
        
#         # ---- Observed TT: pure surface picks (NO t0 shift) ------------------------
#         obs_tt_maps = []
#         for e in range(self.num_events):
#             surface_row = gt_tt_maps[e, 0, :]                 # (W,)
#             observed = np.tile(surface_row, (H, 1))           # (H,W)
# #             if self.blur_sigma_px and self.blur_sigma_px > 0:
# #                 observed = gaussian_filter(observed, sigma=self.blur_sigma_px)
#             obs_tt_maps.append(observed)
#         obs_tt_maps = np.stack(obs_tt_maps, axis=0)  # (E,H,W)

#         # ---- Observed TT: surface picks + per-event origin-time shift --------------
#         obs_tt_maps = []
#         low, high = self.origin_time_shift_range
#         for e in range(self.num_events):
#             surface_row = gt_tt_maps[e, 0, :]  # (W,)
#             observed = np.tile(surface_row, (H, 1))  # (H,W)
#             # Add Δt (unknown origin time; constant over map) to simulate sequential events
#             delta_t = rng.uniform(low, high)
#             observed = observed + delta_t
#             # optional slight blur (mimic preprocessing / interpolation)
#             if self.blur_sigma_px and self.blur_sigma_px > 0:
#                 observed = gaussian_filter(observed, sigma=self.blur_sigma_px)
#             obs_tt_maps.append(observed)
#         obs_tt_maps = np.stack(obs_tt_maps, axis=0)  # (E,H,W)
# #         obs_tt_maps = np.clip(obs_tt_maps, dataset.TT_MIN, dataset.TT_MAX) # clamp before normalizing



        # 1. Define the horizontal indices for your 3 wells (Left, Center, Right)
        x_wells = np.array([0, W // 2, W - 1])

        # 2. Extract the 1D velocity profiles at these three specific columns
        # Resulting shape: (H, 3)
        v_wells = velocity_model[:, x_wells]

        # 3. Create an interpolator that works along the horizontal axis (axis=1)
        interpolator = interp1d(x_wells, v_wells, kind='linear', axis=1)

        # 4. Generate the full range of horizontal indices to interpolate over
        x_full = np.arange(W)

        # 5. Evaluate the interpolator to get the fully interpolated 2D model
        # Resulting shape: (H, W)
        well_map = interpolator(x_full)

#         # Build a 1D layered model by averaging over x for each depth
#         v_1d_col = velocity_model.mean(axis=1, keepdims=True)   # (H,1)
#         well_map = np.tile(v_1d_col, (1, W))              # (H,W) horizontally constant
        
#         # ---- “Random well” input: repeat a random column across x ------------------
#         x_idx = rng.integers(0, W)
#         well_col = velocity_model[:, x_idx]                 # (H,)
#         well_map = np.tile(well_col[:, None], (1, W))       # (H,W)

#         # ---- 4. Condition: Central Well Profile (from Normalized GT) ---------------
#         center_idx = W // 2
#         v_center_col = velocity_model[:, center_idx] 
#         well_map = np.tile(v_center_col[:, None], (1, W))

        # ---- (Optional) keep a smoothed velocity GT for target only ----------------
        smooth_velocity_model = gaussian_filter(well_map, sigma=5)
        
#         print( f'velocity_model.min() = {velocity_model.min()}' )
#         print( f'velocity_model.max() = {velocity_model.max()}' )

        # ---- Normalize -------------------------------------------------------------
        norm_gt_tt  = self.tt_normalizer(gt_tt_maps)                # (E,H,W)
        norm_obs_tt = self.tt_normalizer(obs_tt_maps)               # (E,H,W)
        norm_vel    = self.model_normalizer(velocity_model)         # (H,W)
        norm_smooth = self.model_normalizer(smooth_velocity_model)  # (H,W)
        norm_well   = self.model_normalizer(well_map)               # (H,W)
        
#         print( f'velocity_model.min() = {velocity_model.min()}' )
#         print( f'velocity_model.max() = {velocity_model.max()}' )

        # ---- Assemble model I/O ----------------------------------------------------
        # INPUT c1: observed TT (E channels) + random well column (1 channel)
#         c1 = np.concatenate([norm_obs_tt, norm_well[None, ...]], axis=0)  # (E+1,H,W)
        c1 = np.concatenate([norm_obs_tt, norm_smooth[None, ...]], axis=0)  # (E+1,H,W)

        # TARGET c0: true TT (E channels) + true velocity (1 channel)
        c0 = np.concatenate([norm_gt_tt, norm_vel[None, ...]], axis=0)    # (E+1,H,W)

        # CONDITION: removed (keep a zero channel to preserve shapes)
        # If/when you switch to unconditional i2sb-small, you can ignore this entirely.
#         cond = np.zeros_like(norm_smooth[None, ...], dtype=np.float32)      # (1,H,W) zeros
#         cond = norm_well[None, ...] # 1-channel condition    
#         cond = norm_smooth[None, ...]
        cond = np.concatenate([norm_obs_tt, norm_smooth[None, ...]], axis=0)  # (E+1,H,W)

        # ---- Torchify --------------------------------------------------------------
        c0   = torch.from_numpy(c0).float()
        c1   = torch.from_numpy(c1).float()
        cond = torch.from_numpy(cond).float()
        
#         # TEMPORARY: Return raw and normalized arrays for min/max checking
#         return (
#             c0, c1, cond, 
#             velocity_model, norm_vel,      # Velocity
#             gt_tt_maps, norm_gt_tt,        # Ground Truth Travel Times
#             obs_tt_maps, norm_obs_tt       # Observed Travel Times
#         )
        return c0, c1, cond

    
    
def precompute_microseismic_split(opt, log, split, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = build_lmdb_dataset(opt, log, split)
    
    dataset = MicroseismicDataset(
        sample_list=base_dataset.sample_list,
        lmdb_data=base_dataset.lmdb_data,
        opt=opt,
        return_raw_for_debug=False
    )

    for idx in range(len(dataset)):
        c0, c1, cond = dataset[idx]
        sample = {"c0": c0, "c1": c1, "cond": cond}

        torch.save(sample, out_dir / f"{idx:06d}.pt")

        if idx % 100 == 0:
            log.info(f"[{split}] precomputed {idx}/{len(dataset)}")    
    

    
# Modify the builder to pass the debug flag
def build_microseismic_dataset(opt, log, split, return_raw_for_debug=False):
    
    base_dataset = build_lmdb_dataset(opt, log, split)
    
    microseismic_dataset = MicroseismicDataset(
        sample_list=base_dataset.sample_list,
        lmdb_data=base_dataset.lmdb_data,
        opt=opt,
        return_raw_for_debug=return_raw_for_debug
    )
    
    return microseismic_dataset

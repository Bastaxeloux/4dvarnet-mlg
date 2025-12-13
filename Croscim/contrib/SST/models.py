import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from datetime import datetime
import time
from src.utils import get_last_time_wei, get_frcst_time_wei, get_linear_time_wei
from src.models import Lit4dVarNet
from contrib.SST.load_data import *
from contrib.SST.visualization import save_training_figures
from dataclasses import dataclass
from collections import Counter
from scipy.interpolate import RegularGridInterpolator
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class sBatch:
    input: torch.Tensor
    tgt: torch.Tensor

def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # set to eval mode
    return model

class Lit4dVarNet_SST(Lit4dVarNet):

    def __init__(self,
            optim_weight,
            prior_weight,
            domain_limits,
            persist_rw=True, 
            frcst_lead=0,
            multires=[1], 
            tgt_vars=["tgt_sst"],  # merged of slstr and aasti. (slstr if both present)
            norm_tgt_vars=["slstr_av", "aasti_av"],  # we keep them for normalization
            norm_stats_covs=None,
            *args, **kwargs):

        # IMPORTANT : optim_weight, srnn_weight, rec_weight are now multi-resolution dictionnaries
        # ex : optim_weight = {
        #           "patch_x10": np.array(...),
        #           "patch_x3": np.array(...),
        #           "patch_x1": np.array(...),
        #      }

        super().__init__(*args, **kwargs)

        # Save hyperparameters for TensorBoard (excluding large objects)
        self.save_hyperparameters(ignore=['optim_weight', 'prior_weight'])

        # Désactiver le logging automatique de certaines métriques PyTorch Lightning
        # epoch et hp_metric sont loggués manuellement sous general/
        self.log_epoch = False  # Désactive le logging automatique de 'epoch'

        # Timing tracking for profiling
        self.timing_stats = {}
        self.batch_start_time = None
        self.last_step_time = None
        self.step_times = {}
        self.last_losses = {}
        self.current_batch_in_epoch = 0

        self.var_groups = VAR_GROUPS
        self.covariates = COVARIATES
        self.tgt_vars = tgt_vars
        self.norm_tgt_vars = norm_tgt_vars

        self.frcst_lead = frcst_lead
        self.domain_limits = domain_limits
        self.multires = multires
         
        #self.maxlen_daw = self.trainer.datamodule.test_dataloader()[f"patch_x{self.multires[0]}"].dataset.patch_dims["time"]
        self.maxlen_daw = 15
         
        # we choose to take 15 => 9 => 5 to alwais have an odd number of timesteps (for central time)
        self.len_daw = {
            10: 15,  # x10 res: full window (15 days)
            3: 9,    # x3 res: 9 days (after DAW crop from x10)
            1: 5,    # x1 res: 5 days (after DAW crop from x3)
        }

        self._norm_stats_cov = norm_stats_covs

        # IMPORTANT : register weights as buffers. Or they wont be trained
        self.optim_weight = {}
        for key, weight_array in optim_weight.items():  # key = "patch_x10", etc.
            buffer_name = f"_optim_weight_{key}"
            weight_tensor = torch.from_numpy(weight_array).to("cuda")
            self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
            self.optim_weight[key] = getattr(self, buffer_name)
        self.prior_weight = {}
        for key, weight_array in prior_weight.items():  # key = "patch_x10", etc.
            buffer_name = f"_prior_weight_{key}"
            weight_tensor = torch.from_numpy(weight_array).to("cuda")
            self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
            self.prior_weight[key] = getattr(self, buffer_name)


        self.equivalence_map = {"sst": ["sst", "SST", "sea_surface_temperature", "av"]}
        self._sanity_check_started = False
        
        # Timing tracking 
        self.batch_start_time = None
        self.step_times = {} 

    def on_sanity_check_start(self):
        """Just a print to indicate sanity check start"""
        if self.global_rank == 0:
            print("\nSANITY CHECK: Validating model structure...")
        self._sanity_check_started = True
    
    def on_sanity_check_end(self):
        """Called when the sanity check ends"""
        if self.global_rank == 0:
            print("Sanity check completed\n")

    def on_train_epoch_start(self):
        """Reset batch counter at the start of each epoch"""
        if self.global_rank == 0:
            self.current_batch_in_epoch = 0

    def on_train_batch_start(self, batch, batch_idx):
        """Track batch start time."""
        if self.global_rank == 0:
            # Measure time since last batch ended (= data loading time)
            if hasattr(self, 'last_batch_end_time'):
                self.data_loading_time = time.time() - self.last_batch_end_time
                print(f"\n\n[BATCH {batch_idx+1}] Data loaded in {self.data_loading_time:.1f}s, now processing...", flush=True)
            else:
                print(f"\n\n[BATCH {batch_idx+1}] Starting...", flush=True)
                self.data_loading_time = 0.0

            self.batch_start_time = time.time()
            self.step_times = {}
            self.last_step_time = self.batch_start_time
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Track when batch processing ends (to measure data loading time for next batch)."""
        if self.global_rank == 0:
            self.last_batch_end_time = time.time()
    
    def _track_time(self, step_name):
        """Helper to track time for each step - only records if > 10ms."""
        if self.global_rank == 0 and self.last_step_time is not None:
            current_time = time.time()
            elapsed = current_time - self.last_step_time
            # Only record meaningful times (> 10ms)
            if elapsed > 0.01:
                self.step_times[step_name] = elapsed
            self.last_step_time = current_time

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @property
    def norm_stats_covs(self):
        if self._norm_stats_covs is not None:
            return self._norm_stats_covs
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats_covs()
        return (0., 1.)

    def configure_optimizers(self):
        if self.opt_fn is not None:
            return self.opt_fn(self)
        else:
            params = []
            for model in self.solver.solvers.values():
                params += list(filter(lambda p: p.requires_grad, model.parameters()))
            opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
            return {
               "optimizer": opt,
               "lr_scheduler": {
                   "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100),
                   "interval": "step",  # Update LR at each training step (not epoch)
                   "frequency": 1,
               }
            }

    def crop_daw(self, item_dict, res):
        """
        Crop ALL variables in the dict to a DAW (determined by the input res)
        Curently symmetric cropping. It is subject to change, in case we want to do nowcasting
        PARAMETERS:
        -----------
        item_dict : dict
            Dictionary containing tensors with shape (B, T, H, W) or other shapes
        res : int
            Resolution factor (10, 3, or 1) determining target temporal length
            
        RETURNS:
        --------
        dict : Modified item_dict with all 4D tensors cropped to target temporal length
        """
        target_length = self.len_daw[res]
        
        for var, data in item_dict.items():
            # Crop any 4D tensor with temporal dimension > 1
            if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                current_length = data.shape[1]
                if current_length > target_length:
                    crop_total = current_length - target_length
                    start_idx = crop_total // 2
                    end_idx = start_idx + target_length
                    item_dict[var] = data[:, start_idx:end_idx, :, :]
        
        return item_dict

    def modify_multires_batch(self, batch):
        """
        Applique un masquage temporel sur toutes les résolutions du batch multi-échelle.
        
        NOTE: We do NOT crop observations here. All resolutions start with full 15T data.
        Cropping to resolution-specific timesteps (9T for x3, 5T for x1) happens AFTER
        the coarse prediction is available, in update_batch_as_anomaly().
        """
        # print(f"\n[modify_multires_batch] Starting batch modification")
        for key, item in batch.items():
            if not key.startswith("patch_x"):
                continue 
            # print(f"[modify_multires_batch] Processing {key}")
            
            # Convert TrainingItem (NamedTuple) to dict for modification
            item_dict = item._asdict()
            
            new_item = {}

            for var in item_dict:
                data = item_dict[var]
                if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                    # Masquage temporel (on suppose dim=1 correspond au temps)
                    if self.frcst_lead is not None and self.frcst_lead > 0:
                        data[:, -self.frcst_lead:, :, :] = torch.nan
                    new_item[var] = data.to(device)
                else:
                    new_item[var] = data  # gardé tel quel (land_mask, latv, lonv...)
            
            # Reconstruct TrainingItem from modified dict
            batch[key] = type(item)(**new_item)

        return batch

    def modify_batch(self, batch, res):
        """
        Applique un masquage temporel sur le batch
        """
        item_dict = batch._asdict()
        item_dict = self.crop_daw(item_dict, res)
        new_item = {}
        for var in item_dict:
            data = item_dict[var]
            if isinstance(data, torch.Tensor) and data.ndim == 4 and data.shape[1] > 1:
                # Masquage temporel (on suppose dim=1 correspond au temps)
                if self.frcst_lead is not None and self.frcst_lead > 0:
                    data[:, -self.frcst_lead:, :, :] = torch.nan
                new_item[var] = data.to(device)
            else:
                new_item[var] = data  # gardé tel quel (land_mask, latv, lonv, lat_geo, lon_geo...)
        batch = type(batch)(**new_item)
        return batch

    def format_batch_for_solver(self, batch):
        """
        A partir d'un batch (namedtuple), renvoie une concaténation de tenseurs, pour l'entrée du solver
        Returns : dict with 'input' and 'tgt' tensors for solver
            - input: concatenated tensor of shape (B, C, H, W)
            - tgt: target SST tensor of shape (B, T, H, W)
        C varies by resolution due to temporal cropping. 
            - x10 : 139 channels (8 satsx15T + 1 covx15T + 4 spatialx1T)
            - x3  :  85 channels (8 satsx9T + 1 covx9T + 4 spatialx1T)
            - x1  :  49 channels (8 satsx5T + 1 covx5T + 4 spatialx1T)
        """
        # print(f"\n[format_batch_for_solver] Starting batch formatting")
        input_tensors = []
        
        # Concatenate satellite observations (var_groups: aasti, avhrr, pmw, slstr)
        for group, vars_ in self.var_groups.items():
            for var in vars_:
                key = f"{group}_{var}"
                if hasattr(batch, key):
                    t = getattr(batch, key)
                    # print(f"[format_batch_for_solver] {key}: {t.shape}")
                    input_tensors.append(t)
    
        # Concatenate covariates (sea_ice_fraction)
        for cov in self.covariates:
            if hasattr(batch, cov):
                t = getattr(batch, cov)
                # print(f"[format_batch_for_solver] {cov}: {t.shape}")
                input_tensors.append(t)
        
        # Add spatial/temporal metadata (4 channels: lat, lon, surfmask, time)
        # These are 2D (B, Y, X), need to expand to (B, 1, Y, X) to concat with 4D tensors
        # They represent static information, so we add them as single-channel features
        spatial_temporal_vars = ['lat', 'lon', 'surfmask', 'time']
        for var_name in spatial_temporal_vars:
            if hasattr(batch, var_name):
                spatial_tensor = getattr(batch, var_name)
                # Expand from (B, Y, X) to (B, 1, Y, X) to match temporal dimension
                if spatial_tensor.ndim == 3:
                    spatial_tensor = spatial_tensor.unsqueeze(1)  # Add channel dimension
                # print(f"[format_batch_for_solver] {var_name}: {spatial_tensor.shape}")
                input_tensors.append(spatial_tensor)
        
        # print(f"[format_batch_for_solver] Total input tensor shapes before concat: {[t.shape for t in input_tensors]}")
        
        tgt_tensors = []
        for var in self.tgt_vars:
            if hasattr(batch, var):
                t = getattr(batch, var)
                # print(f"[format_batch_for_solver] tgt {var}: {t.shape}")
                tgt_tensors.append(t)
        
        input_cat = torch.cat(input_tensors, dim=1).float()
        tgt_cat = torch.cat(tgt_tensors, dim=1).float()
        # print(f"[format_batch_for_solver] Final input shape: {input_cat.shape}")
        # print(f"[format_batch_for_solver] Final tgt shape: {tgt_cat.shape}")
    
        return sBatch(
                     input=torch.cat(input_tensors, dim=1).float(),
                     tgt=torch.cat(tgt_tensors, dim=1).float()
                     )

    def update_batch_as_anomaly(self, batch, out, verbose=False):
        """
        Compute anomalies (observations - coarse_prediction) for residual learning
        
        NOTE: Only tgt_sst is used for actual prediction. The other target variables 
        (tgt_aasti_av, tgt_slstr_av) are kept only for evaluation/plotting purposes
        
        STEPS :
            1. Crop obs 15T -> 9T (for x3) or 5T (for x1)
            2. Compute anomaly: obs_cropped - coarse_prediction
            3. Store anomalies back in batch for next resolution training
        """
        batch_dict = batch._asdict()
        
        if verbose:
            print(f"\n[update_batch_as_anomaly] Starting anomaly update")
            print(f"[update_batch_as_anomaly] batch_dict vars BEFORE update:")
            for var in batch_dict:
                if isinstance(batch_dict[var], torch.Tensor) and batch_dict[var].ndim == 4:
                    print(f"  {var}: {batch_dict[var].shape}")
        
        coarse_prediction = out["tgt_sst"]
        if verbose:
            print(f"[update_batch_as_anomaly] Processing tgt_sst prediction, shape={coarse_prediction.shape}")
        
        n_pred_timesteps = coarse_prediction.shape[1]  # 15, 9, or 5
        satellite_prefixes = ["aasti", "avhrr", "pmw", "slstr"]
        if verbose:
            print(f"[update_batch_as_anomaly] Will update all {len(satellite_prefixes)} satellites with tgt_sst prediction")
        
        # pour chaque satellite, on met à jour _av et _std
        for sat_prefix in satellite_prefixes:
            batch_var_av = f"{sat_prefix}_av"
            batch_var_std = f"{sat_prefix}_std"
            
            # Process _av: crop observation to match prediction, then compute anomaly
            if batch_var_av in batch_dict:
                batch_data_av = batch_dict[batch_var_av]
                
                if verbose:
                    print(f"[update_batch_as_anomaly] Updating {batch_var_av}, before shape={batch_data_av.shape}")
                
                if isinstance(batch_data_av, torch.Tensor) and batch_data_av.ndim == 4:
                    n_batch_timesteps = batch_data_av.shape[1]
                    
                    # Crop observation to match prediction timesteps
                    if n_batch_timesteps > n_pred_timesteps:
                        crop_total = n_batch_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        batch_data_av_cropped = batch_data_av[:, start_idx:end_idx, :, :]
                        if verbose:
                            print(f"[update_batch_as_anomaly]   Crop for alignment: {n_batch_timesteps} -> {n_pred_timesteps}")
                    else:
                        batch_data_av_cropped = batch_data_av
                    
                    # Compute anomaly: observation - prediction
                    anomaly = batch_data_av_cropped - coarse_prediction
                    batch_dict[batch_var_av] = anomaly
                    if verbose:
                        print(f"[update_batch_as_anomaly]   {batch_var_av} stored as anomaly (cropped): {batch_dict[batch_var_av].shape}")
            
            # Process _std: crop to match prediction timesteps (no anomaly, just temporal alignment)
            if batch_var_std in batch_dict:
                batch_data_std = batch_dict[batch_var_std]
                if verbose:
                    print(f"[update_batch_as_anomaly] Updating {batch_var_std}, before shape={batch_data_std.shape}")
                
                if isinstance(batch_data_std, torch.Tensor) and batch_data_std.ndim == 4:
                    n_batch_timesteps = batch_data_std.shape[1]
                    
                    if n_batch_timesteps > n_pred_timesteps:
                        crop_total = n_batch_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        batch_data_std_cropped = batch_data_std[:, start_idx:end_idx, :, :]
                        batch_dict[batch_var_std] = batch_data_std_cropped
                        if verbose:
                            print(f"[update_batch_as_anomaly]   Crop for alignment: {n_batch_timesteps} -> {n_pred_timesteps}")
                            print(f"[update_batch_as_anomaly]   {batch_var_std} stored cropped (for consistency): {batch_dict[batch_var_std].shape}")
                    else:
                        if verbose:
                            print(f"[update_batch_as_anomaly]   {batch_var_std} kept unchanged: {batch_data_std.shape}")
        
        # Crop covariates to match prediction timesteps
        for cov_var in ["sea_ice_fraction"]:
            if cov_var in batch_dict:
                cov_data = batch_dict[cov_var]
                if isinstance(cov_data, torch.Tensor) and cov_data.ndim == 4:
                    n_cov_timesteps = cov_data.shape[1]
                    
                    if n_cov_timesteps > n_pred_timesteps:
                        crop_total = n_cov_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        cov_data_cropped = cov_data[:, start_idx:end_idx, :, :]
                        batch_dict[cov_var] = cov_data_cropped
                        if verbose:
                            print(f"[update_batch_as_anomaly]   {cov_var} cropped: {cov_data.shape} -> {batch_dict[cov_var].shape}")
        
        # Crop target variables to match prediction timesteps
        for tgt_var in ["tgt_sst", "tgt_aasti_av", "tgt_slstr_av"]:
            if tgt_var in batch_dict:
                tgt_data = batch_dict[tgt_var]
                if isinstance(tgt_data, torch.Tensor) and tgt_data.ndim == 4:
                    n_tgt_timesteps = tgt_data.shape[1]
                    
                    if n_tgt_timesteps > n_pred_timesteps:
                        crop_total = n_tgt_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        tgt_data_cropped = tgt_data[:, start_idx:end_idx, :, :]
                        batch_dict[tgt_var] = tgt_data_cropped
                        if verbose:
                            print(f"[update_batch_as_anomaly]   {tgt_var} cropped: {tgt_data.shape} => {batch_dict[tgt_var].shape}")
        
        if verbose:
            print(f"[update_batch_as_anomaly] batch_dict vars AFTER update:")
            for var in batch_dict:
                if isinstance(batch_dict[var], torch.Tensor) and batch_dict[var].ndim == 4:
                    print(f"  {var}: {batch_dict[var].shape}")
    
        return type(batch)(**batch_dict)

    def interpolate_torch_orig(self,coarse_dict,
                          xc_coarse, yc_coarse,
                          xc_target, yc_target,
                          mode='bilinear',dtype=torch.float32):
        """
        Interpolate dict of (B, T, H, W) tensors on batch-varying regular grids using torch.vmap.
        coarse_dict: dict of {var_name: (B, T, Hc, Wc)}
        xc_coarse, yc_coarse: (B, Wc), (B, Hc)
        xc_target, yc_target: (B, Wf), (B, Hf)
        Returns: dict of {var_name: (B, T, Hf, Wf)}
        """
    
        def make_normalized_grid(xc_c, yc_c, xc_t, yc_t):
            # Build normalized grid in [-1, 1] for grid_sample
            x_min, x_max = xc_c.min(), xc_c.max()
            y_min, y_max = yc_c.min(), yc_c.max()
            grid_x, grid_y = torch.meshgrid(xc_t, yc_t, indexing='xy')  # (Wf, Hf)
            grid_x = grid_x.permute(1, 0).float()
            grid_y = grid_y.permute(1, 0).float()
            # Normalisation basée sur les extrémités d'INDEX (pas min/max)
            # Cela marche aussi si x_c/y_c décroissent (dénominateur < 0)
            x0, x1 = xc_c[0], xc_c[-1]
            y0, y1 = yc_c[0], yc_c[-1]
            # Evite division par zéro si grille dégénérée
            eps = torch.finfo(dctype := dtype).eps
            dx = torch.clamp(x1 - x0, min=-1e-12, max=-1e-12) if (x1-x0).abs() < eps else (x1 - x0)
            dy = torch.clamp(y1 - y0, min=-1e-12, max=-1e-12) if (y1-y0).abs() < eps else (y1 - y0)
            norm_x = 2.0 * (grid_x - x0) / dx - 1.0
            norm_y = 2.0 * (grid_y - y0) / dy - 1.0
            grid = torch.stack((norm_x, norm_y), dim=-1)  # (Wf, Hf, 2)
            grid = torch.clamp(grid, -1.0001, 1.0001)
            #grid = grid.permute(1, 0, 2)  # (Hf, Wf, 2)
            return grid  # (Hf, Wf, 2)
    
        def interpolate_one_sample(xb, grid):
            # xb: (T, Hc, Wc)
            # grid: (Hf, Wf, 2)
            xb = xb.unsqueeze(1)  # (T, 1, Hc, Wc)
            grid = grid.unsqueeze(0).repeat(xb.shape[0], 1, 1, 1)  # (T, Hf, Wf, 2)
            out = F.grid_sample(xb, grid.to(device),
                                mode=mode, align_corners=True)  # (T, 1, Hf, Wf)
            return out.squeeze(1)  # (T, Hf, Wf)
    
        result = {}
        for var, tensor in coarse_dict.items():
            if (tensor is not None) and (var not in ["time", "lat", "lon"]):
                B = tensor.shape[0]
                grids = []
                for b in range(B):
                    grid = make_normalized_grid(
                        xc_coarse[b], yc_coarse[b],
                        xc_target[b], yc_target[b]
                    )  # (Hf, Wf, 2)
                    grids.append(grid)
                grids = torch.stack(grids, dim=0)  # (B, Hf, Wf, 2)
    
                # vmap interpolation over batch
                out = torch.vmap(interpolate_one_sample)(tensor.to(device),
                                                         grids.to(device))  # (B, T, Hf, Wf)
                result[var] = out
    
        return result
    
    def interpolate_torch(self, coarse_dict, 
                                lon_coarse, lat_coarse, 
                                lon_target, lat_target):
        """
        Interpolate dict of (B, T, Hc, Wc) numpy/tensor arrays onto new target grid (Hf, Wf).
        
        NOTE: The input coordinates are 2D grids created by meshgrid in data_multires.py.
        Since these are regular grids, we extract 1D coordinate vectors to use with
        RegularGridInterpolator (which requires strictly monotonic 1D coordinates).
        
        coarse_dict: dict of {var_name: (B, T, Hc, Wc)}
        lon_coarse: (B, Hc, Wc) 2D longitude grid for each batch (from batch.lon)
        lat_coarse: (B, Hc, Wc) 2D latitude grid for each batch (from batch.lat)
        lon_target: (B, Hf, Wf) 2D target longitude grid for each batch
        lat_target: (B, Hf, Wf) 2D target latitude grid for each batch
        
        Returns: dict of {var_name: (B, T, Hf, Wf)}
        """
        result = {}
        
        for var, tensor in coarse_dict.items():
            if (tensor is None) or (var in ["time", "lat", "lon"]):
                continue
            
            # Convert to numpy if tensor is torch.Tensor
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            
            T, Hc, Wc = tensor.shape[1:]
            B = lat_target.shape[0]
            Hf, Wf = lat_target.shape[1], lon_target.shape[1]
            
            out = np.zeros((B, T, Hf, Wf), dtype=np.float32)
            
            for b in range(B):
                # Extract 2D grids and convert to numpy
                lat_c_2d = lat_coarse[b].cpu().numpy() if hasattr(lat_coarse[b], "cpu") else lat_coarse[b]
                lon_c_2d = lon_coarse[b].cpu().numpy() if hasattr(lon_coarse[b], "cpu") else lon_coarse[b]
                lat_t_2d = lat_target[b].cpu().numpy() if hasattr(lat_target[b], "cpu") else lat_target[b]
                lon_t_2d = lon_target[b].cpu().numpy() if hasattr(lon_target[b], "cpu") else lon_target[b]
                
                # Remove temporal dimension if present (e.g., (1, H, W) -> (H, W))
                # This happens when lon/lat have shape (B, 1, H, W) from the batch
                if lat_c_2d.ndim == 3 and lat_c_2d.shape[0] == 1:
                    lat_c_2d = lat_c_2d.squeeze(0)
                if lon_c_2d.ndim == 3 and lon_c_2d.shape[0] == 1:
                    lon_c_2d = lon_c_2d.squeeze(0)
                if lat_t_2d.ndim == 3 and lat_t_2d.shape[0] == 1:
                    lat_t_2d = lat_t_2d.squeeze(0)
                if lon_t_2d.ndim == 3 and lon_t_2d.shape[0] == 1:
                    lon_t_2d = lon_t_2d.squeeze(0)
                
                # Extract 1D vectors from 2D grids
                # Since grids are created by meshgrid, lat is constant along columns, lon is constant along rows
                lat_c_1d = lat_c_2d[:, 0]  # (Hc,) - extract first column
                lon_c_1d = lon_c_2d[0, :]  # (Wc,) - extract first row
                lat_t_1d = lat_t_2d[:, 0]  # (Hf,)
                lon_t_1d = lon_t_2d[0, :]  # (Wf,)
                
                # DIAGNOSTIC: Check if coordinates are strictly monotonic
                def check_monotonic(arr, name):
                    """Check if array is strictly monotonic (ascending or descending)"""
                    diffs = np.diff(arr)
                    is_ascending = np.all(diffs > 0)
                    is_descending = np.all(diffs < 0)
                    is_monotonic = is_ascending or is_descending
                    
                    if not is_monotonic:
                        print(f"\n[interpolate_torch] ERROR: {name} is NOT strictly monotonic!")
                        print(f"  Shape: {arr.shape}")
                        print(f"  First 10 values: {arr[:10]}")
                        print(f"  Last 10 values: {arr[-10:]}")
                        print(f"  Min: {arr.min()}, Max: {arr.max()}")
                        print(f"  Unique values: {len(np.unique(arr))}/{len(arr)}")
                        
                        # Check for duplicates
                        unique, counts = np.unique(arr, return_counts=True)
                        duplicates = unique[counts > 1]
                        if len(duplicates) > 0:
                            print(f"  Duplicate values: {duplicates[:5]}")
                        
                        # Check diff signs
                        n_positive = np.sum(diffs > 0)
                        n_negative = np.sum(diffs < 0)
                        n_zero = np.sum(diffs == 0)
                        print(f"  Diff stats: {n_positive} positive, {n_negative} negative, {n_zero} zeros")
                        
                        raise ValueError(f"{name} must be strictly ascending or descending for RegularGridInterpolator")
                    
                    return is_ascending
                
                # Validate all coordinate arrays
                lat_c_ascending = check_monotonic(lat_c_1d, f"lat_coarse[batch={b}]")
                lon_c_ascending = check_monotonic(lon_c_1d, f"lon_coarse[batch={b}]")
                
                # DIAGNOSTIC: Check if target points are within source grid bounds
                lat_t_min, lat_t_max = lat_t_1d.min(), lat_t_1d.max()
                lon_t_min, lon_t_max = lon_t_1d.min(), lon_t_1d.max()
                lat_c_min, lat_c_max = lat_c_1d.min(), lat_c_1d.max()
                lon_c_min, lon_c_max = lon_c_1d.min(), lon_c_1d.max()
                
                # Check if target is outside source bounds
                lat_out_of_bounds = (lat_t_min < lat_c_min) or (lat_t_max > lat_c_max)
                lon_out_of_bounds = (lon_t_min < lon_c_min) or (lon_t_max > lon_c_max)
                
                # Print warning for ANY sample with out-of-bounds (not just b==0)
                # if lat_out_of_bounds or lon_out_of_bounds:
                #     print(f"\n[INTERP BOUNDS WARNING] Sample {b}:")
                #     print(f"  Source lat: [{lat_c_min:.2f}, {lat_c_max:.2f}]")
                #     print(f"  Target lat: [{lat_t_min:.2f}, {lat_t_max:.2f}] {'OUT OF BOUNDS!' if lat_out_of_bounds else 'OK'}")
                #     print(f"  Source lon: [{lon_c_min:.2f}, {lon_c_max:.2f}]")
                #     print(f"  Target lon: [{lon_t_min:.2f}, {lon_t_max:.2f}] {'OUT OF BOUNDS!' if lon_out_of_bounds else 'OK'}")
                
                # Create target mesh grid
                Lon_t, Lat_t = np.meshgrid(lon_t_1d, lat_t_1d, indexing="xy")
                target_points = np.stack([Lat_t.ravel(), Lon_t.ravel()], axis=-1)  # (Hf*Wf, 2)
                
                tensor_b = tensor[b]  # (T, Hc, Wc)
                
                for t in range(T):
                    data_t = tensor_b[t]  # (Hc, Wc)
                    
                    # Create interpolator with 1D vectors
                    f_interp = RegularGridInterpolator(
                        (lat_c_1d, lon_c_1d),
                        data_t, 
                        bounds_error=False, fill_value=np.nan
                    )
                    interp_vals = f_interp(target_points).reshape(Hf, Wf)
                    out[b, t] = interp_vals
            
            result[var] = torch.tensor(out).to(device)
        
        return result

    def split_tensor_to_dict(self, tensor):
        """
        Découpe un tenseur interpolé (B, C, H, W) en dictionnaire {var: (B, T, H, W)}.
        Args:
            tensor: torch.Tensor de shape (B, C=T*V, H, W)
        Returns:
            dict {var_name: tensor de shape (B, T, H, W)}
        """
        B, C, H, W = tensor.shape
        V = len(self.tgt_vars)
        time_steps = C//V
        assert C == time_steps * V, f"Expected C={time_steps}x{V}, but got {C}"
        tensor_reshaped = tensor.view(B, V, time_steps, H, W)  # (B, V, T, H, W)
        tensor_reshaped = tensor_reshaped.permute(0, 2, 1, 3, 4)  # (B, T, V, H, W)
        out_dict = {
            var: tensor_reshaped[:, :, i]  # (B, T, H, W)
            for i, var in enumerate(self.tgt_vars)
        }
        return out_dict

    def training_step(self, batch, batch_idx):
        loss = self.multistep(batch, "train")[0]
        
        # Ici on fait un print concis avec toutes les infos utiles a chaque batch 
        if self.global_rank == 0 and self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            
            # GPU/RAM
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 1
            try:
                import psutil
                ram_used = psutil.virtual_memory().used / 1e9
                ram_total = psutil.virtual_memory().total / 1e9
                ram_str = f"RAM:{ram_used:.0f}/{ram_total:.0f}GB"
            except:
                ram_str = "RAM:N/A"
            try:
                batch_size = self.trainer.datamodule.batch_size
            except:
                batch_size = 4
            throughput = batch_size / batch_time if batch_time > 0 else 0
            
            # Résolution entraînée
            epoch = self.current_epoch
            total_batches = self.trainer.limit_train_batches if hasattr(self.trainer, 'limit_train_batches') else 20
            res_idx = min(epoch // (self.trainer.max_epochs // len(self.multires)), len(self.multires) - 1)
            train_res = self.multires[res_idx]

            # Add data loading time if available (step_times already filtered in _track_time)
            timing_dict = {}
            if hasattr(self, 'data_loading_time') and self.data_loading_time > 0.01:
                timing_dict['data_load'] = self.data_loading_time
            timing_dict.update(self.step_times)

            timing_str = " | ".join([f"{k}:{v:.2f}s" for k, v in timing_dict.items()])

            # Format compact: Ep0 B3/20 | x10 | L:245.3 | 12.4s (3.1 samp/s) | GPU:15/47GB | RAM:80/128GB | data_load:0.2s | forward_x10:0.03s | interp_x10->x3:0.23s ...
            print(f"Ep{epoch} B{batch_idx+1}/{total_batches} | x{train_res} | "
                  f"L:{loss:.3f} | {batch_time:.1f}s ({throughput:.1f}samp/s) | "
                  f"GPU:{gpu_mem:.1f}/{gpu_total:.0f}GB | {ram_str} | {timing_str}")

            # Log to TensorBoard - 4 catégories: general/, train/, val/, perf/
            self.current_batch_in_epoch += 1

            # General metrics: epoch et learning rate
            self.log('general/epoch', float(self.current_epoch), on_step=False, on_epoch=True)
            # Get learning rate from optimizer
            try:
                opt = self.optimizers()
                if isinstance(opt, list):
                    opt = opt[0]
                if opt is not None and len(opt.param_groups) > 0:
                    lr = opt.param_groups[0]['lr']
                    self.log('general/lr', lr, on_step=True, on_epoch=False)
            except:
                pass

            # Training progress
            self.log('train/batch_in_epoch', float(self.current_batch_in_epoch), on_step=True, on_epoch=False)

            # Performance metrics
            self.log('perf/throughput_samp_per_sec', throughput, on_step=True, on_epoch=False)
            self.log('perf/gpu_memory_gb', gpu_mem, on_step=True, on_epoch=False)
            self.log('perf/batch_time_sec', batch_time, on_step=True, on_epoch=False)

            # Log losses (from self.last_losses set in step())
            if hasattr(self, 'last_losses') and self.last_losses:
                for key, loss_val in self.last_losses.items():
                    if key.endswith('_ratio'):
                        # Log ratios dans general/
                        self.log(f'general/{key}', loss_val, on_step=True, on_epoch=True)
                    else:
                        # Log losses brutes dans train/
                        self.log(f'train/{key}', loss_val, on_step=True, on_epoch=True, prog_bar=(key=='loss'))

            # Log datamodule hyperparams on first batch
            if self.global_step == 0 and hasattr(self.trainer, 'datamodule'):
                try:
                    dm = self.trainer.datamodule
                    hparams = {
                        'datamodule/batch_size': getattr(dm, 'batch_size', -1),
                        'datamodule/num_workers': getattr(dm, 'num_workers', -1),
                    }
                    self.logger.log_hyperparams(hparams)
                except:
                    pass

        return loss

    def validation_step(self, batch, batch_idx):
        loss, out = self.multistep(batch, "val")
        if self.global_rank == 0 and batch_idx % 5 == 0:
            try:
                import psutil
                ram_gb = psutil.virtual_memory().used / 1e9
                print(f"[VAL] Batch {batch_idx} | Loss:{loss:.3f} | RAM:{ram_gb:.1f}GB", flush=True)
            except:
                pass

        # Log validation losses under val/ category
        if hasattr(self, 'last_losses') and self.last_losses:
            for key, loss_val in self.last_losses.items():
                self.log(f'val/{key}', loss_val, on_step=False, on_epoch=True, prog_bar=(key=='loss'))

        # Stocker un batch aléatoire pour visualisation (seulement le premier batch)
        if batch_idx == 0 and self.global_rank == 0:
            self.val_batch_for_viz = batch
            self.val_pred_for_viz = out
            
            # DIAGNOSTIC: Vérifier la prédiction pour visualisation
            pred_tensor = out.get('patch_x1', {}).get('tgt_sst')
            if pred_tensor is not None:
                n_nan = torch.isnan(pred_tensor).sum().item()
                n_total = pred_tensor.numel()
                valid_mask = ~torch.isnan(pred_tensor)
                if valid_mask.any():
                    pred_min = pred_tensor[valid_mask].min().item()
                    pred_max = pred_tensor[valid_mask].max().item()
                    pred_mean = pred_tensor[valid_mask].mean().item()
                else:
                    pred_min = pred_max = pred_mean = float('nan')
                print(f"[VAL VIZ] Prediction x1: NaN={n_nan}/{n_total} ({100*n_nan/n_total:.1f}%), "
                      f"min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}")

        return loss

    def on_validation_epoch_end(self):
        """Génère les figures de visualisation à la fin de chaque epoch de validation."""
        if self.global_rank == 0 and hasattr(self, 'val_batch_for_viz'):
            try:
                if hasattr(self.trainer, 'logger') and self.trainer.logger is not None:
                    run_dir = Path(self.trainer.logger.log_dir) / "figures"
                else:
                    run_dir = Path("/dmidata/projects/4dvarnet/figures_training") / datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # DIAGNOSTIC: Vérifier la structure de val_pred_for_viz
                # print(f"[VIZ] val_pred_for_viz keys: {list(self.val_pred_for_viz.keys())}")
                # if 'patch_x1' in self.val_pred_for_viz:
                #     print(f"[VIZ] patch_x1 keys: {list(self.val_pred_for_viz['patch_x1'].keys())}")
                
                pred_tensor = self.val_pred_for_viz.get('patch_x1', {}).get('tgt_sst')
                if pred_tensor is None:
                    print(f"[VIZ ERROR] Could not find pred_tensor in val_pred_for_viz!")
                    print(f"[VIZ ERROR] Available keys: {self.val_pred_for_viz.keys()}")
                else:
                    # print(f"[VIZ] pred_tensor shape: {pred_tensor.shape}")
                    pass
                
                save_training_figures(batch=self.val_batch_for_viz, pred=pred_tensor, epoch=self.current_epoch, run_dir=run_dir, batch_idx=0)
                del self.val_batch_for_viz
                del self.val_pred_for_viz
            except Exception as e:
                print(f"[VIZ] Failed to generate figures: {e}")
                import traceback
                traceback.print_exc()

    def forward(self, batch, res=1):
        solver_key = f"solver_x{res}"
        model = self.solver.solvers[solver_key].to(device)
        out = model(batch)
        if self.global_rank == 0 and self.training:
            nan_ratio = (~out.isfinite()).float().mean()
            if nan_ratio > 0.5:
                print(f"\n[forward WARNING] {solver_key} outputs {nan_ratio*100:.1f}% NaN/Inf!")
                print(f"  batch.input finite ratio: {batch.input.isfinite().float().mean():.3f}")
                print(f"  batch.tgt finite ratio: {batch.tgt.isfinite().float().mean():.3f}")
        
        return out

    def on_epoch_start(self):
        epoch = self.current_epoch
        res_idx = min(epoch // (self.trainer.max_epochs // len(self.multires)), len(self.multires) - 1)
        train_res = self.multires[res_idx]

        for res in self.multires:
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            if res == train_res:
                model.train()
                for p in model.parameters():
                    p.requires_grad = True
            else:
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False

    def multistep(self, batch, phase=""):
        """
        boucle sur les résolutions coarse => fine
        """
        # Track preprocessing time
        self._track_time("preproc")

        # ici on applique le crop_daw a toutes les résolutions
        batch = self.modify_multires_batch(batch)
        self._track_time("crop_daw")
        
        out = {}
        total_loss = 0.0

        # on fait, si n resolutions, 1/n des epochs a la premiere res, les 1/n suivantes a la deuxieme, 1/n a la troisieme
        epoch = self.current_epoch
        n_res = len(self.multires)
        total_epochs = self.trainer.max_epochs
        steps_per_res = max(1,total_epochs // n_res)
        res_index = min(epoch // steps_per_res, n_res - 1)  # limit to the last resolution

        train_res = self.multires[res_index]
        if self.global_rank == 0 and phase == "train":
            print(f"[Epoch {epoch}/{total_epochs-1}] Training x{train_res} resolution")
        
        # BOUCLE sur les res dans l'ordre [coarse => fine]
        for i, res in enumerate(self.multires):
            batch_res = batch[f"patch_x{res}"]
            
            if (res==self.multires[0]):
                # PREMIERE RES : prediction directe
                if res==train_res:
                    loss, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
                    total_loss += loss
                    self._track_time(f"forward_x{res}")
                else:
                    with torch.no_grad():
                        _, out[f"patch_x{res}"] = self.step(batch_res, res=res, phase=phase)
                    self._track_time(f"forward_x{res}_nograd")
                
                # DIAGNOSTIC: Vérifier la prédiction x10
                # if self.global_rank == 0 and phase == "val":
                #     for var_name in out[f"patch_x{res}"].keys():
                #         pred_x10 = out[f"patch_x{res}"][var_name]
                #         n_nan = torch.isnan(pred_x10).sum().item()
                #         print(f"[X10 DIAG] var={var_name}: Prediction NaN={n_nan}/{pred_x10.numel()}, shape={pred_x10.shape}")
                #     
                #     # Vérifier les coordonnées du batch x10
                #     print(f"[X10 COORDS] batch_res.lon shape: {batch_res.lon.shape}")
                #     print(f"[X10 COORDS] batch_res.lat shape: {batch_res.lat.shape}")
            
            else:
                # RESOLUTIONS SUIVANTES : utiliser la pred precedente
                coarser_res = self.multires[i-1]  # ex pour x3 : coarser_res = x10
                
                # Use geographic coordinates (in degrees) for interpolation, not normalized coordinates
                lon_target = batch_res.lon_geo
                lat_target = batch_res.lat_geo
                lon_coarse = batch[f"patch_x{coarser_res}"].lon_geo
                lat_coarse = batch[f"patch_x{coarser_res}"].lat_geo
                
                # DIAGNOSTIC: Print coordinate shapes and sample values from batch
                # if self.global_rank == 0 and phase == "train":
                #     print(f"\n[multistep TRAIN] Preparing interpolation from x{coarser_res} to x{res}")
                #     print(f"  lon_coarse shape: {lon_coarse.shape}, sample: [{lon_coarse[0, 0, 0].item():.2f}, {lon_coarse[0, 0, -1].item():.2f}]")
                #     print(f"  lat_coarse shape: {lat_coarse.shape}, sample: [{lat_coarse[0, 0, 0].item():.2f}, {lat_coarse[0, -1, 0].item():.2f}]")
                #     print(f"  lon_target shape: {lon_target.shape}, sample: [{lon_target[0, 0, 0].item():.2f}, {lon_target[0, 0, -1].item():.2f}]")
                #     print(f"  lat_target shape: {lat_target.shape}, sample: [{lat_target[0, 0, 0].item():.2f}, {lat_target[0, -1, 0].item():.2f}]")
                #     
                #     # Check for NaN in the original batch coordinates
                #     if hasattr(lon_coarse, 'isnan'):
                #         n_nan_lon = lon_coarse.isnan().sum().item()
                #         n_nan_lat = lat_coarse.isnan().sum().item()
                #         print(f"  NaN count in lon_coarse: {n_nan_lon}/{lon_coarse.numel()}")
                #         print(f"  NaN count in lat_coarse: {n_nan_lat}/{lat_coarse.numel()}")
                
                # interpoler la pred coarse sur la grille fine
                # DIAGNOSTIC: Vérifier AVANT interpolation (TRAINING ET VALIDATION)
                # if self.global_rank == 0 and res == self.multires[-1]:  # Afficher pour train ET val
                #     print(f"\n[INTERP DEBUG {phase.upper()}] Interpolating x{coarser_res} -> x{res}")
                #     print(f"[INTERP DEBUG {phase.upper()}] Batch size: {lon_coarse.shape[0]}")
                #     
                #     for var_name in out[f"patch_x{coarser_res}"].keys():
                #         coarse_pred = out[f"patch_x{coarser_res}"][var_name]
                #         n_nan = torch.isnan(coarse_pred).sum().item()
                #         print(f"[INTERP DIAG] BEFORE interp x{coarser_res}->x{res}, var={var_name}: NaN={n_nan}/{coarse_pred.numel()}")
                #     
                #     # DIAGNOSTIC: Vérifier les coordonnées ET leurs valeurs réelles
                #     print(f"[INTERP COORDS] lon_coarse shape: {lon_coarse.shape}, lat_coarse: {lat_coarse.shape}")
                #     print(f"[INTERP COORDS] lon_target shape: {lon_target.shape}, lat_target: {lat_target.shape}")
                #     print(f"[INTERP COORDS] Prediction shape: {out[f'patch_x{coarser_res}']['tgt_sst'].shape}")
                #     
                #     # NOUVEAU: Afficher les vraies valeurs de coordonnées (TOUS les samples, pas seulement sample 0)
                #     # Limiter à 1 sample en training pour éviter spam, tous les samples en val
                #     max_samples = 1 if phase == "train" else 4
                #     for b_idx in range(min(max_samples, lon_coarse.shape[0])):
                #         print(f"[INTERP VALUES {phase.upper()}] Sample {b_idx}:")
                #         print(f"  lon_coarse[{b_idx}] range: [{lon_coarse[b_idx].min().item():.2f}, {lon_coarse[b_idx].max().item():.2f}]")
                #         print(f"  lat_coarse[{b_idx}] range: [{lat_coarse[b_idx].min().item():.2f}, {lat_coarse[b_idx].max().item():.2f}]")
                #         print(f"  lon_target[{b_idx}] range: [{lon_target[b_idx].min().item():.2f}, {lon_target[b_idx].max().item():.2f}]")
                #         print(f"  lat_target[{b_idx}] range: [{lat_target[b_idx].min().item():.2f}, {lat_target[b_idx].max().item():.2f}]")
                #     
                #     # Vérifier NaN dans les coordonnées
                #     n_nan_lon_c = torch.isnan(lon_coarse).sum().item()
                #     n_nan_lat_c = torch.isnan(lat_coarse).sum().item()
                #     n_nan_lon_t = torch.isnan(lon_target).sum().item()
                #     n_nan_lat_t = torch.isnan(lat_target).sum().item()
                #     print(f"[INTERP COORDS] NaN in lon_coarse: {n_nan_lon_c}, lat_coarse: {n_nan_lat_c}")
                #     print(f"[INTERP COORDS] NaN in lon_target: {n_nan_lon_t}, lat_target: {n_nan_lat_t}")
                
                out[f"patch_x{coarser_res}_on_x{res}"] = self.interpolate_torch(
                    out[f"patch_x{coarser_res}"],
                    lon_coarse, lat_coarse,
                    lon_target, lat_target
                )
                
                # DIAGNOSTIC: Vérifier APRÈS interpolation (pour toutes les résolutions)
                if self.global_rank == 0:
                    for var_name in out[f"patch_x{coarser_res}_on_x{res}"].keys():
                        interp_pred = out[f"patch_x{coarser_res}_on_x{res}"][var_name]
                        n_nan = torch.isnan(interp_pred).sum().item()
                        print(f"[INTERP DIAG] AFTER interp x{coarser_res}->x{res}, var={var_name}: NaN={n_nan}/{interp_pred.numel()}")
                
                self._track_time(f"interp_x{coarser_res}->x{res}")
                
                # cropper l'interpolation pour matcher les targets
                out[f"patch_x{coarser_res}_on_x{res}"] = self.crop_daw(
                    out[f"patch_x{coarser_res}_on_x{res}"], res
                )
                
                # Transform the batch in anomalies 
                batch_res = self.update_batch_as_anomaly(
                    batch_res, 
                    out[f"patch_x{coarser_res}_on_x{res}"]
                )
                self._track_time(f"anomaly_x{res}")
                
                # Predict the RESIDUAL on the anomaly batch
                if res==train_res:
                    loss, residual = self.step(batch_res, res=res, phase=phase)
                    total_loss+=loss
                    self._track_time(f"forward_x{res}")
                else:
                    with torch.no_grad(): # inference only
                        _, residual = self.step(batch_res, res=res, phase=phase)
                    self._track_time(f"forward_x{res}_nograd")
                
                # RECONSTRUCTION: Add residual to coarse prediction
                # Result: SST_x3 = SST_x10_interpolated + residual_x3
                out[f"patch_x{res}"] = {}
                for var_name in residual.keys():
                    coarse_interp = out[f"patch_x{coarser_res}_on_x{res}"][var_name]
                    resid = residual[var_name]
                    
                    # DIAGNOSTIC: Vérifier avant addition (pour toutes les résolutions en validation)
                    # if self.global_rank == 0 and phase == "val":
                    #     n_nan_coarse = torch.isnan(coarse_interp).sum().item()
                    #     n_nan_resid = torch.isnan(resid).sum().item()
                    #     print(f"[RECONSTRUCTION DIAG] x{coarser_res}->x{res}, var={var_name}")
                    #     print(f"  Coarse interpolated: NaN={n_nan_coarse}/{coarse_interp.numel()}")
                    #     print(f"  Residual: NaN={n_nan_resid}/{resid.numel()}")
                    
                    out[f"patch_x{res}"][var_name] = coarse_interp + resid
                    
                    # DIAGNOSTIC: Vérifier après addition
                    # if self.global_rank == 0 and phase == "val":
                    #     result = out[f"patch_x{res}"][var_name]
                    #     n_nan_result = torch.isnan(result).sum().item()
                    #     print(f"  Result after addition: NaN={n_nan_result}/{result.numel()}")
                
                
        return total_loss, out

    def step(self, batch, res, phase=""):

        loss, out = self.base_step(batch, res=res, phase=phase)
        res_key = f"patch_x{res}"
    
        total_grad_loss = 0.0
        total_srnn_loss = 0.0
    
        for var_name in self.tgt_vars:
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch missing variable: {var_name}")
    
            target = getattr(batch, var_name)
            pred = out[var_name]
    
            tgt_sobel = kfilts.sobel(target)
            pred_sobel = kfilts.sobel(pred)
    
            mask = tgt_sobel.isfinite()
            
            # Get inpainting mask if available
            inpaint_mask_grad = None
            if hasattr(batch, 'inpaint_mask'):
                inpaint_mask_grad = batch.inpaint_mask
    
            # Crop weight to match target temporal dimension
            weight_grad = self.optim_weight[res_key]
            target_length = tgt_sobel.shape[1]
            if weight_grad.shape[0] > target_length:
                crop_total = weight_grad.shape[0] - target_length
                start_idx = crop_total // 2
                weight_grad = weight_grad[start_idx:start_idx + target_length, ...]
            
            # Crop inpaint_mask to match target temporal dimension
            if inpaint_mask_grad is not None:
                if inpaint_mask_grad.shape[1] > target_length:
                    crop_total = inpaint_mask_grad.shape[1] - target_length
                    start_idx = crop_total // 2
                    inpaint_mask_grad = inpaint_mask_grad[:, start_idx:start_idx + target_length, ...]
    
            grad_loss = self.weighted_mse(
                torch.where(mask, pred_sobel, torch.tensor(float('nan'), device=pred.device)) - tgt_sobel,
                weight_grad,
                inpaint_mask=inpaint_mask_grad
            )
            total_grad_loss += grad_loss
    
        # Prior / SRNN loss: measures how well BilinReconstructor reconstructs the target
        if hasattr(self.solver.solvers[f"solver_x{res}"], "prior_cost"):
            sbatch = self.format_batch_for_solver(batch)
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            prior = model.prior_cost.forward_reconstructor(sbatch.input)
            
            # Crop prior weight to match target temporal dimension
            weight_prior = self.prior_weight[res_key]
            target_length_prior = sbatch.tgt.shape[1]
            if weight_prior.shape[0] > target_length_prior:
                crop_total = weight_prior.shape[0] - target_length_prior
                start_idx = crop_total // 2
                weight_prior = weight_prior[start_idx:start_idx + target_length_prior, ...]
            
            # Compute MSE between target and BilinReconstructor output
            total_prior_loss = self.weighted_mse(sbatch.tgt - prior, weight_prior)
        else:
            total_prior_loss = 0.0

        # Balanced loss: Provisoire weights to balance loss magnitudes
        # MSE ~0.01, Grad ~10-1000 (varies!), Prior ~? -> adjust weights accordingly
        # TODO: Fine-tune based on TensorBoard analysis
        training_loss = 1.0 * loss + 0.001 * total_grad_loss + 0.01 * total_prior_loss

        # Stocker les losses pour le print concis et TensorBoard logging (fait dans training_step/validation_step)
        if phase == "train":
            # Calculer les proportions des losses (avant pondération) pour histogramme
            mse_val = loss.item() if hasattr(loss, 'item') else float(loss)
            grad_val = total_grad_loss.item() if hasattr(total_grad_loss, 'item') else float(total_grad_loss)
            prior_val = total_prior_loss.item() if hasattr(total_prior_loss, 'item') else float(total_prior_loss)
            total_raw = mse_val + grad_val + prior_val

            self.last_losses = {
                'loss': training_loss.item() if hasattr(training_loss, 'item') else float(training_loss),
                'mse': mse_val,
                'grad': grad_val,
                'prior': prior_val,
                # Ratios pour histogramme (proportions relatives)
                'mse_ratio': mse_val / total_raw if total_raw > 0 else 0.0,
                'grad_ratio': grad_val / total_raw if total_raw > 0 else 0.0,
                'prior_ratio': prior_val / total_raw if total_raw > 0 else 0.0
            }
    
        return training_loss, out

    def base_step(self, batch, res, phase=""):
        """
        Compute loss over selected target variables in a multi-variate model.
        Args:
            batch: a NamedTuple with target fields matching tgt_vars.
            phase: string for logging ("train", "val", etc.)
        Returns:
           loss: total loss
           out: model output tensor
        """

        sbatch = self.format_batch_for_solver(batch)

        out = self(batch=sbatch, res=res)  # out is a tensor 
        out = self.split_tensor_to_dict(out)
        res_key = f"patch_x{res}"
        inpaint_mask = None
        if hasattr(batch, 'inpaint_mask'):
            inpaint_mask = batch.inpaint_mask  # (B, T, Y, X)

        total_loss = 0.0
        for i, var_name in enumerate(self.tgt_vars):
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch does not contain variable '{var_name}'")
            target = getattr(batch, var_name)
            pred = out[var_name]  # (B, T, Y, X)
            mask = target.isfinite()
            
            # Crop weight to match target temporal dimension
            weight = self.optim_weight[res_key]
            target_length = target.shape[1]
            if weight.shape[0] > target_length:
                crop_total = weight.shape[0] - target_length
                start_idx = crop_total // 2
                weight = weight[start_idx:start_idx + target_length, ...]
            
            # Crop inpaint_mask to match target temporal dimension
            inpaint_mask_cropped = None
            if inpaint_mask is not None:
                if inpaint_mask.shape[1] > target_length:
                    crop_total = inpaint_mask.shape[1] - target_length
                    start_idx = crop_total // 2
                    inpaint_mask_cropped = inpaint_mask[:, start_idx:start_idx + target_length, ...]
                else:
                    inpaint_mask_cropped = inpaint_mask
            
            # DIAGNOSTIC: Understand why loss is 1000
            err = torch.where(mask, pred, torch.tensor(float('nan'), device=pred.device)) - target
            if self.global_rank == 0 and phase == "train":
                # print(f"\n[base_step DIAGNOSTIC] var={var_name}, res=x{res}")
                # print(f"  target.shape: {target.shape}")
                # print(f"  pred.shape: {pred.shape}")
                # print(f"  weight.shape: {weight.shape}")
                # print(f"  target finite ratio: {target.isfinite().float().mean():.3f}")
                # print(f"  pred finite ratio: {pred.isfinite().float().mean():.3f}")
                # print(f"  err finite ratio: {err.isfinite().float().mean():.3f}")
                # print(f"  weight > 0 ratio: {(weight > 0).float().mean():.3f}")
                # print(f"  weight min/max: {weight.min():.4f} / {weight.max():.4f}")
                
                # Check what weighted_mse will see
                err_w = err * weight[None, ...]
                non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
                err_num = err.isfinite() & ~non_zeros
                # print(f"  err_num.sum() (pixels that will be used): {err_num.sum()}")
                
            loss = self.weighted_mse(err, weight, inpaint_mask=inpaint_mask_cropped)
            total_loss += loss

        return total_loss, out

    def reconstruct(self, dl, items, daw, time, weight=None):
        """
        takes as input a list of tensor of dimensions (V, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
        items: list of torch tensor corresponding to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        if weight is None:
            weight = np.ones(list(dl.dataset.patch_dims.values()))
        weight = torch.tensor(weight)

        nvars = items[0].shape[0]

        result_tensor = torch.full((nvars, 1, dl.dataset.da_dims['lat'], dl.dataset.da_dims['lon']),
                                   float('nan'))
        count_tensor = torch.zeros((nvars, 1, dl.dataset.da_dims['lat'], dl.dataset.da_dims['lon']))

        coords = dl.dataset.get_coords()[(daw*len(items)):((daw+1)*len(items))]

        for idx, item in enumerate(items):
            c = coords[idx]
            iy = [np.where(dl.dataset.lat_1d == y)[0][0] for y in c.lat.values]
            ix = [np.where(dl.dataset.lon_1d == x)[0][0] for x in c.lon.values]
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] = torch.where(torch.isnan(result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]),
                                                                              0.,
                                                                              result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1])
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += torch.squeeze(item * weight)
            count_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += weight

        result_tensor /= np.maximum(count_tensor, 1e-6)
        result_da = xr.DataArray(
            result_tensor,
            dims=[f'v{i}' for i in range(nvars - len(coords[0].dims))] + ["time", "lat", "lon"],
            coords={
                "time": [time],
                "lon": dl.dataset.lon_1d,
                "lat": dl.dataset.lat_1d,
                "lon_2d": (["lat", "lon"], dl.dataset.lon_2d),
                "lat_2d": (["lat", "lon"], dl.dataset.lat_2d)
            }
        )
        return result_da

    def aggregate_batches_one_domain(self, idx_daw, idx_rec,
                                     test_data, 
                                     dataloader_idx=None,
                                     use_datamodule=False):

        dl = self.trainer.test_dataloaders[self.dataloader_keys[dataloader_idx]]
        
        res = self.multires[dataloader_idx]
        last = self.len_daw[res]
        res_key = f"patch_x{res}"

        netcdf_final = []
                                       
        for i in idx_rec:
            time = dl.dataset.times[-last:][idx_daw+i]
            print("Reconstructing LEADTIME "+str(i))
            if isinstance(dl,list):
                dl = dl[0]
            nbatch = len(test_data)
            if use_datamodule:
                rec_da = dl.dataset.reconstruct(
                            [ test_data[j][:,[i],:,:].cpu() for j in range(nbatch) ],
                            idx_daw, time,
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:]
                    )
            else:
                rec_da = self.reconstruct(dl,
                            [ test_data[j][:,[i],:,:].cpu() for j in range(nbatch) ],
                            idx_daw, time,
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:]
                    )
            test_data_ldt = rec_da.assign_coords(
                dict(v0=self.test_quantities)
            ).to_dataset(dim='v0')
            # crop (if necessary) 
            test_data_ldt = test_data_ldt.sel(**(self.domain_limits or {}))
            # stack each time 
            netcdf_final.append(test_data_ldt)

        # merge all time steps for final NetCDFs
        return xr.concat(netcdf_final, dim="time").sortby("time")

    def aggregate_batches(self, idx_rec, 
                          test_data, test_times,
                          dataloader_idx=None,
                          metrics=False,
                          write_netcdf=False,
                          use_datamodule=False):

        res = self.multires[dataloader_idx]

        # test_times est maintenant un tensor 1D de timesteps centraux: shape (N,)
        # Chaque élément est un scalaire (le timestep central d'un patch)
        if isinstance(test_times, torch.Tensor):
            time_values = test_times.cpu().numpy()
        else:
            time_values = np.array([t.cpu().item() if isinstance(t, torch.Tensor) else t for t in test_times])
        
        # DEBUG: Check shape and ensure 1D
        # print(f"[AGGREGATE DEBUG] time_values type: {type(time_values)}, shape: {time_values.shape if hasattr(time_values, 'shape') else 'N/A'}")
        if hasattr(time_values, 'ndim') and time_values.ndim > 1:
            # print(f"[AGGREGATE DEBUG] WARNING: time_values is {time_values.ndim}D! Flattening...")
            time_values = time_values.flatten()
        
        # Trouver les temps uniques et assigner un ID à chaque
        unique_times_map = {}
        daws = []
        for t_val in time_values:
            # Ensure t_val is a scalar
            if isinstance(t_val, np.ndarray):
                t_val = t_val.item()  # Convert single-element array to scalar
            # t_val est un scalaire (float ou int)
            if t_val not in unique_times_map:
                unique_times_map[t_val] = len(unique_times_map)
            daws.append(unique_times_map[t_val])
        
        daws = np.array(daws)
        
        # print(f"[AGGREGATE] Found {len(unique_times_map)} unique central timesteps from {len(time_values)} patches")

        netcdf_final = []

        def unnormalize(varname, data):
            group, var = varname.split("_")
            stats = self.norm_stats[group][var]
            if stats["type"] == "zscore":
                return data * stats["std"] + stats["mean"]
            elif stats["type"] == "minmax":
                return data * (stats["max"] - stats["min"]) + stats["min"]
            else:
                raise ValueError(f"Unknown normalization type for {varname}")

        # daws est maintenant un numpy array
        for idx_daw in np.unique(daws):  # [0, 1, 2, ...]
            sel_daw = np.where(daws == idx_daw)[0]
            # DIAGNOSTIC: Verify indices are in range
            if len(test_data) == 0:
                print(f"[AGGREGATE ERROR] test_data is empty! Cannot aggregate.")
                continue
            if sel_daw.max() >= len(test_data):
                print(f"[AGGREGATE ERROR] Index out of range!")
                print(f"  sel_daw range: [{sel_daw.min()}, {sel_daw.max()}]")
                print(f"  test_data length: {len(test_data)}")
                print(f"  daws length: {len(daws)}")
                print(f"  time_values length: {len(time_values)}")
                # Clip indices to valid range
                sel_daw = sel_daw[sel_daw < len(test_data)]
                if len(sel_daw) == 0:
                    print(f"[AGGREGATE ERROR] No valid indices after clipping, skipping this daw.")
                    continue
            test_data_sel = [test_data[i] for i in sel_daw.tolist()]
            test_data_uniq = self.aggregate_batches_one_domain(idx_daw, idx_rec,
                                                               test_data_sel,
                                                               dataloader_idx,
                                                               use_datamodule)
            # prepare unnormalization for metrics and storage
            test_data_unnorm = test_data_uniq.copy(deep=False)
            for i, var in enumerate(self.tgt_vars):
                norm_var = self.norm_tgt_vars[i]
                _, var = norm_var.split("_")
                test_data_unnorm = test_data_unnorm.update({f"pred_{var}" : (("time", "lat", "lon"),
                                                             unnormalize(norm_var, test_data_uniq[f"pred_{var}"].data))})
                test_data_unnorm = test_data_unnorm.update({f"tgt_{var}" : (("time", "lat", "lon"),
                                                             unnormalize(norm_var, test_data_uniq[f"tgt_{var}"].data))})
            if metrics:
                metric_data = test_data_unnorm.pipe(self.pre_metric_fn),
                metrics = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                print(metrics.to_frame(name="Metrics").to_markdown())
            # save NetCDFs
            time = [ datetime.datetime.strptime(str(t)[:10], "%Y-%m-%d").strftime("%Y%m%d") for t in test_data_unnorm.time.data ]
            file = f'test_data_{time[0]}_{time[-1]}_patch_x{res}.nc'
            if self.logger and write_netcdf:
                 test_data_unnorm.to_netcdf(Path(self.logger.log_dir) / file)
                 print(Path(self.trainer.log_dir) / file)
                 if metrics:
                     self.logger.log_metrics(metrics.to_dict())
            # stack each daw
            netcdf_final.append(test_data_uniq)

        # merge all time steps in a dictionary
        return { f"daw_{i}": nc for i, nc in enumerate(netcdf_final) }
        #return xr.concat(netcdf_final, dim="daw").assign_coords(daw=torch.unique(daws)).sortby("daw")

    def convert_xr_to_batch(self, coarse, batch, spatial_sel=False):
        """
        Convert an xarray.Dataset (coarse) to a dictionary of PyTorch tensors,
        matching the batch's temporal indices.
        Args:
            coarse (xr.Dataset): xarray with dims (time, lat, lon)
            batch (dict): Dictionary with keys 'time', 'lat', 'lon' etc.,
                          values are tensors of shape (B, T, H, W)
        Returns:
            coarse_dict: dict with same keys as coarse.data_vars, each of shape (B, T, H, W)
        """
        times = batch.time.cpu().numpy().astype('datetime64[s]').astype('datetime64[ns]')
        times = times.astype('datetime64[D]').astype('datetime64[ns]')
        nbatch = len(batch.time)  # batch.time: shape (B, T)
        # batch.lat/lon are 2D: (B, nlat, nlon)
        T, H, W = batch.time.shape[1], batch.lat.shape[1], batch.lon.shape[2]
        coarse_dict = {}
        # Pour chaque variable du Dataset (add lat_geo and lon_geo for interpolation)
        for var in self.tgt_vars + ["time", "lat", "lon", "lat_geo", "lon_geo"]:
            B_array = []
            for i in range(nbatch):
                times_i = np.squeeze(times[i])  # (T,)
                # Extract 1D coords from 2D grids: (nlat, nlon)
                # Use lat_geo/lon_geo (geographic) if available, else fall back to lat/lon (normalized)
                if hasattr(batch, 'lon_geo') and hasattr(batch, 'lat_geo'):
                    lons_i = batch.lon_geo[i, 0, :].cpu().numpy()  # First row -> (nlon,)
                    lats_i = batch.lat_geo[i, :, 0].cpu().numpy()  # First col -> (nlat,)
                else:
                    lons_i = batch.lon[i, 0, :].cpu().numpy()  # First row -> (nlon,)
                    lats_i = batch.lat[i, :, 0].cpu().numpy()  # First col -> (nlat,)
                # temporal selection
                matching_key = [
                                key
                                for key, ds in coarse.items()
                                if set(ds.time.values) == set(times_i)
                                ][0]
                sel_time = coarse[matching_key]
                # spatial selection
                if spatial_sel:
                    lon_is_descending = sel_time.lon[0] > sel_time.lon[-1]
                    lat_is_descending = sel_time.lat[0] > sel_time.lat[-1]
                    lon_start, lon_end = sorted([lons_i.min(), lons_i.max()],
                                          reverse=lon_is_descending)
                    lat_start, lat_end = sorted([lats_i.min(), lats_i.max()],
                                          reverse=lat_is_descending)
                    sel_patch = sel_time.sel(
                                  lon=slice(lon_start, lon_end),
                                  lat=slice(lat_start, lat_end)
                                        )
                else:
                    sel_patch = sel_time
                
                # Handle lat_geo and lon_geo (same as lat and lon from xarray)
                if var == "lat_geo":
                    arr = sel_patch["lat"].values  # Use lat from xarray
                elif var == "lon_geo":
                    arr = sel_patch["lon"].values  # Use lon from xarray
                else:
                    arr = sel_patch[var].values  # (T, H, W)
                
                if var == "time":
                    arr = arr.astype('datetime64[ns]').astype('int64')
                if var in ["time", "lat", "lon", "lat_geo", "lon_geo"]:
                    arr = np.expand_dims(arr,axis=0)
                B_array.append(torch.from_numpy(arr).float())
            coarse_dict[var] = torch.stack(B_array, dim=0)
            fields = batch._fields
            # Construire un nouveau dict avec les valeurs de coarse_dict
            complete_dict = {field: coarse_dict.get(field, None) for field in fields}
        return  type(batch)(**complete_dict)

    def on_test_start(self):
        # Stocker les dataloader keys dans l'ordre des indices
        self.dataloader_keys = list(self.trainer.test_dataloaders.keys())
    
        # Calculer le nombre de batchs par dataloader
        self.num_test_batches = {
            i: len(dl)
            for i, dl in enumerate(self.trainer.test_dataloaders.values())
        }

    def is_last_batch(self, batch_idx, dataloader_idx):
        total_batches = self.num_test_batches[dataloader_idx]
        return batch_idx == total_batches - 1
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):

        res = self.multires[dataloader_idx] #identifie quelle resolution on traite (idx 0 pour res 10, 1 pour res 3 et 2 pour res 1)
        res_key = f"patch_x{res}"
        last = self.len_daw[res]

        # print(f"Dataloader_{dataloader_idx}, Batch_{batch_idx}, res_{res}")
        if (dataloader_idx == 0) and (batch_idx == 0) :
            self.test_data = {}
            self.test_times = {}
            self.aggregate_results = {}

        if batch_idx == 0:
            self.test_data[res_key] = []
            self.test_times[res_key] = []
            
        batch = self.modify_batch(batch, res)
        # anomaly conversion
        if dataloader_idx > 0:
            coarser_res = self.multires[dataloader_idx-1]
            # project coarser_res batch on res batch
            # Use geographic coordinates (in degrees) for interpolation
            lon_target = batch.lon_geo[:, 0, :]  # (B, nlon) - geographic degrees
            lat_target = batch.lat_geo[:, :, 0]  # (B, nlat) - geographic degrees
            # identify batch daw / coarse daw equivalence for selection
            coarse = self.aggregate_results[f"patch_x{coarser_res}"]
            coarse = {
               k: v.isel(time=np.arange(self.len_daw[coarser_res]-last,
                                        self.len_daw[coarser_res]))
               for k, v in coarse.items()
            }
            coarse = self.convert_xr_to_batch(coarse, batch)
            lon_coarse = coarse.lon_geo[:, 0, :]  # (B, nlon) - geographic degrees
            lat_coarse = coarse.lat_geo[:, :, 0]  # (B, nlat) - geographic degrees
            itrp_coarse = self.interpolate_torch(coarse._asdict(),
                                                 lon_coarse, lat_coarse,
                                                 lon_target, lat_target)
            #itrp_coarse = self.crop_daw(itrp_coarse,res)
            # modify batch to work on anomaly compared to coarser resolution
            batch = self.update_batch_as_anomaly(batch, itrp_coarse)

        sbatch = self.format_batch_for_solver(batch)
        out = self(batch=sbatch, res=res)
        out = self.split_tensor_to_dict(out)
        
        # DIAGNOSTIC: Vérifier si la prédiction est vide/NaN
        # if batch_idx == 0:  # Seulement premier batch pour ne pas spammer
        #     for var in self.tgt_vars:
        #         pred = out[var]
        #         n_nan = torch.isnan(pred).sum().item()
        #         n_total = pred.numel()
        #         # Utiliser masque pour min/max sans NaN
        #         valid_mask = ~torch.isnan(pred)
        #         if valid_mask.any():
        #             pred_min = pred[valid_mask].min().item()
        #             pred_max = pred[valid_mask].max().item()
        #             pred_mean = pred[valid_mask].mean().item()
        #         else:
        #             pred_min = pred_max = pred_mean = float('nan')
        #         print(f"[TEST DIAG res={res}] {var}: NaN={n_nan}/{n_total} ({100*n_nan/n_total:.1f}%), "
        #               f"min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}")
        
        # add coarser resolution to output
        if dataloader_idx > 0:
            out = {k: out[k] + itrp_coarse[k] for k in out}
            #out = {k: itrp_coarse[k] for k in out}
        for i, var in enumerate(self.tgt_vars):
            # Adapter surfmask à la dimension temporelle de out[var]. batch.surfmask peut être (B, H, W) ou (B, T, H, W)
            n_timesteps = out[var].shape[1]
            if batch.surfmask.ndim == 3:
                surfmask_slice = batch.surfmask.unsqueeze(1).expand(-1, n_timesteps, -1, -1) # surfmask est (B, H, W) -> expand à (B, T, H, W)
            elif batch.surfmask.ndim == 4:
                surfmask_slice = batch.surfmask[:, :n_timesteps, :, :] # surfmask est (B, T, H, W) -> slice temporel si nécessaire
            else:
                raise ValueError(f"surfmask has unexpected ndim={batch.surfmask.ndim}")
            
            out[var] = torch.where(surfmask_slice==0., np.nan, out[var])
            # surfmask: 0=terre (mettre NaN), 1=ocean, 2=eau-glace, 3=glace (garder valeurs)

        # Stockage des sorties et des cibles
        # Unnormalization is done in aggregate
        out_norm, tgt_norm = {}, {}
        for i, var in enumerate(self.tgt_vars):
            pred = out[var] 
            out_norm[var] = pred
            if dataloader_idx == 0:
                tgt_norm[var] = getattr(batch, var)
            else:
                tgt_norm[var] = getattr(batch, var) + itrp_coarse[var]
        
        combined = list(out_norm.values()) + list(tgt_norm.values())
        stacked = torch.stack(combined, dim=1)

        # stacked has shape (B,V,T,H,W) with V the number of variables
        self.test_data[res_key].append(stacked)
        
        # Stocker uniquement le timestep CENTRAL pour l'agrégation
        # batch.time shape: (B, T) -> prendre le timestep central (index T//2)
        central_time_idx = batch.time.shape[1] // 2
        central_times = batch.time[:, central_time_idx]  # Shape: (B,) - un timestep par patch
        self.test_times[res_key].append(central_times)

        # if last batch, agreggate (as an xarray dataset with the estimation for a given resolution)
        if self.is_last_batch(batch_idx, dataloader_idx):
            # Flatten test_data: [[tensor1], [tensor2], ...] -> [tensor1, tensor2, ...]
            self.test_data[res_key] = list(itertools.chain(*self.test_data[res_key]))
            # Concatenate all central times into a single 1D tensor
            self.test_times[res_key] = torch.cat(self.test_times[res_key], dim=0)  # Shape: (N,) où N = total patches
            if dataloader_idx == (len(self.multires)-1):
                #idx_rec = np.arange(batch.time.shape[-1]-self.frcst_lead+1,
                #                    batch.time.shape[-1])
                idx_rec = np.arange(batch.time.shape[-1])
                write_netcdf = True
            else:
                idx_rec = np.arange(batch.time.shape[-1])
                write_netcdf = True
            # the idea behind: the aggregation is :
            # on the full window for coarser resolutions
            # only for nowcast/forecast lead times for final resolution
            self.aggregate_results[res_key] = self.aggregate_batches(idx_rec, self.test_data[res_key], self.test_times[res_key],
                                                                     dataloader_idx, metrics=False, write_netcdf=write_netcdf)
        batch, out = None, None

    @property
    def test_quantities(self):
        return [prefix + var.split("_", 1)[-1] for prefix in ["pred_", "tgt_"] for var in self.tgt_vars]

    def on_test_epoch_end(self):
        """Génère des visualisations et logs à la fin du test."""
        print("=" * 80)
        print("[TEST] on_test_epoch_end triggered")
        print("=" * 80)
        
        # Vérifier si on a des résultats agrégés
        if not hasattr(self, 'aggregate_results') or not self.aggregate_results:
            print("[TEST] No aggregate_results found, skipping visualization")
            return
        
        # Récupérer la résolution finale (x1)
        final_res = self.multires[-1]
        final_res_key = f"patch_x{final_res}"
        
        if final_res_key not in self.aggregate_results:
            print(f"[TEST] No results for {final_res_key}, skipping visualization")
            return
        
        # Les résultats sont un dict de xarray.Dataset par DAW
        # Avec mode single_day, il n'y a qu'une seule DAW (daw_0)
        results_dict = self.aggregate_results[final_res_key]
        
        if 'daw_0' not in results_dict:
            print(f"[TEST] No daw_0 in results, available keys: {list(results_dict.keys())}")
            return
        
        final_data = results_dict['daw_0']  # xarray.Dataset
        
        # Log des informations sur le test
        print(f"[TEST] Final reconstruction ({final_res_key}):")
        print(f"  Time range: {final_data.time.values[0]} to {final_data.time.values[-1]}")
        print(f"  Spatial shape: {final_data.dims['lat']} × {final_data.dims['lon']}")
        print(f"  Variables: {list(final_data.data_vars)}")
        
        # Créer des visualisations de la carte complète
        if self.global_rank == 0:
            try:
                from contrib.SST.visualization import plot_test_reconstruction
                
                if hasattr(self.trainer, 'logger') and self.trainer.logger is not None:
                    save_dir = Path(self.trainer.logger.log_dir) / "test_results"
                else:
                    save_dir = Path("/dmidata/projects/4dvarnet/test_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
                
                save_dir.mkdir(parents=True, exist_ok=True)
                
                plot_test_reconstruction(final_data, save_dir)
                print(f"[TEST] Visualizations saved to {save_dir}")
                
            except Exception as e:
                print(f"[TEST] Failed to generate test visualizations: {e}")
                import traceback
                traceback.print_exc()

    def on_load_checkpoint(self, checkpoint):
        """
        very useful when shapes of the patches/weights between
        training and inference
        """
        for key in self.state_dict().keys():
            if key.startswith("rec_weight") or key.startswith("optim_weight") or key.startswith("prior_weight"):
                print(key)
                checkpoint["state_dict"][key] = self.state_dict()[key]

import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from datetime import datetime as dt
import time
from src.utils import get_last_time_wei, get_frcst_time_wei, get_linear_time_wei
from src.models import Lit4dVarNet
from contrib.SST.load_data import *
from dataclasses import dataclass
from collections import Counter
from scipy.interpolate import RegularGridInterpolator
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def detach_to_cpu(obj):
    """Recursively detach tensors and move to CPU to prevent memory leaks.

    This is critical for storing intermediate results without keeping
    the computation graph alive on GPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    elif hasattr(obj, '_fields'):
        # NamedTuple - must check BEFORE regular tuple check
        # NamedTuples have _fields attribute with field names
        return type(obj)(*[detach_to_cpu(v) for v in obj])
    elif isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    elif hasattr(obj, '__dict__'):
        # For dataclass-like objects
        new_obj = type(obj).__new__(type(obj))
        for k, v in obj.__dict__.items():
            setattr(new_obj, k, detach_to_cpu(v))
        return new_obj
    else:
        return obj

@dataclass
class sBatch:
    input: torch.Tensor
    tgt: torch.Tensor
    inpaint_mask: torch.Tensor = None  # Pour SSL : 1 = pixel masqué artificiellement

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
            outputs_dir=None,  # Chemin centralisé pour les outputs (validation, test)
            persist_rw=True,
            frcst_lead=0,
            multires=[1],
            tgt_vars=["tgt_sst"],  # merged of slstr and aasti. (slstr if both present)
            norm_tgt_vars=["slstr_av", "aasti_av"],  # we keep them for normalization
            norm_stats_covs=None,
            epochs_per_res_cycle=None,  # If set and max_epochs divisible by this*3, use cyclic training
            loss_weights=None,  # Poids des composantes de loss: {mse: 1.0, grad: 0.001, prior: 0.05}
            inpaint_weight_factor=1.0,  # Boost pour les pixels masqués artificiellement
            *args, **kwargs):

        # IMPORTANT : optim_weight, srnn_weight, rec_weight are now multi-resolution dictionnaries
        # ex : optim_weight = {
        #           "patch_x10": np.array(...),
        #           "patch_x3": np.array(...),
        #           "patch_x1": np.array(...),
        #      }

        super().__init__(*args, **kwargs)

        # Save hyperparameters for TensorBoard (excluding large objects and nn.Module instances)
        self.save_hyperparameters(ignore=['optim_weight', 'prior_weight', 'solver'])

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
        self.epochs_per_res_cycle = epochs_per_res_cycle
        self.outputs_dir = Path(outputs_dir) if outputs_dir else Path("/dmidata/projects/4dvarnet/outputs")

        # Loss weights configuration (configurable via YAML)
        default_loss_weights = {'mse': 1.0, 'grad': 0.001, 'prior': 0.05}
        self.loss_weights = {**default_loss_weights, **(loss_weights or {})}
        self.inpaint_weight_factor = inpaint_weight_factor

        #self.maxlen_daw = self.trainer.datamodule.test_dataloader()[f"patch_x{self.multires[0]}"].dataset.patch_dims["time"]
        self.maxlen_daw = 15
         
        # we choose to take 15 => 9 => 5 to alwais have an odd number of timesteps (for central time)
        self.len_daw = {
            10: 15,  # x10 res: full window (15 days)
            3: 9,    # x3 res: 9 days (after DAW crop from x10)
            1: 5,    # x1 res: 5 days (after DAW crop from x3)
        }

        self._norm_stats_cov = norm_stats_covs

        # IMPORTANT : register weights as buffers. They will be moved to correct device by Lightning
        # Do NOT use .to("cuda") here - it would force cuda:0 and break DDP multi-GPU
        # Store buffer NAMES, not tensor references (tensors are recreated on .to(device))
        self._optim_weight_keys = []
        for key, weight_array in optim_weight.items():  # key = "patch_x10", etc.
            buffer_name = f"_optim_weight_{key}"
            weight_tensor = torch.from_numpy(weight_array).float()
            self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
            self._optim_weight_keys.append(key)
        self._prior_weight_keys = []
        for key, weight_array in prior_weight.items():  # key = "patch_x10", etc.
            buffer_name = f"_prior_weight_{key}"
            weight_tensor = torch.from_numpy(weight_array).float()
            self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
            self._prior_weight_keys.append(key)


        self.equivalence_map = {"sst": ["sst", "SST", "sea_surface_temperature", "av"]}
        self._sanity_check_started = False
        
        # Timing tracking
        self.batch_start_time = None
        self.step_times = {}

    def get_optim_weight(self, key):
        """Get optim weight buffer by key, always returns tensor on correct device"""
        return getattr(self, f"_optim_weight_{key}")

    def get_prior_weight(self, key):
        """Get prior weight buffer by key, always returns tensor on correct device"""
        return getattr(self, f"_prior_weight_{key}")

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
        """Track batch start time for performance logging."""
        if self.global_rank == 0:
            # Measure time since last batch ended (= data loading time)
            if hasattr(self, 'last_batch_end_time'):
                self.data_loading_time = time.time() - self.last_batch_end_time
            else:
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

    def get_current_resolution_idx(self, epoch=None):
        """Get the resolution index for the current epoch.

        If epochs_per_res_cycle is set and max_epochs is divisible by (epochs_per_res_cycle * 3),
        use cyclic training: alternate through resolutions every epochs_per_res_cycle epochs.

        Example with epochs_per_res_cycle=4 and max_epochs=36:
            epochs 0-3:   x10 (res_idx=0)
            epochs 4-7:   x3  (res_idx=1)
            epochs 8-11:  x1  (res_idx=2)
            epochs 12-15: x10 (res_idx=0)  <- cycle repeats
            ...

        Otherwise, use original behavior: 1/3 of epochs per resolution.
        """
        if epoch is None:
            epoch = self.current_epoch

        max_epochs = self.trainer.max_epochs
        n_res = len(self.multires)  # 3

        # Check if cyclic training is enabled and valid
        if (self.epochs_per_res_cycle is not None
            and self.epochs_per_res_cycle > 0
            and max_epochs % (self.epochs_per_res_cycle * n_res) == 0):
            # Cyclic training: epochs_per_res_cycle epochs per resolution, repeating
            cycle_length = self.epochs_per_res_cycle * n_res  # e.g., 4*3=12
            pos_in_cycle = epoch % cycle_length
            res_idx = pos_in_cycle // self.epochs_per_res_cycle
        else:
            # Original behavior: 1/3 of epochs per resolution
            steps_per_res = max(1, max_epochs // n_res)
            res_idx = min(epoch // steps_per_res, n_res - 1)

        return res_idx

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
                   # StepLR: divise le LR par gamma tous les step_size epochs
                   # Aligné sur les cycles de 12 epochs (4 epochs × 3 résolutions)
                   # Epochs 0-11: lr=1e-3, Epochs 12-23: lr=5e-4, Epochs 24-35: lr=2.5e-4
                   "scheduler": torch.optim.lr_scheduler.StepLR(opt, step_size=12, gamma=0.5),
                   "interval": "epoch",  # Update LR à chaque epoch (aligné sur cycles résolution)
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
            # CRITICAL: Also crop time_indices (2D tensor: B, T)
            elif var == "time_indices" and isinstance(data, torch.Tensor) and data.ndim == 2:
                current_length = data.shape[1]
                if current_length > target_length:
                    crop_total = current_length - target_length
                    start_idx = crop_total // 2
                    end_idx = start_idx + target_length
                    item_dict[var] = data[:, start_idx:end_idx]
        
        return item_dict

    def modify_multires_batch(self, batch):
        """
        Applique un masquage temporel sur toutes les résolutions du batch multi-échelle.
        
        NOTE: We do NOT crop observations here. All resolutions start with full 15T data.
        Cropping to resolution-specific timesteps (9T for x3, 5T for x1) happens AFTER
        the coarse prediction is available, in update_batch_as_anomaly().
        """
        for key, item in batch.items():
            if not key.startswith("patch_x"):
                continue 
            
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
        
        STRUCTURE DE L'INPUT (NEW - pour permettre Φ(state) dynamique):
            [fusion_masquée (0:T), avhrr (T:T+2*T), pmw (T+2*T:T+4*T), covariates, spatial (4 canaux)]
        
        C varies by resolution due to temporal cropping:
            - x10 : 124 channels (fusion×15T + avhrr×2×15T + pmw×2×15T + 1 cov×15T + 4 spatial)
                    = 15 + 30 + 30 + 15 + 4 = 124
            - x3  :  70 channels (fusion×9T + avhrr×2×9T + pmw×2×9T + 1 cov×9T + 4 spatial)
                    = 9 + 18 + 18 + 9 + 4 = 70
            - x1  :  34 channels (fusion×5T + avhrr×2×5T + pmw×2×5T + 1 cov×5T + 4 spatial)
                    = 5 + 10 + 10 + 5 + 4 = 34
        
        NOTE CRITIQUE: On garde la FUSION masquée (tgt_sst après inpainting) au lieu de slstr + aasti.
        Cela permet au BilinReconstructor de recevoir [state, covariates] où state et fusion ont la même dim T.
        """
        input_tensors = []
        
        # 1. Ajouter la fusion masquée en premier (dimension T selon résolution)
        if hasattr(batch, 'tgt_sst'):
            input_tensors.append(batch.tgt_sst)
        else:
            raise RuntimeError("batch.tgt_sst manquant - requis pour construction de l'input")
        
        # 2. Concatenate satellite observations (avhrr, pmw seulement - on exclut aasti et slstr)
        for group, vars_ in self.var_groups.items():
            if group in ['aasti', 'slstr']:
                continue  # Skip - déjà dans fusion
            for var in vars_:
                key = f"{group}_{var}"
                if hasattr(batch, key):
                    t = getattr(batch, key)
                    input_tensors.append(t)
    
        # Concatenate covariates (sea_ice_fraction)
        for cov in self.covariates:
            if hasattr(batch, cov):
                t = getattr(batch, cov)
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
                input_tensors.append(spatial_tensor)
        
        tgt_tensors = []
        for var in self.tgt_vars:
            if hasattr(batch, var):
                t = getattr(batch, var)
                tgt_tensors.append(t)
                
        inpaint_mask = None
        if hasattr(batch, 'inpaint_mask'):
            inpaint_mask = getattr(batch, 'inpaint_mask') # (B, T, H, W)
            
        return sBatch(input=torch.cat(input_tensors, dim=1).float(),
                        tgt=torch.cat(tgt_tensors, dim=1).float(),
                        inpaint_mask=inpaint_mask)

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
        
        coarse_prediction = out["tgt_sst"]
        n_pred_timesteps = coarse_prediction.shape[1]  # 15, 9, or 5
        satellite_prefixes = ["aasti", "avhrr", "pmw", "slstr"]
        
        # pour chaque satellite, on met à jour _av et _std
        for sat_prefix in satellite_prefixes:
            batch_var_av = f"{sat_prefix}_av"
            batch_var_std = f"{sat_prefix}_std"
            
            # Process _av: crop observation to match prediction, then compute anomaly
            if batch_var_av in batch_dict:
                batch_data_av = batch_dict[batch_var_av]
                
                if isinstance(batch_data_av, torch.Tensor) and batch_data_av.ndim == 4:
                    n_batch_timesteps = batch_data_av.shape[1]
                    
                    # Crop observation to match prediction timesteps
                    if n_batch_timesteps > n_pred_timesteps:
                        crop_total = n_batch_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        batch_data_av_cropped = batch_data_av[:, start_idx:end_idx, :, :]
                    else:
                        batch_data_av_cropped = batch_data_av
                    
                    # Compute anomaly: observation - prediction
                    anomaly = batch_data_av_cropped - coarse_prediction
                    batch_dict[batch_var_av] = anomaly
            
            # Process _std: crop to match prediction timesteps (no anomaly, just temporal alignment)
            if batch_var_std in batch_dict:
                batch_data_std = batch_dict[batch_var_std]
                
                if isinstance(batch_data_std, torch.Tensor) and batch_data_std.ndim == 4:
                    n_batch_timesteps = batch_data_std.shape[1]
                    
                    if n_batch_timesteps > n_pred_timesteps:
                        crop_total = n_batch_timesteps - n_pred_timesteps
                        start_idx = crop_total // 2
                        end_idx = start_idx + n_pred_timesteps
                        batch_data_std_cropped = batch_data_std[:, start_idx:end_idx, :, :]
                        batch_dict[batch_var_std] = batch_data_std_cropped
        
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
        
        # === SSL FIX === Crop inpaint_mask to match prediction timesteps
        # CRITICAL: inpaint_mask must have same temporal dimension as batch.tgt in solver
        # x10: 15T → 15T (no crop)
        # x3:  15T → 9T (crop)
        # x1:  9T  → 5T (crop)
        if "inpaint_mask" in batch_dict:
            mask_data = batch_dict["inpaint_mask"]
            if isinstance(mask_data, torch.Tensor) and mask_data.ndim == 4:
                n_mask_timesteps = mask_data.shape[1]
                
                if n_mask_timesteps > n_pred_timesteps:
                    crop_total = n_mask_timesteps - n_pred_timesteps
                    start_idx = crop_total // 2
                    end_idx = start_idx + n_pred_timesteps
                    mask_data_cropped = mask_data[:, start_idx:end_idx, :, :]
                    batch_dict["inpaint_mask"] = mask_data_cropped
    
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
            if (tensor is None) or (var in ["time", "lat", "lon", "lat_geo", "lon_geo"]):
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
                # lat_c_ascending = check_monotonic(lat_c_1d, f"lat_coarse[batch={b}]")
                # lon_c_ascending = check_monotonic(lon_c_1d, f"lon_coarse[batch={b}]")
                
                # # DIAGNOSTIC: Check if target points are within source grid bounds
                # lat_t_min, lat_t_max = lat_t_1d.min(), lat_t_1d.max()
                # lon_t_min, lon_t_max = lon_t_1d.min(), lon_t_1d.max()
                # lat_c_min, lat_c_max = lat_c_1d.min(), lat_c_1d.max()
                # lon_c_min, lon_c_max = lon_c_1d.min(), lon_c_1d.max()
                
                # # Check if target is outside source bounds
                # lat_out_of_bounds = (lat_t_min < lat_c_min) or (lat_t_max > lat_c_max)
                # lon_out_of_bounds = (lon_t_min < lon_c_min) or (lon_t_max > lon_c_max)
                
                # Print warning for ANY sample with out-of-bounds (not just b==0)
                # if lat_out_of_bounds or lon_out_of_bounds:
                #     print(f"\n[INTERP BOUNDS WARNING] Sample {b}:")
                #     print(f"  Source lat: [{lat_c_min:.2f}, {lat_c_max:.2f}]")
                #     print(f"  Target lat: [{lat_t_min:.2f}, {lat_t_max:.2f}] {'OUT OF BOUNDS!' if lat_out_of_bounds else 'OK'}")
                #     print(f"  Source lon: [{lon_c_min:.2f}, {lon_c_max:.2f}]")
                #     print(f"  Target lon: [{lon_t_min:.2f}, {lon_t_max:.2f}] {'OUT OF BOUNDS!' if lon_out_of_bounds else 'OK'}")
                
                # # Count how many target points are out of bounds
                # n_lat_below = np.sum(lat_t_1d < lat_c_min)
                # n_lat_above = np.sum(lat_t_1d > lat_c_max)
                # n_lon_below = np.sum(lon_t_1d < lon_c_min)
                # n_lon_above = np.sum(lon_t_1d > lon_c_max)
                # if b == 0 and (n_lat_below + n_lat_above + n_lon_below + n_lon_above) > 0:
                #     print(f"[BOUNDS DETAIL] Sample {b}: lat_below={n_lat_below}, lat_above={n_lat_above}, lon_below={n_lon_below}, lon_above={n_lon_above}")
                #     print(f"  -> This will cause ~{(n_lat_below + n_lat_above) * len(lon_t_1d) + (n_lon_below + n_lon_above) * len(lat_t_1d)} points to be NaN per timestep")

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
            res_idx = self.get_current_resolution_idx()
            train_res = self.multires[res_idx]

            # Add data loading time if available (step_times already filtered in _track_time)
            timing_dict = {}
            if hasattr(self, 'data_loading_time') and self.data_loading_time > 0.01:
                timing_dict['data_load'] = self.data_loading_time
            timing_dict.update(self.step_times)

            # timing_str = " | ".join([f"{k}:{v:.2f}s" for k, v in timing_dict.items()])

            # Log to TensorBoard - 4 catégories: general/, train/, val/, perf/
            self.current_batch_in_epoch += 1

            # General metrics: epoch, resolution, and learning rate
            self.log('general/epoch', float(self.current_epoch), on_step=False, on_epoch=True, sync_dist=True)
            self.log('general/train_resolution', float(train_res), on_step=False, on_epoch=True, sync_dist=True)
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
                        self.log(f'general/{key}', loss_val, on_step=True, on_epoch=True, sync_dist=True)
                    else:
                        # Log losses brutes dans train/
                        # 4 losses principales: mse, grad, prior, loss (pondérée)
                        # 2 losses détaillées: mse_interp (X_B̄), mse_recons (X_B)
                        self.log(f'train/{key}', loss_val, on_step=True, on_epoch=True, prog_bar=(key=='loss'), sync_dist=True)

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
                print(f"\n[VAL] Batch {batch_idx} | Loss:{loss:.3f} | RAM:{ram_gb:.1f}GB", flush=True)
            except:
                pass

        # Log validation losses under val/ category
        if hasattr(self, 'last_losses') and self.last_losses:
            for key, loss_val in self.last_losses.items():
                self.log(f'val/{key}', loss_val, on_step=False, on_epoch=True, prog_bar=(key=='loss'), sync_dist=True)

        # Collecter TOUS les batches pour visualisation (16 patches au total)
        if self.global_rank == 0:
            if batch_idx == 0:
                self.val_batches_for_viz = []
                self.val_preds_for_viz = []

            # IMPORTANT: detach and move to CPU to prevent GPU memory leak
            # Without this, computation graph stays alive and GPU memory grows each epoch
            self.val_batches_for_viz.append(detach_to_cpu(batch))
            self.val_preds_for_viz.append(detach_to_cpu(out))
            
            if isinstance(batch, dict) and 'patch_x1' in batch:
                batch_x1 = batch['patch_x1']
                tgt_sst = batch_x1.get('tgt_sst') if isinstance(batch_x1, dict) else getattr(batch_x1, 'tgt_sst', None)
                if tgt_sst is not None:
                    batch_size = tgt_sst.shape[0]
                    t_mid = tgt_sst.shape[1] // 2
                    n_patches_so_far = batch_idx * batch_size + batch_size

                    if batch_idx == 0:
                        for i in range(min(4, batch_size)):
                            patch_i = tgt_sst[i, t_mid, :, :]
                            valid_mask = ~torch.isnan(patch_i)
                            # if valid_mask.any():
                            #     mean_i = patch_i[valid_mask].mean().item()
                            #     std_i = patch_i[valid_mask].std().item()
                            #     # print(f"[VAL DEBUG] Sample {i}: mean={mean_i:.4f}, std={std_i:.4f}")
                            # else:
                            #     # print(f"[VAL DEBUG] Sample {i}: ALL NaN!")
            
            # DIAGNOSTIC: Vérifier la prédiction
            pred_tensor = out.get('patch_x1', {}).get('tgt_sst')
            if pred_tensor is not None and batch_idx == 0:
                n_nan = torch.isnan(pred_tensor).sum().item()
                n_total = pred_tensor.numel()
                valid_mask = ~torch.isnan(pred_tensor)
                if valid_mask.any():
                    pred_min = pred_tensor[valid_mask].min().item()
                    pred_max = pred_tensor[valid_mask].max().item()
                    pred_mean = pred_tensor[valid_mask].mean().item()
                else:
                    pred_min = pred_max = pred_mean = float('nan')

        return loss

    def on_validation_epoch_end(self):
        """Génère les figures de visualisation à la fin de chaque epoch de validation."""
        if self.trainer.sanity_checking:
            return
        if self.global_rank == 0 and hasattr(self, 'val_batches_for_viz'):
            try:
                # Créer run_id une seule fois pour tout le training
                if not hasattr(self, 'train_run_id'):
                    self.train_run_id = dt.now().strftime("%Y%m%d_%H%M%S")
                save_dir = self.outputs_dir / self.train_run_id / "validation" / f"epoch_{self.current_epoch:03d}"
                save_dir.mkdir(parents=True, exist_ok=True)
                
                # Extraire les tensors de prédiction de chaque batch
                pred_tensors = []
                for pred_dict in self.val_preds_for_viz:
                    pred_tensor = pred_dict.get('patch_x1', {}).get('tgt_sst')
                    if pred_tensor is None:
                        print(f"[VIZ ERROR] Could not find pred_tensor in batch!")
                    else:
                        pred_tensors.append(pred_tensor)
                
                if len(pred_tensors) == len(self.val_batches_for_viz):
                    # Appeler la fonction de visualisation pour tous les patches
                    from contrib.SST.visualization import save_validation_patches, save_validation_patches_multires
                    save_validation_patches(
                        batches_list=self.val_batches_for_viz,
                        preds_list=pred_tensors,
                        save_dir=save_dir,
                        epoch=self.current_epoch
                    )
                    # Nouveau: Visualisation multi-résolution x10 → x3 → x1
                    save_validation_patches_multires(
                        batches_list=self.val_batches_for_viz,
                        preds_list=self.val_preds_for_viz,  # Dict complet avec toutes les résolutions
                        save_dir=save_dir,
                        epoch=self.current_epoch
                    )
                else:
                    print(f"[VIZ ERROR] Mismatch: {len(self.val_batches_for_viz)} batches but {len(pred_tensors)} predictions")

                # Nettoyer
                del self.val_batches_for_viz
                del self.val_preds_for_viz
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
        res_idx = self.get_current_resolution_idx()
        train_res = self.multires[res_idx]

        # Log training resolution at epoch start
        if self.global_rank == 0:
            max_epochs = self.trainer.max_epochs
            n_res = len(self.multires)
            is_cyclic = (self.epochs_per_res_cycle is not None
                        and self.epochs_per_res_cycle > 0
                        and max_epochs % (self.epochs_per_res_cycle * n_res) == 0)
            if is_cyclic:
                cycle_num = epoch // (self.epochs_per_res_cycle * n_res) + 1
                total_cycles = max_epochs // (self.epochs_per_res_cycle * n_res)
                print(f"\n[Epoch {epoch}/{max_epochs-1}] Training x{train_res} (cycle {cycle_num}/{total_cycles}, cyclic mode)")
            else:
                print(f"\n[Epoch {epoch}/{max_epochs-1}] Training x{train_res} (standard mode)")

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

        # Determine which resolution to train (with gradients) this epoch
        res_index = self.get_current_resolution_idx()
        train_res = self.multires[res_index]
        # if self.global_rank == 0 and phase == "train":
        #     print(f"[Epoch {epoch}/{total_epochs-1}] Training x{train_res} resolution")
        
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
                
                # DIAGNOSTIC: Vérifier AVANT interpolation
                if self.global_rank == 0 and phase == "test" and res in [3, 1]:
                    for var_name in out[f"patch_x{coarser_res}"].keys():
                        coarse_pred = out[f"patch_x{coarser_res}"][var_name]
                        n_nan = torch.isnan(coarse_pred).sum().item()
                        print(f"[INTERP] BEFORE x{coarser_res}->x{res} | {var_name}: NaN={n_nan}/{coarse_pred.numel()} ({100*n_nan/coarse_pred.numel():.1f}%)")
                
                out[f"patch_x{coarser_res}_on_x{res}"] = self.interpolate_torch(
                    out[f"patch_x{coarser_res}"],
                    lon_coarse, lat_coarse,
                    lon_target, lat_target
                )
                
                # DIAGNOSTIC: Vérifier APRÈS interpolation (pour toutes les résolutions)
                if self.global_rank == 0 and phase == "test" and res in [3, 1]:
                    for var_name in out[f"patch_x{coarser_res}_on_x{res}"].keys():
                        interp_pred = out[f"patch_x{coarser_res}_on_x{res}"][var_name]
                        n_nan = torch.isnan(interp_pred).sum().item()
                        print(f"[INTERP] AFTER  x{coarser_res}->x{res} | {var_name}: NaN={n_nan}/{interp_pred.numel()} ({100*n_nan/interp_pred.numel():.1f}%)")
                
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

        # Pour la visualisation multi-résolution: interpoler x10 directement vers x1
        # Permet de visualiser la progression x10 → x3 → x1 sur la même grille
        if len(self.multires) >= 3 and f"patch_x{self.multires[0]}" in out:
            res_x10 = self.multires[0]  # 10
            res_x1 = self.multires[-1]  # 1
            if f"patch_x{res_x10}_on_x{res_x1}" not in out:
                # Récupérer coordonnées x10 et x1 depuis le batch
                lon_x10 = batch[f"patch_x{res_x10}"].lon_geo
                lat_x10 = batch[f"patch_x{res_x10}"].lat_geo
                lon_x1 = batch[f"patch_x{res_x1}"].lon_geo
                lat_x1 = batch[f"patch_x{res_x1}"].lat_geo

                # Interpoler x10 directement vers grille x1
                out[f"patch_x{res_x10}_on_x{res_x1}"] = self.interpolate_torch(
                    out[f"patch_x{res_x10}"],
                    lon_x10, lat_x10,
                    lon_x1, lat_x1
                )
                # Cropper pour matcher les dimensions temporelles de x1
                out[f"patch_x{res_x10}_on_x{res_x1}"] = self.crop_daw(
                    out[f"patch_x{res_x10}_on_x{res_x1}"], res_x1
                )

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

            # DIAGNOSTIC disabled for DDP (causes sync issues)
            # if self.global_rank == 0 and phase == "train":
            #     n_pixels_total = target.numel()
            #     n_valid_target = target.isfinite().sum().item()
            #     n_valid_sobel = mask.sum().item()
            #     pct_target = 100 * n_valid_target / n_pixels_total
            #     pct_sobel = 100 * n_valid_sobel / n_pixels_total
            #     print(f"[GRAD DIAG] var={var_name} | target valid: {pct_target:.1f}% | sobel valid: {pct_sobel:.1f}% (erosion: {pct_target-pct_sobel:.1f}%)")
            
            # Get inpainting mask if available
            inpaint_mask_grad = None
            if hasattr(batch, 'inpaint_mask'):
                inpaint_mask_grad = batch.inpaint_mask
    
            # Crop weight to match target temporal dimension
            weight_grad = self.get_optim_weight(res_key)
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
    
            # === GRAD LOSS : Gradients Sobel sur TOUS pixels valides (régularisation spatiale) ===
            # grad_loss = ||Sobel(pred) - Sobel(target)||^2 sur target.isfinite()
            # weighted_mse applique inpaint_mask pour pondérer (pas filtrer)
            grad_loss = self.weighted_mse(
                torch.where(mask, pred_sobel, torch.tensor(float('nan'), device=pred.device)) - tgt_sobel,
                weight_grad,
                inpaint_mask=inpaint_mask_grad,
                inpaint_weight_factor=self.inpaint_weight_factor
            )

            # DIAGNOSTIC disabled for DDP
            # if self.global_rank == 0 and phase == "train":
            #     print(f"[GRAD DIAG] grad_loss={grad_loss.item():.4f}")

            total_grad_loss += grad_loss
    
        # Prior / SRNN loss: measures how well BilinReconstructor reconstructs the target
        # NOUVELLE VERSION : Φ([state_final, covariates]) au lieu de Φ(input fixe)
        if hasattr(self.solver.solvers[f"solver_x{res}"], "prior_cost"):
            sbatch = self.format_batch_for_solver(batch)
            model = self.solver.solvers[f"solver_x{res}"].to(device)
            
            # Extraire state_final du output du solver
            state_final = out[tgt_key]  # (B, T, H, W)
            T = state_final.shape[1]
            
            # Construire dynamic_input : [state_final, covariates + spatial]
            # sbatch.input structure: [fusion_masquée (0:T), avhrr, pmw, covariates, spatial]
            covariables_and_spatial = sbatch.input[:, T:, :, :]  # (B, dim_in - T, H, W)
            dynamic_input = torch.cat([state_final, covariables_and_spatial], dim=1)  # (B, dim_in, H, W)
            
            # Φ([state_final, covs]) - prior dynamique !
            prior = model.prior_cost.forward_reconstructor(dynamic_input)
            
            # Crop prior weight to match target temporal dimension
            weight_prior = self.get_prior_weight(res_key)
            target_length_prior = state_final.shape[1]
            if weight_prior.shape[0] > target_length_prior:
                crop_total = weight_prior.shape[0] - target_length_prior
                start_idx = crop_total // 2
                weight_prior = weight_prior[start_idx:start_idx + target_length_prior, ...]

            # === PRIOR LOSS CORRECT : TOUJOURS sur tous pixels (régularisation) ===
            # prior_loss = ||state_final - φ([state_final, covs])||² sur tous pixels valides
            # C'est une régularisation, pas une loss de fidélité
            mask_prior = state_final.isfinite()
            n_valid_prior = mask_prior.sum()
            if n_valid_prior > 0:
                err = state_final - prior  # CHANGÉ: state_final au lieu de sbatch.tgt
                weighted_err = err * weight_prior[None, ...]
                total_prior_loss = (weighted_err[mask_prior] ** 2).sum() / n_valid_prior
            else:
                total_prior_loss = torch.tensor(0.0, device=prior.device, requires_grad=True)
        else:
            total_prior_loss = 0.0

        # Balanced loss: poids configurables via YAML (self.loss_weights)
        w = self.loss_weights
        training_loss = w['mse'] * loss + w['grad'] * total_grad_loss + w['prior'] * total_prior_loss

        # Stocker les losses pour le print concis et TensorBoard logging (fait dans training_step/validation_step)
        if phase == "train":
            # Récupérer les losses stockées dans base_step
            interp_val = self._step_losses.get('interp', 0.0) if hasattr(self, '_step_losses') else 0.0
            recons_val = self._step_losses.get('recons', 0.0) if hasattr(self, '_step_losses') else 0.0
            mse_val = loss.item() if hasattr(loss, 'item') else float(loss)
            grad_val = total_grad_loss.item() if hasattr(total_grad_loss, 'item') else float(total_grad_loss)
            prior_val = total_prior_loss.item() if hasattr(total_prior_loss, 'item') else float(total_prior_loss)
            total_raw = mse_val + grad_val + prior_val

            self.last_losses = {
                'loss': training_loss.item() if hasattr(training_loss, 'item') else float(training_loss),
                'mse': mse_val,  # Total MSE (interp + recons)
                'mse_interp': interp_val,  # Interpolation uniquement (X_B̄)
                'mse_recons': recons_val,  # Reconstruction uniquement (X_B)
                'grad': grad_val,  # Gradients Sobel (tous pixels)
                'prior': prior_val,  # Prior loss (tous pixels)
                # Ratios pour histogramme (proportions relatives)
                'mse_ratio': mse_val / total_raw if total_raw > 0 else 0.0,
                'grad_ratio': grad_val / total_raw if total_raw > 0 else 0.0,
                'prior_ratio': prior_val / total_raw if total_raw > 0 else 0.0
            }
            
            # Réinitialiser pour le prochain batch
            self._step_losses = {'interp': 0.0, 'recons': 0.0}
    
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
            inpaint_mask = batch.inpaint_mask

        total_loss = 0.0
        for i, var_name in enumerate(self.tgt_vars):
            if not hasattr(batch, var_name):
                raise ValueError(f"Batch does not contain variable '{var_name}'")
            target = getattr(batch, var_name)
            pred = out[var_name]  # (B, T, Y, X)
            mask = target.isfinite()
            
            # Crop weight to match target temporal dimension
            weight = self.get_optim_weight(res_key)
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
            
            # ==== SSL CORRECT : loss_interp (X_B̄) + loss_recons (X_B) ====
            # loss_interp : pixels masqués (capacité d'interpolation)
            # loss_recons : pixels visibles (fidélité aux observations)
            
            # DEBUG: Premier batch seulement
            if not hasattr(self, '_debug_base_step_printed'):
                self._debug_base_step_printed = True
                print(f"\n[DEBUG base_step] phase={phase}, res={res}")
                print(f"[DEBUG base_step] target shape: {target.shape}")
                print(f"[DEBUG base_step] pred shape: {pred.shape}")
                print(f"[DEBUG base_step] weight shape: {weight.shape}")
                if inpaint_mask_cropped is not None:
                    print(f"[DEBUG base_step] inpaint_mask_cropped shape: {inpaint_mask_cropped.shape}")
                    print(f"[DEBUG base_step] inpaint_mask sum: {inpaint_mask_cropped.sum().item()}/{inpaint_mask_cropped.numel()}")
                    print(f"[DEBUG base_step] Mode: SSL (loss_interp on masked + loss_recons on visible)")
                else:
                    print(f"[DEBUG base_step] inpaint_mask: None")
                    print(f"[DEBUG base_step] Mode: INFERENCE (loss on all valid pixels)\n")
            
            # Loss interpolation : pixels masqués (X_B̄)
            loss_interp = torch.tensor(0.0, device=pred.device, requires_grad=True)
            if inpaint_mask_cropped is not None and inpaint_mask_cropped.sum() > 0:
                interp_mask = (inpaint_mask_cropped > 0) & target.isfinite()
                n_interp = interp_mask.sum()
                if n_interp > 0:
                    err = pred - target
                    weighted_err = err * weight[None, ...]
                    loss_interp = (weighted_err[interp_mask] ** 2).sum() / n_interp
            
            # Loss reconstruction : pixels visibles (X_B)
            loss_recons = torch.tensor(0.0, device=pred.device, requires_grad=True)
            if inpaint_mask_cropped is not None:
                recons_mask = (~(inpaint_mask_cropped > 0)) & target.isfinite()
                n_recons = recons_mask.sum()
                if n_recons > 0:
                    err = pred - target
                    weighted_err = err * weight[None, ...]
                    loss_recons = (weighted_err[recons_mask] ** 2).sum() / n_recons
            else:
                # Mode inference : tous pixels valides
                mask = target.isfinite()
                n_valid = mask.sum()
                if n_valid > 0:
                    err = pred - target
                    weighted_err = err * weight[None, ...]
                    loss_recons = (weighted_err[mask] ** 2).sum() / n_valid
            
            # Loss totale = interpolation + reconstruction
            loss = loss_interp + loss_recons
            
            # Stocker pour logs
            if not hasattr(self, '_step_losses'):
                self._step_losses = {'interp': 0.0, 'recons': 0.0}
            self._step_losses['interp'] += loss_interp.item() if hasattr(loss_interp, 'item') else float(loss_interp)
            self._step_losses['recons'] += loss_recons.item() if hasattr(loss_recons, 'item') else float(loss_recons)
                    
            total_loss += loss

        return total_loss, out

    def reconstruct(self, dl, items, daw, time, weight=None, save_patches_dir=None, res=None):
        """
        takes as input a list of tensor of dimensions (V, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims
        items: list of torch tensor corresponding to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting
        save_patches_dir: if not None, save individual patches for debugging
        res: resolution (10, 3, or 1) for labeling the coverage plot
        """

        if weight is None:
            weight = np.ones(list(dl.dataset.patch_dims.values()))
        weight = torch.tensor(weight)

        nvars = items[0].shape[0]

        result_tensor = torch.full((nvars, 1, dl.dataset.da_dims['lat'], dl.dataset.da_dims['lon']),
                                   float('nan'))
        count_tensor = torch.zeros((nvars, 1, dl.dataset.da_dims['lat'], dl.dataset.da_dims['lon']))

        # Get coordinates: for single day mode, use all coords; otherwise slice by daw
        all_coords = dl.dataset.get_coords()
        if len(all_coords) == len(items):
            coords = all_coords
        else:
            coords = all_coords[(daw*len(items)):((daw+1)*len(items))]

        # Collect patch positions for visualization
        patch_positions = []

        for idx, item in enumerate(items):
            c = coords[idx]
            iy = [np.where(dl.dataset.lat_1d == y)[0][0] for y in c.lat.values]
            ix = [np.where(dl.dataset.lon_1d == x)[0][0] for x in c.lon.values]

            # Save individual patches for inspection (optional) - ONLY for x10 to avoid too many files
            if save_patches_dir is not None and res == 10:
                patch_dir = Path(save_patches_dir) / "patches"
                patch_dir.mkdir(parents=True, exist_ok=True)

                # Save first variable (pred_sst) as image
                patch_data = item[0].cpu().numpy()  # First var, shape (T, H, W) or (H, W)
                if patch_data.ndim == 3:
                    patch_data = patch_data[patch_data.shape[0]//2]  # Middle timestep

                # Use fixed indices for surfmask slicing
                iy_start, iy_end = iy[0], iy[-1]
                ix_start, ix_end = ix[0], ix[-1]
                surfmask_patch = dl.dataset.mask[iy_start:iy_end+1, ix_start:ix_end+1]
                lat_range = (c.lat.values.min(), c.lat.values.max())
                lon_range = (c.lon.values.min(), c.lon.values.max())
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                im1 = ax1.imshow(patch_data, origin='lower', cmap='RdYlBu_r')
                ax1.set_title(f"Patch {idx} pred_sst: lat=[{lat_range[0]:.1f},{lat_range[1]:.1f}], lon=[{lon_range[0]:.1f},{lon_range[1]:.1f}]")
                plt.colorbar(im1, ax=ax1)
                im2 = ax2.imshow(surfmask_patch, origin='lower', cmap='tab10', vmin=0, vmax=3)
                ax2.set_title(f"Surfmask (0=land, 1=ocean, 2=ice-water, 3=ice)")
                cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
                cbar2.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
                fig.savefig(patch_dir / f"patch_{idx:03d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)

            # CRITICAL FIX: Use iy/ix values directly, not as indices
            # iy/ix are already grid indices (e.g., [0,1,2,...,255] or [104,105,...,359])
            # NOT list positions to index again!
            iy_start, iy_end = iy[0], iy[-1]
            ix_start, ix_end = ix[0], ix[-1]

            # Store patch position for rectangle overlay
            patch_positions.append((iy_start, iy_end, ix_start, ix_end))

            result_tensor[:, 0, iy_start:iy_end+1, ix_start:ix_end+1] = torch.where(
                torch.isnan(result_tensor[:, 0, iy_start:iy_end+1, ix_start:ix_end+1]),
                0.,
                result_tensor[:, 0, iy_start:iy_end+1, ix_start:ix_end+1])
            result_tensor[:, 0, iy_start:iy_end+1, ix_start:ix_end+1] += torch.squeeze(item * weight)
            count_tensor[:, 0, iy_start:iy_end+1, ix_start:ix_end+1] += weight

        # Normalize by count (weighted average) - FIXED: removed double division
        result_tensor = torch.where(count_tensor > 0, result_tensor / count_tensor, result_tensor)
        coverage = (count_tensor > 0).float().mean()
        # Optional: save diagnostic plot
        if save_patches_dir is not None:
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            # Resolution label for title
            res_label = f"x{res}" if res is not None else "unknown"

            # 1. Coverage map (count_tensor) - SANS colorbar
            ax = axes[0, 0]
            ax.imshow(count_tensor[0, 0].numpy(), origin='lower', cmap='viridis', aspect='auto')
            ax.set_title(f'Coverage (count_tensor) - {coverage*100:.1f}% non-zero - {res_label}')
            ax.set_xlabel('lon idx'); ax.set_ylabel('lat idx')

            # 2. Prediction (first variable = pred_sst) WITH PATCH RECTANGLES
            ax = axes[0, 1]
            data_pred = result_tensor[0, 0].numpy()

            # 4. Target (second variable = tgt_sst) - calculer min/max commun avec pred
            ax_tgt = axes[1, 1]
            if nvars > 1:
                data_tgt = result_tensor[1, 0].numpy()
                # Calculer vmin/vmax commun pour pred et tgt
                vmin_common = min(np.nanmin(data_pred), np.nanmin(data_tgt))
                vmax_common = max(np.nanmax(data_pred), np.nanmax(data_tgt))
            else:
                vmin_common, vmax_common = -3, 3

            # Plot pred avec range commun
            ax.imshow(data_pred, origin='lower', cmap='RdYlBu_r', aspect='auto', vmin=vmin_common, vmax=vmax_common)
            ax.set_title(f'pred_sst - range [{vmin_common:.2f}, {vmax_common:.2f}] (normalized) - {res_label}')
            ax.set_xlabel('lon idx'); ax.set_ylabel('lat idx')
            plt.colorbar(ax.images[0], ax=ax)

            # Draw patch rectangles on prediction
            for (iy_start, iy_end, ix_start, ix_end) in patch_positions:
                height = iy_end - iy_start + 1
                width = ix_end - ix_start + 1
                rect = Rectangle((ix_start, iy_start), width, height,
                                linewidth=1, edgecolor='cyan', facecolor='none', alpha=0.7)
                ax.add_patch(rect)

            # 3. Valid data mask
            ax = axes[1, 0]
            valid = ~np.isnan(data_pred)
            ax.imshow(valid.astype(float), origin='lower', cmap='gray', aspect='auto')
            ax.set_title(f'Valid data mask (white=data, black=NaN)')
            ax.set_xlabel('lon idx'); ax.set_ylabel('lat idx')

            # Plot tgt avec range commun
            if nvars > 1:
                ax_tgt.imshow(data_tgt, origin='lower', cmap='RdYlBu_r', aspect='auto', vmin=vmin_common, vmax=vmax_common)
                ax_tgt.set_title(f'tgt_sst - range [{vmin_common:.2f}, {vmax_common:.2f}] (normalized)')
                ax_tgt.set_xlabel('lon idx'); ax_tgt.set_ylabel('lat idx')
                plt.colorbar(ax_tgt.images[0], ax=ax_tgt)
            else:
                # Fallback: show surfmask if available
                if hasattr(dl.dataset, 'mask'):
                    im = ax_tgt.imshow(dl.dataset.mask, origin='lower', cmap='tab10', vmin=0, vmax=3, aspect='auto')
                    ax_tgt.set_title('Surfmask (0=land, 1=ocean, 2=ice-water, 3=ice)')
                    ax_tgt.set_xlabel('lon idx'); ax_tgt.set_ylabel('lat idx')
                    cbar = plt.colorbar(im, ax=ax_tgt, ticks=[0, 1, 2, 3])
                    cbar.set_ticklabels(['Land', 'Ocean', 'Ice-water', 'Ice'])
                else:
                    ax_tgt.text(0.5, 0.5, 'No target data available', ha='center', va='center', transform=ax_tgt.transAxes)

            plt.tight_layout()
            fig_path = save_patches_dir / f'reconstruction_coverage_{res_label}_daw{daw}.png'
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"  [COVERAGE] Saved {res_label} coverage plot: {fig_path.name}")
            plt.close(fig)

        # result_tensor shape: (nvars, 1, lat, lon)
        
        data_vars = {}
        # Les noms des variabels sont dans self.test_quantities (pred_sst, tgt_sst, etc)
        # Assumons que l'ordre est respecté
        
        var_names = self.test_quantities
        if len(var_names) != nvars:
            print(f"[RECONSTRUCT WARNING] nvars={nvars} but only {len(var_names)} names found: {var_names}")
            # Fallback names
            if nvars > len(var_names):
                for i in range(len(var_names), nvars):
                    var_names.append(f"var_{i}")
        
        for i in range(nvars):
            # Extraire (1, lat, lon) -> (lat, lon)
            grid = result_tensor[i, 0, :, :]
            # Ajouter dimension time -> (1, lat, lon)
            grid = grid.unsqueeze(0)
            
            vname = var_names[i]
            data_vars[vname] = (["time", "lat", "lon"], grid.cpu().numpy())
            
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": [time],
                "lon": dl.dataset.lon_1d,
                "lat": dl.dataset.lat_1d,
                "lon_2d": (["lat", "lon"], dl.dataset.lon_2d),
                "lat_2d": (["lat", "lon"], dl.dataset.lat_2d)
            }
        )
        return ds

    def aggregate_batches_one_domain(self, idx_daw, idx_rec,
                                     test_data, 
                                     dataloader_idx=None,
                                     use_datamodule=False):

        dl = self.trainer.test_dataloaders[self.dataloader_keys[dataloader_idx]]
        
        res = self.multires[dataloader_idx]
        last = self.len_daw[res]
        res_key = f"patch_x{res}"

        netcdf_final = []
                                       
        for i in tqdm(idx_rec, desc="Reconstructing lead times", leave=True):
            time = dl.dataset.times[-last:][idx_daw+i]
            #print("Reconstructing LEADTIME "+str(i))
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
                # Save patches for ALL resolutions (x10, x3, x1) for coverage visualization
                save_dir = None
                if i == 0:  # Only for first timestep
                    save_dir = self.outputs_dir / self.test_run_id / "test" / "coverage"
                    save_dir.mkdir(parents=True, exist_ok=True)

                rec_da = self.reconstruct(dl,
                            [ test_data[j][:,[i],:,:].cpu() for j in range(nbatch) ],
                            idx_daw, time,
                            self.rec_weight[res_key].cpu().numpy()[[i],:,:],
                            save_patches_dir=save_dir,
                            res=res
                    )
            # rec_da est déjà un Dataset avec les variables (pred_sst, tgt_sst, etc.)
            test_data_ldt = rec_da
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
        if hasattr(time_values, 'ndim') and time_values.ndim > 1:
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

        netcdf_final = []

        def unnormalize(varname, data):
            if varname == "tgt_sst":
                stats = self.norm_stats["tgt_sst"]
            else:
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
            # Construire le dict de variables dénormalisées
            unnorm_vars = {}
            for var_full in self.tgt_vars:  # ex: "tgt_sst"
                _, var_name = var_full.split("_", 1)  # ex: "sst"
                # Dénormaliser prédiction ET target
                unnorm_vars[f"pred_{var_name}"] = (("time", "lat", "lon"),
                                                    unnormalize(var_full, test_data_uniq[f"pred_{var_name}"].data))
                unnorm_vars[f"tgt_{var_name}"] = (("time", "lat", "lon"),
                                                   unnormalize(var_full, test_data_uniq[f"tgt_{var_name}"].data))

            # Créer un nouveau dataset avec les variables dénormalisées
            test_data_unnorm = test_data_uniq.assign(unnorm_vars)
            if metrics:
                metric_data = test_data_unnorm.pipe(self.pre_metric_fn),
                metrics = pd.Series({
                    metric_n: metric_fn(metric_data)
                    for metric_n, metric_fn in self.metrics.items()
                })
                print(metrics.to_frame(name="Metrics").to_markdown())
            # save NetCDFs
            time = [ dt.strptime(str(t)[:10], "%Y-%m-%d").strftime("%Y%m%d") for t in test_data_unnorm.time.data ]
            file = f'test_data_{time[0]}_{time[-1]}_patch_x{res}.nc'
            if self.logger and write_netcdf:
                netcdf_dir = self.outputs_dir / self.test_run_id / "test" / "netcdf"
                netcdf_dir.mkdir(parents=True, exist_ok=True)
                out_path = netcdf_dir / file
                test_data_unnorm.to_netcdf(out_path)
                print("\n")
                print(out_path)
                print("\n")
                if metrics:
                    self.logger.log_metrics(metrics.to_dict())
            # stack each daw
            netcdf_final.append(test_data_uniq)

        # merge all time steps in a dictionary
        return { f"daw_{i}": nc for i, nc in enumerate(netcdf_final) }
        #return xr.concat(netcdf_final, dim="daw").assign_coords(daw=torch.unique(daws)).sortby("daw")

    def convert_xr_to_batch(self, coarse, batch, spatial_sel=False, verbose=False):
        """
        Convert an xarray.Dataset (coarse) to a dictionary of PyTorch tensors,
        matching the batch's spatial extent.
        Args:
            coarse (xr.Dataset): dict of xarray with dims (time, lat, lon)
            batch (TrainingItem): Batch with spatial coordinates (lat_geo, lon_geo)
            spatial_sel (bool): If True, crop coarse spatially to match batch bounds
            verbose (bool): If True, print debug info
        Returns:
            coarse_dict: dict with same keys as coarse.data_vars, each of shape (B, T, H, W)
        """
        # CRITICAL FIX: Use timestamps from coarse datasets, NOT from batch
        # When interpolating x3→x1, batch has 9 timesteps but coarse (after crop) has only 5
        # We should use the temporal extent of coarse, not batch
        
        # Get timestamps from first dataset in coarse (all should have same times after crop)
        first_key = list(coarse.keys())[0]
        coarse_times = coarse[first_key].time.values  # (T,) - datetime64[ns]
        T = len(coarse_times)
        
        # Get batch size and spatial dimensions from batch
        nbatch = batch.lat_geo.shape[0] if hasattr(batch, 'lat_geo') else batch.lat.shape[0]
        H, W = batch.lat.shape[1], batch.lon.shape[2]
        
        coarse_dict = {}
        # Pour chaque variable du Dataset (add lat_geo and lon_geo for interpolation)
        # CRITICAL FIX: Map tgt_vars to pred_vars in the xarray!
        # self.tgt_vars = ["tgt_sst"] but xarray contains "pred_sst" (prediction) and "tgt_sst" (target with clouds)
        # We want to interpolate the PREDICTION, not the target!
        for var in self.tgt_vars + ["time", "lat", "lon", "lat_geo", "lon_geo"]:
            # Map target variable names to prediction variable names in the xarray
            if var.startswith("tgt_"):
                xr_var = "pred_" + var[4:]  # "tgt_sst" -> "pred_sst"
            else:
                xr_var = var
            B_array = []
            for i in range(nbatch):
                # Extract 1D coords from 2D grids: (nlat, nlon)
                # Use lat_geo/lon_geo (geographic) if available, else fall back to lat/lon (normalized)
                if hasattr(batch, 'lon_geo') and hasattr(batch, 'lat_geo'):
                    lons_i = batch.lon_geo[i, 0, :].cpu().numpy()  # First row -> (nlon,)
                    lats_i = batch.lat_geo[i, :, 0].cpu().numpy()  # First col -> (nlat,)
                else:
                    lons_i = batch.lon[i, 0, :].cpu().numpy()  # First row -> (nlon,)
                    lats_i = batch.lat[i, :, 0].cpu().numpy()  # First col -> (nlat,)
                
                # temporal selection: all datasets in coarse should have same timestamps after crop
                # No need to search - just use first dataset
                if verbose and i == 0:  # Only print for first batch item
                    print(f"[CONVERT] Using coarse times (length={T}): {coarse_times[:min(3,T)]}...{coarse_times[-min(3,T):]}")
                
                # Use first dataset (all have same temporal extent after crop)
                sel_time = coarse[first_key]
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
                
                # Handle lat_geo and lon_geo: reconstruct 2D meshgrid from 1D xarray coords
                if var == "lat_geo":
                    lat_1d = sel_patch["lat"].values  # (nlat,) - 1D from xarray
                    lon_1d = sel_patch["lon"].values  # (nlon,) - 1D from xarray
                    if i == 0 and verbose:  # Print only for first batch item
                        print(f"[MESHGRID] Creating lat_geo: lat_1d.shape={lat_1d.shape}, lon_1d.shape={lon_1d.shape}")
                    lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)  # Create 2D grids
                    arr = lat_mesh  # (nlat, nlon) - 2D grid
                    if i == 0 and verbose:
                        print(f"[MESHGRID] lat_geo (meshgrid) shape: {arr.shape}")
                elif var == "lon_geo":
                    lat_1d = sel_patch["lat"].values  # (nlat,)
                    lon_1d = sel_patch["lon"].values  # (nlon,)
                    lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)  # Create 2D grids
                    arr = lon_mesh  # (nlat, nlon) - 2D grid
                    if i == 0 and verbose:
                        print(f"[MESHGRID] lon_geo (meshgrid) shape: {arr.shape}")
                else:
                    arr = sel_patch[xr_var].values  # (T, H, W) - use xr_var to get pred_sst instead of tgt_sst
                
                if var == "time":
                    arr = arr.astype('datetime64[ns]').astype('int64')
                if var in ["time", "lat", "lon"]:
                    # Only expand time/lat/lon (normalized coords), NOT lat_geo/lon_geo (already 2D meshgrid)
                    arr = np.expand_dims(arr,axis=0)
                B_array.append(torch.from_numpy(arr).float())
            coarse_dict[var] = torch.stack(B_array, dim=0)
            fields = batch._fields
            # Construire un nouveau dict avec les valeurs de coarse_dict
            complete_dict = {field: coarse_dict.get(field, None) for field in fields}
        return  type(batch)(**complete_dict)

    def on_test_start(self):
        # Créer un run_id unique pour toute la session de test
        self.test_run_id = dt.now().strftime("%Y%m%d_%H%M%S")
        
        # Stocker les dataloader keys dans l'ordre des indices
        self.dataloader_keys = list(self.trainer.test_dataloaders.keys())
    
        # Calculer le nombre de batchs par dataloader
        self.num_test_batches = {
            i: len(dl)
            for i, dl in enumerate(self.trainer.test_dataloaders.values())}
        
        # Initialiser le stockage pour la visualisation des patches
        self.viz_patches = {10: [], 3: [], 1: []}

    def is_last_batch(self, batch_idx, dataloader_idx):
        """
        Détermine si c'est le dernier batch du dataloader actuel.
        Prend en compte limit_test_batches si défini dans le trainer.
        """
        total_batches = self.num_test_batches[dataloader_idx]

        # Prendre en compte limit_test_batches si défini
        if hasattr(self.trainer, 'limit_test_batches') and self.trainer.limit_test_batches:
            limit = self.trainer.limit_test_batches
            if isinstance(limit, int) and limit > 0:
                total_batches = min(total_batches, limit)
            elif isinstance(limit, float) and 0.0 < limit < 1.0:
                # Cas où limit est une fraction (ex: 0.1 = 10% des batches)
                total_batches = max(1, int(total_batches * limit))
        return batch_idx == total_batches - 1
    
    def test_step(self, batch, batch_idx, dataloader_idx=None):

        res = self.multires[dataloader_idx] #identifie quelle resolution on traite (idx 0 pour res 10, 1 pour res 3 et 2 pour res 1)
        res_key = f"patch_x{res}"
        last = self.len_daw[res]

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
            # Use geographic coordinates (in degrees) for interpolation - pass full 2D grids
            lon_target = batch.lon_geo  # (B, nlat, nlon) - 2D grid in geographic degrees
            lat_target = batch.lat_geo  # (B, nlat, nlon) - 2D grid in geographic degrees
            
            # identify batch daw / coarse daw equivalence for selection
            coarse = self.aggregate_results[f"patch_x{coarser_res}"]
            
            # Apply SYMMETRIC temporal crop (centered on target day)
            coarse = {
               k: v.isel(time=slice((self.len_daw[coarser_res] - last) // 2,
                                   (self.len_daw[coarser_res] - last) // 2 + last))
               for k, v in coarse.items()
            }
            
            # if batch_idx == 0:
            #     for k, v in list(coarse.items())[:1]:  # Just first key
            #         print(f"[TEST_STEP] coarse['{k}'] time length AFTER crop: {len(v.time)}")
            
            coarse = self.convert_xr_to_batch(coarse, batch, verbose=False)
            lon_coarse = coarse.lon_geo
            lat_coarse = coarse.lat_geo

            itrp_coarse = self.interpolate_torch(coarse._asdict(),
                                                 lon_coarse, lat_coarse,
                                                 lon_target, lat_target)

            #itrp_coarse = self.crop_daw(itrp_coarse,res)
            # modify batch to work on anomaly compared to coarser resolution
            batch = self.update_batch_as_anomaly(batch, itrp_coarse)

        sbatch = self.format_batch_for_solver(batch)
        out = self(batch=sbatch, res=res)
        out = self.split_tensor_to_dict(out)
        
        # SAVE RAW SOLVER OUTPUT (avant addition avec coarse) pour collection de patches
        out_raw_solver = {k: v.clone() for k, v in out.items()}
        
        # add coarser resolution to output
        itrp_coarse_for_viz = itrp_coarse if dataloader_idx > 0 else None

        if dataloader_idx > 0:
            # IMPORTANT a fix: Il faut nettoyer les NaN pixelisés de itrp_coarse AVANT addition
            # car sinon les NaN viennent de l'interpolation depuis x3/x10 et se propagent
            for var in self.tgt_vars:
                n_timesteps = itrp_coarse[var].shape[1]
                surfmask_slice = (batch.surfmask.unsqueeze(1).expand(-1, n_timesteps, -1, -1)
                                 if batch.surfmask.ndim == 3
                                 else batch.surfmask[:, :n_timesteps, :, :])
                # 1) Remplacer TOUS les NaN par 0 (anomalie nulle là où x3/x10 avait des terres)
                itrp_coarse[var] = torch.nan_to_num(itrp_coarse[var], nan=0.0)
                itrp_coarse[var] = torch.where(surfmask_slice == 0., torch.tensor(float('nan'), device=itrp_coarse[var].device), itrp_coarse[var])

            out = {k: out[k] + itrp_coarse[k] for k in out}
            out_after_add = {k: v.clone() for k, v in out.items()}
        else:
            out_after_add = None
            
        for i, var in enumerate(self.tgt_vars):
            # Adapter surfmask à la dimension temporelle de out[var]
            n_timesteps = out[var].shape[1]
            surfmask_slice = (batch.surfmask.unsqueeze(1).expand(-1, n_timesteps, -1, -1) if batch.surfmask.ndim == 3
                             else batch.surfmask[:,:n_timesteps,:,:])
            # surfmask: 0=terre (mettre NaN), 1=ocean, 2=eau-glace, 3=glace (garder valeurs)
            out[var] = torch.where(surfmask_slice == 0., np.nan, out[var])

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

        # --- COLLECTION POUR VISUALISATION ---
        # On collecte tous les patches, le tri se fera à la fin
        try:
            # On suppose que la première variable est la SST
            main_var = self.tgt_vars[0]
            
            # Indices temporels
            t_mid_pred = out_norm[main_var].shape[1] // 2
            t_mid_tgt = tgt_norm[main_var].shape[1] // 2
            
            batch_size = out_norm[main_var].shape[0]
            
            # Récupérer PMW si disponible
            pmw_key = 'pmw_av'
            if hasattr(batch, pmw_key):
                pmw_tensor = getattr(batch, pmw_key)
                t_mid_pmw = pmw_tensor.shape[1] // 2
            else:
                pmw_tensor = None

            for b in range(batch_size):
                # Extraire Target (SST)
                # Note: tgt_norm contient déjà la somme (res + coarse) pour les résolutions fines
                tgt_img = tgt_norm[main_var][b, t_mid_tgt, :, :].detach().cpu().numpy()
                
                # Extraire Prediction FINALE (après masque)
                pred_img = out_norm[main_var][b, t_mid_pred, :, :].detach().cpu().numpy()
                if pred_img.ndim == 3: pred_img = pred_img[0] # Handle channel dim if present
                
                # DIAGNOSTIC x3 et x1: Extraire les 3 étapes intermédiaires
                # Étape 1: Sortie solver brute (résiduel)
                if out_raw_solver is not None and main_var in out_raw_solver:
                    solver_raw_img = out_raw_solver[main_var][b, t_mid_pred, :, :].detach().cpu().numpy()
                    if solver_raw_img.ndim == 3: solver_raw_img = solver_raw_img[0]
                else:
                    solver_raw_img = None

                # Étape 2: Interpolation coarse (avant addition)
                if dataloader_idx > 0 and itrp_coarse_for_viz is not None and main_var in itrp_coarse_for_viz:
                    itrp_coarse_img = itrp_coarse_for_viz[main_var][b, t_mid_pred, :, :].detach().cpu().numpy()
                    if itrp_coarse_img.ndim == 3: itrp_coarse_img = itrp_coarse_img[0]
                else:
                    itrp_coarse_img = None

                # Étape 3: Après addition (solver + interp)
                if dataloader_idx > 0 and out_after_add is not None and main_var in out_after_add:
                    pred_after_add_img = out_after_add[main_var][b, t_mid_pred, :, :].detach().cpu().numpy()
                    if pred_after_add_img.ndim == 3: pred_after_add_img = pred_after_add_img[0]
                else:
                    pred_after_add_img = None
                
                # FLIP vertical pour correspondre à la convention géographique (nord en haut)
                # PyTorch stocke avec Y croissant vers le bas, mais lat géo croît vers le nord
                tgt_img = np.flipud(tgt_img)
                pred_img = np.flipud(pred_img)
                if solver_raw_img is not None:
                    solver_raw_img = np.flipud(solver_raw_img)
                if itrp_coarse_img is not None:
                    itrp_coarse_img = np.flipud(itrp_coarse_img)
                if pred_after_add_img is not None:
                    pred_after_add_img = np.flipud(pred_after_add_img)
                
                # Extraire surfmask pour diagnostic (on skip PMW pour économiser mémoire)
                if hasattr(batch, 'surfmask'):
                    surfmask_tensor = batch.surfmask
                    if surfmask_tensor.ndim == 3:
                        surfmask_img = surfmask_tensor[b, :, :].detach().cpu().numpy()
                    elif surfmask_tensor.ndim == 4:
                        surfmask_img = surfmask_tensor[b, 0, :, :].detach().cpu().numpy()
                    else:
                        surfmask_img = None
                    
                    # FLIP surfmask aussi pour correspondre aux autres images
                    if surfmask_img is not None:
                        surfmask_img = np.flipud(surfmask_img)
                else:
                    surfmask_img = None
                
                # ID unique pour le patch
                patch_id = len(self.viz_patches[res])
                
                if len(self.viz_patches[res]) < 140:
                    self.viz_patches[res].append({
                        'id': patch_id,
                        'target': tgt_img,
                        'prediction': pred_img,
                        'solver_raw': solver_raw_img,        # Étape 1: sortie solver brute
                        'itrp_coarse': itrp_coarse_img,      # Étape 2: interpolation coarse
                        'pred_after_add': pred_after_add_img,  # Étape 3: solver + interp
                        'surfmask': surfmask_img,
                        'res': res
                    })
        except Exception as e:
            if batch_idx == 0:
                print(f"[VIZ WARNING] Failed to collect patches for res {res}: {e}")
        # -------------------------------------

        # stacked has shape (B,V,T,H,W) with V the number of variables
        # CRITICAL: detach and move to CPU to avoid GPU memory accumulation (OOM after ~1000 patches)
        self.test_data[res_key].append(stacked.detach().cpu())

        # Stocker uniquement le timestep CENTRAL pour l'agrégation
        # batch.time shape: (B, nlat, nlon) - grille spatiale remplie d'une valeur unique (jour normalisé)
        # On prend un seul pixel car tous ont la même valeur
        central_times = batch.time[:, 0, 0]  # Shape: (B,) - une valeur par patch
        self.test_times[res_key].append(central_times.cpu())  # Also move to CPU

        # if last batch, agreggate (as an xarray dataset with the estimation for a given resolution)
        if self.is_last_batch(batch_idx, dataloader_idx):
            # Flatten test_data: [[tensor1], [tensor2], ...] -> [tensor1, tensor2, ...]
            self.test_data[res_key] = list(itertools.chain(*self.test_data[res_key]))
            # Concatenate all central times into a single 1D tensor
            self.test_times[res_key] = torch.cat(self.test_times[res_key], dim=0)  # Shape: (N,) où N = total patches
            # last = self.len_daw[res] contient la vraie longueur temporelle (15, 9, ou 5)
            # batch.time est une grille spatiale (B, nlat, nlon), pas temporelle
            if dataloader_idx == (len(self.multires)-1):
                idx_rec = np.arange(last)
                write_netcdf = True
            else:
                idx_rec = np.arange(last)
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
        print(f"\n")
        print("[VISUALIZATION]")
        
        save_dir = self.outputs_dir / self.test_run_id / "test" / "analysis"
        save_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self, 'viz_patches'):
            from contrib.SST.visualization import plot_patch_analysis, plot_spectral_analysis
            patches_config = {10: 4, 3: 8, 1: 16}
            max_land_fraction = 0.75  # Exclure les patchs avec plus de 75% de terre

            for res, patches in self.viz_patches.items():
                if not patches:
                    continue

                # Filtrer les patchs avec trop de terre (surfmask==0 → terre)
                filtered_patches = []
                for p in patches:
                    if 'surfmask' in p and p['surfmask'] is not None:
                        land_fraction = np.mean(p['surfmask'] == 0)
                        if land_fraction <= max_land_fraction:
                            filtered_patches.append(p)
                    else:
                        filtered_patches.append(p)  # Garder si pas de surfmask

                n_total = len(patches)
                n_filtered = len(filtered_patches)
                n_select = patches_config.get(res, 4)

                # Sélection avec pas régulier parmi les patchs filtrés
                if n_filtered <= n_select:
                    selected_patches = filtered_patches
                else:
                    selected_indices = np.linspace(0, n_filtered - 1, n_select, dtype=int)
                    selected_patches = [filtered_patches[i] for i in selected_indices]

                print(f"  [x{res}] Plotting {len(selected_patches)}/{n_filtered} patches (filtered from {n_total}, max {int(max_land_fraction*100)}% land)")
                
                # Plot Patch Analysis (Target, Pred, PMW, Error)
                try:
                    plot_patch_analysis(selected_patches, save_dir, title_suffix=f"res_x{res}")
                except Exception as e:
                    print(f"    Patch analysis failed: {e}")
                try:
                    plot_spectral_analysis(patches, save_dir, title_suffix=f"res_x{res}")
                except Exception as e:
                    print(f"    Spectral analysis failed: {e}")
        # 2. Visualisation de la reconstruction globale (existante)
        # Vérifier si on a des résultats agrégés
        if not hasattr(self, 'aggregate_results') or not self.aggregate_results:
            print("[TEST] No aggregate_results found, skipping global reconstruction visualization")
            return
        
        # Récupérer la résolution finale (x1)
        final_res = self.multires[-1]
        final_res_key = f"patch_x{final_res}"
        
        if final_res_key not in self.aggregate_results:
            print(f"[TEST] No results for {final_res_key}, skipping global reconstruction visualization")
            return
        
        # Les résultats sont un dict de xarray.Dataset par DAW
        # Avec mode single_day, il n'y a qu'une seule DAW (daw_0)
        results_dict = self.aggregate_results[final_res_key]
        
        if 'daw_0' not in results_dict:
            print(f"[TEST] No daw_0 in results, available keys: {list(results_dict.keys())}")
            return
        
        final_data = results_dict['daw_0']  # xarray.Dataset
        
        # Log des informations sur le test (simplifié et aéré)
        print(f"\n[RECONSTRUCTION] {final_res_key}")
        print(f"  Time: {final_data.time.values[0]} -> {final_data.time.values[-1]}")
        print(f"  Grid: {final_data.sizes['lat']} x {final_data.sizes['lon']}")
        print(f"  Variables: {', '.join(list(final_data.data_vars))}")
        print(f"\n")
        
        # Créer des visualisations de la carte complète
        if self.global_rank == 0:
            try:
                from contrib.SST.visualization import plot_test_reconstruction
                plot_test_reconstruction(final_data, save_dir)
                
            except Exception as e:
                print(f"[TEST ERROR] Failed to generate visualizations: {e}")
                import traceback
                traceback.print_exc()

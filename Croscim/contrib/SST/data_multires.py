from contrib.SST.data import XrDataset, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES, concatenate, fast_coarsen_xr, fast_pool
from contrib.SST.load_data import organize_by_resolution
import torch
import torch.nn.functional as F
import pandas as pd
import xarray as xr
import os
import numpy as np

import gc


def pad_dataset(ds, pad_lat=0, pad_lon=0):
    """
    Pad SST Dataset (time, lat, lon) avec extension des coordonnées 1D.
    """
    dy = float((ds.lat[1] - ds.lat[0]).item())
    dx = float((ds.lon[1] - ds.lon[0]).item())

    pad_lat_before = pad_lat // 2
    pad_lat_after = pad_lat - pad_lat_before
    pad_lon_before = pad_lon // 2
    pad_lon_after = pad_lon - pad_lon_before

    ds_padded = ds.pad(lat=(pad_lat_before, pad_lat_after), lon=(pad_lon_before, pad_lon_after), constant_values=np.nan)

    new_lat = np.concatenate([ds.lat[0].item() - dy * np.arange(pad_lat_before, 0, -1), ds.lat.values, ds.lat[-1].item() + dy * np.arange(1, pad_lat_after + 1)])
    new_lon = np.concatenate([ds.lon[0].item() - dx * np.arange(pad_lon_before, 0, -1), ds.lon.values, ds.lon[-1].item() + dx * np.arange(1, pad_lon_after + 1)])

    ds_padded = ds_padded.assign_coords(lat=("lat", new_lat), lon=("lon", new_lon))

    return ds_padded

class XrDatasetMultiResTrain(XrDataset):

    def __init__(self, multires=[10, 3, 1], precomputed=True, *args, **kwargs):
        # import os
        # print(f"[INIT] Worker PID={os.getpid()} initializing XrDatasetMultiResTrain", flush=True)
        
        # Extract enable_patch_filtering before passing to parent
        # (parent class doesn't accept this argument)
        self.enable_patch_filtering = kwargs.pop('enable_patch_filtering', True)
        
        # print(f"[INIT] Worker PID={os.getpid()} calling super().__init__()", flush=True)
        super().__init__(*args, **kwargs)
        # print(f"[INIT] Worker PID={os.getpid()} super().__init__() DONE", flush=True)
        
        # Save postpro_fn for later, but remove it from parent to avoid applying it in super().__getitem__()
        self.saved_postpro_fn = self.postpro_fn
        self.postpro_fn = None  # Disable postpro in parent class
        
        # Note: We handle patch filtering ourselves in __getitem__() by temporarily disabling
        # parent's filtering before calling super().__getitem__() to prevent recursive retries
        
        self.multires = multires
        self.precomputed = precomputed
        self.enlarged_dims = {}
        for factor in self.multires:
            # When precomputed=True, we load from pre-coarsened files (x1, x3, x10). Each file is already at its target resolution, so enlarged patch is always 256x256
            # When precomputed=False, we load from x1 files and coarsen on-the-fly
            if self.precomputed and factor > 1:
                self.enlarged_dims[factor] = {'lat': 256, 'lon': 256}
            else:
                self.enlarged_dims[factor] = {'lat': 256 * factor, 'lon': 256 * factor}

    def extract_enlarged_patch_from_datasets(self, sl, factor):
        """
        Extrait un patch élargi centré, simple et clair (basé sur CROSCIM logic).
        
        Args:
            sl: Dict de slices x1 {'time': slice(...), 'lat': slice(...), 'lon': slice(...)}
            factor: Facteur de résolution (3 ou 10)
        
        Logic:
            1. Calculer centre en x1-pixels, convertir à pixels de la résolution cible
            2. Clipper aux limites du fichier
            3. Charger données + appliquer pooling au masque si nécessaire
            4. Pader si région est plus petite que 256x256
        
        IMPORTANT: self.da_dims['lat/lon'] sont les dimensions du fichier x1 (le parent).
        - En mode precomputed: on charge depuis x3/x10 pré-coarsifiés (dimensions 1/factor)
        - En mode pooled: on charge depuis x1 (dimensions complètes)
        """
        # Step 1: Calculate center (x1-pixels)
        lat_center_x1 = (sl["lat"].start + sl["lat"].stop) // 2
        lon_center_x1 = (sl["lon"].start + sl["lon"].stop) // 2
        
        # Determine if we're loading from precomputed files or pooling from x1
        is_precomputed_mode = (self.precomputed and self.is_multiresolution and factor > 1)
        
        if is_precomputed_mode:
            # Precomputed: files are pre-coarsened, so divide indices by factor
            lat_center = lat_center_x1 // factor
            lon_center = lon_center_x1 // factor
            target_lat_dim = self.da_dims["lat"] // factor
            target_lon_dim = self.da_dims["lon"] // factor
        else:
            # Pooled: loading from x1, use indices as-is
            lat_center = lat_center_x1
            lon_center = lon_center_x1
            target_lat_dim = self.da_dims["lat"]
            target_lon_dim = self.da_dims["lon"]
        
        enlarged_lat = self.enlarged_dims[factor]['lat']
        enlarged_lon = self.enlarged_dims[factor]['lon']
        
        lat_start = max(0, lat_center - enlarged_lat // 2)
        lat_end = min(lat_start + enlarged_lat, target_lat_dim)
        lon_start = max(0, lon_center - enlarged_lon // 2)
        lon_end = min(lon_start + enlarged_lon, target_lon_dim)
        
        # Step 2: Extract mask and apply pooling if needed
        if hasattr(self.mask, 'isel'):
            mask_slice = self.mask.isel(lon=slice(lon_start, lon_end), 
                                       lat=slice(lat_start, lat_end))
        else:
            mask_slice = xr.DataArray(self.mask[lat_start:lat_end, lon_start:lon_end])
        
        # Apply pooling to mask: factor for precomputed=False (x1 -> coarsen), 1 for precomputed=True
        pooling_factor = 1 if (self.precomputed and self.is_multiresolution and factor > 1) else factor
        item_mask = fast_pool(mask_slice, pooling_factor, pooling_factor, mode="binary")
        
        # Step 3: Load data
        if self.load_data:
            sst_ds = self.full_sst.isel(
                time=sl["time"],
                lat=slice(lat_start, lat_end),
                lon=slice(lon_start, lon_end))
        else:
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            slices = {
                "lat": slice(lat_start, lat_end),
                "lon": slice(lon_start, lon_end)
            }
            all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]
            if self.precomputed and self.is_multiresolution:
                sst_daily_paths_for_res = self.sst_daily_paths_by_resolution.get(factor, self.sst_daily_paths)
                resize_factor = 1
            else:
                sst_daily_paths_for_res = self.sst_daily_paths
                resize_factor = factor
            
            time_indices_list = list(time_indices) if isinstance(time_indices, np.ndarray) else time_indices
            paths_to_load = [sst_daily_paths_for_res[i] for i in time_indices_list]
            
            sst_ds = concatenate(
                paths_to_load,
                var_list=all_sst_vars + COVARIATES,
                slices=slices,
                type_coords="index",
                resize=resize_factor,
                domain_limits=self.domain_limits
            )
        
        # Step 4: Pad if necessary
        expected_shape = (self.patch_dims['time'], 256, 256)
        first_var = list(sst_ds.data_vars)[0]
        actual_shape = sst_ds[first_var].shape
        
        if actual_shape != expected_shape:
            pad_t = expected_shape[0] - actual_shape[0]
            pad_lat = expected_shape[1] - actual_shape[1]
            pad_lon = expected_shape[2] - actual_shape[2]
            
            if pad_lat > 0 or pad_lon > 0:
                # Padding: NaN for data variables, but extrapolate coordinates
                pad_lat_before = pad_lat // 2
                pad_lat_after = pad_lat - pad_lat_before
                pad_lon_before = pad_lon // 2
                pad_lon_after = pad_lon - pad_lon_before
                
                # Pad data variables with NaN
                sst_ds = sst_ds.pad(
                    lat=(pad_lat_before, pad_lat_after),
                    lon=(pad_lon_before, pad_lon_after),
                    constant_values=np.nan
                )
                
                # Fix coordinates: extrapolate lat/lon instead of using NaN
                if pad_lat > 0:
                    lat_vals = sst_ds.lat.values
                    # Find valid (non-NaN) lat values (they are in the middle after padding)
                    valid_mask = ~np.isnan(lat_vals)
                    if valid_mask.any():
                        valid_indices = np.where(valid_mask)[0]
                        first_valid_idx = valid_indices[0]
                        last_valid_idx = valid_indices[-1]
                        
                        valid_lats = lat_vals[valid_mask]
                        lat_step = np.diff(valid_lats).mean() if len(valid_lats) > 1 else 0.1
                        
                        # Extrapolate before first valid value
                        for i in range(first_valid_idx):
                            lat_vals[i] = valid_lats[0] - (first_valid_idx - i) * lat_step
                        
                        # Extrapolate after last valid value
                        for i in range(last_valid_idx + 1, len(lat_vals)):
                            lat_vals[i] = valid_lats[-1] + (i - last_valid_idx) * lat_step
                        
                        sst_ds['lat'] = lat_vals
                
                if pad_lon > 0:
                    lon_vals = sst_ds.lon.values
                    valid_mask = ~np.isnan(lon_vals)
                    if valid_mask.any():
                        valid_indices = np.where(valid_mask)[0]
                        first_valid_idx = valid_indices[0]
                        last_valid_idx = valid_indices[-1]
                        
                        valid_lons = lon_vals[valid_mask]
                        lon_step = np.diff(valid_lons).mean() if len(valid_lons) > 1 else 0.1
                        
                        # Extrapolate before first valid value
                        for i in range(first_valid_idx):
                            lon_vals[i] = valid_lons[0] - (first_valid_idx - i) * lon_step
                        
                        # Extrapolate after last valid value
                        for i in range(last_valid_idx + 1, len(lon_vals)):
                            lon_vals[i] = valid_lons[-1] + (i - last_valid_idx) * lon_step
                        
                        sst_ds['lon'] = lon_vals
            
            # Adjust mask to match padded size
            actual_lat = len(sst_ds.lat)
            actual_lon = len(sst_ds.lon)
            mask_lat, mask_lon = item_mask.shape[-2:] if item_mask.ndim == 3 else item_mask.shape
            
            if actual_lat > mask_lat or actual_lon > mask_lon:
                pad_lat_needed = actual_lat - mask_lat
                pad_lon_needed = actual_lon - mask_lon
                if item_mask.ndim == 3:
                    item_mask = np.pad(item_mask, ((0, 0), (pad_lat_needed//2, (pad_lat_needed+1)//2), (pad_lon_needed//2, (pad_lon_needed+1)//2)), mode='constant', constant_values=1)
                else:
                    item_mask = np.pad(item_mask, ((pad_lat_needed//2, (pad_lat_needed+1)//2), (pad_lon_needed//2, (pad_lon_needed+1)//2)), mode='constant', constant_values=1)
        
        # Step 5: Assemble output dict
        sample = {}
        
        # Add satellite variables
        for sat_name in ['aasti', 'avhrr', 'pmw', 'slstr']:
            for var in VAR_GROUPS[sat_name]:
                var_key = f"{sat_name}_{var}"
                if var_key in sst_ds:
                    sample[var_key] = sst_ds[var_key].values
        
        # Add covariates
        for cov in COVARIATES:
            if cov in sst_ds:
                sample[cov] = sst_ds[cov].values
        
        # Add metadata
        sample["surfmask"] = np.expand_dims(item_mask, axis=0)
        
        lon_1d = sst_ds.lon.values
        lat_1d = sst_ds.lat.values
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        sample["lat"] = lat_2d
        sample["lon"] = lon_2d
        
        # Create time channel as a 2D grid (same as in data.py)
        # Use the center timestep's day of year, normalized [0, 1]
        import pandas as pd
        center_time_idx = len(sst_ds.time) // 2
        center_time = sst_ds.time.values[center_time_idx]
        day_of_year = pd.Timestamp(center_time).dayofyear  # 1-366
        time_value = day_of_year / 366.0  # Normalized [0, 1]
        time_channel = np.full((lat_2d.shape[0], lat_2d.shape[1]), time_value, dtype=np.float32)
        sample["time"] = time_channel  # (nlat, nlon) - 2D grid
        
        sample["inpaint_mask"] = np.zeros_like(sample["surfmask"])
        
        # Add target variables
        for tgt_var in self.tgt_vars:
            if tgt_var in sample:
                sample[f"tgt_{tgt_var}"] = sample[tgt_var]
        if self.tgt_vars:
            sample["tgt_sst"] = sample.get(f"tgt_{self.tgt_vars[0]}", np.zeros_like(sample["surfmask"]))
        
        # Return raw dict - postprocessing (normalization + TrainingItem creation) 
        # will be applied later in __getitem__() using saved_postpro_fn
        return sample


    def __getitem__(self, idx):
        """
        Retourne un dict avec patches multi-résolution IMBRIQUÉS:
        - patch_x10: 256x256 pixels à résolution x10
        - patch_x3: 256x256 pixels à résolution x3
        - patch_x1: 256x256 pixels à résolution x1
        
        Note: Filtering is handled here to avoid double-application of postpro_fn during retries
        """
        # Track memory at start
        # import psutil
        # import os
        # if not hasattr(self, '_mem_counter'):
        #     self._mem_counter = 0
        # self._mem_counter += 1
        # 
        # if self._mem_counter % 20 == 0:
        #     ram_gb = psutil.virtual_memory().used / 1e9
        #     print(f"[MEM START] Worker PID={os.getpid()} __getitem__ #{self._mem_counter} | RAM:{ram_gb:.1f}GB", flush=True)
        
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                        self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }
        
        # Temporarily disable parent's filtering to prevent recursive retry issues
        # (parent's retry would call our __getitem__ which returns multi-res dict)
        parent_filtering_state = getattr(self, 'enable_patch_filtering', True)
        self.enable_patch_filtering = False
        
        # Extract x1 patch (postpro_fn disabled in parent, so returns dict)
        hr_sample = super().__getitem__(idx)
        
        # Restore our filtering state
        self.enable_patch_filtering = parent_filtering_state
        
        out = {}
        out[f"patch_x{self.resize}"] = hr_sample
        # extract lower resolution patches (x3, x10)
        for factor in self.multires[:-1]:
            enlarged_patch = self.extract_enlarged_patch_from_datasets(sl, factor // self.resize)
            out[f"patch_x{factor}"] = enlarged_patch
        
        # Apply patch filtering on hr_sample (x1 resolution) before normalization
        enable_filtering = getattr(self, 'enable_patch_filtering', True)
        max_retries = getattr(self, 'max_patch_retries', 50)
        
        # Track retries for this sample (increment on each rejection)
        if not hasattr(self, '_current_sample_retries'):
            self._current_sample_retries = 0
        
        patch_stats = None
        
        if enable_filtering and hasattr(self, 'is_valid_patch'):
            is_valid, reason, patch_stats = self.is_valid_patch(hr_sample)
            if not is_valid:
                if not hasattr(self, '_rejection_count'):
                    self._rejection_count = 0
                self._rejection_count += 1
                self._current_sample_retries += 1
                
                if self._rejection_count % 20 == 0:
                    print(f"[{self._rejection_count} patches rejetés] Dernier rejet: {reason}")
                
                if self._current_sample_retries < max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx)
                else:
                    print(f"WARNING: {max_retries} rejets consécutifs, on garde le patch malgré: {reason}")
                    self._rejection_count = 0
                    self._current_sample_retries = 0
        
        # Patch valide trouvé, afficher les stats
        if patch_stats:
            print(f"✓ Patch valide (retries={self._current_sample_retries}) | mean={patch_stats['mean']:.2f}°C, std={patch_stats['std']:.2f}°C, ocean={patch_stats['ocean_pct']:.1f}%")
        
        self._rejection_count = 0
        self._current_sample_retries = 0  # Reset pour le prochain sample

        if self.saved_postpro_fn is not None:
            for key in out:
                if isinstance(out[key], dict):  # Should be a dict from load_patch_at_resolution
                    # print(f"[DEBUG __getitem__] Applying saved_postpro_fn to out['{key}']")
                    out[key] = self.saved_postpro_fn(out[key])

        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        
        if self._log_counter % 10 == 1:  # Log every 10th batch
            from contrib.SST.load_data import log_batch_load
            # Estimate data size from x10 patch
            patch_x10 = out.get('patch_x10')
            if isinstance(patch_x10, dict) and 'tgt_sst' in patch_x10:
                # Count channels: all satellites + covariates
                n_channels = sum(len(vars) for vars in self.var_groups.values()) + len(self.covariates) if hasattr(self, 'var_groups') else 8
                spatial_h, spatial_w = 256, 256
                data_mb = (n_channels * self.patch_dims['time'] * spatial_h * spatial_w * 4) / 1e6  # 4 bytes per float32
                log_batch_load(
                    batch_idx=self._log_counter,
                    batch_size=getattr(self, 'batch_size', 4),
                    timesteps=self.patch_dims['time'],
                    spatial_shape=f"{spatial_h}x{spatial_w}",
                    data_size_mb=data_mb
                )
        
        # Track memory at end
        # if self._mem_counter % 20 == 0:
        #     ram_gb = psutil.virtual_memory().used / 1e9
        #     print(f"[MEM END] Worker PID={os.getpid()} __getitem__ #{self._mem_counter} DONE | RAM:{ram_gb:.1f}GB", flush=True)
        gc.collect()
        return out

class XrDatasetMultiResTest:
    """
    Dataset pour le mode test : contient N XrDataset avec coarsening contrôlé par multires
    Chaque sortie est un TrainingItem avec suffixe de résolution.
    """
    def __init__(self, multires=[1], *args, **kwargs):
        self.datasets = {}
        for res in multires:
            kwargs["resize"] = res
            self.datasets[res] = XrDataset(load_data=True,*args, **kwargs)

    def get_dataloader_dict(self, batch_size=1, **loader_kwargs):
        from torch.utils.data import DataLoader
        return {res: DataLoader(ds, batch_size=batch_size, **loader_kwargs)
                for res, ds in self.datasets.items()}

class BaseDataModuleMultiRes(BaseDataModule):
    
    def __init__(self, sst_daily_paths, multires=[1], covariates_paths=None,
                 covariates=None, mask_path=None,
                 domain_name='sst_multires', domains=None,
                 precomputed=True, res=5.0, norm_stats=None, norm_stats_covs=None,
                 *args, **kwargs):
        if covariates_paths is None:
            covariates_paths = []
        if covariates is None:
            covariates = COVARIATES
        if domains is None:
            domains = {
                'train': {'time': slice(None, None)},
                'val': {'time': slice(None, None)},
                'test': {'time': slice(None, None)}
            }
        
        # Si sst_daily_paths est un dossier, scanner pour trouver tous les fichiers
        if isinstance(sst_daily_paths, str):
            from pathlib import Path
            import glob
            path = Path(sst_daily_paths)
            if path.is_dir():
                # Scanner tous les sous-dossiers (années) pour trouver les fichiers .zarr
                all_files = []
                for year_dir in sorted(path.iterdir()):
                    if year_dir.is_dir():
                        all_files.extend(sorted(year_dir.glob('*_x1.zarr')))
                sst_daily_paths = [str(f) for f in all_files]
            elif not path.exists():
                raise FileNotFoundError(f"Path does not exist: {sst_daily_paths}")
        
        super().__init__(sst_paths=sst_daily_paths,
                        covariates_paths=covariates_paths, covariates=covariates,
                        mask_path=mask_path, domain_name=domain_name, domains=domains,
                        norm_stats=norm_stats, norm_stats_covs=norm_stats_covs,
                        *args, **kwargs)
        # self.sst_paths est déjà défini par le parent
        self.multires = multires
        self.resize = self.multires[-1]
        self.precomputed = precomputed
        self.res = res

    def save_batch_as_NetCDF_multires(self, batch_dict, ibatch, patch_dims_dict, save_dir="/dmidata/users/malegu/PREPROC/"):
        # """
        # Sauvegarde un batch multi-résolution en fichiers NetCDF séparés
        # Args:
        #     batch_dict: dict de {f"patch_x{res}": TrainingItem}
        #     patch_dims_dict: dict de {res: {"time": ..., "lat": ..., "lon": ...}}
        #     save_dir: Répertoire de sauvegarde
        # """
        # os.makedirs(save_dir, exist_ok=True)
        # for key, batch in batch_dict.items():
        #     try: factor = int(key.split("x")[-1])
        #     except:
        #         print(f"Warning: can't parse resolution factor in {key}")
        #         continue
        #     patch_dims = patch_dims_dict[factor]
        #     data_vars = {}
        #     # Variables SST (4 satellites)
        #     for var in VAR_GROUPS["sst"]:
        #         var_name = f"sst_{var}"
        #         if hasattr(batch, var_name):
        #             tensor = getattr(batch, var_name)
        #             if torch.is_tensor(tensor) and tensor.ndim == 4:
        #                 data_vars[var_name] = (
        #                     ('sample', 'time', 'lat', 'lon'),
        #                     tensor.detach().cpu())
        #     # Covariables (sea_ice_fraction)
        #     for cov in COVARIATES:
        #         if hasattr(batch, cov):
        #             tensor = getattr(batch, cov)
        #             if torch.is_tensor(tensor) and tensor.ndim == 4:
        #                 data_vars[cov] = (
        #                     ('sample', 'time', 'lat', 'lon'),
        #                     tensor.detach().cpu())
        #     # Coordonnées et masque
        #     data_vars.update({
        #         'times': (('sample', 'time'),torch.squeeze(batch.time, dim=1).detach().cpu().numpy().astype("datetime64[s]")),
        #         'lats': (('sample', 'lat'),torch.squeeze(batch.lat_coord, dim=1).detach().cpu()),
        #         'lons': (('sample', 'lon'),torch.squeeze(batch.lon_coord, dim=1).detach().cpu()),
        #         'lat': (('sample', 'lat', 'lon'),torch.squeeze(batch.lat, dim=1).detach().cpu()),
        #         'lon': (('sample', 'lat', 'lon'),torch.squeeze(batch.lon, dim=1).detach().cpu()),
        #         'land_mask': (('sample', 'lat', 'lon'),torch.squeeze(batch.land_mask, dim=1).detach().cpu())})
        #     # Variables cibles (tgt_xxx)
        #     for var in VAR_GROUPS["sst"]:
        #         new_key = f"tgt_{var}"
        #         if hasattr(batch, new_key):
        #             tensor = getattr(batch, new_key)
        #             if torch.is_tensor(tensor) and tensor.ndim == 4:
        #                 data_vars[new_key] = (
        #                     ('sample', 'time', 'lat', 'lon'),
        #                     tensor.detach().cpu())
        #     # Coordonnées du Dataset
        #     coords = {
        #         'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
        #         'time': np.arange(patch_dims['time']),
        #         'lat': np.arange(patch_dims['lat']),
        #         'lon': np.arange(patch_dims['lon'])}
        #     # Construction et sauvegarde
        #     ds = xr.Dataset(data_vars=data_vars, coords=coords)
        #     save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}_x{factor}.nc")
        #     ds.to_netcdf(save_path)
        #     print(f"Saved {save_path}")
        raise NotImplementedError("Function save_batch_as_NetCDF_multires is not used in SST")

    def setup(self, stage='test'):
        """
        Configure datasets pour train/val/test avec multi-résolution
        
        Args:
            stage: 'train', 'val', ou 'test'
        """
        
        def select_paths(files, times, fmt="%Y%m%d"):
            # Si slice(None, None) : utiliser tous les fichiers
            if isinstance(times, slice) and times.start is None and times.stop is None:
                files_sorted = np.sort(files)
                time_vals = []
                for f in files_sorted:
                    basename = os.path.basename(f)
                    date_str = basename[:10]
                    time_vals.append(pd.to_datetime(date_str, format="%Y%m%d%H"))
                return files_sorted, np.array(time_vals)
            # Sinon : filtrage par date_range
            if isinstance(times, list):
                dates, time_vals = [], []
                for sl in times:
                    start, end = sl.start, sl.stop
                    dts = pd.date_range(start, end)
                    dates.extend(dts.strftime(fmt).tolist())
                    time_vals.extend(dts.tolist())
            else:
                start, end = times.start, times.stop
                dts = pd.date_range(start, end)
                dates = dts.strftime(fmt).tolist()
                time_vals = dts.tolist()
            
            files = np.sort([f for f in files if any(date in f for date in dates)])
            return files, np.array(time_vals)
        
        def create_dataset(split):
            """
            Crée XrDataset multi-résolution pour un split donné
            Args: split: 'train', 'val', ou 'test'
            Returns: XrDatasetMultiResTrain ou XrDatasetMultiResTest
            """
            sst_paths, times = select_paths(self.sst_paths, self.domains[split]['time'], fmt="%Y%m%d")
            if self.precomputed:
                sst_paths = organize_by_resolution(sst_paths)
            if split == "test":
                XrDatasetMultiRes = XrDatasetMultiResTest
            else:
                XrDatasetMultiRes = XrDatasetMultiResTrain
            
            return XrDatasetMultiRes(
                multires=self.multires,
                sst_daily_paths=sst_paths,
                tgt_vars=self.tgt_vars,
                mask=self.mask,
                times=times,
                precomputed=self.precomputed,
                **self.xrds_kw,
                postpro_fn=self.post_fn(rand_obs=(split == 'train')),
                res=self.res,
                pad=self.pads[0 if split == 'train' else 1 if split == 'val' else 2],
                stride_test=(split != 'train'),
                resize=self.resize
            )
        
        # Créer datasets pour chaque split
        self.train_ds = create_dataset('train')
        self.val_ds = create_dataset('val')
        self.test_ds = create_dataset('test')
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return {f"patch_x{res}": torch.utils.data.DataLoader(ds, shuffle=False, **self.dl_kw) for res, ds in self.test_ds.datasets.items()}
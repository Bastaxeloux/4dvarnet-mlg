import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import torch
import xarray as xr
import itertools
import functools as ft 
from collections import namedtuple
from torch.utils.data import  ConcatDataset
import multiprocessing
import gc
from random import sample
import contrib
from contrib.SST.load_data import *
import datetime
import pyresample
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import os
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
import zarr
import time
import logging
import random


def create_training_item(var_groups, covariates, tgt_vars):
    """
    Dynamically create a TrainingItem with variables from var_groups and covariates
    """
    fields = []
    # Variables satellites
    for group in var_groups:
        for var in var_groups[group]:
            fields.append(f"{group}_{var}")
    # Target variables (éviter les doublons)
    tgt_fields = set()
    for group, variables in VAR_GROUPS.items():
        for var in variables:
            var_key = f"{group}_{var}"
            if var_key in tgt_vars:
                tgt_fields.add(f"tgt_{group}_{var}")
    fields.extend(sorted(tgt_fields))
    fields.append('tgt_sst')
    fields.append('tgt_sst_full')  # Version complète (avant inpainting) pour évaluation SSL
    fields.extend(covariates)
    fields.extend(['lat', 'lon', 'surfmask', "time"])
    fields.append('inpaint_mask')  # 1=removed by inpainting, 0=kept
    # Add geographic coordinates (in degrees) for interpolation (not used as input channels)
    fields.extend(['lat_geo', 'lon_geo'])
    # Add temporal metadata (actual timestamps, not sent to solver)
    fields.append('time_indices')
    return namedtuple("TrainingItem", fields)

TrainingItem = create_training_item(VAR_GROUPS, COVARIATES, tgt_vars=["slstr_av", "aasti_av"])

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass


def find_idx(coords,c):
    return np.where(coords==c)[0][0]

def pad_batch_with_coords(ds, sl, global_xc, global_yc, global_lon, global_lat):
    # """
    # Pads an xarray Dataset `ds` so that its yc/xc match a window from the global coords.
    # Missing values are NaN-filled, and lon/lat are taken from the global reference.

    # Parameters
    # ----------
    # ds : xr.Dataset
    #     Input patch (must have coords 'xc' and 'yc').
    # sl: slices for coordinates
    # global_xc, global_yc : 1D array-like
    #     Full-resolution reference coordinates for xc and yc.
    # global_lon, global_lat : 2D array-like
    #     Reference longitude and latitude on (yc, xc) grid.

    # Returns
    # -------
    # ds_padded : xr.Dataset
    #     Dataset aligned on the padded coords with NaN padding.
    # """

    # ix = [find_idx(global_xc, x) for x in global_xc[sl["xc"].start:sl["xc"].stop]]
    # iy = [find_idx(global_yc, y) for y in global_yc[sl["yc"].start:sl["yc"].stop]]
    
    # # Create padded coordinate window from global arrays
    # padded_coords = {
    #     "time": ds.time,
    #     "xc": global_xc[sl["xc"].start:sl["xc"].stop],
    #     "yc": global_yc[sl["yc"].start:sl["yc"].stop],
    #     "lon": (["yc", "xc"], global_lon[iy[0]: iy[-1] + 1, ix[0]: ix[-1] + 1]),
    #     "lat": (["yc", "xc"], global_lat[iy[0]: iy[-1] + 1, ix[0]: ix[-1] + 1]),
    # }

    # # Create a template Dataset with the padded coords
    # padded_template = xr.Dataset(coords=padded_coords)

    # # Align => ensures missing coords in ds become NaNs
    # _, ds_padded = xr.align(padded_template, ds, join="left")

    # return ds_padded
    raise NotImplementedError("pad_batch_with_coords is not implemented.")

class XrDataset(torch.utils.data.Dataset):

    def __init__(self, sst_daily_paths,  # Daily SST NetCDF files
                 tgt_vars,
                 mask, times,
                 patch_dims, domain_limits=None, strides=None,
                 strides_test=None, postpro_fn=None,
                 resize=1, res=5.0, pad=False, stride_test=False,  # res=5km for SST
                 precomputed=True,  # For multi-resolution: use precomputed files or pool on-the-fly
                 load_data=False, domain=None, verbose=False, **kwargs):

        super().__init__()
        self.verbose = verbose
        if self.verbose:
            print("[DEBUG] XrDataset SST __init__ started")
        
        self.postpro_fn = postpro_fn
        self.precomputed = precomputed  # Store for potential use in subclasses
        # Multi-resolution: dict {1: paths, 3: paths, 10: paths} or list (legacy)
        if isinstance(sst_daily_paths, dict):
            self.is_multiresolution = True
            self.sst_daily_paths_by_resolution = sst_daily_paths

            if resize in sst_daily_paths:
                self.sst_daily_paths = sst_daily_paths[resize]
            else:
                # Résolution manquante → fallback x1 + coarsening
                print(f"\n WARNING: x{resize} non trouvé, disponible: {list(sst_daily_paths.keys())}")
                print(f" WARNING: Fallback: coarsening à la volée depuis x1\n")
                self.sst_daily_paths = sst_daily_paths.get(1, list(sst_daily_paths.values())[0])
                self.precomputed = False  # Force coarsening
        else:
            self.is_multiresolution = False
            self.sst_daily_paths = sst_daily_paths
            self.sst_daily_paths_by_resolution = None
        
        self.tgt_vars = tgt_vars
        self.times = times
        self.patch_dims = patch_dims
        self.strides = strides or {}
        if stride_test:
            self.strides = strides_test or {}

        self.domain_limits = domain_limits
        self.res = res * resize
        self.pad = pad
        self.load_data = load_data
        self.domain = domain
        self.resize = resize
        
        # Patch filtering control: default True for training, can be disabled for test
        self.enable_patch_filtering = kwargs.pop('enable_patch_filtering', True)
        
        # Load first file to get grid structure
        first_file = str(self.sst_daily_paths[0])
        if first_file.endswith('.zarr'):
            # Pure zarr to avoid Dask threading issues
            store = zarr.open(first_file, mode='r')
            self.lon_1d = np.array(store['lon'][:])
            self.lat_1d = np.array(store['lat'][:])
            if mask is None:
                self.mask = np.array(store['surfmask'][:])
        else:
            # NetCDF fallback
            print("[WARNING] Using xarray to open NetCDF file, which may cause multiprocessing issues.")
            sst_base = xr.open_dataset(first_file, chunks=None)
            self.lon_1d = sst_base.lon.values
            self.lat_1d = sst_base.lat.values
            if mask is None:
                self.mask = sst_base.surfmask.values
            sst_base.close()
        
        # Create 2D meshgrid for each pixel to have lat/lon info
        self.lon_2d, self.lat_2d = np.meshgrid(self.lon_1d, self.lat_1d)
        
        if mask is not None:
            self.mask = mask.sel(**(domain_limits or {})) if domain_limits else mask

        # Resize: coarsening if precomputed=False, else files already at correct resolution
        if self.resize != 1 and not self.precomputed:
            print(f"  Grid: Coarsening x1 -> x{self.resize} ({len(self.lat_1d)}x{len(self.lon_1d)} -> {len(self.lat_1d)//self.resize}x{len(self.lon_1d)//self.resize})")
            self.lat_1d = self.lat_1d[::self.resize]
            self.lon_1d = self.lon_1d[::self.resize]
            self.lon_2d, self.lat_2d = np.meshgrid(self.lon_1d, self.lat_1d)
            if mask is not None:
                # CRITICAL FIX: Use max pooling for surfmask downsampling
                # If ANY pixel in a block is ocean (>0), the downsampled pixel should be ocean
                # This avoids creating artificial land blocks in coarse resolutions
                from scipy.ndimage import maximum_filter
                self.mask = maximum_filter(self.mask, size=(self.resize, self.resize))[::self.resize, ::self.resize]
                # Alternative explanation: prend le MAX dans chaque bloc NxN, puis sample
                # surfmask: 0=land, 1=ocean, 2=ice-water, 3=ice
                # MAX garantit que si 1 pixel océan existe dans le bloc, on garde océan
        elif self.resize != 1 and self.precomputed:
            print(f"  Grid: Loaded precomputed ({len(self.lat_1d)}x{len(self.lon_1d)})")
        
        if self.verbose:
            print(f"[DEBUG] Grid shape - lat_1d: {self.lat_1d.shape}, lon_1d: {self.lon_1d.shape}")
            print(f"[DEBUG] Grid 2D - lat_2d: {self.lat_2d.shape}, lon_2d: {self.lon_2d.shape}")
            print(f"[DEBUG] Mask shape: {self.mask.shape if hasattr(self.mask, 'shape') else 'N/A'}")
        
        # Padding 
        if self.pad:
            pad_x = self._find_pad(self.patch_dims.get('lon', self.patch_dims.get('xc', 240)), 
                                   self.strides.get('lon', self.strides.get('xc', 1)), 
                                   len(self.lon_1d))
            pad_y = self._find_pad(self.patch_dims.get('lat', self.patch_dims.get('yc', 240)), 
                                   self.strides.get('lat', self.strides.get('yc', 1)), 
                                   len(self.lat_1d))
            self.lon_2d = np.pad(self.lon_2d, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), mode="edge")
            self.lat_2d = np.pad(self.lat_2d, ((pad_y[0], pad_y[1]), (pad_x[0], pad_x[1])), mode="edge")
            self.lon_1d = np.linspace(self.lon_1d[0] - pad_x[0]*self.res, 
                                       self.lon_1d[-1] + pad_x[1]*self.res, 
                                       len(self.lon_1d)+sum(pad_x))
            self.lat_1d = np.linspace(self.lat_1d[0] + pad_y[0]*self.res, 
                                       self.lat_1d[-1] - pad_y[1]*self.res, 
                                       len(self.lat_1d)+sum(pad_y))

        # Dimensions for patch extraction
        nt, nlat, nlon = (len(self.times), len(self.lat_1d), len(self.lon_1d))
        self.da_dims = dict(time=nt, lat=nlat, lon=nlon)
        
        # Calculate number of patches in each dimension
        self.ds_size = {
            dim: max((self.da_dims[dim] - self.patch_dims.get(dim, 1)) // self.strides.get(dim, 1) + 1, 0)
            for dim in self.patch_dims
        }

    def _find_pad(self, sl, st, N):
        k = np.floor(N/st)
        if N>((k*st)+(sl-st)):
            pad = (k+1)*st + (sl-st) - N
        elif N<((k*st)+(sl-st)):
            pad = (k*st) + (sl-st) - N
        else:
            pad = 0
        return int(pad/2), int(pad-int(pad/2))
    
    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self, limit=None):
        coords_list = []
        if limit is None:
            indices = range(len(self))
        else:
            indices = np.random.choice(len(self), size=limit, replace=False)

        for idx in indices:
            sl = {
                dim: slice(self.strides.get(dim, 1) * idx_dim,
                           self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
                for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
            }
            
            # Adapt for lat/lon dimensions
            lat_slice = sl.get("lat", sl.get("yc", slice(None)))
            lon_slice = sl.get("lon", sl.get("xc", slice(None)))
            
            coords = xr.Dataset(coords=dict(
                lon=self.lon_1d[lon_slice],
                lat=self.lat_1d[lat_slice],
                time=self.times[sl["time"].start:sl["time"].stop],
                lon_2d=(["lat", "lon"], self.lon_2d[lat_slice, lon_slice]),
                lat_2d=(["lat", "lon"], self.lat_2d[lat_slice, lon_slice]),
            )).transpose("time", "lat", "lon")
            coords_list.append(coords)
        return coords_list

    def find_patches_in_ocean(self):
        """
        Vectorized version. Valid patches = patches containing at least one ocean/ice pixel (mask = 1, 2, or 3)
        """
        from scipy.ndimage import maximum_filter
        
        if self.verbose:
            print("[DEBUG] Starting vectorized patch finding...")
        
        # Create binary mask: 1 = valid (ocean/ice), 0 = invalid (land)
        valid_mask = ((self.mask == 1) | (self.mask == 2) | (self.mask == 3)).astype(np.uint8)
        
        # Get patch and stride sizes
        patch_h = self.patch_dims.get('lat', self.patch_dims.get('yc', 256))
        patch_w = self.patch_dims.get('lon', self.patch_dims.get('xc', 256))
        stride_h = self.strides.get('lat', self.strides.get('yc', 20))
        stride_w = self.strides.get('lon', self.strides.get('xc', 20))
        
        if self.verbose:
            print(f"[DEBUG] Patch size: {patch_h}x{patch_w}, stride: {stride_h}x{stride_w}")
            print(f"[DEBUG] Mask shape: {valid_mask.shape}")
        
        # Apply maximum filter: for each pixel, check if there's a valid pixel
        # within a patch_size window centered on it
        filtered = maximum_filter(valid_mask, size=(patch_h, patch_w), mode='constant', cval=0)
        
        # Sample at stride intervals (top-left corner of each potential patch)
        sampled = filtered[::stride_h, ::stride_w]
        
        # Find indices where patches are valid (contain at least one valid pixel)
        idx_ocean = np.where(sampled.flatten() > 0)[0]
        
        if self.verbose:
            print(f"[DEBUG] Found {len(idx_ocean)} valid patches out of {sampled.size} possible")
            print(f"[DEBUG] Coverage: {100*len(idx_ocean)/sampled.size:.1f}%")
        
        return idx_ocean

    def interpolate_dataset(self, target_grid, ds, var_list, prefix=None):
        """
        Interpolates variables from ds onto the target grid.
    
        Args:
            target_grid: either a tuple (xc, yc) for regular grid
                         or a pyresample SwathDefinition for irregular grid
            ds: xarray.Dataset with variables to interpolate
            var_list: list of variable names to interpolate
            prefix: optional prefix for output keys
    
        Returns:
            dict of interpolated numpy arrays (shape: [time, yc, xc])
        """
        data_out = {}
    
        use_regular_grid = isinstance(target_grid, tuple) and len(target_grid) == 2
        isel_time = ds.sizes["time"]
    
        for var in var_list:
            if var not in ds:
                continue
    
            key = f"{prefix}_{var}" if prefix is not None else var
    
            if use_regular_grid:
                # Regular grid interpolation
                xc_target, yc_target = target_grid
                interpolated = ds[var].interp(xc=("xc", xc_target), yc=("yc", yc_target))
                data_out[key] = interpolated.values
            else:
                # Irregular grid using pyresample
                swath_def_target = target_grid
                src_def = pyresample.geometry.SwathDefinition(lons=ds.lon.values, lats=ds.lat.values)
                interpolated = np.stack([
                    pyresample.kd_tree.resample_nearest(
                        src_def,
                        ds[var].isel(time=i).values,
                        swath_def_target,
                        radius_of_influence=30000,
                        fill_value=np.nan
                    ) for i in range(isel_time)
                ])
                data_out[key] = interpolated

        return data_out

    def __getitem__(self, idx):
        t_start_total = time.time()

        # Calculate spatial and temporal slices
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                       self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        # Extract slices (adapt for lat/lon dimensions)
        time_slice = sl["time"]
        lat_slice = sl.get("lat", sl.get("yc", slice(None))) # use of CROSCIM notation if needed
        lon_slice = sl.get("lon", sl.get("xc", slice(None)))

        # Get mask for this patch
        item_mask = self.mask[lat_slice, lon_slice]

        # Load SST data for temporal window (no load_data option, always on the fly)
        time_indices = np.arange(time_slice.start, time_slice.stop)
        sst_files = [self.sst_daily_paths[t_idx] for t_idx in time_indices]

        t_after_slices = time.time()

        # Load patches using zarr
        import os
        data_list = []
        t_zarr_total = 0
        for t_idx, sst_file in enumerate(sst_files):
            t_zarr_start = time.time()
            store = zarr.open(str(sst_file), mode='r')
            t_after_open = time.time()
            patches_t = {}
            for var_name in store.array_keys():
                arr = store[var_name]
                if arr.ndim != 2:
                    continue
                patches_t[var_name] = np.array(arr[lat_slice, lon_slice])
            t_after_read = time.time()
            t_zarr_total += (t_after_read - t_zarr_start)

            # Log individual file load time (only first 3 files to avoid spam)
            # Désactivé : Logging de timing trop volumineux
            # if t_idx < 3:
            #     t_open_ms = (t_after_open - t_zarr_start) * 1000
            #     t_read_ms = (t_after_read - t_after_open) * 1000
            #     self.timing_logger.info(f"idx={idx} | STEP=zarr_file_{t_idx} | open={t_open_ms:.1f}ms | read={t_read_ms:.1f}ms")

            data_list.append(patches_t)
        
        t_after_zarr_loop = time.time()

        # Stack patches along time dimension
        all_vars = {}
        for var_name in data_list[0].keys():
            stacked = np.stack([d[var_name] for d in data_list], axis=0)
            all_vars[var_name] = stacked  # Keep as numpy array, not xarray

        t_after_stack = time.time()

        # Free data_list immediately to save memory
        del data_list

        # Don't create xarray Dataset to avoid memory leaks - use dict of numpy arrays instead
        nt = len(time_indices)
        
        # Extract lat/lon coordinates for this patch
        lat_patch = self.lat_2d[lat_slice, lon_slice]  # (nlat, nlon)
        lon_patch = self.lon_2d[lat_slice, lon_slice]  # (nlat, nlon)
        
        # Assemble data channels (TOTAL = 9 : 4 sats x 2 + sea_ice_fraction)
        full_input = {}
        
        # Add satellite data (all 4 satellites: aasti, avhrr, pmw, slstr)
        for sat_name in ['aasti', 'avhrr', 'pmw', 'slstr']:
            for var in VAR_GROUPS[sat_name]:  # ['av', 'std']
                var_key = f"{sat_name}_{var}"
                if var_key in all_vars:
                    full_input[var_key] = all_vars[var_key]  # Already numpy array: shape (nt, nlat, nlon)
                else:
                    # Fill with NaN if missing
                    nlat, nlon = lat_patch.shape
                    full_input[var_key] = np.full((nt, nlat, nlon), np.nan, dtype=np.float32)
                    if self.verbose:
                        print(f"[WARNING] Variable {var_key} not found, filling with NaN")
                        # In theory this should not happen, because i checked before that everything was complete
        
        # Add covariate (sea_ice_fraction)
        if 'sea_ice_fraction' in all_vars:
            full_input['sea_ice_fraction'] = all_vars['sea_ice_fraction']
        else:
            nlat, nlon = lat_patch.shape
            full_input['sea_ice_fraction'] = np.zeros((nt, nlat, nlon), dtype=np.float32)
        
        # Free all_vars immediately after copying to full_input
        del all_vars
        
        # Add spatial/temporal metadata as channels
        nt, nlat, nlon = full_input['aasti_av'].shape
        
        # Coordonnées spatiales normalisées
        lat_channel = (lat_patch / 90.0).astype(np.float32)  # in [-1, 1]
        lon_channel = (lon_patch / 180.0).astype(np.float32) # idem
        
        # Pour le Time channel, on prendra le jour dans l'année (1-366) du centre de la fenêtre, normalisé [0, 1]
        center_time_idx = nt // 2
        center_time = time_indices[center_time_idx]
        day_of_year = pd.Timestamp(center_time).dayofyear  # 1-366
        time_value = day_of_year / 366.0  # Normalisé [0, 1]
        time_channel = np.full((nlat, nlon), time_value, dtype=np.float32)  # (nlat, nlon)
        
        full_input['lat'] = lat_channel  # (nlat, nlon) - normalized [-1, 1]
        full_input['lon'] = lon_channel  # (nlat, nlon) - normalized [-1, 1]
        full_input['time'] = time_channel  # (nlat, nlon)
        full_input['surfmask'] = item_mask.astype(np.float32)  # (nlat, nlon)
        
        # Store geographic coordinates (in degrees) for interpolation (NOT used as input channels)
        full_input['lat_geo'] = lat_patch.astype(np.float32)  # (nlat, nlon) - geographic degrees
        full_input['lon_geo'] = lon_patch.astype(np.float32)  # (nlat, nlon) - geographic degrees
        
        
        
        # Desormais on passe a la target !
        # on va utiliser slstr. Mais slstr n'est pas disponible aux pôles, ou on utilisera donc aasti.
        slstr_av = full_input['slstr_av']
        aasti_av = full_input['aasti_av']
        
        # slstr has priority where not NaN
        tgt_sst = np.where(~np.isnan(slstr_av), slstr_av, aasti_av)
        full_input['tgt_sst'] = tgt_sst

        # we also keep each component for normalization + time and lat/lon. It will be deleted before training
        full_input['tgt_slstr_av'] = slstr_av
        full_input['tgt_aasti_av'] = aasti_av
        full_input["lat_coords"] = self.lat_1d[lat_slice]
        full_input["lon_coords"] = self.lon_1d[lon_slice]

        # Store actual timestamps as metadata (not a spatial channel)
        # Used for temporal matching in convert_xr_to_batch, not sent to solver
        # Convert to float64 for PyTorch compatibility (collate requires numeric types)
        time_vals = self.times[time_slice]
        full_input["time_indices"] = np.array(time_vals, dtype='datetime64[ns]').astype('float64')

        # this is where we will put the target with 50% removal if parameter is set (done in apply_norm())
        full_input["inpaint_mask"] = np.zeros((nt, nlat, nlon), dtype=np.float32)

        t_after_build_input = time.time()

        enable_filtering = getattr(self, 'enable_patch_filtering', True)
        max_retries = getattr(self, 'max_patch_retries', 50)
        
        t_before_validation = time.time()

        if enable_filtering:
            is_valid, reason, stats = self.is_valid_patch(full_input)
            t_after_validation = time.time()

            if not is_valid:
                if not hasattr(self, '_rejection_count'):
                    self._rejection_count = 0
                self._rejection_count += 1
                if self._rejection_count == 1:
                    print(f"[ATTENTION] Patch filtered: {reason} | Stats: {stats}. Problematic if you are in Test mode !!")
                if self._rejection_count % 20 == 0:
                    print(f"[{self._rejection_count} patches rejetés] Dernier rejet: {reason}")

                if self._rejection_count < max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx)
                else:
                    print(f"WARNING: {max_retries} rejets consécutifs, on garde le patch malgré: {reason}")
                    self._rejection_count = 0
        else:
            t_after_validation = time.time()

        # t_before_postpro = time.time()
        if self.postpro_fn is not None:
            full_input = self.postpro_fn(full_input)
        # t_after_postpro = time.time()

        # Log timing summary
        # t_total = t_after_postpro - t_start_total
        # t_slices = (t_after_slices - t_start_total) * 1000
        # t_zarr = (t_after_zarr_loop - t_after_slices) * 1000
        # t_stack = (t_after_stack - t_after_zarr_loop) * 1000
        # t_build = (t_after_build_input - t_after_stack) * 1000
        # t_valid = (t_after_validation - t_before_validation) * 1000
        # t_postpro = (t_after_postpro - t_before_postpro) * 1000

        # Désactivé : Logging de timing trop volumineux
        # self.timing_logger.info(
        #     f"idx={idx} | TOTAL={t_total*1000:.1f}ms | "
        #     f"slices={t_slices:.1f}ms | zarr_io={t_zarr:.1f}ms | "
        #     f"stack={t_stack:.1f}ms | build={t_build:.1f}ms | "
        #     f"valid={t_valid:.1f}ms | postpro={t_postpro:.1f}ms"
        # )

        return full_input
    
    def is_valid_patch(self, patch_data_dict, min_valid_ratio=0.08, min_variance=0.05, min_ocean_ratio=0.05):
        """
        Vérifie si un patch est valide pour l'entraînement.
        Returns:
            tuple: (is_valid, rejection_reason, stats)
                - is_valid: bool, True si le patch est valide
                - rejection_reason: str ou None, raison du rejet si invalide
                - stats: dict avec {mean, std, ocean_pct} du patch (ou None si invalide)
        """
        stats = {'mean': None, 'std': None, 'ocean_pct': None}

        # tgt_sst (target réelle, AVANT normalisation)
        if 'tgt_sst' not in patch_data_dict:
            raise KeyError("tgt_sst manquant dans patch_data_dict")
        data = patch_data_dict['tgt_sst']

        # Critère 1: Au moins min_valid_ratio% de données valides
        valid_ratio = np.sum(~np.isnan(data)) / data.size
        if valid_ratio < min_valid_ratio:
            return False, f"not_enough_data (valid_ratio={valid_ratio:.2%} < {min_valid_ratio:.2%})", None

        # Critère 2: Variance minimale (éviter patches uniformes)
        var = np.nanvar(data)
        if var < min_variance:
            return False, f"low_variance (var={var:.4f} < {min_variance:.4f})", None

        # Calculer les stats pour le logging (en °C, AVANT normalisation)
        stats['mean'] = float(np.nanmean(data))
        stats['std'] = float(np.nanstd(data))

        # Critère 3: Ratio océan suffisant
        if 'surfmask' not in patch_data_dict:
            raise KeyError("surfmask manquant dans patch_data_dict")

        mask = patch_data_dict['surfmask']
        # surfmask peut avoir shape (1, nlat, nlon) ou (nlat, nlon)
        if mask.ndim == 3:
            mask = mask[0]  # Prendre premier timestep si 3D

        # surfmask: 0=terre, 1=ocean, 2=interface eau-glace, 3=glace
        # On garde: ocean (1) + interface eau-glace (2) + glace (3)
        ocean_pixels = np.sum((mask == 1) | (mask == 2) | (mask == 3))
        total_pixels = mask.size
        ocean_ratio = ocean_pixels / total_pixels
        stats['ocean_pct'] = float(ocean_ratio * 100)

        if ocean_ratio < min_ocean_ratio:
            return False, f"not_enough_ocean (ocean_ratio={ocean_ratio:.2%} < {min_ocean_ratio:.2%})", None

        return True, None, stats

    def reconstruct(self, batches, index_time, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

        batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, index_time, weight)

    def reconstruct_from_items(self, items, index_time, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
            weight = np.expand_dims(weight, 0)

        nvars = items[0].shape[0]
        # Adapt for lat/lon dimensions
        nlat = self.da_dims.get('lat', self.da_dims.get('yc'))
        nlon = self.da_dims.get('lon', self.da_dims.get('xc'))
        result_tensor = np.zeros((nvars, 1, nlat, nlon))
        count_tensor = np.zeros((nvars, 1, nlat, nlon))

        coords = self.get_coords()

        for idx, item in enumerate(items):
            c = coords[idx]
            iy = [np.where(self.lat_1d == y)[0][0] for y in c.lat.values]
            ix = [np.where(self.lon_1d == x)[0][0] for x in c.lon.values]
            result_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += item * weight
            count_tensor[:, 0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += weight

        result_tensor /= np.maximum(count_tensor, 1e-6)
        result_da = xr.DataArray(
            result_tensor,
            dims=[f'v{i}' for i in range(nvars)] + ["time", "lat", "lon"],
            coords={
                "time": [self.times[index_time]],
                "lon": self.lon_1d,
                "lat": self.lat_1d,
                "lon_2d": (("lat", "lon"), self.lon_2d),
                "lat_2d": (("lat", "lon"), self.lat_2d)
            }
        )
        return result_da

class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))
    
        return xr.concat(rec_das,dim="time")


class XrDatasetSingleDay(XrDataset):
    """
    Dataset spécialisé pour test sur UNE seule journée cible.
    
    Sélectionne aléatoirement une journée dans la plage fournie et ne génère que les patches spatiaux pour reconstruire cette journée (une seule fenêtre temporelle).
    
    Hérite de XrDataset mais surcharge __init__ pour:
    - Sélectionner une journée aléatoire valide (avec marge temporelle suffisante)
    - Extraire uniquement la fenêtre de patch_dims['time'] jours centrée sur cette journée
    - Forcer stride temporel = patch_dims['time'] pour n'avoir qu'une seule fenêtre
    """
    
    def __init__(self, *args, **kwargs):
        times = kwargs.get('times')
        sst_daily_paths = kwargs.get('sst_daily_paths')
        patch_dims = kwargs.get('patch_dims', {'time': 15})
        if times is None or sst_daily_paths is None:
            raise ValueError("XrDatasetSingleDay requires 'times' and 'sst_daily_paths'")
        nt = len(times)
        patch_t = patch_dims.get('time', 15)
        
        # Calculer la plage valide pour la journée centrale
        margin = patch_t // 2
        valid_start = margin
        valid_end = nt - margin
        
        if valid_end <= valid_start:
            raise ValueError(
                f"Not enough days in test period ({nt}) for patch_dims['time']={patch_t}. "
                f"Need at least {patch_t} days.")

        # Si test_date_idx est fourni dans kwargs, on l'utilise, sinon random
        target_day_idx = kwargs.pop('test_date_idx', None)
        if target_day_idx is None:
            target_day_idx = random.randint(valid_start, valid_end - 1)
        else:
            # Valider que l'index fourni est dans la plage valide
            if not (valid_start <= target_day_idx < valid_end):
                print(f"WARNING: test_date_idx={target_day_idx} hors plage [{valid_start}, {valid_end}), utilisation de {valid_start}")
                target_day_idx = valid_start

        window_start = target_day_idx - margin
        window_end = window_start + patch_t
        
        times_windowed = times[window_start:window_end]
        if isinstance(sst_daily_paths, dict):
            # Multi-résolution: filtrer chaque résolution
            sst_daily_paths_windowed = {
                res: paths[window_start:window_end] 
                for res, paths in sst_daily_paths.items()
            }
        else:
            # Mono-résolution: filtrer directement la liste
            sst_daily_paths_windowed = sst_daily_paths[window_start:window_end]
        
        kwargs['times'] = times_windowed
        kwargs['sst_daily_paths'] = sst_daily_paths_windowed
        
        # Forcer stride temporel = patch_dims pour n'avoir qu'UNE fenêtre temporelle
        # IMPORTANT: Copier le dict pour ne pas modifier l'original (shared entre train/val/test)
        strides = kwargs.get('strides', {}).copy()
        strides['time'] = patch_t
        kwargs['strides'] = strides
        
        # IMPORTANT: Désactiver stride_test pour garder les strides spatiaux normaux (64×64)
        # sinon stride_test=True remplacerait les strides par strides_test (vide = stride 1)
        kwargs['stride_test'] = False
        
        target_date = times_windowed[margin]
        print(f"  Target day: {target_date}")
        print(f"  Temporal window: {times_windowed[0]} to {times_windowed[-1]} ({len(times_windowed)} days)\n")
        super().__init__(*args, **kwargs)


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, sst_paths,
                 covariates_paths, covariates,
                 tgt_vars,
                 mask_path,
                 domain_name, domains,
                 xrds_kw, dl_kw, 
                 norm_stats, norm_stats_covs,
                 aug_kw=None, res=500, pads=[False,False,False], 
                 resize=1,
                 **kwargs):
        
        super().__init__()
        self.sst_paths = sst_paths
        self.covariates_paths = covariates_paths
        self.covariates = covariates
        self.tgt_vars = tgt_vars
        self.mask_path = mask_path
        self.domain_name = domain_name
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self.res = res
        self.pads = pads
        self.resize = resize
        self._norm_stats = norm_stats  # Satellite variables normalization (VAR_GROUPS)
        self._norm_stats_covs = norm_stats_covs  # Covariate normalization (COVARIATES)

        self.resize = resize
        # Load base grid from first SST file to get lat/lon
        first_file = str(self.sst_paths[0])
        if first_file.endswith('.zarr'):
            # print(f"\n[DEBUG] Tentative ouverture Zarr: '{first_file}'")
            # print(f"\n[DEBUG] Le fichier existe il ? {os.path.exists(first_file)}")
            store = zarr.open(first_file, mode='r')
            # --- DEBUT DU PATCH MOUCHARD ---
            try:
                self.lon = np.array(store['lon'][:])
            except KeyError:
                print("\n" + "!"*60)
                print(f"  ERREUR CRITIQUE DÉTECTÉE !")
                print(f"  Le code a planté car il ne trouve pas 'lon'.")
                print(f"  Le fichier Zarr en cours d'ouverture est : {store.store.path if hasattr(store, 'store') and hasattr(store.store, 'path') else store}")
                print(f"  Les clés trouvées dans ce fichier sont : {list(store.keys())}")
                print("!"*60 + "\n")
                raise  # On fait replanter le code pour arrêter tout
            # --- FIN DU PATCH MOUCHARD ---
            self.lat = np.array(store['lat'][:])
        else:
            sst_base = xr.open_dataset(first_file)
            self.lon = sst_base.lon.values
            self.lat = sst_base.lat.values
            sst_base.close()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

        # For SST, surfmask is already in the NetCDF files, no need to build land mask
        # mask_path is kept for backward compatibility but not used
        self.mask = None

    def build_land_mask(self):
        # mask = xr.Dataset(
        #                 coords={
        #                     "xc": self.xc,
        #                     "yc": self.yc,
        #                     "lon": (["yc","xc"], self.lon),
        #                     "lat": (["yc","xc"], self.lat)
        #                     })
        # land_mask = np.zeros((len(self.yc),len(self.xc)))
        # land_50m = cfeature.NaturalEarthFeature('physical','land','10m')
        # land_polygons_cartopy = list(land_50m.geometries())
        # land_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=land_polygons_cartopy)
        # step_yc = np.concatenate((np.arange(len(self.yc),step=1000),np.array([len(self.yc)])))
        # step_xc = np.concatenate((np.arange(len(self.xc),step=1000),np.array([len(self.xc)])))
        # for i in range(len(step_yc)-1):
        #     for j in range(len(step_xc)-1):
        #         lon = self.lon[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]]
        #         lat = self.lat[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]]
        #         nlat, nlon = lon.shape
        #         points = GeoSeries(gpd.points_from_xy(lon.flatten(), lat.flatten()))
        #         points_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
        #         joined = gpd.sjoin(points_gdf, land_gdf, how='left', predicate='within')
        #         part_land_mask = np.reshape(np.array(joined['index_right'].notnull().to_list()),(nlat,nlon))
        #         land_mask[step_yc[i]:step_yc[i+1],step_xc[j]:step_xc[j+1]] = part_land_mask
        # mask = mask.update({"mask":(("yc","xc"),land_mask)})
        # encoding = {
        #            var: {"zlib": True, "complevel": 6}  # 9 = compression maximale
        #            for var in mask.data_vars
        #            }
        # mask.to_netcdf(
        #               self.mask_path,
        #               format="NETCDF4",
        #               engine="netcdf4",
        #               encoding=encoding
        # )
        # return mask.mask 
        raise NotImplementedError("Land mask building is not implemented for SST datasets.")
    
    def norm_stats(self):
        return self._norm_stats

    def norm_stats_covs(self):
        return self._norm_stats_covs

    def post_fn(self, rand_obs=False):
        norm_sats = self._norm_stats
        norm_covs = self._norm_stats_covs

        def normalize_var(x, stats):
            if stats['type'] == 'zscore':
                return (x - stats['mean']) / stats['std']
            elif stats['type'] == 'minmax':
                return (x - stats['min']) / (stats['max'] - stats['min'])
            elif stats['type'] is None:
                return x
            else:
                raise ValueError(f"Unknown normalization type {stats['type']} for variable")

        def generate_random_obs_mask(gt_item):
            """
            Randomly mask ~50% of valid observations with rectangles.
            Input:
                gt_item (np.ndarray): should be of shape (15,256,256) or similar
            Returns:
                tuple: (masked_data, inpaint_mask)
                    - masked_data: gt_item with removed pixels set to NaN
                    - inpaint_mask: binary mask (1=removed by inpainting, 0=kept)
            """
            obs_mask_item = ~np.isnan(gt_item)
            inpaint_mask = np.zeros_like(gt_item, dtype=np.float32)
            _obs_item = gt_item.copy()
            dtime, dyc, dxc = gt_item.shape
            for t in range(dtime):
                if np.sum(obs_mask_item[t]) > 0.02 * dyc * dxc:
                    obs_objectif = 0.5 * np.sum(obs_mask_item[t])
                    initial_valid = obs_mask_item[t].copy()
                    while np.sum(obs_mask_item[t]) >= obs_objectif:
                        half_h = np.random.randint(2,10)
                        half_w = np.random.randint(2,10)
                        yc = np.random.randint(0, dyc)
                        xc = np.random.randint(0, dxc)
                        obs_mask_item[t, max(0,yc-half_h):min(dyc,yc+half_h+1),
                                         max(0,xc-half_w):min(dxc,xc+half_w+1)] = 0
                    # pixels that were initially valid and have been removed
                    inpaint_mask[t] = (initial_valid & ~obs_mask_item[t]).astype(np.float32)
            masked_data = np.where(obs_mask_item, _obs_item, np.nan)
            return masked_data, inpaint_mask
            
        def apply_norm(item):
            """
            Normalize a batch item according to norm_stats and norm_stats_covs.
            """
            # If item is already a TrainingItem (from retry)
            if isinstance(item, tuple) and hasattr(item, '_fields'):
                # Check if it has tgt_sst_full field (new version)
                if hasattr(item, 'tgt_sst_full'):
                    return item
                else:
                    # Old version without tgt_sst_full, convert to dict and rebuild
                    item = item._asdict()
            
            # Remove coordinate metadata (not part of TrainingItem)
            for coord_key in ['lat_coords', 'lon_coords']:
                item.pop(coord_key, None)
            
            # Add tgt_sst_full placeholder if not present
            if 'tgt_sst_full' not in item:
                item['tgt_sst_full'] = None
            
            data = TrainingItem(**item)
            inpaint_masks = []

            # Normalize target 
            for group, variables in VAR_GROUPS.items():
                for var in variables:
                    var_key = f"{group}_{var}"
                    new_key = f"tgt_{group}_{var}"
                    if hasattr(data, new_key) and (var_key in self.tgt_vars):
                        var_data = getattr(data, new_key)
                        norm_params = norm_sats[group][var]
                        var_data = normalize_var(var_data, norm_params)
                        data = data._replace(**{new_key: var_data})
            
            # CRITICAL FIX: Récupérer les versions BRUTES avant normalisation
            slstr_raw = getattr(data, 'slstr_av', None)
            aasti_raw = getattr(data, 'aasti_av', None)

            if slstr_raw is None and aasti_raw is None:
                raise RuntimeError('No slstr_av or aasti_av present to build tgt_sst')

            # Créer la fusion BRUTE (avant inpainting et normalisation)
            if slstr_raw is None:
                raw_tgt_sst_full = aasti_raw.copy()
            elif aasti_raw is None:
                raw_tgt_sst_full = slstr_raw.copy()
            else:
                raw_tgt_sst_full = np.where(~np.isnan(slstr_raw), slstr_raw, aasti_raw)

            # Require global stats for the fused target
            if 'tgt_sst' not in norm_sats:
                raise RuntimeError("norm_stats missing 'tgt_sst'. Run compute_statistics to add fused-target stats before training.")

            tgt_stats = norm_sats['tgt_sst']

            # tgt_sst_full : Fusion COMPLÈTE (avant inpainting) pour l'évaluation SSL
            tgt_sst_full_normalized = normalize_var(raw_tgt_sst_full, tgt_stats)
            data = data._replace(tgt_sst_full=tgt_sst_full_normalized)

            # Check validity
            if not np.all(np.isnan(tgt_sst_full_normalized)):
                mn = np.nanmin(tgt_sst_full_normalized)
                mx = np.nanmax(tgt_sst_full_normalized)
                assert mn > -5, f"tgt_sst_full min={mn:.2f} trop faible (attendu > -5)"
                assert mx < 5,  f"tgt_sst_full max={mx:.2f} trop élevé (attendu < 5)"

            # Normalisation input (Inpainting seulement sur aasti_av et slstr_av)
            inpaint_mask_slstr = None
            inpaint_mask_aasti = None
            for group, variables in VAR_GROUPS.items():
                for var in variables:
                    var_key = f"{group}_{var}"
                    if hasattr(data, var_key):
                            var_data = getattr(data, var_key)
                            # Appliquer inpainting AVANT normalisation et stocker le masque
                            if rand_obs and var_key in ['aasti_av', 'slstr_av']:
                                var_data, var_inpaint_mask = generate_random_obs_mask(var_data)
                                inpaint_masks.append(var_inpaint_mask)
                                if var_key == 'slstr_av':
                                    inpaint_mask_slstr = var_inpaint_mask
                                elif var_key == 'aasti_av':
                                    inpaint_mask_aasti = var_inpaint_mask
                            norm_params = norm_sats[group][var]
                            var_data = normalize_var(var_data, norm_params)
                            data = data._replace(**{var_key: var_data})

            # Normalisation des covariates
            for cov in COVARIATES:
                if hasattr(data, cov):
                        norm_params = norm_covs[cov]
                        cov_data = normalize_var(getattr(data, cov), norm_params)
                        data = data._replace(**{cov: cov_data})

            # Calculer le masque d'inpainting global (union de tous les masques)
            if rand_obs and len(inpaint_masks) > 0:
                global_inpaint_mask = np.maximum.reduce(inpaint_masks) # un pixel est inpainté si au moins une var l'est
            else:
                if hasattr(data, 'aasti_av'):
                    ref_shape = getattr(data, 'aasti_av').shape  # (nt, nlat, nlon)
                    global_inpaint_mask = np.zeros(ref_shape, dtype=np.float32)
                else:
                    global_inpaint_mask = np.zeros_like(data.surfmask)

            data = data._replace(inpaint_mask=global_inpaint_mask)

            # CRITICAL FIX: Créer tgt_sst depuis la fusion BRUTE (avant normalisation)
            # puis appliquer le même inpainting que sur les inputs
            raw_tgt_sst_masked = raw_tgt_sst_full.copy()

            # Appliquer les masques d'inpainting si nécessaire
            if rand_obs:
                if inpaint_mask_slstr is not None or inpaint_mask_aasti is not None:
                    # Appliquer le masque global à la fusion brute
                    raw_tgt_sst_masked = np.where(global_inpaint_mask == 0, raw_tgt_sst_masked, np.nan)

            # Normaliser la fusion masquée avec tgt_stats (comme tgt_sst_full)
            tgt_sst_normalized = normalize_var(raw_tgt_sst_masked, tgt_stats)
            data = data._replace(tgt_sst=tgt_sst_normalized)
            
            # NOTE: tgt_sst_full (complète) a déjà été créée avant l'inpainting
            # tgt_sst (masquée) vient d'être créée après l'inpainting
            # Pour SSL: input utilise tgt_sst (masquée), target utilise tgt_sst_full (complète)
            
            data = data._replace(lat=normalize_var(data.lat, {"type": "minmax", "min": -90, "max": 90}))
            data = data._replace(lon=normalize_var(data.lon, {"type": "minmax", "min": -180, "max": 180}))
            return data

        return ft.partial(ft.reduce, lambda i, f: f(i), [apply_norm])

    def save_batch_as_NetCDF(self, batch, ibatch, patch_dims, save_dir="/dmidata/users/maxb/PREPROC/"):
        # """
        # Save a batch in NetCDF format, adapted to VAR_GROUPS logic
        # """
        # # Variables à sauvegarder
        # data_vars = {}
        # # Variables satellites (asip, cimr, cristal)
        # for group in VAR_GROUPS:
        #     for var in VAR_GROUPS[group]:
        #         if hasattr(batch, "{group}_{var}"):
        #             data_vars[var] = (('sample', 'time', 'yc', 'xc'), getattr(batch, var).detach().cpu())
        # # Covariates
        # for cov in COVARIATES:
        #     if hasattr(batch, cov):
        #         tensor = getattr(batch, cov)
        #         if torch.is_tensor(tensor) and tensor.ndim == 4:
        #             data_vars[cov] = (('sample', 'time', 'yc', 'xc'), tensor.detach().cpu())
        # # target variables
        # for target in self.tgt_vars:
        #     if hasattr(batch, target):
        #         data_vars[target] = (('sample', 'time', 'yc', 'xc'), getattr(batch, target).detach().cpu())
        # # Coordonnées et masque
        # data_vars.update({
        #     'times': (('sample', 'time'), torch.squeeze(batch.time).detach().cpu().numpy().astype("datetime64[s]")),
        #     'ycs': (('sample', 'yc'), torch.squeeze(batch.yc.detach().cpu())),
        #     'xcs': (('sample', 'xc'), torch.squeeze(batch.xc.detach().cpu())),
        #     'lat': (('sample', 'yc', 'xc'), torch.squeeze(batch.lat).detach().cpu()),
        #     'lon': (('sample', 'yc', 'xc'), torch.squeeze(batch.lon).detach().cpu()),
        #     'land_mask': (('sample', 'yc', 'xc'), torch.squeeze(batch.land_mask).detach().cpu()),})
        # coords = {
        #     'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
        #     'time': np.arange(patch_dims['time']),
        #     'yc': np.arange(patch_dims['yc']),
        #     'xc': np.arange(patch_dims['xc'])}
        # # Sauvegarde
        # ds = xr.Dataset(data_vars=data_vars, coords=coords)
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}.nc")
        # ds.to_netcdf(save_path)
        raise NotImplementedError("save_batch_as_NetCDF is not used in SST")

    def setup(self, stage='test'):

        def select_paths(files, times, fmt="%Y%m%d"):
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
            sst_paths, times = select_paths(self.sst_paths, self.domains[split]['time'])
            cov_paths, _ = select_paths(self.covariates_paths, self.domains[split]['time'], fmt="%Y-%m-%d")
            return XrDataset(
                sst_paths=sst_paths,
                covariates_paths=cov_paths,
                covariates=COVARIATES,
                tgt_vars=self.tgt_vars,
                mask=self.mask,
                times=times,
                **self.xrds_kw,
                postpro_fn=self.post_fn(rand_obs=(split=='train')),
                res=self.res,
                pad=self.pads[0 if split == 'train' else 1 if split == 'val' else 2],
                resize = self.resize,
                stride_test=(split != 'train'),
                load_data=(split == 'test'),
            )

        self.train_ds = create_dataset('train')
        self.val_ds = create_dataset('val')
        self.test_ds = create_dataset('test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        # Pas de shuffle pour validation (reproductibilité)
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

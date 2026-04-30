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
import logging
import time
import warnings
import sys
from datetime import datetime
from src.utils import extract_encompassing_patch

from torch.utils.data import Sampler


def _worker_init_fn(worker_id):
    """Initialize worker and suppress multiprocessing cleanup warnings."""
    warnings.filterwarnings('ignore', message='.*Device or resource busy.*')
    warnings.filterwarnings('ignore', category=ResourceWarning)

    # CRITICAL: Disable Dask threading to avoid deadlocks with DDP
    # xr.open_zarr() creates Dask threads that conflict with DDP multiprocessing
    import dask
    dask.config.set(scheduler='synchronous')

    import multiprocessing.util as mp_util
    def _silent_remove_temp_dir(rmtree_func, tempdir):
        """Silently remove temp directory, ignoring resource busy errors.

        Args:
            rmtree_func: The rmtree function to use (passed by multiprocessing)
            tempdir: The temporary directory path to remove
        """
        try:
            # Call rmtree without onerror/onexc - let our try/except handle it
            rmtree_func(tempdir)
        except OSError as e:
            if e.errno != 16:  # Not "Device or resource busy"
                raise
            # Silently ignore cleanup errors - OS will clean up later
    mp_util._remove_temp_dir = _silent_remove_temp_dir


# OBSOLÈTE: pad_dataset déplacé dans src/utils.py pour centralisation
# def pad_dataset(ds, pad_lat=0, pad_lon=0):
#     """Pad SST Dataset (time, lat, lon) avec extension des coordonnées 1D."""
#     raise NotImplementedError("pad_dataset moved to src.utils - import from there")


class XrDatasetMultiResTrain(XrDataset):

    def __init__(self, multires=[10, 3, 1], precomputed=True, *args, **kwargs):
        # import os
        # print(f"[INIT] Worker PID={os.getpid()} initializing XrDatasetMultiResTrain", flush=True)

        # Extract enable_patch_filtering before passing to parent
        # (parent class doesn't accept this argument)
        self.enable_patch_filtering = kwargs.pop('enable_patch_filtering', True)
        # Seuils explicites pour is_valid_patch (sinon valeurs par défaut de la méthode).
        self.patch_filter_kwargs = kwargs.pop('patch_filter_kwargs', {}) or {}

        # Log filtering status for training
        filter_status = "ENABLED" if self.enable_patch_filtering else "DISABLED"
        thresholds_str = (
            f" (thresholds={self.patch_filter_kwargs})"
            if self.patch_filter_kwargs else ""
        )
        print(f"[XrDatasetMultiResTrain] Patch filtering {filter_status}{thresholds_str}")

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

    def _setup_timing_logger(self):
        """Configure file logger for timing profiling (multiprocessing-safe)"""
        pass
        # self.timing_logger = logging.getLogger(f'timing_profile_multires_worker_{os.getpid()}')
        # self.timing_logger.setLevel(logging.INFO)
        # self.timing_logger.handlers.clear()

        # log_file = 'timings_multires.log'
        # file_handler = logging.FileHandler(log_file, mode='a')
        # file_handler.setLevel(logging.INFO)

        # formatter = logging.Formatter('%(asctime)s | PID=%(process)d | %(message)s')
        # file_handler.setFormatter(formatter)

        # self.timing_logger.addHandler(file_handler)
        # self.timing_logger.propagate = False

    def __getitem__(self, idx):
        """
        Retourne un dict avec patches multi-résolution IMBRIQUÉS géographiquement:
        - patch_x1: 256x256 pixels à résolution x1 (haute résolution, référence)
        - patch_x3: 256x256 pixels à résolution x3 (englobe x1 géographiquement)
        - patch_x10: 256x256 pixels à résolution x10 (englobe x3 géographiquement)

        Strategy : Extract from FINEST to COARSEST to ensure geographic nesting
        1. Extract x1 FIRST (reference patch from balanced dataset)
        2. Find x3 that ENCOMPASSES x1 geographically
        3. Find x10 that ENCOMPASSES x3 geographically
        """
        # DEBUG: Print idx to track what's being requested
        if not hasattr(self, '_getitem_counter'):
            self._getitem_counter = 0
        self._getitem_counter += 1
        
        # Debug: Track worker initialization (first 10 calls only)
        # if self._getitem_counter <= 10:
        #     print(f"[Worker PID={os.getpid()}] Loading idx={idx}")
        
        t_start_total = time.time()

        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                        self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        t_after_slices = time.time()

        # Temporarily disable parent's filtering to prevent recursive retry issues
        # when loading the high-res patch (x1) via super().__getitem__()
        # This is needed because we do our own validation AFTER collecting all resolutions
        parent_filtering_state = getattr(self, 'enable_patch_filtering', True)
        self.enable_patch_filtering = False

        out = {}
        lowres_times = {}

        # Extraire x1 et récupérer ses bounds géographiques
        t_x1_start = time.time()
        hr_sample = super().__getitem__(idx)
        out[f"patch_x{self.resize}"] = hr_sample
        
        lat_geo_x1 = hr_sample['lat_geo']
        lon_geo_x1 = hr_sample['lon_geo']
        lat_bounds_x1 = (float(lat_geo_x1.min()), float(lat_geo_x1.max()))
        lon_bounds_x1 = (float(lon_geo_x1.min()), float(lon_geo_x1.max()))
        
        t_x1_end = time.time()
        lowres_times[f"x{self.resize}"] = (t_x1_end - t_x1_start) * 1000
        
        # Pour chaque résolution plus grossière, extraire patch englobant
        coarser_factors = sorted([f for f in self.multires if f > self.resize])
        prev_lat_bounds = lat_bounds_x1
        prev_lon_bounds = lon_bounds_x1
        
        for factor in coarser_factors:
            t_factor_start = time.time()
            
            enlarged_patch = extract_encompassing_patch(
                dataset_obj=self,sl=sl,factor=factor // self.resize,lat_bounds=prev_lat_bounds,lon_bounds=prev_lon_bounds,
                VAR_GROUPS=VAR_GROUPS,COVARIATES=COVARIATES,patch_dims=self.patch_dims,tgt_vars=self.tgt_vars)
            
            out[f"patch_x{factor}"] = enlarged_patch
            
            # Mettre à jour bounds pour la prochaine résolution plus grossière
            lat_geo = enlarged_patch['lat_geo']
            lon_geo = enlarged_patch['lon_geo']
            prev_lat_bounds = (float(lat_geo.min()), float(lat_geo.max()))
            prev_lon_bounds = (float(lon_geo.min()), float(lon_geo.max()))
            
            t_factor_end = time.time()
            lowres_times[f"x{factor}"] = (t_factor_end - t_factor_start) * 1000
        
        t_after_lowres = time.time()
        
        # Restore parent's filtering state before validation
        self.enable_patch_filtering = parent_filtering_state
        
        # Apply patch filtering on hr_sample (x1 resolution) before normalization
        t_before_validation = time.time()
        enable_filtering = getattr(self, 'enable_patch_filtering', True)
        max_retries = getattr(self, 'max_patch_retries', 50)

        # Track retries for this sample (increment on each rejection)
        if not hasattr(self, '_current_sample_retries'):
            self._current_sample_retries = 0

        # patch_stats = None
        # resolution = f"x{self.resize}"

        if enable_filtering and hasattr(self, 'is_valid_patch'):
            filter_kwargs = getattr(self, 'patch_filter_kwargs', {}) or {}
            is_valid, reason, patch_stats = self.is_valid_patch(hr_sample, **filter_kwargs)
            t_after_validation = time.time()

            if not is_valid:
                if not hasattr(self, '_rejection_count'):
                    self._rejection_count = 0
                self._rejection_count += 1
                self._current_sample_retries += 1
                if self._current_sample_retries < max_retries:
                    new_idx = np.random.randint(0, len(self))
                    return self.__getitem__(new_idx)
                else:
                    print(f"WARNING: {max_retries} rejets consécutifs, on garde le patch malgré: {reason}")
                    self._rejection_count = 0
                    self._current_sample_retries = 0
        else:
            t_after_validation = time.time()

        self._rejection_count = 0
        self._current_sample_retries = 0

        t_before_postpro = time.time()
        if self.saved_postpro_fn is not None:
            for key in out:
                if isinstance(out[key], dict):  # Should be a dict from load_patch_at_resolution
                    # print(f"[DEBUG __getitem__] Applying saved_postpro_fn to out['{key}']")
                    out[key] = self.saved_postpro_fn(out[key])
        t_after_postpro = time.time()

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

        # Log timing summary
        t_end_total = time.time()
        t_total = (t_end_total - t_start_total) * 1000
        t_slices = (t_after_slices - t_start_total) * 1000
        t_valid = (t_after_validation - t_before_validation) * 1000
        t_postpro = (t_after_postpro - t_before_postpro) * 1000

        # Build lowres details string (includes all resolutions now)
        lowres_detail = " | ".join([f"{k}={v:.1f}ms" for k, v in lowres_times.items()])

        # Désactivé : Logging de timing trop volumineux
        # self.timing_logger.info(
        #     f"idx={idx} | TOTAL={t_total:.1f}ms | "
        #     f"slices={t_slices:.1f}ms | {lowres_detail} | "
        #     f"valid={t_valid:.1f}ms | postpro={t_postpro:.1f}ms")
        
        gc.collect()
        return out


class _XrDatasetMultiResTestBase:
    """
    Classe de base pour les datasets multi-résolution en mode test (XrDatasetMultiResTestMultiDay et XrDatasetMultiResTestSingleDay).
    """
    DATASET_CLASS = None
    def __init__(self, multires=[1], *args, **kwargs):
        if self.DATASET_CLASS is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set DATASET_CLASS class attribute")
        
        # Extract enable_patch_filtering: default False for test (we want to reconstruct ALL patches)
        enable_patch_filtering = kwargs.pop('enable_patch_filtering', False)

        # Extract strides_by_res if provided
        strides_by_res = kwargs.pop('strides_by_res', None)

        # Log filtering status for test mode
        filter_status = "ENABLED" if enable_patch_filtering else "DISABLED"
        print(f"[DATASET CONFIG] Patch filtering {filter_status} for test mode\n")

        self.datasets = {}
        for res in multires:
            kwargs_copy = kwargs.copy()
            kwargs_copy["resize"] = res
            kwargs_copy["enable_patch_filtering"] = enable_patch_filtering

            # Override strides if strides_by_res is provided for this resolution
            if strides_by_res is not None and res in strides_by_res:
                kwargs_copy["strides"] = strides_by_res[res]
                print(f"--- Resolution x{res} Configuration ---")
                print(f"  Strides: {strides_by_res[res]}")

            self.datasets[res] = self.DATASET_CLASS(load_data=True, *args, **kwargs_copy)
    
    def get_dataloader_dict(self, batch_size=1, **loader_kwargs):
        """
        Retourne un dict de DataLoaders, un par résolution.
        Returns:
            dict: {resolution: DataLoader} ex: {1: DataLoader(...), 3: DataLoader(...), 10: DataLoader(...)}
        """
        from torch.utils.data import DataLoader
        return {res: DataLoader(ds, batch_size=batch_size, **loader_kwargs)
                for res, ds in self.datasets.items()}


class XrDatasetMultiResTestMultiDay(_XrDatasetMultiResTestBase):
    """
    Dataset multi-résolution pour test sur PLUSIEURS JOURS.
    """
    DATASET_CLASS = XrDataset

class XrDatasetMultiResTestSingleDay(_XrDatasetMultiResTestBase):
    """
    Dataset multi-résolution pour test sur UNE SEULE JOURNÉE.
    Utilise XrDatasetSingleDay pour chaque résolution, garantissant que toutes les résolutions travaillent sur la même journée cible.
    """
    def __init__(self, multires=[1], *args, **kwargs):
        from contrib.SST.data import XrDatasetSingleDay
        self.DATASET_CLASS = XrDatasetSingleDay
        super().__init__(multires=multires, *args, **kwargs)

XrDatasetMultiResTest = XrDatasetMultiResTestMultiDay


def debug_collate_fn(batch_list):
    """
    DEBUG collate function that prints what it receives and returns default collate
    """
    print(f"\n[DEBUG COLLATE] Received batch_list with {len(batch_list)} samples")
    for i, sample in enumerate(batch_list[:2]):  # Print first 2 samples
        print(f"[DEBUG COLLATE] Sample {i} keys: {sample.keys()}")
        for res_key in sample.keys():
            # Access TrainingItem attributes directly (it's a namedtuple)
            patch_item = sample[res_key]
            if hasattr(patch_item, 'lat_geo'):
                lat = patch_item.lat_geo
                print(f"  {res_key}: lat_geo range [{lat.min():.2f}, {lat.max():.2f}]")
    
    # Use default collate
    from torch.utils.data._utils.collate import default_collate
    result = default_collate(batch_list)
    
    print(f"[DEBUG COLLATE] After default_collate, result keys: {result.keys()}")
    for res_key in list(result.keys())[:2]:  # Print first 2 resolutions
        if hasattr(result[res_key], 'lat_geo'):
            lat = result[res_key].lat_geo
            print(f"  {res_key}: lat_geo shape {lat.shape}, sample 0 range [{lat[0].min():.2f}, {lat[0].max():.2f}]")
    
    return result


class BaseDataModuleMultiRes(BaseDataModule):

    def __init__(self, sst_daily_paths, multires=[1], covariates_paths=None, covariates=None, mask_path=None, domain_name='sst_multires',
                 domains=None, precomputed=True, res=5.0, norm_stats=None, norm_stats_covs=None,
                 patch_filter=None, val_set_dir=None, n_viz=16, n_loss=48, rebuild_val_set=False,
                 val_set_seed=42, val_set_max_scan=2000, val_candidate_budget=None,
                 val_set_num_workers=0,
                 *args, **kwargs):
        if covariates_paths is None:
            covariates_paths = []
        if covariates is None:
            covariates = COVARIATES
        if domains is None:
            domains = {
                'train': {'time': slice(None, None)},
                'val': {'time': slice(None, None)},
                'test': {'time': slice(None, None)}}

        # Patch filtering thresholds par split. Si patch_filter est None,
        # is_valid_patch utilise ses defaults (0.08 / 0.05 / 0.05).
        # Structure attendue (dict, optionnel) :
        #   patch_filter:
        #     train:    {min_valid_ratio, min_variance, min_ocean_ratio}
        #     val_viz:  {...}   # filtre dur (qualité visuelle)
        #     val_loss: {...}   # filtre bas (représentativité loss)
        self.patch_filter = patch_filter or {}
        self.val_set_dir = val_set_dir
        self.n_viz = n_viz
        self.n_loss = n_loss
        self.rebuild_val_set = rebuild_val_set
        self.val_set_seed = val_set_seed
        self.val_set_max_scan = val_set_max_scan
        self.val_candidate_budget = val_candidate_budget if val_candidate_budget is not None else val_set_max_scan
        self.val_set_num_workers = val_set_num_workers
        # Si sst_daily_paths est un dossier, scanner pour trouver tous les fichiers
        if isinstance(sst_daily_paths, str):
            from pathlib import Path
            import glob
            path = Path(sst_daily_paths)
            if path.is_dir():
                # Scanner tous les sous-dossiers (années) pour trouver TOUS les fichiers .zarr (x1, x3, x10)
                all_files = []
                for year_dir in sorted(path.iterdir()):
                    if year_dir.is_dir():
                        all_files.extend(sorted(year_dir.glob('*_x*.zarr')))
                sst_daily_paths = [str(f) for f in all_files]
            elif not path.exists():
                raise FileNotFoundError(f"Path does not exist: {sst_daily_paths}")
        
        super().__init__(sst_paths=sst_daily_paths, covariates_paths=covariates_paths, covariates=covariates, mask_path=mask_path, 
                         domain_name=domain_name, domains=domains, norm_stats=norm_stats, norm_stats_covs=norm_stats_covs, *args, **kwargs)
        self.multires = multires
        self.resize = self.multires[-1]
        self.precomputed = precomputed
        self.res = res

    def setup(self, stage='test'):
        """
        Configure datasets pour train/val/test avec multi-résolution
        Args:
            stage: 'train', 'val', ou 'test'
        """
        # CRITICAL: Disable Dask threading globally to avoid deadlocks with DDP
        # xr.open_zarr() creates Dask threads that conflict with DDP multiprocessing
        import dask
        dask.config.set(scheduler='synchronous')

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
            else:
                start, end = times.start, times.stop
                dts = pd.date_range(start, end)
                dates = dts.strftime(fmt).tolist()
                
            # 1 : Filtrer les fichiers en fonction des dates souhaitées
            selected_files = np.sort([f for f in files if any(date in f for date in dates)])
            # 2 : Extraire les dates directement des noms de fichiers sélectionnés
            time_vals_from_files = []
            for f in selected_files:
                basename = os.path.basename(f)
                # Extrait 'YYYYMMDD' du nom de fichier comme '2024010112_...'
                date_str = basename[:8] 
                time_vals_from_files.append(pd.to_datetime(date_str, format=fmt))
            return selected_files, np.array(time_vals_from_files)

        
        def create_dataset(split):
            """
            Crée XrDataset multi-résolution pour un split donné
            Args: split: 'train', 'val', ou 'test'
            Returns: XrDatasetMultiResTrain, XrDatasetMultiResTestMultiDay, ou XrDatasetMultiResTestSingleDay
            """
            sst_paths, times = select_paths(self.sst_paths, self.domains[split]['time'], fmt="%Y%m%d")
            if self.precomputed:
                sst_paths = organize_by_resolution(sst_paths)
                if isinstance(sst_paths, dict):
                    res_summary = {res: f"{len(paths)} files" for res, paths in sst_paths.items()}
                    print(f"\n[MULTI-RES {split.upper()}] Fichiers trouvés: {res_summary}")
                    # Recalculer times depuis la résolution de base (sinon 3× trop d'entrées)
                    base_res = self.multires[-1]
                    if base_res in sst_paths:
                        times = np.array([pd.to_datetime(os.path.basename(f)[:8], format="%Y%m%d")
                                          for f in sst_paths[base_res]])
            # Choisir la classe appropriée selon le split et le mode single_day
            if split == "test":
                test_single_day = self.xrds_kw.get('test_single_day', False)
                if test_single_day:
                    XrDatasetMultiRes = XrDatasetMultiResTestSingleDay
                else:
                    XrDatasetMultiRes = XrDatasetMultiResTestMultiDay
            else: 
                XrDatasetMultiRes = XrDatasetMultiResTrain
            
            # Filtrer les kwargs pour ne pas passer 'test_single_day' aux datasets non-test
            xrds_kw_filtered = {k: v for k, v in self.xrds_kw.items() if k != 'test_single_day'}
            
            # Désactiver le filtrage des patches en mode test : on veut reconstruire TOUT le domaine,
            # même les patches vides, car on a au moins l'info de PMW et des résolutions supérieures
            if split == 'test':
                xrds_kw_filtered['enable_patch_filtering'] = False
            elif split == 'val':
                # build_validation_set fait son propre filtrage. On désactive celui du dataset
                # pour ne pas que les retries internes masquent des candidats.
                xrds_kw_filtered['enable_patch_filtering'] = False
            elif split == 'train':
                # Seuils explicites pour le filtrage train (sinon les defaults de
                # is_valid_patch s'appliquent : 0.08 / 0.05 / 0.05).
                if 'train' in self.patch_filter:
                    xrds_kw_filtered['patch_filter_kwargs'] = dict(self.patch_filter['train'])

            return XrDatasetMultiRes(
                multires=self.multires, sst_daily_paths=sst_paths, tgt_vars=self.tgt_vars, mask=self.mask,times=times,
                precomputed=self.precomputed,**xrds_kw_filtered, postpro_fn=self.post_fn(rand_obs=(split == 'train')),
                res=self.res, pad=self.pads[0 if split == 'train' else 1 if split == 'val' else 2],
                stride_test=(split != 'train'), resize=self.resize)
        # Chargement conditionnel selon le stage
        if stage == 'fit' or stage is None:
            self.train_ds = create_dataset('train')
            self.val_ds = create_dataset('val')

            # CRITICAL FIX: Disable Dask threading before validation patch selection
            # to avoid deadlock with multiprocessing (xr.open_zarr creates threads)
            import dask
            with dask.config.set(scheduler='synchronous'):
                self.validation_indices = self._build_or_load_val_set()

        if stage == 'test' or stage is None:
            self.test_ds = create_dataset('test')
    
    def _build_or_load_val_set(self):
        """Construit (ou recharge) le set de validation figé.

        Lit ``self.val_set_dir/val_indices.json`` s'il existe et que
        ``rebuild_val_set`` est False. Sinon scanne ``val_ds`` via
        :func:`build_val_set.build_validation_set`, écrit le JSON et les
        histogrammes, puis renvoie la liste d'indices.

        L'ordre est strict : les ``n_viz`` premiers indices sont les patchs
        viz, les ``n_loss`` suivants sont les patchs loss.
        """
        from pathlib import Path
        from contrib.SST.build_val_set import (
            FilterThresholds,
            build_validation_set,
            check_val_cache_compatible,
            load_validation_indices,
        )

        if self.val_set_dir is None:
            raise ValueError(
                "val_set_dir doit être fourni dans la config datamodule "
                "pour activer le set de validation figé."
            )
        out_dir = Path(self.val_set_dir)
        json_path = out_dir / "val_indices.json"

        viz_cfg = self.patch_filter.get('val_viz') or {}
        loss_cfg = self.patch_filter.get('val_loss') or {}

        filter_viz = FilterThresholds(
            min_valid_ratio=viz_cfg.get('min_valid_ratio', 0.50),
            min_variance=viz_cfg.get('min_variance', 0.30),
            min_ocean_ratio=viz_cfg.get('min_ocean_ratio', 0.50),
        )
        filter_loss = FilterThresholds(
            min_valid_ratio=loss_cfg.get('min_valid_ratio', 0.02),
            min_variance=loss_cfg.get('min_variance', 0.05),
            min_ocean_ratio=loss_cfg.get('min_ocean_ratio', 0.05),
        )

        candidate_budget = self.val_candidate_budget

        def cache_status(ignore_rebuild=False):
            if not json_path.exists() or (self.rebuild_val_set and not ignore_rebuild):
                return False, "cache absent ou rebuild_val_set=true"
            return check_val_cache_compatible(
                json_path,
                n_viz=self.n_viz,
                n_loss=self.n_loss,
                dataset_len=len(self.val_ds),
                candidate_budget=candidate_budget,
                seed=self.val_set_seed,
                filter_viz=filter_viz,
                filter_loss=filter_loss,
            )

        rank, world_size, dist_ready = self._distributed_context()
        if rank != 0:
            if dist_ready:
                import torch.distributed as dist
                print(f"[VAL SET] Rank {rank}/{world_size} waiting for rank 0 cache")
                dist.barrier()
            else:
                self._wait_for_val_cache(
                    json_path,
                    lambda: cache_status(ignore_rebuild=True),
                )

            compatible, reason = check_val_cache_compatible(
                json_path,
                n_viz=self.n_viz,
                n_loss=self.n_loss,
                dataset_len=len(self.val_ds),
                candidate_budget=candidate_budget,
                seed=self.val_set_seed,
                filter_viz=filter_viz,
                filter_loss=filter_loss,
            )
            if not compatible:
                raise RuntimeError(f"[VAL SET] Rank {rank} cache incompatible after wait: {reason}")
            indices = load_validation_indices(json_path)
            print(f"[VAL SET] Rank {rank} loaded {len(indices)} indices from {json_path}")
            return indices

        compatible, reason = cache_status()
        if compatible:
            indices = load_validation_indices(json_path)
            print(
                f"[VAL SET] Loaded {len(indices)} indices from {json_path} "
                f"(rebuild_val_set=False)"
            )
        else:
            if json_path.exists():
                print(f"[VAL SET] Cache incompatible, reconstruction forcée : {reason}")
            indices = build_validation_set(
                val_ds=self.val_ds,
                output_dir=out_dir,
                n_viz=self.n_viz,
                n_loss=self.n_loss,
                filter_viz=filter_viz,
                filter_loss=filter_loss,
                max_scan=self.val_set_max_scan,
                candidate_budget=candidate_budget,
                num_workers=self.val_set_num_workers,
                seed=self.val_set_seed,
            )

        if dist_ready:
            import torch.distributed as dist
            dist.barrier()
        return indices

    @staticmethod
    def _distributed_context():
        """Return (rank, world_size, dist_ready) without forcing dist init."""
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        try:
            import torch.distributed as dist
            dist_ready = dist.is_available() and dist.is_initialized()
            if dist_ready:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            return rank, world_size, dist_ready
        except Exception:
            return rank, world_size, False

    @staticmethod
    def _wait_for_val_cache(json_path, cache_status, timeout_s=1800, poll_s=5):
        """Poll fallback for rank processes when torch.distributed is not ready."""
        start = time.time()
        while time.time() - start < timeout_s:
            compatible, _ = cache_status()
            if compatible:
                return
            time.sleep(poll_s)
        raise TimeoutError(
            f"[VAL SET] Timed out waiting for compatible validation cache at {json_path}"
        )
    
    def train_dataloader(self):
        # === DEBUG DDP (décommenter si blocage sur Gefion) ===
        # import torch.distributed as dist
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # print(f"\n[DEBUG RANK {rank}] train_dataloader() called - dataset len={len(self.train_ds)}", flush=True)
        # === FIN DEBUG DDP ===

        return torch.utils.data.DataLoader(
            self.train_ds,
            # Ne PAS mettre shuffle=True pour DDP !
            # En DDP, PyTorch Lightning ajoute automatiquement un DistributedSampler(shuffle=True)
            # Mettre shuffle=True explicitement cause un conflit → deadlock
            worker_init_fn=_worker_init_fn,
            **self.dl_kw
        )

    def val_dataloader(self):
        class FixedIndicesSampler(Sampler):
            def __init__(self, indices):
                self.indices = indices
            def __iter__(self):
                return iter(self.indices)
            def __len__(self):
                return len(self.indices)
        sampler = FixedIndicesSampler(self.validation_indices)
        dl_kw_no_shuffle = {k: v for k, v in self.dl_kw.items() if k != 'shuffle'}
        return torch.utils.data.DataLoader(
            self.val_ds, 
            sampler=sampler, 
            worker_init_fn=_worker_init_fn,
            **dl_kw_no_shuffle
        )
    
    def test_dataloader(self):
        return {
            f"patch_x{res}": torch.utils.data.DataLoader(
                ds, 
                shuffle=False, 
                worker_init_fn=_worker_init_fn,
                **self.dl_kw
            ) 
            for res, ds in self.test_ds.datasets.items()
        }

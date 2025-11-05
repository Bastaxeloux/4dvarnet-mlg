from contrib.SST.data import XrDataset, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES, concatenate, fast_coarsen_xr, fast_pool
from contrib.SST.load_data import organize_by_resolution
import torch
import torch.nn.functional as F
import pandas as pd
import xarray as xr
import os
import numpy as np


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
        super().__init__(*args, **kwargs)
        self.multires = multires
        self.precomputed = precomputed
        self.enlarged_dims = {}
        for factor in self.multires:
            self.enlarged_dims[factor] = {'lat': 256 * factor, 'lon': 256 * factor}

    def extract_enlarged_patch_from_datasets(self, sl, factor):
        """
        Extrait un patch élargi à partir du dataset SST unifié
        
        Args:
            sl: Slices dict avec keys 'time', 'lat', 'lon'
            factor: Facteur de résolution (1, 3, ou 10)
            
        Returns:
            sample: dict avec toutes les variables SST normalisées
        """
        lat_center = (sl["lat"].start + sl["lat"].stop) // 2  # we compute the center
        lon_center = (sl["lon"].start + sl["lon"].stop) // 2
        enlarged_lat = self.enlarged_dims[factor]['lat'] # then we get enlarged dims
        enlarged_lon = self.enlarged_dims[factor]['lon']
        

        lat_start = max(0, lat_center - enlarged_lat // 2)
        lat_end = min(lat_start + enlarged_lat, self.da_dims["lat"] - 1)
        lon_start = max(0, lon_center - enlarged_lon // 2)
        lon_end = min(lon_start + enlarged_lon, self.da_dims["lon"] - 1)
        
        if hasattr(self.mask, 'isel'):
            mask_slice = self.mask.isel(lon=slice(lon_start, lon_end), lat=slice(lat_start, lat_end))
        else: 
            mask_slice = xr.DataArray(self.mask[lat_start:lat_end, lon_start:lon_end])
        
        item_mask = fast_pool(mask_slice, factor, factor, mode="binary")
        
        if self.load_data: # si tout est en mémoire
            sst_ds = self.full_sst.isel(
                time=sl["time"],
                lon=slice(lon_start, lon_end),
                lat=slice(lat_start, lat_end))
        else: # sinon on the fly
            time_indices = np.arange(sl["time"].start, sl["time"].stop)
            lon_coord = self.lon_1d if hasattr(self, 'lon_1d') else self.lon
            lat_coord = self.lat_1d if hasattr(self, 'lat_1d') else self.lat
            slices = {
                "lon": slice(lon_coord[lon_start], lon_coord[lon_end]),
                "lat": slice(lat_coord[lat_start], lat_coord[lat_end])
            }
            all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]
            if self.precomputed and self.is_multiresolution:
                sst_daily_paths_for_res = self.sst_daily_paths_by_resolution.get(factor, self.sst_daily_paths)
                resize_factor = 1
            else:
                sst_daily_paths_for_res = self.sst_daily_paths
                resize_factor = factor
            
            # Convert time_indices to list for proper indexing (works with both list and dict/array paths)
            time_indices_list = list(time_indices) if isinstance(time_indices, np.ndarray) else time_indices
            paths_to_load = [sst_daily_paths_for_res[i] for i in time_indices_list]
            
            sst_ds = concatenate(
                paths_to_load,
                var_list=all_sst_vars + COVARIATES,
                slices=slices,
                type_coords="coords",
                resize=resize_factor,
                domain_limits=self.domain_limits
            )

        expected_shape = (self.patch_dims['time'], 256, 256)
        first_var = list(sst_ds.data_vars)[0]
        actual_shape = sst_ds[first_var].shape
        
        # Handle shape mismatch: crop or pad as needed
        if actual_shape != expected_shape:
            # First, handle negative padding (crop if needed)
            pad_t = expected_shape[0] - actual_shape[0]
            pad_lat = expected_shape[1] - actual_shape[1]
            pad_lon = expected_shape[2] - actual_shape[2]
            
            # Crop if arrays are too large
            if pad_lat < 0 or pad_lon < 0:
                lat_start = max(0, (-pad_lat) // 2) if pad_lat < 0 else 0
                lon_start = max(0, (-pad_lon) // 2) if pad_lon < 0 else 0
                lat_end = lat_start + min(256, actual_shape[1])
                lon_end = lon_start + min(256, actual_shape[2])
                
                sst_ds = sst_ds.isel(lat=slice(lat_start, lat_end), lon=slice(lon_start, lon_end))
                if item_mask.ndim == 3:
                    item_mask = item_mask[:, lat_start:lat_end, lon_start:lon_end]
                else:
                    item_mask = item_mask[lat_start:lat_end, lon_start:lon_end]
                
                # Recalculate shape and padding needed
                actual_shape = sst_ds[first_var].shape
                pad_lat = expected_shape[1] - actual_shape[1]
                pad_lon = expected_shape[2] - actual_shape[2]
            
            # Then pad if needed
            if pad_lat > 0 or pad_lon > 0:
                sst_ds = pad_dataset(sst_ds, pad_lat=pad_lat, pad_lon=pad_lon)
            
            # Pad mask to match the EXACT dataset dimensions after padding
            actual_lat = len(sst_ds.lat)
            actual_lon = len(sst_ds.lon)
            mask_lat, mask_lon = item_mask.shape[-2:] if item_mask.ndim == 3 else item_mask.shape
            pad_lat_needed = actual_lat - mask_lat
            pad_lon_needed = actual_lon - mask_lon
            if pad_lat_needed > 0 or pad_lon_needed > 0:
                if item_mask.ndim == 3:
                    item_mask = np.pad(item_mask, ((0, 0), (0, pad_lat_needed), (0, pad_lon_needed)), mode='constant', constant_values=1)
                else:
                    item_mask = np.pad(item_mask, ((0, pad_lat_needed), (0, pad_lon_needed)), mode='constant', constant_values=1)
        sst_ds = sst_ds.assign({"mask": (("lat", "lon"), item_mask)})
        sst_ds['mask'] = sst_ds['mask'].fillna(1)
        item_mask = sst_ds.mask.data
        
        # sample is the dict to return
        sample = {}
        for sat_name in ['aasti', 'avhrr', 'pmw', 'slstr']:
            for var in VAR_GROUPS[sat_name]:  # ['av', 'std']
                var_key = f"{sat_name}_{var}"
                if var_key in sst_ds:
                    sample[var_key] = sst_ds[var_key].values
        
        # Add covariates
        for cov in COVARIATES:
            if cov in sst_ds:
                sample[cov] = sst_ds[cov].values
        
        # TrainingItem required fields
        sample["surfmask"] = np.expand_dims(item_mask, axis=0)

        lon_1d = sst_ds.lon.values
        lat_1d = sst_ds.lat.values
        lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        sample["lat"] = lat_2d
        sample["lon"] = lon_2d
        sample["time"] = np.expand_dims(sst_ds.time.values.astype('float64') / 1e9, axis=0)
        sample["inpaint_mask"] = np.zeros_like(sample["surfmask"])
        
        # Target variables
        for tgt_var in self.tgt_vars:
            if tgt_var in sample:
                sample[f"tgt_{tgt_var}"] = sample[tgt_var]
        if self.tgt_vars:
            sample["tgt_sst"] = sample.get(f"tgt_{self.tgt_vars[0]}", np.zeros_like(sample["surfmask"]))

        # apply post-processing if defined
        if self.postpro_fn is not None:
            sample = self.postpro_fn(sample)
        else:
            # If no post-processing, still need to convert dict to TrainingItem
            from contrib.SST.data import TrainingItem
            sample = TrainingItem(**sample)
        
        return sample

    def __getitem__(self, idx):
        """
        Retourne un dict avec patches multi-résolution IMBRIQUÉS:
        - patch_x10: 256x256 pixels à résolution x10 (contexte large)
        - patch_x3: 256x256 pixels à résolution x3 (contexte moyen, centré dans x10)
        - patch_x1: 256x256 pixels à résolution x1 (haute res, centré dans x3)
        
        Les patches sont extraits du même centre géographique avec des amplifications spatiales différentes,
        créant un véritable imbrication où x1 ⊂ x3 ⊂ x10.
        """
        # Get base coordinates for highest resolution (x1)
        if self.subsel_patch:
            idx = self.idx_patches_in_ocean[idx]
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                        self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(), np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        # Extract x1 patch (high resolution, 256x256 at res x1)
        hr_sample = super().__getitem__(idx)
        out = {}
        out[f"patch_x{self.resize}"] = hr_sample  # x1 (haute résolution)
        
        # extract lower resolution patches (x3, x10)
        # These are extracted from the SAME geographic center but with larger spatial extent,
        # then cropped back to 256x256, ensuring that x1 remains centered within them
        for factor in self.multires[:-1]:
            enlarged_patch = self.extract_enlarged_patch_from_datasets(sl, factor // self.resize)
            out[f"patch_x{factor}"] = enlarged_patch
        
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
            self.datasets[res] = XrDataset(load_data=True,
                                           *args, **kwargs)

    def get_dataloader_dict(self, batch_size=1, **loader_kwargs):
        from torch.utils.data import DataLoader
        return {res: DataLoader(ds, batch_size=batch_size, **loader_kwargs)
                for res, ds in self.datasets.items()}

class BaseDataModuleMultiRes(BaseDataModule):
    
    def __init__(self, 
                 multires=[1],
                 covariates_paths=None,
                 covariates=None,
                 mask_path=None,
                 domain_name='sst_multires',
                 domains=None,
                 precomputed=True,
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
        super().__init__(
            covariates_paths=covariates_paths,
            covariates=covariates,
            mask_path=mask_path,
            domain_name=domain_name,
            domains=domains,
            *args, **kwargs
        )
        
        self.multires = multires
        self.resize = self.multires[-1]
        self.precomputed = precomputed

    def save_batch_as_NetCDF_multires(self, batch_dict, ibatch, patch_dims_dict, save_dir="/dmidata/users/malegu/PREPROC/"):
        """
        Sauvegarde un batch multi-résolution en fichiers NetCDF séparés
        
        Args:
            batch_dict: dict de {f"patch_x{res}": TrainingItem}
            patch_dims_dict: dict de {res: {"time": ..., "lat": ..., "lon": ...}}
            save_dir: Répertoire de sauvegarde
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for key, batch in batch_dict.items():
            # Extraire facteur de résolution (ex: x10 -> 10)
            try:
                factor = int(key.split("x")[-1])
            except:
                print(f"Warning: can't parse resolution factor in {key}")
                continue
            
            patch_dims = patch_dims_dict[factor]
            data_vars = {}
            
            # Variables SST (4 satellites)
            for var in VAR_GROUPS["sst"]:
                var_name = f"sst_{var}"
                if hasattr(batch, var_name):
                    tensor = getattr(batch, var_name)
                    if torch.is_tensor(tensor) and tensor.ndim == 4:
                        data_vars[var_name] = (
                            ('sample', 'time', 'lat', 'lon'),
                            tensor.detach().cpu()
                        )
            
            # Covariables (sea_ice_fraction)
            for cov in COVARIATES:
                if hasattr(batch, cov):
                    tensor = getattr(batch, cov)
                    if torch.is_tensor(tensor) and tensor.ndim == 4:
                        data_vars[cov] = (
                            ('sample', 'time', 'lat', 'lon'),
                            tensor.detach().cpu()
                        )
            
            # Coordonnées et masque
            data_vars.update({
                'times': (
                    ('sample', 'time'),
                    torch.squeeze(batch.time, dim=1).detach().cpu().numpy().astype("datetime64[s]")
                ),
                'lats': (
                    ('sample', 'lat'),
                    torch.squeeze(batch.lat_coord, dim=1).detach().cpu()
                ),
                'lons': (
                    ('sample', 'lon'),
                    torch.squeeze(batch.lon_coord, dim=1).detach().cpu()
                ),
                'lat': (
                    ('sample', 'lat', 'lon'),
                    torch.squeeze(batch.lat, dim=1).detach().cpu()
                ),
                'lon': (
                    ('sample', 'lat', 'lon'),
                    torch.squeeze(batch.lon, dim=1).detach().cpu()
                ),
                'land_mask': (
                    ('sample', 'lat', 'lon'),
                    torch.squeeze(batch.land_mask, dim=1).detach().cpu()
                ),
            })
            
            # Variables cibles (tgt_xxx)
            for var in VAR_GROUPS["sst"]:
                new_key = f"tgt_{var}"
                if hasattr(batch, new_key):
                    tensor = getattr(batch, new_key)
                    if torch.is_tensor(tensor) and tensor.ndim == 4:
                        data_vars[new_key] = (
                            ('sample', 'time', 'lat', 'lon'),
                            tensor.detach().cpu()
                        )
            
            # Coordonnées du Dataset
            coords = {
                'sample': np.arange(list(data_vars.values())[0][1].shape[0]),
                'time': np.arange(patch_dims['time']),
                'lat': np.arange(patch_dims['lat']),
                'lon': np.arange(patch_dims['lon'])
            }
            
            # Construction et sauvegarde
            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            save_path = os.path.join(save_dir, f"preproc_batch_{ibatch}_x{factor}.nc")
            ds.to_netcdf(save_path)
            print(f"Saved {save_path}")

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
                resize=self.resize,
                subsel_patch_path=(
                    f"{self.subsel_path}/patch_in_ocean_{split}_{self.domain_name}_"
                    f"patch_{self.xrds_kw['patch_dims']['lat']}_{self.xrds_kw['strides']['lat']}_"
                    f"resize_x{self.resize}.txt"
                )
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



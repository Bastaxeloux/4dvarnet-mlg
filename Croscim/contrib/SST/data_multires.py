from contrib.SST.data import XrDataset, BaseDataModule
from contrib.SST.load_data import VAR_GROUPS, COVARIATES, concatenate, fast_coarsen_xr, fast_pool
import torch
import torch.nn.functional as F
import pandas as pd
import xarray as xr
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import numpy as np

def compute_cell_edges(arr):
    """
    Convert center-based coordinates (1D or 2D) to edge-based coordinates for pcolormesh.
    Returns array with shape+1 in each axis.
    """
    if arr.ndim == 1:
        # 1D version: compute edges between points
        edges = (arr[:-1] + arr[1:]) / 2
        first = arr[0] - (edges[0] - arr[0])
        last = arr[-1] + (arr[-1] - edges[-1])
        return np.concatenate([[first], edges, [last]])

    elif arr.ndim == 2:
        ny, nx = arr.shape
        edges = np.zeros((ny + 1, nx + 1), dtype=arr.dtype)

        # Interpolate internal edges
        edges[1:-1, 1:-1] = 0.25 * (
            arr[:-1, :-1] + arr[1:, :-1] + arr[:-1, 1:] + arr[1:, 1:]
        )

        # Extrapolate borders
        edges[0, 1:-1] = edges[1, 1:-1] - (edges[2, 1:-1] - edges[1, 1:-1])
        edges[-1, 1:-1] = edges[-2, 1:-1] + (edges[-2, 1:-1] - edges[-3, 1:-1])
        edges[1:-1, 0] = edges[1:-1, 1] - (edges[1:-1, 2] - edges[1:-1, 1])
        edges[1:-1, -1] = edges[1:-1, -2] + (edges[1:-1, -2] - edges[1:-1, -3])

        # Corners
        edges[0, 0] = edges[1, 0] - (edges[2, 0] - edges[1, 0])
        edges[0, -1] = edges[1, -1] - (edges[2, -1] - edges[1, -1])
        edges[-1, 0] = edges[-2, 0] + (edges[-2, 0] - edges[-3, 0])
        edges[-1, -1] = edges[-2, -1] + (edges[-2, -1] - edges[-3, -1])

        return edges

    else:
        raise ValueError("Unsupported dimension for computing edges.")


def plot_multires(batch_dict, var, t_idx=7, resolution_colors=None, title="Multi-resolution", save_dir="figs"):
    """
    Visualise patches multi-résolution sur carte géographique
    """
    if resolution_colors is None:
        resolution_colors = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Si var est une liste, on fait un subplot par variable
    if isinstance(var, list):
        plot_multires_comparison(batch_dict, var, t_idx, resolution_colors, title, save_dir)
        return
    
    # Déterminer le nombre de timesteps
    first_key = list(batch_dict.keys())[0]
    first_item = batch_dict[first_key]
    first_data = getattr(first_item, var, None)
    if first_data is None:
        print(f"Variable '{var}' non trouvée dans {first_key}")
        return
    
    n_timesteps = first_data.shape[0] if first_data.ndim == 3 else 1
    if 0 <= t_idx < n_timesteps:
        timesteps_to_plot = [t_idx]
    else:
        timesteps_to_plot = list(range(n_timesteps))
        print(f"t_idx={t_idx} hors limites [0, {n_timesteps-1}] → génération de {n_timesteps} plots")

    for t in timesteps_to_plot:
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())  
        ax.add_feature(cfeature.LAND, zorder=0, facecolor='lightgray')
        ax.coastlines()

        pcm = None

        for key in sorted(batch_dict.keys(), key=lambda k: int(k.split("_x")[-1])):
            try:
                factor = int(key.split("_x")[-1])
            except ValueError:
                print(f"Ignoring malformed key: {key}")
                continue

            item = batch_dict[key]
            color = resolution_colors.get(factor, f"C{factor % 10}")

            try:
                data = getattr(item, var)
                lon = getattr(item, "lon")
                lat = getattr(item, "lat")
            except AttributeError:
                print(f"Variable manquante dans {key}.")
                continue

            if data.ndim == 3:
                data = data[t, :, :]  # Prendre le timestep t

            if lon.ndim == 3:
                lon = lon[0, :, :]
                lat = lat[0, :, :]
            elif lon.ndim == 2:
                pass 
            elif lon.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            
            lon_edges = compute_cell_edges(lon)
            lat_edges = compute_cell_edges(lat)

            lon_edges = compute_cell_edges(lon)
            lat_edges = compute_cell_edges(lat)

            # Détection des sauts de longitude (wraps à +/-180°)
            delta_lon_h = np.abs(np.diff(lon_edges, axis=1))
            jump_mask_h = delta_lon_h[:, :-1] > 180

            delta_lon_v = np.abs(np.diff(lon_edges, axis=0))
            jump_mask_v = delta_lon_v[:-1, :] > 180

            # Masque final pour cellules contenant un saut
            mask_bad = np.zeros_like(data, dtype=bool)
            mask_bad[:, :-1] |= jump_mask_h
            mask_bad[:-1, :] |= jump_mask_v

            data_masked = np.ma.array(data, mask=mask_bad)

            pcm = ax.pcolormesh(
                lon_edges, lat_edges, data_masked,
                transform=ccrs.PlateCarree(),
                cmap='RdYlBu_r', shading='flat', alpha=0.7, zorder=1
            )

            # Tracer les contours du patch
            bottom = np.column_stack((lon_edges[0, :], lat_edges[0, :]))
            right = np.column_stack((lon_edges[:, -1], lat_edges[:, -1]))
            top = np.column_stack((lon_edges[-1, ::-1], lat_edges[-1, ::-1]))
            left = np.column_stack((lon_edges[::-1, 0], lat_edges[::-1, 0]))
            contour = np.vstack([bottom, right[1:], top[1:], left[1:], bottom[0:1]])

            ax.plot(contour[:, 0], contour[:, 1],
                    transform=ccrs.PlateCarree(), color=color,
                    linewidth=2, label=f"Patch x{factor}", zorder=2)

        plot_title = f"{title} - {var} (t={t})" if len(timesteps_to_plot) > 1 else f"{title} - {var}"
        ax.set_title(plot_title)
        if pcm is not None:
            plt.colorbar(pcm, ax=ax, orientation='vertical', shrink=0.5, label=var)
        ax.legend()

        if len(timesteps_to_plot) > 1:
            save_path = os.path.join(save_dir, f"multires_{var}_t{t:02d}.png")
        else:
            save_path = os.path.join(save_dir, f"multires_{var}.png")
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


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

    def __init__(self, multires=[10, 3, 1], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multires = multires
        self.enlarged_dims = {}
        for factor in self.multires:
            self.enlarged_dims[factor] = {'lat': 256 * factor, 'lon': 256 * factor}

    def coarsen_patch(self, patch, target_shape):
        """
        Coarsen a patch by adaptive average pooling to target shape.
        Input: (T, Y, X)
        """
        # patch = torch.as_tensor(patch).float().unsqueeze(0)  # Add batch dim
        # coarsened = F.adaptive_avg_pool2d(patch, target_shape)
        # return coarsened.squeeze(0).numpy()
        raise NotImplementedError("coarsen_patch is no longer used.")

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
            sst_ds = concatenate(
                self.sst_daily_paths[time_indices],
                var_list=all_sst_vars + COVARIATES,
                slices=slices,
                type_coords="coords",
                resize=factor,
                domain_limits=self.domain_limits
            )

        expected_shape = (self.patch_dims['time'], 256, 256)
        first_var = list(sst_ds.data_vars)[0]
        actual_shape = sst_ds[first_var].shape
        
        if actual_shape != expected_shape:
            pad_t = expected_shape[0] - actual_shape[0]
            pad_lat = expected_shape[1] - actual_shape[1]
            pad_lon = expected_shape[2] - actual_shape[2]
            sst_ds = sst_ds.assign({"mask": (("lat", "lon"), item_mask)})
            sst_ds = pad_dataset(sst_ds, pad_lat=pad_lat, pad_lon=pad_lon)
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
        sample["lat"] = np.expand_dims(sst_ds.lat.values, axis=0)
        sample["lon"] = np.expand_dims(sst_ds.lon.values, axis=0)
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
        
        return sample

    def __getitem__(self, idx):
        """
        Retourne un dict avec patches multi-résolution
        {
            "patch_x1": TrainingItem (256x256),
            "patch_x3": TrainingItem (256x256 from 768x768 context),
            "patch_x10": TrainingItem (256x256 from 2560x2560 context)
        }
        """
        hr_sample = super().__getitem__(idx)
        if self.subsel_patch:
            idx = self.idx_patches_in_ocean[idx]
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx_dim,
                        self.strides.get(dim, 1) * idx_dim + self.patch_dims[dim])
            for dim, idx_dim in zip(self.ds_size.keys(),np.unravel_index(idx, tuple(self.ds_size.values())))
        }

        out = {}
        out[f"patch_x{self.resize}"] = hr_sample  # x1 (haute résolution)
        
        # extract lower resolution patches
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
            
            Args:
                split: 'train', 'val', ou 'test'
                
            Returns:
                XrDatasetMultiResTrain ou XrDatasetMultiResTest
            """
            sst_paths, times = select_paths(self.sst_paths,self.domains[split]['time'],fmt="%Y%m%d")
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



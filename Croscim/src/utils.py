import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import functools as ft
import metpy.calc as mpcalc
import kornia
import pandas as pd
import xrft
import torch
import pyinterp
import pyinterp.fill
import pyinterp.backends.xarray
import src.data
import xarray as xr
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from PIL import Image

from src.ose.mod_inout import *
from src.ose.mod_interp import *
from src.ose.mod_stats import *
from src.ose.mod_spectral import *
from src.ose.mod_plot import *
from src.ose.utils import *

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def encompassing_patch(lat_1d, lon_1d, inner_lat_bounds, inner_lon_bounds, patch_size=256):
    """
    Trouve les indices d'un patch qui englobe géographiquement une région donnée.
    
    Cette fonction recherche un patch de taille patch_size x patch_size qui contient
    complètement la région définie par inner_lat_bounds et inner_lon_bounds.
    Elle gère les cas particuliers :
    - Pôles : si la région est proche d'un pôle, le patch sera décalé vers le bord
    - Dateline : si la région traverse la ligne de changement de date (180°/-180°), 
    la recherche est effectuée en tenant compte de la circularité des longitudes
    
    Args:
        lat_1d: array 1D des latitudes (en degrés) à la résolution cible
        lon_1d: array 1D des longitudes (en degrés) à la résolution cible
        inner_lat_bounds: tuple (lat_min, lat_max) de la région à englober (degrés)
        inner_lon_bounds: tuple (lon_min, lon_max) de la région à englober (degrés)
        patch_size: taille du patch à extraire (défaut=256)
    
    Returns:
        tuple (lat_slice, lon_slice) : slices pour extraire le patch englobant
    
    Raises:
        ValueError: Si aucun patch englobant ne peut être trouvé (ne devrait jamais arriver)
    
    Example:
        # Trouver patch x3 qui englobe un patch x1
        >>> lat_slice_x3, lon_slice_x3 = encompassing_patch(
        ...     lat_1d[::3], lon_1d[::3],  # Coordonnées x3
        ...     (patch_x1['lat_geo'].min(), patch_x1['lat_geo'].max()),
        ...     (patch_x1['lon_geo'].min(), patch_x1['lon_geo'].max()),
        ...     patch_size=256
        ... )
    """
    lat_min_inner, lat_max_inner = inner_lat_bounds
    lon_min_inner, lon_max_inner = inner_lon_bounds

    # DEBUG: Active pour premiers appels uniquement
    debug = False
    # if not hasattr(encompassing_patch, '_debug_count'):
    #     encompassing_patch._debug_count = 0
    # if encompassing_patch._debug_count < 2:
    #     encompassing_patch._debug_count += 1
    #     debug = True
    #     print(f"\n{'='*80}", flush=True)
    #     print(f"[encompassing_patch DEBUG #{encompassing_patch._debug_count}]", flush=True)
    #     print(f"  INPUT: Search for lat=[{lat_min_inner:.2f}, {lat_max_inner:.2f}], lon=[{lon_min_inner:.2f}, {lon_max_inner:.2f}]", flush=True)
    #     print(f"  GRID:  lat_1d len={len(lat_1d)}, range=[{lat_1d.min():.2f}, {lat_1d.max():.2f}]", flush=True)
    #     print(f"  GRID:  lon_1d len={len(lon_1d)}, range=[{lon_1d.min():.2f}, {lon_1d.max():.2f}]", flush=True)    # 1: Trouver tous les indices qui contiennent la région intérieure
    # Tolérance pour comparaison flottante
    tol = 1e-5
    
    # Recherche en latitude (simple, pas de circularité)
    # Important: on cherche des points de grille qui ENGLOBENT les bounds intérieures,
    # même si ces bounds dépassent légèrement la grille (cas des pôles avec grilles grossières).
    # On tolère que inner_lat dépasse lat_1d aux extrémités car géographiquement 
    # le patch couvre le même domaine, c'est juste la résolution qui diffère.
    
    # Trouver le point le plus au sud qui est <= lat_min_inner (ou le plus au sud de la grille)
    lat_candidates_min = np.where(lat_1d <= lat_min_inner + tol)[0]
    if len(lat_candidates_min) == 0:
        # lat_min_inner est plus au sud que tous les points de la grille
        # → prendre le point le plus au sud
        lat_candidates_min = np.array([0])
    
    # Trouver le point le plus au nord qui est >= lat_max_inner (ou le plus au nord de la grille)
    lat_candidates_max = np.where(lat_1d >= lat_max_inner - tol)[0]
    if len(lat_candidates_max) == 0:
        # lat_max_inner est plus au nord que tous les points de la grille
        # → prendre le point le plus au nord
        lat_candidates_max = np.array([len(lat_1d) - 1])
    
    # Recherche en longitude (gère la circularité de la dateline)
    # Même logique que pour les latitudes: tolérer que inner_lon dépasse légèrement lon_1d
    # Cas 1: Pas de wraparound (région normale)
    if lon_min_inner <= lon_max_inner:
        lon_candidates_min = np.where(lon_1d <= lon_min_inner + tol)[0]
        if len(lon_candidates_min) == 0:
            lon_candidates_min = np.array([0])
        
        lon_candidates_max = np.where(lon_1d >= lon_max_inner - tol)[0]
        if len(lon_candidates_max) == 0:
            lon_candidates_max = np.array([len(lon_1d) - 1])
    # Cas 2: Wraparound (région traverse la dateline, ex: 170° à -170°)
    else:
        # Chercher indices à l'est (>= lon_min) OU à l'ouest (<= lon_max)
        lon_candidates_min = np.where(lon_1d <= lon_min_inner + tol)[0]
        if len(lon_candidates_min) == 0:
            lon_candidates_min = np.array([0])
        
        lon_candidates_max = np.where(lon_1d >= lon_max_inner - tol)[0]
        if len(lon_candidates_max) == 0:
            lon_candidates_max = np.array([len(lon_1d) - 1])
    
    # 2: Calculer le centre idéal du patch englobant
    # Calculer le centre GÉOGRAPHIQUE de la région à englober (en degrés) puis trouver l'indice de grille le plus proche de ce centre
    lat_center_geo = (lat_min_inner + lat_max_inner) / 2.0
    lon_center_geo = (lon_min_inner + lon_max_inner) / 2.0
    
    # Trouver les indices les plus proches des centres géographiques
    lat_center_ideal = int(np.argmin(np.abs(lat_1d - lat_center_geo)))
    lon_center_ideal = int(np.argmin(np.abs(lon_1d - lon_center_geo)))
    
    # if debug:
    #     print(f"  STEP 1: Computed geographic center: lat={lat_center_geo:.2f}, lon={lon_center_geo:.2f}", flush=True)
    #     print(f"  STEP 2: Found nearest grid indices: lat_idx={lat_center_ideal}, lon_idx={lon_center_ideal}", flush=True)
    #     print(f"  STEP 3: Grid points at those indices: lat={lat_1d[lat_center_ideal]:.2f}, lon={lon_1d[lon_center_ideal]:.2f}", flush=True)

    if len(lat_1d) < patch_size:
        raise ValueError(f"Grid too small: lat_1d has {len(lat_1d)} points but need {patch_size}")
    if len(lon_1d) < patch_size:
        raise ValueError(f"Grid too small: lon_1d has {len(lon_1d)} points but need {patch_size}")
    
    # 3: Calculer les indices du patch en respectant les limites du domaine
    # GARANTIR que le patch fait exactement patch_size x patch_size
    lat_start = max(0, lat_center_ideal - patch_size // 2)
    lat_start = min(lat_start, len(lat_1d) - patch_size)  # Ne pas dépasser à droite
    lat_end = lat_start + patch_size
    
    lon_start = max(0, lon_center_ideal - patch_size // 2)
    lon_start = min(lon_start, len(lon_1d) - patch_size)  # Ne pas dépasser à droite
    lon_end = lon_start + patch_size
    
    # if debug:
    #     print(f"  STEP 4: Computed slice indices: lat[{lat_start}:{lat_end}], lon[{lon_start}:{lon_end}]", flush=True)
    #     print(f"  STEP 5: Final patch geographic bounds:", flush=True)
    #     print(f"          lat=[{lat_1d[lat_start]:.2f}, {lat_1d[lat_end-1]:.2f}]", flush=True)
    #     print(f"          lon=[{lon_1d[lon_start]:.2f}, {lon_1d[lon_end-1]:.2f}]", flush=True)
    #     print(f"  STEP 6: Span check: lat_span={(lat_1d[lat_end-1] - lat_1d[lat_start]):.2f}°, lon_span={(lon_1d[lon_end-1] - lon_1d[lon_start]):.2f}°", flush=True)
    #     print(f"{'='*80}\n", flush=True)
    
    # Vérification finale: le patch englobe-t-il bien la région ?
    patch_lat_min = lat_1d[lat_start]
    patch_lat_max = lat_1d[lat_end - 1]
    patch_lon_min = lon_1d[lon_start]
    patch_lon_max = lon_1d[lon_end - 1]
    
    # Vérifier que le patch englobe les bounds intérieures (avec tolérance raisonnable)
    # Tolérance: ~1 degré pour absorber les différences dues à la grille discrète
    # Si le patch est vraiment mal positionné, l'écart sera >> 1 degré
    safety_tol = 0.2  # degré
    
    # Vérification latitude
    lat_contains_min = (patch_lat_min <= lat_min_inner + safety_tol) or (lat_start == 0)
    lat_contains_max = (patch_lat_max >= lat_max_inner - safety_tol) or (lat_end == len(lat_1d))
    lat_ok = lat_contains_min and lat_contains_max
    
    # Vérification longitude (avec gestion dateline)
    if lon_min_inner <= lon_max_inner:
        lon_contains_min = (patch_lon_min <= lon_min_inner + safety_tol) or (lon_start == 0)
        lon_contains_max = (patch_lon_max >= lon_max_inner - safety_tol) or (lon_end == len(lon_1d))
        lon_ok = lon_contains_min and lon_contains_max
    else:
        # Cas dateline: vérifier que le patch contient SOIT les deux extrémités SOIT toute la plage
        lon_ok = (
            (patch_lon_min <= lon_min_inner + safety_tol and patch_lon_max >= 180.0 - safety_tol) or
            (patch_lon_min <= -180.0 + safety_tol and patch_lon_max >= lon_max_inner - safety_tol) or
            (patch_lon_max - patch_lon_min >= 350.0)  # Patch couvre presque tout le globe
        )
    
    if not (lat_ok and lon_ok):
        raise ValueError(
            f"Encompassing patch does not contain inner region!\n"
            f"  Inner bounds: lat=[{lat_min_inner:.2f}, {lat_max_inner:.2f}], "
            f"lon=[{lon_min_inner:.2f}, {lon_max_inner:.2f}]\n"
            f"  Patch bounds: lat=[{patch_lat_min:.2f}, {patch_lat_max:.2f}], "
            f"lon=[{patch_lon_min:.2f}, {patch_lon_max:.2f}]\n"
            f"  Slices: lat={lat_start}:{lat_end}, lon={lon_start}:{lon_end}\n"
            f"  (tolerance={safety_tol}° applied)"
        )
    
    return slice(lat_start, lat_end), slice(lon_start, lon_end)


def extract_encompassing_patch(
    dataset_obj, sl, factor, lat_bounds, lon_bounds,
    VAR_GROUPS, COVARIATES, patch_dims, tgt_vars
):
    """
    Extrait un patch SST multi-résolution qui ENGLOBE géographiquement une région donnée.
    
    Cette fonction combine :
    1. Recherche géographique via encompassing_patch() (garantit l'imbrication)
    2. Chargement des données SST depuis fichiers ou mémoire
    3. Assembly du dict de sortie avec toutes les variables
    
    Args:
        dataset_obj: Instance de XrDataset (pour accès aux attributs)
        sl: dict de slices {'time': slice(...), 'lat': slice(...), 'lon': slice(...)}
        factor: Facteur de résolution (3 ou 10)
        lat_bounds: tuple (lat_min, lat_max) de la région à englober (degrés)
        lon_bounds: tuple (lon_min, lon_max) de la région à englober (degrés)
        VAR_GROUPS: dict {satellite: [variables]} ex: {'aasti': ['sst'], ...}
        COVARIATES: list de covariables ex: ['sea_ice_fraction']
        patch_dims: dict {'time': nt, 'lat': 256, 'lon': 256}
        tgt_vars: list de variables cibles ex: ['aasti_sst']
        time_indices: Optional explicit time indices to use instead of sl["time"]
    
    Returns:
        dict: Sample avec clés 'aasti_sst', 'lat_geo', 'lon_geo', 'surfmask', etc.
    
    Raises:
        ValueError: Si encompassing_patch échoue (région hors du domaine)
    """
    import pandas as pd
    import xarray as xr
    from contrib.SST.load_data import concatenate, fast_pool
    
    # Étape 1: Déterminer coordonnées 1D pour cette résolution
    is_precomputed_mode = (
        dataset_obj.precomputed and 
        dataset_obj.is_multiresolution and 
        factor > 1
    )
    
    if is_precomputed_mode:
        # CRITICAL FIX: Use dataset's reference coordinates, not file-specific ones!
        # Files may have slightly different coordinates causing massive misalignment
        lat_1d = dataset_obj.lat_1d[::factor]
        lon_1d = dataset_obj.lon_1d[::factor]
        target_patch_size = 256  # Patch final = 256×256
    else:
        # Mode pooling: coordonnées x1 (coarsening après chargement)
        # IMPORTANT: chercher un patch plus grand qui sera poolé ensuite
        lat_1d = dataset_obj.lat_1d
        lon_1d = dataset_obj.lon_1d
        target_patch_size = 256 * factor  # Ex: 256*3=768 → poolé en 256
    
    # Étape 2: Trouver indices du patch englobant
    try:
        lat_slice, lon_slice = encompassing_patch(
            lat_1d, lon_1d,
            lat_bounds, lon_bounds,
            patch_size=target_patch_size
        )
    except ValueError as e:
        print(f"[ERROR] encompassing_patch failed for factor={factor}: {e}")
        print(f"  lat_bounds={lat_bounds}, lon_bounds={lon_bounds}")
        print(f"  lat_1d range=[{lat_1d.min():.2f}, {lat_1d.max():.2f}]")
        print(f"  lon_1d range=[{lon_1d.min():.2f}, {lon_1d.max():.2f}]")
        raise
    
    lat_start, lat_end = lat_slice.start, lat_slice.stop
    lon_start, lon_end = lon_slice.start, lon_slice.stop
    
    # Étape 3: Extraire et pooler le masque
    if hasattr(dataset_obj.mask, 'isel'):
        mask_slice = dataset_obj.mask.isel(
            lon=slice(lon_start, lon_end),
            lat=slice(lat_start, lat_end)
        )
    else:
        mask_slice = xr.DataArray(dataset_obj.mask[lat_start:lat_end, lon_start:lon_end])
    
    pooling_factor = 1 if is_precomputed_mode else factor
    item_mask = fast_pool(mask_slice, pooling_factor, pooling_factor, mode="binary")
    
    # Étape 4: Charger données SST
    if dataset_obj.load_data:
        # Charger depuis dataset en mémoire
        sst_ds = dataset_obj.full_sst.isel(
            time=sl["time"],
            lat=slice(lat_start, lat_end),
            lon=slice(lon_start, lon_end)
        )
    else:
        # Charger depuis fichiers journaliers
        time_indices = np.arange(sl["time"].start, sl["time"].stop)
        slices = {
            "lat": slice(lat_start, lat_end),
            "lon": slice(lon_start, lon_end)
        }
        all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]
        
        if is_precomputed_mode:
            sst_daily_paths_for_res = dataset_obj.sst_daily_paths_by_resolution.get(
                factor, dataset_obj.sst_daily_paths
            )
            resize_factor = 1
        else:
            sst_daily_paths_for_res = dataset_obj.sst_daily_paths
            resize_factor = factor
        
        time_indices_list = list(time_indices) if isinstance(time_indices, np.ndarray) else time_indices
        paths_to_load = [sst_daily_paths_for_res[i] for i in time_indices_list]
        
        sst_ds = concatenate(
            paths_to_load,
            var_list=all_sst_vars + COVARIATES,
            slices=slices,
            type_coords="index",
            resize=resize_factor,
            domain_limits=dataset_obj.domain_limits
        )
    
    # Étape 5: Vérification de la taille (devrait toujours être 256x256 grâce à encompassing_patch)
    expected_shape = (patch_dims['time'], 256, 256)
    first_var = list(sst_ds.data_vars)[0]
    actual_shape = sst_ds[first_var].shape
    
    if actual_shape[1:] != expected_shape[1:]:
        # NORMALEMENT ceci ne devrait JAMAIS arriver car encompassing_patch garantit 256x256
        raise ValueError(
            f"BUG: encompassing_patch devrait garantir 256x256 mais on a {actual_shape[1:]}!\n"
            f"  factor={factor}, lat_bounds={lat_bounds}, lon_bounds={lon_bounds}\n"
            f"  Cela indique un problème dans encompassing_patch() ou les coordonnées 1D"
        )
    sample = {}
    for sat_name in ['aasti', 'avhrr', 'pmw', 'slstr']:
        for var in VAR_GROUPS[sat_name]:
            var_key = f"{sat_name}_{var}"
            if var_key in sst_ds:
                sample[var_key] = sst_ds[var_key].values
    
    # Covariables
    for cov in COVARIATES:
        if cov in sst_ds:
            sample[cov] = sst_ds[cov].values
    

    sample["surfmask"] = np.expand_dims(item_mask, axis=0)
    
    # CORRECTIF: Utiliser les coordonnées de référence avec les slices calculés par encompassing_patch
    # au lieu des coordonnées du dataset slicé (qui sont incorrectes)
    lon_1d_correct = lon_1d[lon_start:lon_end]
    lat_1d_correct = lat_1d[lat_start:lat_end]
    lon_2d, lat_2d = np.meshgrid(lon_1d_correct, lat_1d_correct)
    sample["lat"] = (lat_2d / 90.0).astype(np.float32)
    sample["lon"] = (lon_2d / 180.0).astype(np.float32)
    sample["lat_geo"] = lat_2d.astype(np.float32)
    sample["lon_geo"] = lon_2d.astype(np.float32)
    
    center_time_idx = len(sst_ds.time) // 2
    center_time = sst_ds.time.values[center_time_idx]
    day_of_year = pd.Timestamp(center_time).dayofyear
    time_value = day_of_year / 366.0
    time_channel = np.full((lat_2d.shape[0], lat_2d.shape[1]), time_value, dtype=np.float32)
    sample["time"] = time_channel
    
    sample["inpaint_mask"] = np.zeros_like(sample["surfmask"])

    for tgt_var in tgt_vars:
        if tgt_var in sample:
            sample[f"tgt_{tgt_var}"] = sample[tgt_var]
    if tgt_vars:
        sample["tgt_sst"] = sample.get(f"tgt_{tgt_vars[0]}", np.zeros_like(sample["surfmask"]))
    return sample

def pipe(inp, fns):
    for f in fns:
        inp = f(inp)
    return inp

def kwgetattr(obj, name):
    return getattr(obj, name)

def callmap(inp, fns):
    return [fn(inp) for fn in fns]

def half_lr_adam(lit_mod, lr):
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )

def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }


def remove_nan(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")

    da.transpose("lon", "lat", "time")[:, :] = pyinterp.fill.gauss_seidel(
        pyinterp.backends.xarray.Grid3D(da)
    )[1]
    return da


def get_constant_crop(patch_dims, crop, dim_order=["time", "lat", "lon"]):
    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight


def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    pw = get_constant_crop(patch_dims, crop)

    t_msk = kornia.filters.get_hanning_kernel1d(patch_dims["time"])

    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()

def get_rectangular_hanning_mask(patch_dims, crop, dim_order=["time", "lat", "lon"]):
    
    print(patch_dims)
    M, N = (patch_dims["yc"], patch_dims["xc"])
    hann_1d_M = np.hanning(M)  # 1D Hanning window for rows
    hann_1d_N = np.hanning(N)  # 1D Hanning window for columns
    patch_weight = np.repeat(np.outer(hann_1d_M, hann_1d_N)[None,:], patch_dims["time"], axis=0)
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight

def get_triang_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_linear_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

def get_frcst_time_wei(patch_dims, offset=0, frcst_lead=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + np.where(t<(patch_dims["time"]-(frcst_lead+1)),t,(patch_dims["time"]-(frcst_lead+1))) \
             - np.min((patch_dims["time"]-(frcst_lead+1),patch_dims["time"])))/np.min((patch_dims["time"]-(frcst_lead+1),patch_dims["time"])) ) * pw
        ),
        patch_dims.values(),
    )

def get_last_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            pw * (t == (patch_dims["time"]-1) )
        ),
        patch_dims.values(),
    )

def get_center_time_wei(patch_dims, offset=0, **crop_kw):
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            pw * (t == np.floor(patch_dims["time"]/2)) 
        ),
        patch_dims.values(),
    )

def get_uniform_time_wei(patch_dims, offset=0, constant_crop=True, **crop_kw):
    if constant_crop:
        pw = get_constant_crop(patch_dims, **crop_kw)
    else:
        pw = get_rectangular_hanning_mask(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            pw * 1
        ),
        patch_dims.values(),
    )

def duplicate_wei(weights):
    return np.concatenate((weights,weights),axis=0)

def load_enatl(*args, obs_from_tgt=True, **kwargs):
    # ds = xr.open_dataset('../sla-data-registry/qdata/enatl_wo_tide.nc')
    # print(ds)
    # return ds.rename(nadir_obs='input', ssh='tgt').to_array().transpose('variable', 'time', 'lat', 'lon').sortby('variable')
    ssh = xr.open_zarr('../sla-data-registry/enatl_preproc/truth_SLA_SSH_NATL60.zarr/').ssh
    nadirs = xr.open_zarr('../sla-data-registry/enatl_preproc/SLA_SSH_5nadirs.zarr/').ssh
    ssh = ssh.interp(
        lon=np.arange(ssh.lon.min(), ssh.lon.max(), 1/20),
        lat=np.arange(ssh.lat.min(), ssh.lat.max(), 1/20)
    )
    nadirs = nadirs.interp(time=ssh.time, method='nearest')\
        .interp(lat=ssh.lat, lon=ssh.lon, method='zero')
    ds =  xr.Dataset(dict(input=nadirs, tgt=(ssh.dims, ssh.values)), nadirs.coords)
    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.transpose(*ds.input.dims).where(np.isfinite(ds.input), np.nan))
    return ds.transpose('time', 'lat', 'lon').to_array().load().sortby('variable')


def load_altimetry_data(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        # .assign(ssh=lambda ds: ds.ssh.coarsen(lon=2, lat=2).mean().interp(lat=ds.lat, lon=ds.lon))
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            tgt=lambda ds: remove_nan(ds.ssh),
        )    
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_altimetry_data_fast(path, obs_from_tgt=False, var_obs="nadir_obs", var_gt='ssh'):
    
    ds = xr.merge([
             xr.open_dataset(path).rename_vars({var_obs:"input"}),
             xr.open_dataset(path).rename_vars({var_gt:"tgt"})]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')
    
    #ds = ds.update({"tgt":(("time","lat","lon"),remove_nan(ds.tgt))})

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
    
    return ds

def load_altimetry_data_ose(path, obs_from_tgt=False, var_obs="ssh", var_gt='ssh', fast=True):

    if not fast:
        ds =  (
            xr.open_dataset(path).load().assign(
               input=lambda ds: ds.ssh,
               tgt=lambda ds: ds.ssh,
            )    
        )

        return (
            ds[[*src.data.TrainingItem._fields]]
            .transpose("time", "lat", "lon")
            .to_array())
 
    else:
        ds = xr.merge([
             xr.open_dataset(path).rename_vars({var_obs:"input"}),
             xr.open_dataset(path).rename_vars({var_gt:"tgt"})]
           ,compat='override')[[*src.data.TrainingItem._fields]].transpose('time', 'lat', 'lon')

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))

    return ds


def load_altimetry_data_woi(path, obs_from_tgt=False):
    ds =  (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds.nadir_obs,
            oi=lambda ds: ds.ssh_mod,
            tgt=lambda ds: remove_nan(ds.ssh),
        )
    )

    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
        
    return (
        ds[[*src.data_notebook_woi.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def load_altimetry_data_fast_woi(path, obs_from_tgt=False, var_obs="nadir_mod", var_oi = "oi_ssh_mod", var_gt='ssh'):
    
    ds = xr.merge([
             xr.open_dataset(path).rename_vars({var_obs:"input"}),
             xr.open_dataset(path).rename_vars({var_gt:"tgt"}),
             xr.open_dataset(path).rename_vars({var_oi:"oi"})]
           ,compat='override')[[*src.data_notebook_woi_fast.TrainingItem._fields]].transpose('time', 'lat', 'lon')
    
    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.where(np.isfinite(ds.input), np.nan))
        
    ds = ds.update({'tgt':(('time','lat','lon'),np.where(ds.tgt.data==0.,np.nan,ds.tgt.data))})
    ds = ds.update({'oi':(('time','lat','lon'),np.where(np.abs(ds.oi.data)>1e3,np.nan,ds.oi.data))})

    ds = ds.update({'tgt':(('time','lat','lon'),ds.tgt.data-ds.oi.data)})
    ds = ds.update({'input':(('time','lat','lon'),ds.input.data-ds.oi.data)})

    return ds

def load_natl_data(
        path_obs="/DATASET/eNATL/eNATL60_BLB002_SSH_nadirs/eNATL60-BLB002-7nadirs-2009-2010-1_20.nc",
        path_gt="/DATASET/eNATL/eNATL60_BLB002_SSH_nadirs/eNATL60-BLB002-ssh-2009-2010-1_20.nc",
        obs_var='input',
        gt_var='ssh',
        **kwargs
    ):
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).transpose('time', 'lat', 'lon')

def load_dc_data(**kwargs):
    path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
    path_obs ="NATL60/NATL/data_new/dataset_nadir_0d.nc"


def load_full_natl_data(
        path_obs="../sla-data-registry/CalData/cal_data_new_errs.nc",
        path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
        obs_var='five_nadirs',
        gt_var='ssh',
        **kwargs
    ):
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        # .isel(time=slice(0, -1))
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).to_array().sortby('variable')


def rmse_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    try:
        return rmse_based_scores(ds[ref_variable], ds[study_variable])[2:]
    except:
        return [np.nan, np.nan]

def psd_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    try:
        return psd_based_scores(ds[ref_variable], ds[study_variable])[1:]
    except:
        return [np.nan, np.nan]

def rmse_based_scores(da_rec, da_ref):
    rmse_t = (
        1.0
        - (((da_rec - da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
        / (((da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
    )
    rmse_xy = (((da_rec - da_ref) ** 2).mean(dim=("time"))) ** 0.5
    rmse_t = rmse_t.rename("rmse_t")
    rmse_xy = rmse_xy.rename("rmse_xy")
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = (
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (((da_ref) ** 2).mean()) ** 0.5
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5).item(),
        np.round(reconstruction_error_stability_metric, 5).item(),
    )


def psd_based_scores(da_rec, da_ref):
    err = da_rec - da_ref
    err["time"] = (err.time - err.time[0]) / np.timedelta64(1, "D")
    signal = da_ref
    signal["time"] = (signal.time - signal.time[0]) / np.timedelta64(1, "D")
    psd_err = xrft.power_spectrum(
        err, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    mean_psd_signal = psd_signal.mean(dim="lat").where(
        (psd_signal.freq_lon > 0.0) & (psd_signal.freq_time > 0), drop=True
    )
    mean_psd_err = psd_err.mean(dim="lat").where(
        (psd_err.freq_lon > 0.0) & (psd_err.freq_time > 0), drop=True
    )
    psd_based_score = 1.0 - mean_psd_err / mean_psd_signal
    level = [0.5]
    cs = plt.contour(
        1.0 / psd_based_score.freq_lon.values,
        1.0 / psd_based_score.freq_time.values,
        psd_based_score,
        level,
    )
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    psd_da = 1.0 - mean_psd_err / mean_psd_signal
    psd_da.name = "psd_score"
    return (
        psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, 3).item(),
        np.round(shortest_temporal_wavelength_resolved, 3).item(),
    )

def overlay_img(bgfile, fgfile):
    bg = Image.open(bgfile)  
    w, h = bg.size
    y, x = np.mgrid[0:h, 0:w]
    fg = Image.open(fgfile)
    fg = fg.resize((w,h)).convert("RGBA")
    datas = fg.getdata() 
    newData = [] 
    for item in datas: 
        if item[0] == 255 and item[1] == 255 and item[2] == 255:  # finding white colour by its RGB value 
            # storing a transparent value when we find a black colour 
            newData.append((255, 255, 255, 0)) 
        else: 
            newData.append(item)  # other colours remain unchanged         
    fg.putdata(newData) 
    img = Image.alpha_composite(bg,fg)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img)
    ax.axis('off')
    plt.savefig(bgfile)

def plot_simu_daw(gt,simu1,simu2,simu3,simu4,simu5,lon,lat,resfile,figsize):
    # crs = ccrs.Orthographic(-30,45)
    # vmax = 1.5
    # vmin = -1.5
    # cm = plt.cm.viridis
    # norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # extent = [np.min(lon),np.max(lon),np.min(lat),np.max(lat)]
    # fig = plt.figure(figsize=(10,10))
    # gs = gridspec.GridSpec(6, 5)
    
    # gs1 = GridSpec(1, 5, top=0.4)
    # gs2 = GridSpec(5, 5, botton=0.5)
    # gs2.update(wspace=0.05,hspace=0.05)
    
    # title = ['','','','','','']
    # for k in range(5):
    #     ax1 = fig.add_subplot(gs1[0, k], projection=crs)
    #     ax2 = fig.add_subplot(gs2[0, k], projection=crs)
    #     ax3 = fig.add_subplot(gs2[1, k], projection=crs)
    #     ax4 = fig.add_subplot(gs2[2, k], projection=crs)
    #     ax5 = fig.add_subplot(gs2[3, k], projection=crs)
    #     ax6 = fig.add_subplot(gs2[4, k], projection=crs)
    #     plot(ax1, lon, lat, gt[:,:,k].values, title[0], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    #     plot(ax2, lon, lat, simu1[:,:,k].values, title[1], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    #     plot(ax3, lon, lat, simu2[:,:,k].values, title[2], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    #     plot(ax4, lon, lat, simu3[:,:,k].values, title[3], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    #     plot(ax5, lon, lat, simu4[:,:,k].values, title[4], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    #     plot(ax6, lon, lat, simu5[:,:,k].values, title[5], extent=extent, cmap=cm, norm=norm, colorbar=False,fmt=False)
    # # Colorbar
    # cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.01])
    # sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # sm._A = []
    # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=3.0)
    # my_dpi = 96
    # plt.savefig(resfile,bbox_inches="tight",figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)    # save the figure
    # fig = plt.gcf()
    # plt.close()             # close the figure
    # return fig
    raise NotImplementedError("Function plot_simu_daw is no longer used.")

def compute_ose_metrics(test_data, alontrack_independent_dataset='/homes/m19beauc/4dvarnet-starter/src/ose/dt_gulfstream_c2_phy_l3_20161201-20180131_285-315_23-53.nc', time_min='2017-01-01', time_max='2017-12-31'):
 
    lon_min = 295.
    lon_max = 305.
    lat_min = 33.
    lat_max = 43.
    is_circle = False

    # Outputs
    bin_lat_step = 1.
    bin_lon_step = 1.
    bin_time_step = '1D'

    # Spectral parameter
    # C2 parameter
    delta_t = 0.9434  # s
    velocity = 6.77   # km/s
    delta_x = velocity * delta_t
    lenght_scale = 1000 # km
   
    file= '/Odyssey/private/m19beauc/IMT/4DVarNet_outputs/SPDE/file_4dvarnet_for_metrics.nc'
    test_data = test_data.update({'ssh':(('time','lat','lon'),test_data.out.data)})
    test_data.to_netcdf(file)

    # independent along-track
    # Read along-track
    ds_alongtrack = read_l3_dataset(alontrack_independent_dataset, 
                                           lon_min=lon_min, 
                                           lon_max=lon_max, 
                                           lat_min=lat_min, 
                                           lat_max=lat_max, 
                                           time_min=time_min, 
                                           time_max=time_max)

    res = interp_on_alongtrack(file,
                              ds_alongtrack,
                              lon_min=lon_min,
                              lon_max=lon_max,
                              lat_min=lat_min,
                              lat_max=lat_max,
                              time_min=time_min,
                              time_max=time_max,
                              is_circle=is_circle)
    time_alongtrack, lat_alongtrack, lon_alongtrack, ssh_alongtrack, ssh_interp = res
    
 
    # Compute spatial and temporal statistics
    leaderboard_nrmse, leaderboard_nrmse_std = compute_stats(time_alongtrack, 
                                                         lat_alongtrack, 
                                                         lon_alongtrack, 
                                                         ssh_alongtrack, 
                                                         ssh_interp, 
                                                         bin_lon_step,
                                                         bin_lat_step, 
                                                         bin_time_step,
                                                         output_filename='/Odyssey/private/m19beauc/spa_stat.nc',
                                                         output_filename_timeseries='/Odyssey/private/m19beauc/TS.nc')
    
    # Compute spectral scores
    compute_spectral_scores(time_alongtrack, 
                        lat_alongtrack, 
                        lon_alongtrack, 
                        ssh_alongtrack, 
                        ssh_interp, 
                        lenght_scale,
                        delta_x,
                        delta_t,
                        '/Odyssey/private/m19beauc/spectrum.nc')    
    
    leaderboard_psds_score = -999
    leaderboard_psds_score = plot_psd_score('/Odyssey/private/m19beauc/spectrum.nc')  

    os.remove('/Odyssey/private/m19beauc/spa_stat.nc')
    os.remove('/Odyssey/private/m19beauc/spectrum.nc')
    os.remove('/Odyssey/private/m19beauc/TS.nc')
    os.remove(file)  
    
    return leaderboard_nrmse, leaderboard_nrmse_std, int(leaderboard_psds_score)

def diagnostics(lit_mod, test_domain):
    test_data = lit_mod.test_data.sel(test_domain)
    return diagnostics_from_ds(test_data, test_domain)


def diagnostics_from_ds(test_data, test_domain):
    test_data = test_data.sel(test_domain)
    metrics = {
        "RMSE (m)": test_data.pipe(lambda ds: (ds.out - ds.tgt))
        .pipe(lambda da: da**2)
        .mean()
        .pipe(np.sqrt)
        .item(),
        **dict(
            zip(
                ["λx", "λt"],
                test_data.pipe(lambda ds: psd_based_scores(ds.out, ds.tgt)[1:]),
            )
        ),
        **dict(
            zip(
                ["μ", "σ"],
                test_data.pipe(lambda ds: rmse_based_scores(ds.out, ds.tgt)[2:]),
            )
        ),
    }
    return pd.Series(metrics, name="osse_metrics")


def test_osse(trainer, lit_mod, osse_dm, osse_test_domain, ckpt, diag_data_dir=None):
    lit_mod.norm_stats = osse_dm.norm_stats()
    trainer.test(lit_mod, datamodule=osse_dm, ckpt_path=ckpt)
    osse_tdat = lit_mod.test_data[['out', 'ssh']]
    osse_metrics = diagnostics_from_ds(
        osse_tdat, test_domain=osse_test_domain
    )

    print(osse_metrics.to_markdown())

    if diag_data_dir is not None:
        osse_metrics.to_csv(diag_data_dir / "osse_metrics.csv")
        if (diag_data_dir / "osse_test_data.nc").exists():
            xr.open_dataset(diag_data_dir / "osse_test_data.nc").close()
        osse_tdat.to_netcdf(diag_data_dir / "osse_test_data.nc")

    return osse_metrics



def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path):
    metrics = []
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        rmse = (
            lit_mod.test_data.pipe(lambda ds: (ds.out - ds.ssh))
            .pipe(lambda da: da**2)
            .mean()
            .pipe(np.sqrt)
            .item()
        )
        lx, lt = psd_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[1:]
        mu, sig = rmse_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[2:]

        metrics.append(dict(ckpt=ckpt, rmse=rmse, lx=lx, lt=lt, mu=mu, sig=sig))

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(out=f"out_{i}")
        else:
            test_data = test_data.assign(**{f"out_{i}": lit_mod.test_data.out})
        test_data[f"out_{i}"] = test_data[f"out_{i}"].assign_attrs(
            ckpt=str(ckpt)
        )

    metric_df = pd.DataFrame(metrics)
    print(metric_df.to_markdown())
    print(metric_df.describe().to_markdown())
    metric_df.to_csv(save_path + "/metrics.csv")
    test_data.to_netcdf(save_path + "ens_out.nc")


def add_geo_attrs(da):
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    return da


def vort(da):
    return mpcalc.vorticity(
        *mpcalc.geostrophic_wind(
            da.pipe(add_geo_attrs).assign_attrs(units="m").metpy.quantify()
        )
    ).metpy.dequantify()


def geo_energy(da):
    return np.hypot(*mpcalc.geostrophic_wind(da.pipe(add_geo_attrs))).metpy.dequantify()


def best_ckpt(xp_dir):
    _, xpn = load_cfg(xp_dir)
    if xpn is None:
        return None
    print(Path(xp_dir) / xpn / 'checkpoints')
    ckpt_last = max(
        (Path(xp_dir) / xpn / 'checkpoints').glob("*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    cbs = torch.load(ckpt_last)["callbacks"]
    ckpt_cb = cbs[next(k for k in cbs.keys() if "ModelCheckpoint" in k)]
    return ckpt_cb["best_model_path"]


def load_cfg(xp_dir):
    hydra_cfg = OmegaConf.load(Path(xp_dir) / ".hydra/hydra.yaml").hydra
    cfg = OmegaConf.load(Path(xp_dir) / ".hydra/config.yaml")
    OmegaConf.register_new_resolver(
        "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
    )
    try:
        OmegaConf.resolve(cfg)
        OmegaConf.resolve(cfg)
    except Exception as e:
        return None, None

    return cfg, OmegaConf.select(hydra_cfg, "runtime.choices.xp")

def pad_dataset(ds, pad_lat=0, pad_lon=0):
    # """
    # Pad SST Dataset (time, lat, lon) avec extension des coordonnées 1D.
    
    # Utilisé dans extract_encompassing_patch() pour ajuster la taille des patches
    # lorsque la région extraite est plus petite que 256x256.
    
    # Args:
    #     ds: xarray Dataset à padder
    #     pad_lat: nombre de pixels à ajouter en latitude
    #     pad_lon: nombre de pixels à ajouter en longitude
    
    # Returns:
    #     xarray Dataset paddé avec coordonnées extrapolées
    # """
    # dy = float((ds.lat[1] - ds.lat[0]).item())
    # dx = float((ds.lon[1] - ds.lon[0]).item())

    # pad_lat_before = pad_lat // 2
    # pad_lat_after = pad_lat - pad_lat_before
    # pad_lon_before = pad_lon // 2
    # pad_lon_after = pad_lon - pad_lon_before

    # ds_padded = ds.pad(
    #     lat=(pad_lat_before, pad_lat_after), 
    #     lon=(pad_lon_before, pad_lon_after), 
    #     constant_values=np.nan
    # )

    # new_lat = np.concatenate([
    #     ds.lat[0].item() - dy * np.arange(pad_lat_before, 0, -1),
    #     ds.lat.values,
    #     ds.lat[-1].item() + dy * np.arange(1, pad_lat_after + 1)
    # ])
    # new_lon = np.concatenate([
    #     ds.lon[0].item() - dx * np.arange(pad_lon_before, 0, -1),
    #     ds.lon.values,
    #     ds.lon[-1].item() + dx * np.arange(1, pad_lon_after + 1)
    # ])

    # ds_padded = ds_padded.assign_coords(lat=("lat", new_lat), lon=("lon", new_lon))

    # return ds_padded
    raise NotImplementedError("pad_dataset is no longer used.")

def cosanneal_vae_lr_adam(lit_mod, lr, T_max=100, weight_decay=0., train_vae=False):
    # if train_vae:
    #     opt = torch.optim.Adam(
    #     [
    #         {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
    #         #{"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
    #         {"params": lit_mod.solver.gen_mod.parameters(), "lr": lr},
    #         #{"params": lit_mod.solver.lambda_obs, "lr": lr / 2},
    #         #{"params": lit_mod.solver.lambda_reg, "lr": lr / 2},
    #     ], weight_decay=weight_decay
    #     )
    # else:
    #     opt = torch.optim.Adam(
    #     [
    #         {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
    #         #{"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
    #         {"params": lit_mod.solver.lambda_obs, "lr": 1e-1},
    #         {"params": lit_mod.solver.lambda_reg, "lr": lr/2},
    #     ], weight_decay=weight_decay
    #     )
    # return {
    #     "optimizer": opt,
    #     "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    # }
    raise NotImplementedError("cosanneal_vae_lr_adam is no longer used.")

def cosanneal_spde_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    # opt = torch.optim.Adam(
    #     [
    #         {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.nll.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.nlpobs.parameters(), "lr": lr / 2},
    #     ], weight_decay=weight_decay
    # )
    # return {
    #     "optimizer": opt,
    #     "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    # }
    raise NotImplementedError("cosanneal_spde_lr_adam is no longer used.")

def cosanneal_spde_lr_adam_winit2(lit_mod, lr, T_max=100, weight_decay=0.):
    # opt = torch.optim.Adam(
    #         [
    #          #{"params": lit_mod.solver.parameters(), "lr": lr},
    #          {"params": lit_mod.solver2.parameters(), "lr": lr},
    #         ],weight_decay=weight_decay)
    # return {
    #     "optimizer": opt,
    #     "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    # }
    raise NotImplementedError("cosanneal_spde_lr_adam_winit2 is no longer used.")

def cosanneal_spde_lr_adam_winit(lit_mod, lr, T_max=100, weight_decay=0., epoch_start_opt2=50):
    # opt1 = torch.optim.Adam([{"params": lit_mod.solver2.parameters(), "lr": lr},],weight_decay=weight_decay)
    # opt2 = torch.optim.Adam([{"params": lit_mod.solver.parameters(), "lr": lr},],weight_decay=weight_decay)
    # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=T_max)
    # #scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=T_max)
    # lambda2 = lambda epoch: 10**float(-( np.max([epoch-epoch_start_opt2,0])//35))
    # scheduler2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda = lambda2)
    # return  [opt1, opt2], [scheduler1, scheduler2]
    raise NotImplementedError("cosanneal_spde_lr_adam_winit is no longer used.")

def cosanneal_score_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    # opt = torch.optim.Adam(
    #     [
    #         {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.score_model.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.nlpobs.parameters(), "lr": lr / 2},
    #     ], weight_decay=weight_decay
    # )
    # return {
    #     "optimizers": opt,
    #     "lr_schedulers": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    # }
    raise NotImplementedError("cosanneal_score_lr_adam is no longer used.")

def cosanneal_lr_lion(lit_mod, lr, T_max=100):
    # import lion_pytorch
    # opt = lion_pytorch.Lion(
    #     [
    #         {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
    #         {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
    #     ], weight_decay=1e-3
    # )
    # return {
    #     "optimizer": opt,
    #     "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    # }
    raise NotImplementedError("cosanneal_lr_lion is no longer used.")

def extract_enlarged_patch_from_datasets(self, sl, factor, lat_bounds=None, lon_bounds=None):
        # """
        # Extrait un patch élargi qui ENGLOBE géographiquement le patch x1.
        
        # Args:
        #     sl: Slices temporels et spatiaux du patch x1
        #     factor: Facteur de résolution (3 ou 10)
        #     lat_bounds: (lat_min, lat_max) du patch x1 en degrés (optional, pour imbrication)
        #     lon_bounds: (lon_min, lon_max) du patch x1 en degrés (optional, pour imbrication)
        
        # Args:
        #     sl: Dict de slices x1 {'time': slice(...), 'lat': slice(...), 'lon': slice(...)}
        #     factor: Facteur de résolution (3 ou 10)
        
        # Logic:
        #     1. Calculer centre en x1-pixels, convertir à pixels de la résolution cible
        #     2. Clipper aux limites du fichier
        #     3. Charger données + appliquer pooling au masque si nécessaire
        #     4. Pader si région est plus petite que 256x256
        
        # IMPORTANT: self.da_dims['lat/lon'] sont les dimensions du fichier x1 (le parent).
        # - En mode precomputed: on charge depuis x3/x10 pré-coarsifiés (dimensions 1/factor)
        # - En mode pooled: on charge depuis x1 (dimensions complètes)
        # """
        # # Determine if we're loading from precomputed files or pooling from x1
        # is_precomputed_mode = (self.precomputed and self.is_multiresolution and factor > 1)
        
        # # Step 1: Find pixel indices that encompass the geographic bounds of x1 patch
        # if lat_bounds is not None and lon_bounds is not None:
        #     lat_min_x1, lat_max_x1 = lat_bounds
        #     lon_min_x1, lon_max_x1 = lon_bounds
            
        #     # Get coordinate arrays for this resolution
        #     if is_precomputed_mode:
        #         # Use pre-coarsened coordinates
        #         if hasattr(self, 'datasets') and factor in self.datasets:
        #             lat_1d = self.datasets[factor].lat_1d
        #             lon_1d = self.datasets[factor].lon_1d
        #         else:
        #             # Fallback: subsample from x1
        #             lat_1d = self.lat_1d[::factor]
        #             lon_1d = self.lon_1d[::factor]
        #         target_lat_dim = len(lat_1d)
        #         target_lon_dim = len(lon_1d)
        #     else:
        #         # Use x1 coordinates
        #         lat_1d = self.lat_1d
        #         lon_1d = self.lon_1d
        #         target_lat_dim = self.da_dims["lat"]
        #         target_lon_dim = self.da_dims["lon"]
            
        #     # Find indices that contain the x1 patch bounds
        #     # We want to ENCOMPASS the x1 patch, so use floor for min and ceil for max
        #     lat_indices = np.where((lat_1d >= lat_min_x1 - 1e-5) & (lat_1d <= lat_max_x1 + 1e-5))[0]
        #     lon_indices = np.where((lon_1d >= lon_min_x1 - 1e-5) & (lon_1d <= lon_max_x1 + 1e-5))[0]
            
        #     if len(lat_indices) == 0 or len(lon_indices) == 0:
        #         # Fallback to center-based approach if geographic matching fails
        #         print(f"[WARNING] No geographic overlap found for factor={factor}. Falling back to center-based extraction.")
        #         lat_center_x1 = (sl["lat"].start + sl["lat"].stop) // 2
        #         lon_center_x1 = (sl["lon"].start + sl["lon"].stop) // 2
        #         if is_precomputed_mode:
        #             lat_center = lat_center_x1 // factor
        #             lon_center = lon_center_x1 // factor
        #         else:
        #             lat_center = lat_center_x1
        #             lon_center = lon_center_x1
        #     else:
        #         # Use geographic bounds to define the patch
        #         lat_start = max(0, lat_indices[0])
        #         lat_end = min(lat_indices[-1] + 1, target_lat_dim)
        #         lon_start = max(0, lon_indices[0])
        #         lon_end = min(lon_indices[-1] + 1, target_lon_dim)
        #         lat_center = (lat_start + lat_end) // 2
        #         lon_center = (lon_start + lon_end) // 2
        # else:
        #     # Fallback: old center-based approach (for backward compatibility)
        #     lat_center_x1 = (sl["lat"].start + sl["lat"].stop) // 2
        #     lon_center_x1 = (sl["lon"].start + sl["lon"].stop) // 2
            
        #     if is_precomputed_mode:
        #         lat_center = lat_center_x1 // factor
        #         lon_center = lon_center_x1 // factor
        #         target_lat_dim = self.da_dims["lat"] // factor
        #         target_lon_dim = self.da_dims["lon"] // factor
        #     else:
        #         lat_center = lat_center_x1
        #         lon_center = lon_center_x1
        #         target_lat_dim = self.da_dims["lat"]
        #         target_lon_dim = self.da_dims["lon"]
        
        # # Step 2: Calculate enlarged patch bounds
        # # If we used geographic bounds, lat/lon_start/end are already set above
        # # Otherwise, calculate them from center
        # if not (lat_bounds is not None and lon_bounds is not None and len(lat_indices) > 0 and len(lon_indices) > 0):
        #     enlarged_lat = self.enlarged_dims[factor]['lat']
        #     enlarged_lon = self.enlarged_dims[factor]['lon']
            
        #     lat_start = max(0, lat_center - enlarged_lat // 2)
        #     lat_end = min(lat_start + enlarged_lat, target_lat_dim)
        #     lon_start = max(0, lon_center - enlarged_lon // 2)
        #     lon_end = min(lon_start + enlarged_lon, target_lon_dim)
        # # else: lat_start, lat_end, lon_start, lon_end already calculated from geographic bounds
        
        # # Step 3: Extract mask and apply pooling if needed
        # if hasattr(self.mask, 'isel'):
        #     mask_slice = self.mask.isel(lon=slice(lon_start, lon_end), 
        #                                lat=slice(lat_start, lat_end))
        # else:
        #     mask_slice = xr.DataArray(self.mask[lat_start:lat_end, lon_start:lon_end])
        
        # # Apply pooling to mask: factor for precomputed=False (x1 -> coarsen), 1 for precomputed=True
        # pooling_factor = 1 if (self.precomputed and self.is_multiresolution and factor > 1) else factor
        # item_mask = fast_pool(mask_slice, pooling_factor, pooling_factor, mode="binary")
        
        # # Step 3: Load data
        # if self.load_data:
        #     sst_ds = self.full_sst.isel(
        #         time=sl["time"],
        #         lat=slice(lat_start, lat_end),
        #         lon=slice(lon_start, lon_end))
        # else:
        #     time_indices = np.arange(sl["time"].start, sl["time"].stop)
        #     slices = {
        #         "lat": slice(lat_start, lat_end),
        #         "lon": slice(lon_start, lon_end)
        #     }
        #     all_sst_vars = [f"{sat}_{var}" for sat in VAR_GROUPS.keys() for var in VAR_GROUPS[sat]]
        #     if self.precomputed and self.is_multiresolution:
        #         sst_daily_paths_for_res = self.sst_daily_paths_by_resolution.get(factor, self.sst_daily_paths)
        #         resize_factor = 1
        #     else:
        #         sst_daily_paths_for_res = self.sst_daily_paths
        #         resize_factor = factor
            
        #     time_indices_list = list(time_indices) if isinstance(time_indices, np.ndarray) else time_indices
        #     paths_to_load = [sst_daily_paths_for_res[i] for i in time_indices_list]
            
        #     sst_ds = concatenate(
        #         paths_to_load,
        #         var_list=all_sst_vars + COVARIATES,
        #         slices=slices,
        #         type_coords="index",
        #         resize=resize_factor,
        #         domain_limits=self.domain_limits
        #     )
        
        # # Step 4: Pad if necessary
        # expected_shape = (self.patch_dims['time'], 256, 256)
        # first_var = list(sst_ds.data_vars)[0]
        # actual_shape = sst_ds[first_var].shape
        
        # if actual_shape != expected_shape:
        #     pad_t = expected_shape[0] - actual_shape[0]
        #     pad_lat = expected_shape[1] - actual_shape[1]
        #     pad_lon = expected_shape[2] - actual_shape[2]
            
        #     if pad_lat > 0 or pad_lon > 0:
        #         # Padding: NaN for data variables, but extrapolate coordinates
        #         pad_lat_before = pad_lat // 2
        #         pad_lat_after = pad_lat - pad_lat_before
        #         pad_lon_before = pad_lon // 2
        #         pad_lon_after = pad_lon - pad_lon_before
                
        #         # Pad data variables with NaN
        #         sst_ds = sst_ds.pad(
        #             lat=(pad_lat_before, pad_lat_after),
        #             lon=(pad_lon_before, pad_lon_after),
        #             constant_values=np.nan
        #         )
                
        #         # Fix coordinates: extrapolate lat/lon instead of using NaN
        #         if pad_lat > 0:
        #             lat_vals = sst_ds.lat.values
        #             # Find valid (non-NaN) lat values (they are in the middle after padding)
        #             valid_mask = ~np.isnan(lat_vals)
        #             if valid_mask.any():
        #                 valid_indices = np.where(valid_mask)[0]
        #                 first_valid_idx = valid_indices[0]
        #                 last_valid_idx = valid_indices[-1]
                        
        #                 valid_lats = lat_vals[valid_mask]
        #                 lat_step = np.diff(valid_lats).mean() if len(valid_lats) > 1 else 0.1
                        
        #                 # Extrapolate before first valid value
        #                 for i in range(first_valid_idx):
        #                     lat_vals[i] = valid_lats[0] - (first_valid_idx - i) * lat_step
                        
        #                 # Extrapolate after last valid value
        #                 for i in range(last_valid_idx + 1, len(lat_vals)):
        #                     lat_vals[i] = valid_lats[-1] + (i - last_valid_idx) * lat_step
                        
        #                 sst_ds['lat'] = lat_vals
                
        #         if pad_lon > 0:
        #             lon_vals = sst_ds.lon.values
        #             valid_mask = ~np.isnan(lon_vals)
        #             if valid_mask.any():
        #                 valid_indices = np.where(valid_mask)[0]
        #                 first_valid_idx = valid_indices[0]
        #                 last_valid_idx = valid_indices[-1]
                        
        #                 valid_lons = lon_vals[valid_mask]
        #                 lon_step = np.diff(valid_lons).mean() if len(valid_lons) > 1 else 0.1
                        
        #                 # Extrapolate before first valid value
        #                 for i in range(first_valid_idx):
        #                     lon_vals[i] = valid_lons[0] - (first_valid_idx - i) * lon_step
                        
        #                 # Extrapolate after last valid value
        #                 for i in range(last_valid_idx + 1, len(lon_vals)):
        #                     lon_vals[i] = valid_lons[-1] + (i - last_valid_idx) * lon_step
                        
        #                 sst_ds['lon'] = lon_vals
            
        #     # Adjust mask to match padded size
        #     actual_lat = len(sst_ds.lat)
        #     actual_lon = len(sst_ds.lon)
        #     mask_lat, mask_lon = item_mask.shape[-2:] if item_mask.ndim == 3 else item_mask.shape
            
        #     if actual_lat > mask_lat or actual_lon > mask_lon:
        #         pad_lat_needed = actual_lat - mask_lat
        #         pad_lon_needed = actual_lon - mask_lon
        #         if item_mask.ndim == 3:
        #             item_mask = np.pad(item_mask, ((0, 0), (pad_lat_needed//2, (pad_lat_needed+1)//2), (pad_lon_needed//2, (pad_lon_needed+1)//2)), mode='constant', constant_values=1)
        #         else:
        #             item_mask = np.pad(item_mask, ((pad_lat_needed//2, (pad_lat_needed+1)//2), (pad_lon_needed//2, (pad_lon_needed+1)//2)), mode='constant', constant_values=1)
        
        # # Step 5: Assemble output dict
        # sample = {}
        
        # # Add satellite variables
        # for sat_name in ['aasti', 'avhrr', 'pmw', 'slstr']:
        #     for var in VAR_GROUPS[sat_name]:
        #         var_key = f"{sat_name}_{var}"
        #         if var_key in sst_ds:
        #             sample[var_key] = sst_ds[var_key].values
        
        # # Add covariates
        # for cov in COVARIATES:
        #     if cov in sst_ds:
        #         sample[cov] = sst_ds[cov].values
        
        # # Add metadata
        # sample["surfmask"] = np.expand_dims(item_mask, axis=0)
        
        # lon_1d = sst_ds.lon.values
        # lat_1d = sst_ds.lat.values
        # lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d)
        
        # # Store both normalized (for network input) and geographic (for interpolation) coordinates
        # # Normalize coordinates for network input channels
        # lat_channel = (lat_2d / 90.0).astype(np.float32)  # in [-1, 1]
        # lon_channel = (lon_2d / 180.0).astype(np.float32)  # in [-1, 1]
        # sample["lat"] = lat_channel  # Normalized for network
        # sample["lon"] = lon_channel  # Normalized for network
        
        # # Store geographic coordinates (in degrees) for interpolation
        # sample["lat_geo"] = lat_2d.astype(np.float32)  # Geographic degrees
        # sample["lon_geo"] = lon_2d.astype(np.float32)  # Geographic degrees
         
        # # Create time channel as a 2D grid (same as in data.py)
        # # Use the center timestep's day of year, normalized [0, 1]
        # import pandas as pd
        # center_time_idx = len(sst_ds.time) // 2
        # center_time = sst_ds.time.values[center_time_idx]
        # day_of_year = pd.Timestamp(center_time).dayofyear  # 1-366
        # time_value = day_of_year / 366.0  # Normalized [0, 1]
        # time_channel = np.full((lat_2d.shape[0], lat_2d.shape[1]), time_value, dtype=np.float32)
        # sample["time"] = time_channel  # (nlat, nlon) - 2D grid
        
        # sample["inpaint_mask"] = np.zeros_like(sample["surfmask"])
        
        # # Add target variables
        # for tgt_var in self.tgt_vars:
        #     if tgt_var in sample:
        #         sample[f"tgt_{tgt_var}"] = sample[tgt_var]
        # if self.tgt_vars:
        #     sample["tgt_sst"] = sample.get(f"tgt_{self.tgt_vars[0]}", np.zeros_like(sample["surfmask"]))
        
        # # Return raw dict - postprocessing (normalization + TrainingItem creation) 
        # # will be applied later in __getitem__() using saved_postpro_fn
        raise NotImplementedError("This function was used in data_multires.py, replaced by encompassing_patch()")
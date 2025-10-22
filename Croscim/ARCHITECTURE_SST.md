# Architecture du projet SST Multi-résolution

### Données
- **Période**: 10-15 ans (2010-2024 prévu)
- **Couverture**: Globe entier (3600×7200 pixels à 5km)
- **Capteurs**: 4 satellites SST
  - `slstr`: SLSTR (Sentinel-3), mid-latitudes
  - `aasti`: AATSR (Envisat), pôles
  - `avhrr`: AVHRR (NOAA), bonne couverture
  - `pmw`: PMW (micro-ondes), global mais moins précis
- **Covariates**: `sea_ice_fraction`
- **Target**: SST fusionnée (slstr prioritaire, aasti aux pôles)

### Résolution temporelle
- **Fenêtre**: 15 jours consécutifs par patch
- **Stride train**: 7 jours (1 fenêtre/semaine)
- **Stride test**: 15 jours (pas de chevauchement)

### Résolution spatiale
- **Patch size**: 240×240 pixels
- **Stride train**: 40 pixels
- **Stride test**: 120 pixels
- **Multi-résolution**: [1×, 5×, 10×]

---

## Structure des données sur disque

```
/dmidata/users/malegu/data/
├── raw/                              # Données brutes (read-only)
│   ├── netcdf_2010/
│   │   ├── 2010010112_13vars.nc
│   │   ├── 2010010200_13vars.nc
│   │   └── ...
│   ├── netcdf_2011/
│   └── ... (jusqu'à 2024)
│
├── masks/                            # Masques (calculés une fois)
│   └── surfmask_global.asc          # Mask océan/terre/glace ASCII
│
├── preprocessed/                     # Pré-calculs (calculés une fois)
│   ├── valid_patches_stride40.npy   # ~15k indices patches valides
│   ├── norm_stats_2010_2022.yaml    # Stats train (avec tgt_sst)
│   └── domain_splits.yaml           # Définition train/val/test
│
└── outputs/                          # Résultats
    ├── models/                       # Checkpoints
    ├── predictions/                  # NetCDF de prédictions
    └── figures/                      # Diagnostics visuels
```

---

## Pipeline de traitement

### PHASE 1 : Setup (une seule fois)

#### 1.1 Calcul des patches valides
```bash
python compute_valid_patches.py \
    --mask /dmidata/users/malegu/data/masks/surfmask_global.asc \
    --patch_size 240 \
    --stride 40 \
    --output /dmidata/users/malegu/data/preprocessed/valid_patches_stride40.npy
```
**Temps**: ~5 secondes (avec vectorisation)  
**Résultat**: Fichier ~50Ko avec ~15k indices

#### 1.2 Calcul des statistiques globales
```bash
python contrib/SST/compute_statistics.py \
    --data_dir /dmidata/users/malegu/data/raw/netcdf_201* \
    --output /dmidata/users/malegu/data/preprocessed/norm_stats_2010_2022.yaml
```
**Temps**: ~1-2 heures (sur 13 ans)  
**Résultat**: YAML avec stats par capteur + `tgt_sst` fusionnée

#### 1.3 Définition des domaines
```yaml
# domain_splits.yaml
train:
  time: ["2010-01-01", "2022-12-31"]  # 13 ans = ~4745 jours
val:
  time: ["2023-01-01", "2023-12-31"]  # 1 an = 365 jours
test:
  time: ["2024-01-01", "2024-12-31"]  # 1 an = 365 jours
```

---

### PHASE 2 : Training

#### Architecture des modules

```
contrib/SST/
├── load_data.py              # Utilitaires (concatenate, coarsen, VAR_GROUPS)
├── utils.py                  # validate_norm_stats, helpers
├── data.py                   # BaseDataModule + XrDataset (lecture à la volée)
├── data_multires.py          # Multi-résolution (hérite de data.py)
├── models.py                 # UNet, 4DVarNet, architectures
├── solver.py                 # 4DVarNet solver
└── compute_statistics.py     # Calcul stats + tgt_sst
```

#### Configuration training (production_10y.yaml)

```yaml
data:
  sst_paths: "/dmidata/users/malegu/data/raw/**/*_13vars.nc"
  mask_path: "/dmidata/users/malegu/data/masks/surfmask_global.asc"
  norm_stats: "/dmidata/users/malegu/data/preprocessed/norm_stats_2010_2022.yaml"
  
  patch_dims: {time: 15, lat: 240, lon: 240}
  strides: {time: 7, lat: 40, lon: 40}
  strides_test: {time: 15, lat: 120, lon: 120}
  
  subsel_patch: true
  subsel_patch_path: "/dmidata/users/malegu/data/preprocessed/valid_patches_stride40.npy"

multires:
  enable: true
  factors: [1, 2, 4]

training:
  patches_per_epoch: 50_000    # Subsampling (sur ~10M disponibles)
  batch_size: 64               # Par GPU
  gpus: 4                      # H100 80GB
  num_workers: 12              # Par GPU
  max_epochs: 100
  early_stopping_patience: 15
```

#### Lancement
```bash
python main.py \
    --config config/xp/SST/production_10y.yaml \
    --gpus 0,1,2,3
```

---

### PHASE 3 : Inference & Validation

```bash
# Test sur 2024
python src/test.py \
    --checkpoint outputs/models/best_model.ckpt \
    --config config/xp/SST/production_10y.yaml \
    --split test

# Génération figures
python contrib/SST/plot_results.py \
    --predictions outputs/predictions/test_2024.nc \
    --output outputs/figures/
```

---

## Estimation des ressources

### Nombre de patches

| Split | Période | Fenêtres temps | Patches spatiaux | Total patches |
|-------|---------|----------------|------------------|---------------|
| Train | 2010-2022 (13 ans) | ~676 (stride=7j) | ~15k | **~10M** |
| Val   | 2023 (1 an) | ~52 | ~15k | **~780k** |
| Test  | 2024 (1 an) | ~52 | ~15k | **~780k** |

### Subsampling training
- Patches disponibles: 10M
- **Patches/epoch: 50k (0.5%)** ← configurable
- Shuffle aléatoire à chaque epoch
- Pas de sur-apprentissage (grande diversité)

### Mémoire

| Composant | Taille |
|-----------|--------|
| 1 TrainingItem (15×240×240×12 floats) | ~41 MB |
| 1 Batch (64 items) | ~2.6 GB |
| 4 GPUs × (batch + modèle) | ~20 GB total |
| **Requis**: 4xH100 80GB | Largement suffisant |

### Temps d'entraînement (H100)

```python
# Avec subsampling
patches_per_epoch = 50,000
effective_batch_size = 256  # 4 GPUs × 64
steps_per_epoch = 50,000 / 256 ≈ 195

# H100 (estimé)
time_per_step ≈ 0.1 sec
time_per_epoch ≈ 20 secondes

# Training complet
100 epochs × 20 sec = 2,000 sec ≈ 33 minutes
Avec early stopping (~30 epochs) ≈ 10 minutes
```

**Note**: Pour ajuster à quelques heures, augmenter `patches_per_epoch` (ex: 500k → ~3h30 par epoch)

---

## Optimisations clés

### 1. Vectorisation du filtrage patches
```python
# Ancienne version: boucle Python → 1-2 jours
for i in range(59000):
    if patch_is_valid(i):
        indices.append(i)

# Nouvelle version: scipy.ndimage → 5 secondes
filtered = maximum_filter(mask, size=(240, 240))
indices = np.where(filtered[::40, ::40] > 0)[0]
```

### 2. Subsampling des patches
```python
# Au lieu d'utiliser 10M patches/epoch
class SubsampledBatchSampler:
    def __init__(self, dataset, batch_size, patches_per_epoch=50000):
        self.indices = np.random.choice(len(dataset), patches_per_epoch, replace=False)
    
    def __iter__(self):
        np.random.shuffle(self.indices)  # Shuffle à chaque epoch
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i:i+self.batch_size]
```

### 3. Lecture lazy des NetCDF
```python
# On lit seulement la région spatiale nécessaire
ds = xr.open_dataset(path)
patch = ds.isel(lat=slice(100, 340), lon=slice(500, 740))  # 240×240 seulement
```

### 4. Cache des résultats coûteux
- `valid_patches.npy`: pré-calculé une fois
- `norm_stats.yaml`: pré-calculé sur train split
- Pas de recalcul à chaque run

---

## Checklist de mise en œuvre

### Phase 1: Setup
- [ ] Créer `compute_valid_patches.py` (vectorisé)
- [ ] Lancer calcul patches → `valid_patches_stride40.npy`
- [ ] Vérifier `compute_statistics.py` inclut `tgt_sst`
- [ ] Lancer calcul stats → `norm_stats_2010_2022.yaml`
- [ ] Créer `domain_splits.yaml`

### Phase 2: Code
- [ ] Optimiser `find_patches_in_ocean()` dans `data.py`
- [ ] Ajouter `SubsampledBatchSampler` dans `data.py`
- [ ] Adapter `data_multires.py` (asip/cimr/cristal → slstr/aasti/avhrr/pmw)
- [ ] Créer config `production_10y.yaml`

### Phase 3: Tests
- [ ] Créer `test_sst_multires_with_plots.py`
- [ ] Test sur 1 mois de données (validation rapide)
- [ ] Test sur 1 an (validation complète)

### Phase 4: Production
- [ ] Training complet 10-15 ans
- [ ] Évaluation sur test set 2024
- [ ] Génération figures publication
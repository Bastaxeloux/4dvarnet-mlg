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
- **Stride test**: a definir

### Résolution spatiale
- **Patch size**: 256×256 pixels
- **Stride train**: a definir
- **Stride test**: a definir
- **Multi-résolution**: [1×, 3×, 10×]

---

## Structure des données sur disque

```
/dmidata/users/malegu/
├── data/
│   ├── netcdf_2010/
│   │   ├── 2010010112_13vars.nc
│   │   ├── 2010010200_13vars.nc
│   │   └── ...
│   ├── netcdf_2011/
│   └── ... (jusqu'à 2024)
```


# FLUX COMPLET AVEC MULTI-RÉSOLUTION (1 batch)

## 1. CHARGEMENT DES DONNÉES (data_multires.py)

```
   DataLoader tire 4 indices : [42, 7234, 15, 89]
        ↓
   XrDatasetMultiResTrain.__getitem__(42)
   │
   ├─ A. Charger patch HAUTE RÉSOLUTION (patch_x2 = resize=1)
   │  │
   │  ├─ super().__getitem__(42)  # Appelle XrDataset.__getitem__
   │  │  ├─ Charge 15 fichiers NetCDF
   │  │  ├─ Extrait patch 240×240
   │  │  └─ Retourne dict: {
   │  │         'slstr_av': (15, 240, 240),
   │  │         'aasti_av': (15, 240, 240),
   │  │         'tgt_sst': (15, 240, 240),
   │  │         'lat': (240, 240),
   │  │         'lon': (240, 240),
   │  │         ...
   │  │      }
   │  │
   │  └─ out['patch_x2'] = hr_sample  # Haute résolution
   │
   ├─ B. Créer patch RÉSOLUTION MOYENNE (patch_x10)
   │  │
   │  ├─ extract_enlarged_patch_from_datasets(sl, factor=5)
   │  │  │
   │  │  ├─ Calculer zone élargie :
   │  │  │  patch 240×240 × 5 = 1200×1200 (contexte plus large)
   │  │  │
   │  │  ├─ Charger cette zone élargie (1200×1200)
   │  │  │
   │  │  ├─ Coarsen (réduire résolution) :
   │  │  │  1200×1200 → 120×120 (pooling factor=10)
   │  │  │
   │  │  └─ Retourne dict: {
   │  │         'slstr_av': (15, 120, 120),
   │  │         'aasti_av': (15, 120, 120),
   │  │         'tgt_sst': (15, 120, 120),
   │  │         ...
   │  │      }
   │  │
   │  └─ out['patch_x10'] = coarsened_sample
   │
   └─ C. Créer patch BASSE RÉSOLUTION (patch_x50)
      │
      ├─ extract_enlarged_patch_from_datasets(sl, factor=25)
      │  │
      │  ├─ Zone encore plus large :
      │  │  patch 240×240 × 25 = 6000×6000 (contexte océan entier)
      │  │
      │  ├─ Charger cette zone (6000×6000)
      │  │
      │  ├─ Coarsen : 6000×6000 → 48×48 (pooling factor=125)
      │  │
      │  └─ Retourne dict: {
      │         'slstr_av': (15, 48, 48),
      │         'aasti_av': (15, 48, 48),
      │         'tgt_sst': (15, 48, 48),
      │         ...
      │      }
      │
      └─ out['patch_x50'] = very_coarse_sample

   ↓ Retourne :
   
   out = {
       'patch_x2': {slstr_av: (15,240,240), ...},   # Détails locaux
       'patch_x10': {slstr_av: (15,120,120), ...},  # Patterns régionaux
       'patch_x50': {slstr_av: (15,48,48), ...},    # Contexte global
   }

   ↓ DataLoader assemble 4 patches

   batch = {
       'patch_x2': TrainingItem(
           slstr_av=(4, 15, 240, 240),  # 4 patches haute résolution
           ...
       ),
       'patch_x10': TrainingItem(
           slstr_av=(4, 15, 120, 120),  # 4 patches résolution moyenne
           ...
       ),
       'patch_x50': TrainingItem(
           slstr_av=(4, 15, 48, 48),    # 4 patches basse résolution
           ...
       ),
   }

   ↓ post_fn(batch) : normalisation sur chaque résolution
   ↓ batch → GPU
```
---

## 2. FORWARD PASS MULTI-RÉSOLUTION (models.py + solver.py)
```
   Lightning appelle : model.training_step(batch, batch_idx)
        ↓
   models.py : Lit4dVarNet_SST.training_step()
        ↓
   self.forward(batch)
   │
   ├─────────────────────────────────────────────────────────────
   │ ÉTAPE 1 : RÉSOLUTION COARSE (48×48, contexte global)
   ├─────────────────────────────────────────────────────────────
   │
   ├─ Extraire patch_x50 du batch
   │  input_x50 = batch['patch_x50']
   │  → input: (4, 9, 15, 48, 48)  # 9 vars (slstr, aasti, avhrr, pmw, sea_ice)
   │  → tgt: (4, 1, 15, 48, 48)     # 1 var (tgt_sst)
   │
   ├─ Passer dans solver_x50
   │  solver.py : GradSolver.forward(input_x50, tgt_x50)
   │  │
   │  ├─ state = init (prédiction initiale)
   │  │  → (4, 1, 15, 48, 48)
   │  │
   │  ├─ Pour 5 steps :
   │  │  ├─ cost = prior_cost(state) + obs_cost(state, input_x50)
   │  │  ├─ grad = autograd.grad(cost, state)
   │  │  │  → (4, 1, 15, 48, 48)
   │  │  ├─ correction = ConvLSTM(grad)
   │  │  │  # ConvLSTM a vue sur TOUT le contexte global
   │  │  └─ state = state - correction
   │  │
   │  └─ Retourne pred_x50 : (4, 1, 15, 48, 48)
   │     → SST prédite avec contexte global
   │
   ├─────────────────────────────────────────────────────────────
   │ INTERPOLATION 1 : 48×48 → 120×120
   ├─────────────────────────────────────────────────────────────
   │
   ├─ Upsampling bilinéaire
   │  pred_x50_upsampled = F.interpolate(pred_x50, size=(120, 120))
   │  → (4, 1, 15, 120, 120)
   │
   ├─────────────────────────────────────────────────────────────
   │ ÉTAPE 2 : RÉSOLUTION MEDIUM (120×120, patterns régionaux)
   ├─────────────────────────────────────────────────────────────
   │
   ├─ Extraire patch_x10 du batch
   │  input_x10 = batch['patch_x10']
   │  → (4, 9, 15, 120, 120)
   │
   ├─ Ajouter prédiction coarse comme prior
   │  input_x10_enriched = concat(input_x10, pred_x50_upsampled)
   │  → (4, 10, 15, 120, 120)  # 9 obs + 1 prédiction coarse
   │
   ├─ Passer dans solver_x10
   │  solver.py : GradSolver.forward(input_x10_enriched, tgt_x10)
   │  │
   │  ├─ state = pred_x50_upsampled  # Partir de la prédiction coarse
   │  │  → (4, 1, 15, 120, 120)
   │  │
   │  ├─ Pour 5 steps :
   │  │  ├─ cost = prior_cost(state) + obs_cost(state, input_x10)
   │  │  ├─ grad = autograd.grad(cost, state)
   │  │  ├─ correction = ConvLSTM(grad)
   │  │  │  # Affine la prédiction avec patterns régionaux
   │  │  └─ state = state - correction
   │  │
   │  └─ Retourne pred_x10 : (4, 1, 15, 120, 120)
   │     → SST prédite avec contexte global + patterns régionaux
   │
   ├─────────────────────────────────────────────────────────────
   │ INTERPOLATION 2 : 120×120 → 240×240
   ├─────────────────────────────────────────────────────────────
   │
   ├─ Upsampling bilinéaire
   │  pred_x10_upsampled = F.interpolate(pred_x10, size=(240, 240))
   │  → (4, 1, 15, 240, 240)
   │
   ├─────────────────────────────────────────────────────────────
   │ ÉTAPE 3 : RÉSOLUTION FINE (240×240, détails locaux)
   ├─────────────────────────────────────────────────────────────
   │
   ├─ Extraire patch_x2 du batch
   │  input_x2 = batch['patch_x2']
   │  → (4, 9, 15, 240, 240)
   │
   ├─ Ajouter prédiction medium comme prior
   │  input_x2_enriched = concat(input_x2, pred_x10_upsampled)
   │  → (4, 10, 15, 240, 240)
   │
   ├─ Passer dans solver_x2
   │  solver.py : GradSolver.forward(input_x2_enriched, tgt_x2)
   │  │
   │  ├─ state = pred_x10_upsampled  # Partir de la prédiction medium
   │  │
   │  ├─ Pour 10 steps :
   │  │  ├─ cost = prior_cost(state) + obs_cost(state, input_x2)
   │  │  ├─ grad = autograd.grad(cost, state)
   │  │  ├─ correction = ConvLSTM(grad)
   │  │  │  # Ajoute les détails fins locaux
   │  │  └─ state = state - correction
   │  │
   │  └─ Retourne pred_x2 : (4, 1, 15, 240, 240)
   │     → SST finale avec TOUT : global + régional + local
   │
   └─ predictions = {
          'patch_x50': pred_x50,  # (4, 1, 15, 48, 48)
          'patch_x10': pred_x10,  # (4, 1, 15, 120, 120)
          'patch_x2': pred_x2,    # (4, 1, 15, 240, 240)
      }
```
---

## 3. CALCUL DE LA LOSS MULTI-RÉSOLUTION
```
   compute_loss(predictions, batch)
   │
   ├─ Loss résolution fine (la plus importante)
   │  loss_x2 = MSE(pred_x2, batch['patch_x2'].tgt_sst)
   │  → Comparaison 240×240
   │
   ├─ Loss résolution medium (aide à la convergence)
   │  tgt_x10_downsampled = downsample(batch['patch_x2'].tgt_sst, factor=2)
   │  loss_x10 = MSE(pred_x10, tgt_x10_downsampled)
   │  → Comparaison 120×120
   │
   ├─ Loss résolution coarse (stabilise l'entraînement)
   │  tgt_x50_downsampled = downsample(batch['patch_x2'].tgt_sst, factor=5)
   │  loss_x50 = MSE(pred_x50, tgt_x50_downsampled)
   │  → Comparaison 48×48
   │
   └─ Loss totale (pondérée)
      loss = loss_x2 + 0.5 * loss_x10 + 0.25 * loss_x50
      # Poids décroissants : résolution fine = priorité
```
---

## 4. BACKWARD (identique, Lightning automatique)
```
   loss.backward() → optimizer.step()
```
---

# Estimation des ressources

### Nombre de patches

| Split | Période | Fenêtres temps | Patches spatiaux | Total patches |
|-------|---------|----------------|------------------|---------------|
| Train | 2010-2022 (13 ans) | ~676 (stride=7j) | ~15k | **~10M** |
| Val   | 2023 (1 an) | ~52 | ~15k | **~780k** |
| Test  | 2024 (1 an) | ~52 | ~15k | **~780k** |

### Subsampling training
- Patches disponibles: 10M
- **Patches/epoch: 50k (0.5%)** <= a choisir
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
steps_per_epoch = 50,000 / 256 = 195

# H100 (estimé)
time_per_step = 0.1 sec
time_per_epoch = 20 secondes

# Training complet
100 epochs * 20 sec = 2,000 sec = 33 minutes
Avec early stopping (~30 epochs) = 10 minutes
```

**Note**: Pour ajuster à quelques heures, augmenter `patches_per_epoch` (ex: 500k => ~3h30 par epoch)
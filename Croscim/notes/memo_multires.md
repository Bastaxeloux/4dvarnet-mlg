# MÉMO MULTI-RÉSOLUTION SST - État des Lieux

**Objectif**: Document de référence pour comprendre la gestion des patchs multi-résolution

---

## 1. OBJECTIF DU PROJET

### 1.1 But Final
Reconstruire la SST (Sea Surface Temperature) sur **tout le globe** en haute résolution (x1 = 5km/pixel) à partir d'observations satellites incomplètes.

### 1.2 Contrainte
- Le réseau ne peut pas traiter tout le globe d'un coup (mémoire GPU limitée). On choisit donc des patchs de **256×256 pixels**
- Les observations satellites ont des **trous** (nuages, pas de passage satellite, etc.)
- Besoin de capturer les **relations multi-échelles** (grande échelle -> détails fins)

### 1.3 Solution Envisagée
Architecture **multi-résolution en cascade** :
- **x10** (50km/px) : Capture les grandes échelles océaniques (courants, fronts)
- **x3** (15km/px) : Ajoute les échelles méso (tourbillons moyens)
- **x1** (5km/px) : Ajoute les détails fins (petits tourbillons, fronts précis)

---

## 2. ARCHITECTURE THÉORIQUE

### 2.1 Mode TRAINING

Pour chaque batch d'entraînement :
1. Tirer un triplet de patchs **géographiquement imbriqués** (x10, x3, x1)
2. Prédire séquentiellement : x10 -> x3 -> x1
3. Calculer la loss et backpropager

#### Schéma d'Imbrication Géographique
```
┌─────────────────────────────────────────┐
│  Patch x10 (256px @ 50km = 12800km)     
│                                         
│    ┌────────────────────────────────┐   
│    │  Patch x3 (256px @ 15km           
│    │          = 3840km)                
│    │                                  
│    │   ┌────────────────────┐          
│    │   │  Patch x1                    
│    │   │  (256px @ 5km                
│    │   │   = 1280km)                  
│    │   └────────────────────┘          
│    └────────────────────────────────┘   
└─────────────────────────────────────────┘
```

**IMPORTANT** : Les 3 patchs doivent être **imbriqués** !

#### Forward Pass Training
```python
# ÉTAPE 1: Prédiction x10 (directe)
Input x10 : obs satellites x10 (aasti, avhrr, pmw, slstr) + covariates
Solver x10 : GradSolver avec n_step=2
Output : pred_x10 (B, 15, 256, 256)

# ÉTAPE 2: Prédiction x3 (anomalie)
pred_x10_interp = interpolate(pred_x10, grid_x3)  # Interpolation géométrique
Input x3 : obs satellites x3 (DIFFÉRENTES de x10!) + covariates
           MAIS converties en ANOMALIES par rapport à pred_x10_interp
Solver x3 : GradSolver avec n_step=3, prédit le RÉSIDU
Output : residual_x3
Reconstruction : pred_x3 = pred_x10_interp + residual_x3

# ÉTAPE 3: Prédiction x1 (anomalie)
pred_x3_interp = interpolate(pred_x3, grid_x1)
Input x1 : obs satellites x1 (ENCORE DIFFÉRENTES!) + covariates
           converties en ANOMALIES par rapport à pred_x3_interp
Solver x1 : GradSolver avec n_step=3, prédit le RÉSIDU
Output : residual_x1
Reconstruction : pred_x1 = pred_x3_interp + residual_x1
```

**Note Cruciale** : Les observatios satellites sont celles de la résolution x1. Les résolutions x3 et x10 ont étés obtenues via les x1.
Pour le train on a précalculé les x3 et x10, afin d'aller plus vite. Lors de l'inférence il faudra faire ce calcul des résolutions supérieures à la volée.

---

### 2.2 Mode TEST

#### Principe
Reconstruire **tout le globe** de manière systématique :

1. **Étape 1** : Prédire et assembler x10 sur tout le globe
   - Découper le globe en patchs x10 (avec overlap)
   - Prédire chaque patch
   - Agréger en une carte globale x10
   - Sauvegarder la carte complète

2. **Étape 2** : Prédire et assembler x3 sur tout le globe
   - Découper le globe en patchs x3
   - Pour chaque patch x3 :
     - Aller chercher la zone correspondante dans la carte x10
     - Interpoler x10 sur grid x3
     - Prédire le résidu x3
     - Additionner : pred_x3 = interp(x10) + residual_x3
   - Agréger en une carte globale x3
   - Sauvegarder

3. **Étape 3** : Prédire et assembler x1 sur tout le globe
   - Idem que x3 mais en partant de la carte x3

#### Schéma Test
```
┌─────────────────────────────────────┐
│  GLOBE x10 (360×720 pixels)         
│  ┌────┬────┬────┬────┬────┐         
│  │ P1 │ P2 │ P3 │ P4 │...          
│  ├────┼────┼────┼────┼────┤         
│  │ P6 │ P7 │ P8 │ P9 │...          
│  └────┴────┴────┴────┴────┘         
│  -> Prédire tous les patchs x10      
│  -> Assembler en carte globale x10   
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  GLOBE x3 (1200×2400 pixels)        
│  ┌──┬──┬──┬──┬──┬──┬──┬──┐         
│  │P1│P2│P3│P4│P5│P6│P7│..         
│  ├──┼──┼──┼──┼──┼──┼──┼──┤         
│  │P8│P9│..│..│..│..│..│..         
│  └──┴──┴──┴──┴──┴──┴──┴──┘         
│  -> Pour chaque patch x3 :           
│    - Interpoler carte x10 locale    
│    - Prédire résidu                 
│    - Additionner                    
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│  GLOBE x1 (3600×7200 pixels)
│  -> Idem que x3
└─────────────────────────────────────┘
```

**Logique Additive Pure** : C'est une **reconstruction en cascade**, pas une prédiction jointe.

---

## 3. CODE ACTUEL

### 3.1 Structure des Datasets

#### Fichiers Zarr
```
/nwp/sst_malegu/data_2024/
├── 2024010112_x1.zarr   # 3600×7200 pixels (5km/px)
├── 2024010112_x3.zarr   # 1200×2400 pixels (15km/px, pré-coarsened)
├── 2024010112_x10.zarr  # 360×720 pixels (50km/px, pré-coarsened)
└── ...
```

#### Datasets Indépendants
```python
# Dans XrDatasetMultiResSingleDay (data_multires.py:503)
self.datasets = {}
for res in [1, 3, 10]:
    kwargs["resize"] = res
    self.datasets[res] = XrDatasetSingleDay(...)

# Chaque dataset a :
self.datasets[1]  : grille x1 (3600 x 7200), patch_dims=(15, 256, 256)
self.datasets[3]  : grille x3 (1200 x 2400), patch_dims=(15, 256, 256)
self.datasets[10] : grille x10 (360 x 720),  patch_dims=(15, 256, 256)
```

#### Système d'Indexation
```python
# XrDataset.__getitem__(idx)
sl = {
    dim: slice(self.strides[dim] * idx_dim,
               self.strides[dim] * idx_dim + self.patch_dims[dim])
    for dim, idx_dim in zip(
        self.ds_size.keys(),
        np.unravel_index(idx, tuple(self.ds_size.values()))
    )
}
```

**Problème** : Chaque dataset (x1, x3, x10) a sa propre grille de patchs. 
-> Le même `idx=42` pointe vers des **zones géographiques différentes** !

**Exemple** :
```
idx=42 dans dataset x10 -> lat=[-80,-70], lon=[-150,-140]
idx=42 dans dataset x3  -> lat=[-85,-75], lon=[+20,+30]   <- PAS PAREIL !
idx=42 dans dataset x1  -> lat=[-90,-80], lon=[+50,+60]   <- ENCORE DIFFÉRENT !
```

---

### 3.2 Principe de l'Imbrication Géographique

Le code garantit désormais que les patchs x1, x3 et x10 sont **géographiquement imbriqués**.

**Fichiers clés** :
- `contrib/SST/data_multires.py` : `XrDatasetMultiResTrain.__getitem__`
- `src/utils.py` : `extract_encompassing_patch`, `encompassing_patch`

**Stratégie** :
1. Extraire d'abord le patch **x1** (référence, haute résolution)
2. Trouver le patch **x3** qui **englobe géographiquement** x1
3. Trouver le patch **x10** qui **englobe géographiquement** x3
4. L'interpolation géométrique aligne les pixels entre résolutions

### 3.3 Fonction `extract_encompassing_patch` (src/utils.py)

```python
def extract_encompassing_patch(
    dataset_obj, sl, factor, lat_bounds, lon_bounds,
    VAR_GROUPS, COVARIATES, patch_dims, tgt_vars
):
    """
    Extrait un patch SST multi-résolution qui ENGLOBE géographiquement une région donnée.
    
    Étapes :
    1. Déterminer coordonnées 1D pour la résolution cible (x3 ou x10)
    2. Appeler encompassing_patch() pour trouver les indices lat/lon
    3. Charger les données SST depuis fichiers Zarr
    4. Assembler le dict de sortie avec toutes les variables
    """
```
La sous-fonction **encompassing_patch** trouve le patch 256×256 optimal :
- Prend en entrée les bounds géographiques à englober
- Cherche le meilleur positionnement (centré si possible, décalé aux bords du globe)
- Retourne les slices lat_slice, lon_slice

### 3.3 Code __getitem__ Multi-Résolution (data_multires.py:99-173)

```python
def __getitem__(self, idx):
    # 1. Extraire patch x1 (référence)
    hr_sample = super().__getitem__(idx)
    lat_bounds_x1 = (float(lat_geo_x1.min()), float(lat_geo_x1.max()))
    lon_bounds_x1 = (float(lon_geo_x1.min()), float(lon_geo_x1.max()))
    
    # 2. Pour chaque résolution plus grossière (x3 puis x10)
    for factor in [3, 10]:
        enlarged_patch = extract_encompassing_patch(
            dataset_obj=self,
            factor=factor,
            lat_bounds=prev_lat_bounds,
            lon_bounds=prev_lon_bounds,
            ...
        )
        # Mettre à jour bounds pour la résolution suivante
        prev_lat_bounds = (enlarged_patch['lat_geo'].min(), ...)
    
    return {'patch_x1': hr_sample, 'patch_x3': ..., 'patch_x10': ...}
```

**Validation** : 5 tests unitaires passent avec des erreurs d'arrondi mineures (< 0.5°) dues à l'interpolation, mais visuellement les patchs sont bien imbriqués.

### 3.3 Code Forward Multi-Résolution

#### `multistep` (models.py:~870)
```python
def multistep(self, batch, phase=""):
    # batch = {'patch_x1': ..., 'patch_x3': ..., 'patch_x10': ...}
    
    out = {}
    
    for i, res in enumerate([10, 3, 1]):  # Ordre coarse -> fine
        batch_res = batch[f"patch_x{res}"]
        
        if res == 10:  # Première résolution
            # Prédiction directe
            loss, out['patch_x10'] = self.step(batch_res, res=10, phase=phase)
        
        else:  # Résolutions suivantes (x3, x1)
            coarser_res = [10, 3, 1][i-1]  # x3->10, x1->3
            
            # 1. Interpoler pred coarse sur grille fine
            lon_target = batch_res.lon_geo  # Coordonnées GÉOGRAPHIQUES
            lat_target = batch_res.lat_geo
            lon_coarse = batch[f"patch_x{coarser_res}"].lon_geo
            lat_coarse = batch[f"patch_x{coarser_res}"].lat_geo
            
            out[f"patch_x{coarser_res}_on_x{res}"] = self.interpolate_torch(
                out[f"patch_x{coarser_res}"],
                lon_coarse, lat_coarse,
                lon_target, lat_target
            )
            # PROBLÈME : Les coordonnées ne correspondent PAS !
            #    -> Interpolation hors limites -> 100% NaN !
            
            # 2. Convertir batch en anomalies
            batch_res = self.update_batch_as_anomaly(
                batch_res,
                out[f"patch_x{coarser_res}_on_x{res}"]
            )
            
            # 3. Prédire le résidu
            loss, residual = self.step(batch_res, res=res, phase=phase)
            
            # 4. Reconstruire : pred = interp + résidu
            out[f"patch_x{res}"] = {
                var: out[f"patch_x{coarser_res}_on_x{res}"][var] + residual[var]
                for var in residual
            }
```

---

### 3.4 Code Test Actuel

#### `test_step` (models.py:~1485)
```python
def test_step(self, batch, batch_idx, dataloader_idx):
    res = self.multires[dataloader_idx]  # 0->x10, 1->x3, 2->x1
    
    if dataloader_idx == 0:  # x10
        # Prédiction directe
        out = self(batch=sbatch, res=10)
    
    else:  # x3 ou x1
        coarser_res = self.multires[dataloader_idx-1]
        
        # 1. Récupérer la carte globale coarse assemblée
        coarse = self.aggregate_results[f"patch_x{coarser_res}"]
        
        # 2. Interpoler sur grille batch actuel
        itrp_coarse = self.interpolate_torch(coarse, ...)
        
        # 3. Convertir batch en anomalies
        batch = self.update_batch_as_anomaly(batch, itrp_coarse)
        
        # 4. Prédire résidu
        out = self(batch=sbatch, res=res)
        
        # 5. Reconstruire : pred = interp + résidu
        out = {k: out[k] + itrp_coarse[k] for k in out}
    
    # Stocker pour agrégation
    self.test_data[res_key].append(stacked)
    self.test_times[res_key].append(central_times)
    
    # Si dernier batch : agréger en carte globale
    if self.is_last_batch(batch_idx, dataloader_idx):
        self.aggregate_results[res_key] = self.aggregate_batches(...)
```

**Logique Test** :
- OUI : Test résolution par résolution (x10 -> x3 -> x1)
- OUI : Utilise la carte globale assemblée pour la résolution précédente
- OUI : Anomalies calculées par rapport à l'interpolation de cette carte
- OUI : Agrégation avec gestion des overlaps (weighted sum)

**MAIS** : Cette logique ne peut fonctionner que si l'entraînement a bien appris les résidus, ce qui n'est pas le cas actuellement à cause du problème d'imbrication !

---

## 5. CE QUI FONCTIONNE ACTUELLEMENT

### 5.1 Architecture GradSolver
- Les 3 solvers (x10, x3, x1) sont bien définis
- L'optimisation à deux niveaux fonctionne
- La loss finale (MSE + gradient + prior) est correcte

### 5.2 Logique d'Anomalies
- `update_batch_as_anomaly` convertit correctement les observations en anomalies
- Les résidus sont prédits (quand l'interpolation fonctionne)
- La reconstruction additive est correcte

### 5.3 Test Single-Day Mode
- Le mode test single-day fonctionne (1577 batches au lieu de 207k)
- Toutes les résolutions utilisent le même jour cible

### 5.4 Interpolation Géométrique
- La fonction `interpolate_torch` est correcte
- Utilise `RegularGridInterpolator` avec coordonnées géographiques
- Retourne NaN pour les points hors limites (comportement attendu)

### 5.5 Agrégation Test
- L'agrégation weighted sum avec gestion des overlaps fonctionne
- Les cartes globales sont assemblées correctement

---

## 6. SOLUTION RETENUE : Extraction Géographique Smart

### 6.1 Principe Général

**Stratégie** : Balayer **TOUS** les patchs x1 du globe et, pour chacun, trouver les patchs x3 et x10 qui l'englobent géographiquement.

**Points Clés** :
1. La grille x1 est la **référence** -> Tous les patchs x1 sont utilisés
2. x3 et x10 **englobent** x1, mais x1 n'est **pas forcément centré** (gestion des bords)
3. Recherche basée sur **coordonnées géographiques** (lat_geo, lon_geo)
4. Calcul **à la volée** (pas de pré-calcul, réaliste pour l'inférence)
5. L'interpolation géométrique gère l'alignement pixel-parfait -> **Pas besoin de positional encoding** (?? ah bon a eclaircir

---

### 6.2 Cas d'Usage

#### Cas 1 : Patch x1 au centre du globe
```
+---------------------------------------+
|          Patch x10                    |
|    +---------------------------+      |
|    |      Patch x3             |      |
|    |   +-----------------+     |      |
|    |   |   Patch x1      |     |      | <- x1 centré
|    |   +-----------------+     |      |
|    +---------------------------+      |
+---------------------------------------+
```
x1 est centré dans x3, x3 est centré dans x10

#### Cas 2 : Patch x1 au bord Nord
```
+---------------------------------------+
+---------------------------+           |
|   +-----------------+     |           |
|   |   Patch x1      |     |           | <- x1 décentré (bord Nord)
|   +-----------------+     |           |
|        Patch x3           |           |
+---------------------------+           |
|            Patch x10                  |
+---------------------------------------+
```
x1 au bord Nord -> x3 aussi au bord Nord -> x10 aussi au bord Nord
Pas de problème : l'interpolation géométrique gère l'alignement

#### Cas 3 : Patch x1 dans un coin
```
+---------------------------------------+
|          Patch x10                    |
+------------------+                    |
|   Patch x3       |                    |
| +----------+     |                    |
| | Patch x1 |     |                    | <- x1 dans le coin NO
| +----------+     |                    |
+------------------+                    |
+---------------------------------------+
```
Fonctionne aussi ! x3 et x10 s'ajustent au coin

---

**FIN DU MÉMO**

_Ce document doit être relu et mis à jour régulièrement au fur et à mesure de l'avancement du projet._


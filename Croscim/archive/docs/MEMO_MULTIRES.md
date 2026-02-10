# MÉMO MULTI-RÉSOLUTION SST - État des Lieux

**Date**: 3 Décembre 2025  
**Objectif**: Document de référence pour comprendre la gestion des patchs multi-résolution

---

## 1. OBJECTIF DU PROJET

### 1.1 But Final
Reconstruire la SST (Sea Surface Temperature) sur **tout le globe** en haute résolution (x1 = 5km/pixel) à partir d'observations satellites incomplètes.

### 1.2 Contrainte
- Le réseau ne peut traiter que des patchs de **256×256 pixels** à la fois (mémoire GPU limitée)
- Les observations satellites ont des **trous** (nuages, pas de passage satellite, etc.)
- Besoin de capturer les **relations multi-échelles** (grande échelle -> détails fins)

### 1.3 Solution Envisagée
Architecture **multi-résolution en cascade** :
- **x10** (50km/px) : Capture les grandes échelles océaniques (courants, fronts)
- **x3** (15km/px) : Ajoute les échelles méso (tourbillons moyens)
- **x1** (5km/px) : Ajoute les détails fins (petits tourbillons, fronts précis)

---

## 2. ARCHITECTURE THÉORIQUE (Ce qu'on VEUT)

### 2.1 Mode TRAINING (Actuel)

#### Principe
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

**IMPORTANT** : Les 3 patchs doivent couvrir la **MÊME zone géographique** !
- x3 est au **centre** de x10
- x1 est au **centre** de x3

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

**Note Cruciale** : Chaque résolution a ses **PROPRES observations satellites** ! 
- Les obs x1 ne sont PAS les obs x10 interpolées
- Ce sont de vraies observations à cette résolution (plus denses spatialement)

---

### 2.2 Mode TEST (Futur - Ce qu'on veut faire)

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
│  GLOBE x1 (3600×7200 pixels)        │
│  -> Idem que x3                      │
└─────────────────────────────────────┘
```

**Logique Additive Pure** : C'est une **reconstruction en cascade**, pas une prédiction jointe.

---

## 3. CODE ACTUEL - ANALYSE DÉTAILLÉE

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

#### Système d'Indexation (LE PROBLÈME !)
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

### 3.2 Code Training Actuel

#### `XrDatasetMultiRes.__getitem__` (data_multires.py:~370)
```python
def __getitem__(self, idx):
    # 1. Calculer slices basées sur idx
    sl = {dim: slice(...) for ...}  # Basé sur grille x1 !
    
    # 2. Extraire patch x1
    hr_sample = super().__getitem__(idx)  # Appelle XrDataset x1
    
    # 3. Extraire patchs x3 et x10
    for factor in [3, 10]:
        enlarged_patch = self.extract_enlarged_patch_from_datasets(sl, factor)
        # PROBLÈME : utilise les mêmes slices pour des grilles différentes !
    
    return {
        'patch_x1': hr_sample,
        'patch_x3': enlarged_patch_x3,
        'patch_x10': enlarged_patch_x10
    }
```

#### `extract_enlarged_patch_from_datasets` (data_multires.py:~120)
```python
def extract_enlarged_patch_from_datasets(self, sl, factor):
    # PROBLÈME : Calcul du centre basé sur indices pixels x1
    lat_center_x1 = (sl["lat"].start + sl["lat"].stop) // 2
    lon_center_x1 = (sl["lon"].start + sl["lon"].stop) // 2
    
    # Division par factor (ne préserve PAS la zone géographique!)
    lat_center = lat_center_x1 // factor
    lon_center = lon_center_x1 // factor
    
    # Exemple :
    # x1 : lat_center_x1 = 384  -> géo ~-50°
    # x10: lat_center = 384//10 = 38 -> géo ~-70° <- PAS LA MÊME ZONE !
```

**Conséquence** : Les 3 patchs sont à des **endroits totalement différents** du globe !

---

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

**État Actuel** : 
- OUI : La logique d'anomalies est correcte
- OUI : La reconstruction additive fonctionne
- NON : Mais l'interpolation échoue car les patchs ne sont pas imbriqués géographiquement
- NON : Résultat : 100% NaN dans les prédictions x3 et x1

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

## 4. PROBLÈMES IDENTIFIÉS

### 4.1 Problème MAJEUR : Patchs Non-Imbriqués (TRAINING)

**Symptôme** :
```
[INTERP BOUNDS WARNING] Batch 0:
  Source lat: [-94.07, -81.32]   # Patch x10
  Target lat: [-88.72, -75.97]   # Patch x3  <- Pas vraiment imbriqué
  
  Source lon: [-165.18, -152.43]
  Target lon: [-115.73, -102.97]  <- Complètement ailleurs !
```

**Cause** :
- Chaque dataset (x1, x3, x10) a sa propre grille de patchs
- Le même `idx` pointe vers des zones géographiques différentes
- `extract_enlarged_patch_from_datasets` utilise des indices de pixels au lieu de coordonnées géographiques

**Conséquence** :
- L'interpolation x10->x3 et x3->x1 retourne 100% NaN
- Les prédictions x3 et x1 sont full NaN
- Le réseau n'apprend RIEN sur les résidus multi-échelles

**Impact** :
- Training complètement cassé pour x3 et x1
- Impossible de tester la logique multi-résolution

---

### 4.2 Gestion des Bords du Globe

**Question Non Résolue** : Comment gérer les patchs qui tombent sur les bords du globe ?

**Cas Problématiques** :
1. **Pôles** : Singularité, projection
2. **Ligne de changement de date** (180°/-180°)
3. **Bords Nord/Sud** (±90°)

**Solution Actuelle** : Aucune gestion spécifique -> Risque de patchs invalides

---

### 4.3 Information de Position Spatiale

**Question** : Comment le réseau sait-il **où** se trouve le patch x1 dans le patch x10 ?

**Réponse Actuelle** : Il ne le sait PAS !
- Les channels d'entrée contiennent `lat` et `lon` normalisés
- MAIS ces coordonnées sont normalisées **localement** au patch (pas globales)
- Le réseau n'a aucune information sur la position relative x1/x3/x10

**Conséquence** : Le réseau ne peut pas apprendre de relations spatiales cohérentes multi-échelles.

**Solution Envisagée** : Ajouter des coordonnées **globales** (pas normalisées) ou un encoding de position.

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
5. L'interpolation géométrique gère l'alignement pixel-parfait -> **Pas besoin de positional encoding**

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
|          Patch x10                    |
+---------------------------+           |
|      Patch x3             |           |
|   +-----------------+     |           |
|   |   Patch x1      |     |           | <- x1 décentré (bord Nord)
|   +-----------------+     |           |
+---------------------------+           |
|                                       |
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
|                                       |
+---------------------------------------+
```
Fonctionne aussi ! x3 et x10 s'ajustent au coin

---

### 6.3 Algorithme Détaillé

#### Fonction `find_encompassing_patch_smart`

```python
def find_encompassing_patch_smart(dataset, inner_lat_bounds, inner_lon_bounds, res_name=""):
    """
    Trouve le patch qui englobe au mieux les bounds données.
    Privilégie le patch où inner est le plus centré (meilleur contexte).
    
    Args:
        dataset: Dataset (x3 ou x10)
        inner_lat_bounds: (min, max) latitude du patch intérieur
        inner_lon_bounds: (min, max) longitude du patch intérieur
        res_name: Nom pour debug (ex: "x3", "x10")
    
    Returns:
        patch: Le patch trouvé qui englobe inner_bounds
        idx: Index du patch dans le dataset
    
    Raises:
        ValueError: Si aucun patch n'englobe inner_bounds
    """
    inner_lat_center = (inner_lat_bounds[0] + inner_lat_bounds[1]) / 2
    inner_lon_center = (inner_lon_bounds[0] + inner_lon_bounds[1]) / 2
    
    best_patch = None
    best_idx = None
    best_score = -float('inf')
    
    # Parcourir tous les patchs du dataset
    for idx in range(len(dataset)):
        patch = dataset[idx]
        
        # Extraire bounds géographiques du patch
        patch_lat_min = float(patch['lat_geo'].min())
        patch_lat_max = float(patch['lat_geo'].max())
        patch_lon_min = float(patch['lon_geo'].min())
        patch_lon_max = float(patch['lon_geo'].max())
        
        # Vérifier si le patch englobe complètement inner_bounds
        lat_ok = (patch_lat_min <= inner_lat_bounds[0]) and (patch_lat_max >= inner_lat_bounds[1])
        lon_ok = (patch_lon_min <= inner_lon_bounds[0]) and (patch_lon_max >= inner_lon_bounds[1])
        
        if lat_ok and lon_ok:
            # Score de centrage : privilégier patchs où inner est centré
            # (meilleur contexte autour de la zone d'intérêt)
            patch_lat_center = (patch_lat_min + patch_lat_max) / 2
            patch_lon_center = (patch_lon_min + patch_lon_max) / 2
            
            lat_dist = abs(patch_lat_center - inner_lat_center)
            lon_dist = abs(patch_lon_center - inner_lon_center)
            score = -(lat_dist + lon_dist)  # Plus proche du centre = meilleur
            
            if score > best_score:
                best_score = score
                best_patch = patch
                best_idx = idx
    
    if best_patch is None:
        raise ValueError(
            f"Aucun patch {res_name} trouvé pour englober "
            f"lat=[{inner_lat_bounds[0]:.2f}, {inner_lat_bounds[1]:.2f}], "
            f"lon=[{inner_lon_bounds[0]:.2f}, {inner_lon_bounds[1]:.2f}]"
        )
    
    return best_patch, best_idx
```

#### Modification de `__getitem__`

```python
def __getitem__(self, idx):
    """
    Extraction multi-résolution avec imbrication géographique garantie.
    
    Stratégie :
    1. Extraire patch x1 (grille x1 = référence complète du globe)
    2. Trouver patch x3 qui englobe géographiquement x1
    3. Trouver patch x10 qui englobe géographiquement x3
    4. L'interpolation géométrique alignera pixel-parfaitement les grilles
    """
    # 1. Extraire patch x1 (référence)
    patch_x1 = self.datasets[1][idx]
    
    # Extraire bounds géographiques de x1
    lat_bounds_x1 = (
        float(patch_x1['lat_geo'].min()),
        float(patch_x1['lat_geo'].max())
    )
    lon_bounds_x1 = (
        float(patch_x1['lon_geo'].min()),
        float(patch_x1['lon_geo'].max())
    )
    
    # 2. Trouver patch x3 qui englobe x1
    patch_x3, idx_x3 = self.find_encompassing_patch_smart(
        self.datasets[3],
        lat_bounds_x1,
        lon_bounds_x1,
        res_name="x3"
    )
    
    # 3. Trouver patch x10 qui englobe x3
    lat_bounds_x3 = (
        float(patch_x3['lat_geo'].min()),
        float(patch_x3['lat_geo'].max())
    )
    lon_bounds_x3 = (
        float(patch_x3['lon_geo'].min()),
        float(patch_x3['lon_geo'].max())
    )
    
    patch_x10, idx_x10 = self.find_encompassing_patch_smart(
        self.datasets[10],
        lat_bounds_x3,
        lon_bounds_x3,
        res_name="x10"
    )
    
    # 4. Retourner le triplet imbriqué
    return {
        'patch_x1': patch_x1,
        'patch_x3': patch_x3,
        'patch_x10': patch_x10
    }
```

---

### 6.4 Pourquoi Pas de Positional Encoding ?

**Question** : Le réseau a-t-il besoin de savoir OÙ se trouve x1 dans x10 ?

**Réponse** : **NON** ! Voici pourquoi :

1. **L'interpolation géométrique fait le job**
   - `interpolate_torch` utilise `lat_geo` et `lon_geo`
   - Alignement **pixel-parfait** entre grilles
   - Chaque pixel x3 sait EXACTEMENT de quels pixels x10 il vient

2. **Le réseau reçoit déjà l'info via les canaux géographiques**
   - `lat` et `lon` (normalisés localement)
   - `lat_geo` et `lon_geo` (coordonnées absolues globales)
   - Le réseau peut apprendre la position via ces canaux

3. **Positional encoding = complexité inutile (pour l'instant)**
   - Ajout de canaux supplémentaires
   - Risque de confusion si mal calibré
   - On pourra l'ajouter PLUS TARD si le réseau a du mal

**Décision** : On commence simple. Si le réseau ne converge pas bien, on pourra ajouter du positional encoding dans une v2.

---

### 6.5 Avantages de Cette Solution

**Couverture complète** : Tous les patchs x1 utilisés (pas de zone oubliée)
**Gestion des bords** : Naturelle, pas de cas spéciaux
**Robustesse** : Basée sur géographie, pas sur indices pixels
**Simplicité** : Pas de pré-calcul, pas de positional encoding complexe
**Réaliste** : Calcul à la volée -> applicable en inférence temps réel
**Flexibilité** : x1 peut être n'importe où dans x3 (centré ou pas)

---

### 6.6 Inconvénients et Mitigations

**Performance** : Recherche linéaire O(n) pour chaque patch
   -> À surveiller. Si trop lent, on pourra optimiser avec un index spatial

**Certains x1 peuvent avoir le même x3/x10**
   -> Pas grave ! C'est normal aux bords. Le réseau verra juste plusieurs contextes x10 similaires

**Bords du globe (pôles, ligne de date)**
   -> À tester. Peut nécessiter une gestion spéciale si problèmes

---

## 7. PLAN D'IMPLÉMENTATION

### Phase 1 : Implémentation du Fix

#### Étape 2.1 : Créer `find_encompassing_patch`
**Fichier** : `contrib/SST/data_multires.py`

**Actions** :
1. Ajouter la méthode dans `XrDatasetMultiResSingleDay`
2. Parcourir tous les patchs du dataset cible
3. Vérifier l'englobement géographique (lat/lon bounds)
4. Calculer un score de centrage (privilégier patchs où inner est centré)
5. Retourner le meilleur patch + son index

**Points d'attention** :
- [ ] Gérer le cas où **aucun patch n'englobe** (ValueError explicite)
- [ ] Convertir tensors en float pour comparaisons (`float(tensor.min())`)
- [ ] Gestion de la **ligne de changement de date** (180°/-180°) ?
- [ ] Gestion des **pôles** (singularités) ?
- [ ] Logging/debug pour vérifier que les bounds sont cohérentes

---

#### Étape 2.2 : Modifier `__getitem__`
**Fichier** : `contrib/SST/data_multires.py`

**Actions** :
1. Supprimer l'ancien code `extract_enlarged_patch_from_datasets`
2. Extraire patch_x1 via `super().__getitem__(idx)`
3. Extraire bounds géo de x1 : `(lat_geo.min(), lat_geo.max())`
4. Appeler `find_encompassing_patch_smart` pour x3
5. Extraire bounds géo de x3
6. Appeler `find_encompassing_patch_smart` pour x10
7. Retourner triplet `{'patch_x1': ..., 'patch_x3': ..., 'patch_x10': ...}`

**Points d'attention** :
- [ ] S'assurer que `lat_geo`/`lon_geo` existent dans tous les patchs
- [ ] Gérer proprement les exceptions (ValueError -> skip patch ou crash ?)
- [ ] Ajouter des **logs de debug** pour vérifier imbrication :
  ```python
  print(f"[NESTING] idx={idx}")
  print(f"  x1:  lat=[{lat_x1_min:.2f}, {lat_x1_max:.2f}]")
  print(f"  x3:  lat=[{lat_x3_min:.2f}, {lat_x3_max:.2f}]")
  print(f"  x10: lat=[{lat_x10_min:.2f}, {lat_x10_max:.2f}]")
  ```
- [ ] Vérifier que l'imbrication est correcte : `lat_x10_min <= lat_x3_min <= lat_x1_min`

---

#### Étape 2.3 : Tests Unitaires
**Fichier** : Créer `tests/test_multires_nesting.py` (optionnel mais recommandé)

**Tests à implémenter** :
1. **Test imbrication géographique** :
   - Tirer 10 patchs aléatoires
   - Vérifier que x3 englobe x1
   - Vérifier que x10 englobe x3

2. **Test bords du globe** :
   - Forcer extraction d'un patch x1 au bord Nord
   - Vérifier qu'on trouve un x3/x10 valide
   - Idem pour bord Sud, Est, Ouest

3. **Test interpolation** :
   - Vérifier que `interpolate_torch` ne retourne plus 100% NaN
   - Vérifier que les bounds source/target se chevauchent

**Points d'attention** :
- [ ] Cas où **pas de x3/x10 englobant** -> Comment gérer ?
- [ ] **Performance** : Mesurer le temps de `__getitem__` (doit rester < 0.5s)
- [ ] **Reproductibilité** : Fixer seed pour tests déterministes

---

### Phase 3 : Validation Training (OBJECTIF)

#### Étape 3.1 : Premier training de test
**Actions** :
1. Lancer training avec **1 epoch, 100 batches**
2. Vérifier que **pas de NaN** dans les prédictions x3/x1
3. Monitorer les losses :
   - `loss_x10` : Doit descendre
   - `loss_x3` : Doit descendre (pas rester à 0 ou exploser)
   - `loss_x1` : Doit descendre
4. Vérifier les diagnostics d'interpolation :
   - Plus de `[INTERP BOUNDS WARNING]` ?
   - `NaN count` dans `coarse_interp` doit être 0

**Points d'attention** :
- [ ] **Memory leak** ? (surveiller GPU memory avec `nvidia-smi`)
- [ ] **Vitesse** : epochs trop lentes -> besoin d'optimiser ?
- [ ] **Convergence** : losses descendent-elles vraiment ?
- [ ] **Overflow** : Valeurs qui explosent (inf, nan) ?

---

#### Étape 3.2 : Validation visuelle
**Actions** :
1. Sauvegarder quelques prédictions après 1 epoch
2. Visualiser :
   - `pred_x10` : Doit ressembler à la vraie SST (coarse)
   - `pred_x3` : Doit ajouter des détails par rapport à x10
   - `pred_x1` : Doit être encore plus détaillé
3. Comparer avec ground truth
4. Vérifier qu'il n'y a pas de **discontinuités** bizarres

**Points d'attention** :
- [ ] **Artefacts** aux bords des patchs ?
- [ ] **Patterns répétitifs** (signe d'overfitting ou bug) ?
- [ ] **Valeurs physiques** cohérentes (SST entre -2°C et 35°C) ?

---

### Phase 4 : Training Complet (SI PHASE 3 OK)

#### Étape 4.1 : Training sur plusieurs epochs
**Actions** :
1. Lancer training complet (10-20 epochs)
2. Surveiller métriques :
   - RMSE sur validation set
   - Gradient flow (pas de vanishing/exploding gradients)
3. Sauvegarder checkpoints régulièrement
4. Comparer avec baseline single-résolution

**Points d'attention** :
- [ ] **Overfitting** : Val loss remonte alors que train loss descend ?
- [ ] **Instabilité** : Losses qui oscillent beaucoup ?
- [ ] **Temps de training** : Trop long -> besoin d'optimiser dataset loading ?

---

### Phase 5 : Test & Inférence (VALIDATION FINALE)

#### Étape 5.1 : Test sur journée complète
**Actions** :
1. Lancer `test_step` sur une journée test
2. Vérifier l'agrégation multi-résolution :
   - Carte x10 globale
   - Carte x3 globale (x10 + résidus x3)
   - Carte x1 globale (x3 + résidus x1)
3. Sauvegarder en NetCDF
4. Calculer métriques globales (RMSE, corrélation, etc.)

**Points d'attention** :
- [ ] **Artefacts d'agrégation** : Bandes, discontinuités aux overlaps ?
- [ ] **Couverture** : Zones du globe manquantes ?
- [ ] **Qualité** : Amélioration réelle par rapport à x10 seul ?

---

### Phase 6 : Optimisations (SI NÉCESSAIRE)

#### Si `find_encompassing_patch_smart` est trop lent :

**Option A : Index spatial pré-calculé**
```python
def _build_spatial_index(self):
    """Construit un index spatial à l'init (une seule fois)."""
    self.spatial_index = {res: [] for res in [3, 10]}
    for res in [3, 10]:
        for idx in range(len(self.datasets[res])):
            patch = self.datasets[res][idx]
            bounds = (
                float(patch['lat_geo'].min()),
                float(patch['lat_geo'].max()),
                float(patch['lon_geo'].min()),
                float(patch['lon_geo'].max())
            )
            self.spatial_index[res].append((idx, bounds))
```

**Option B : R-tree spatial indexing** (librairie `rtree`)
```python
from rtree import index
# Créer index 2D (lat, lon)
# Requêtes d'englobement ultra-rapides
```

**Points d'attention** :
- [ ] **Mémoire** : Index spatial peut être gros (acceptable ?)
- [ ] **Maintenance** : Si datasets changent, recalculer index

---

## 8. CHECKLIST PRÉ-IMPLÉMENTATION

### Code à Modifier
- [ ] `contrib/SST/data_multires.py` :
  - [ ] Ajouter `find_encompassing_patch_smart()`
  - [ ] Modifier `XrDatasetMultiResSingleDay.__getitem__()`
  - [ ] Supprimer `extract_enlarged_patch_from_datasets()` (ancienne méthode)

### Code à Vérifier (Pas de Modif)
- [ ] `contrib/SST/data.py` : Vérifier que `lat_geo`/`lon_geo` sont bien présents
- [ ] `contrib/SST/models.py` : `interpolate_torch` doit rester inchangée
- [ ] `contrib/SST/models.py` : `multistep` doit rester inchangée

### Tests à Préparer
- [ ] Script de test rapide : extraire 10 patchs, vérifier imbrication
- [ ] Logging/debug pour visualiser les bounds
- [ ] Test d'un forward pass complet (1 batch) sans crash

### Risques Identifiés
1. **CRITIQUE : Pas de patch englobant** 
   - Cause probable : Patchs x1 trop proches des pôles
   - Solution : Skip ces patchs ou gérer spécialement

2. **MOYEN : Performance dégradée**
   - Cause : Recherche linéaire O(n) pour chaque patch
   - Solution : Surveiller, optimiser si nécessaire (index spatial)

3. **MOYEN : Ligne de changement de date (180°/-180°)**
   - Cause : Comparaison de longitudes peut échouer
   - Solution : Normaliser longitudes ou gérer cas spécial

4. **FAIBLE : Certains x1 réutilisent le même x3/x10**
   - Cause : Normal aux bords du globe
   - Impact : Aucun (juste moins de diversité)

### Points de Validation
Après implémentation, on doit vérifier :
- [ ] Interpolation ne retourne plus 100% NaN
- [ ] Bounds source/target se chevauchent toujours
- [ ] Prédictions x3/x1 ont des valeurs réalistes (pas NaN, pas inf)
- [ ] Losses descendent au training
- [ ] Temps de chargement acceptable (< 3s par patch)

---

**Prêt à implémenter ?** 

---

## 8. QUESTIONS EN SUSPENS

### 8.1 Training
- Comment gérer les patchs aux pôles ?
- Faut-il ajouter un encoding de position globale ?
- Comment équilibrer les résolutions dans le training progressif ?

### 8.2 Test
- Quel stride optimal pour couvrir le globe sans trop d'overlap ?
- Comment gérer les overlaps dans l'agrégation finale ?
- Faut-il un post-processing (lissage des bords, etc.) ?

### 8.3 Architecture
- Devrait-on passer à une architecture "end-to-end" où pred_x10 est input de solver_x3 ?
- Comment intégrer les incertitudes multi-échelles ?

---

## 9. RÉFÉRENCES CLÉS DANS LE CODE

### Fichiers Principaux
```
contrib/SST/
├── data.py              # XrDataset, is_valid_patch, TrainingItem
├── data_multires.py     # XrDatasetMultiRes, extract_enlarged_patch
├── models.py            # Lit4dVarNet_SST, multistep, test_step
├── solver.py            # GradSolver, BilinReconstructorPriorCost
└── load_data.py         # VAR_GROUPS, COVARIATES

config/xp/SST/
└── multires_lite.yaml   # Config training actuelle
```

### Fonctions Critiques
```python
# TRAINING
XrDatasetMultiRes.__getitem__()         # data_multires.py:~370
  -> extract_enlarged_patch_from_datasets()  # data_multires.py:~120  ATTENTION : CASSÉ

Lit4dVarNet_SST.multistep()             # models.py:~870
  -> interpolate_torch()                    # models.py:~540
  -> update_batch_as_anomaly()             # models.py:~690

# TEST
Lit4dVarNet_SST.test_step()             # models.py:~1485
  -> aggregate_batches()                   # models.py:~1210
```

---

## 10. GLOSSAIRE

- **Patch** : Région rectangulaire de 256×256 pixels extraite du globe
- **Résolution** : Taille d'un pixel en km (x10=50km, x3=15km, x1=5km)
- **Imbrication** : x3 au centre de x10, x1 au centre de x3
- **Anomalie** : Différence entre observation et prédiction coarse interpolée
- **Résidu** : Correction prédite par le réseau (anomalie apprise)
- **Agrégation** : Assemblage des patchs en carte globale (weighted sum des overlaps)
- **DAW** : Data Assimilation Window (fenêtre temporelle de 15 jours)

---

## Annexe : Correction du Bug de Coordonnées pour l'Imbrication Géographique

Lors de l'implémentation du système multi-résolution pour CROSCIM (patches imbriqués x1 ⊂ x3 ⊂ x10), un bug critique a été découvert empêchant l'imbrication géographique correcte des patches.

## Symptômes Observés
- Les patches de différentes résolutions ne s'imbriquaient pas géographiquement
- Coordonnées lat/lon complètement désalignées entre résolutions
- Le patch x1 (haute résolution) n'était pas contenu dans le patch x3, qui n'était pas contenu dans x10
- Échec du masque océanique (`mask_x1_ocean`) car les coordonnées ne correspondaient pas

## Diagnostic : Origine du Bug

### Problème dans `src/utils.py` fonction `extract_encompassing_patch()`

**Ligne problématique (357-363)** :
```python
# CODE BUGUÉ (utilisait les coordonnées locales du fichier zarr)
lon_subset = sst_ds.lon.values  # Coordonnées LOCALES du zarr
lat_subset = sst_ds.lat.values  # Coordonnées LOCALES du zarr
```

**Cause racine** :
- Les fichiers zarr prétraités (`sst_daily_x1/`, `sst_daily_x3/`, `sst_daily_x10/`) ont été créés avec des coordonnées **locales** qui commencent à 0
- Exemple : un patch extrait à `lon=[50:306]` a des coordonnées lon=`[0, 1, 2, ..., 255]` dans le zarr
- Lors de l'extraction d'un patch englobant, le code utilisait ces coordonnées locales au lieu des coordonnées globales de référence

### Impact
- Chaque résolution avait son propre système de coordonnées locales
- Impossible d'aligner géographiquement les patches
- Les masques océaniques basés sur lat/lon ne fonctionnaient pas

## Solution Implémentée

Modification dans `src/utils.py:357-363`
```python
# CODE CORRIGÉ (utilise la grille de référence globale)
lon_subset = lon_1d[lon_start:lon_end]  # Coordonnées GLOBALES
lat_subset = lat_1d[lat_start:lat_end]  # Coordonnées GLOBALES
```

Au lieu d'utiliser `sst_ds.lon.values` (coordonnées locales du zarr), on utilise maintenant les tranches de la **grille de référence globale** :
- `lon_1d` et `lat_1d` sont les coordonnées 1D de la grille globale (par exemple -180° à 180° pour longitude)
- `lon_start:lon_end` et `lat_start:lat_end` sont les indices d'extraction
- On assigne les bonnes coordonnées globales à chaque patch extrait

## Validation Post-Correction
```python
# Vérification de l'imbrication
patch_x1 = batch['patch_x1']
patch_x3 = batch['patch_x3']
patch_x10 = batch['patch_x10']

# Extraction des coordonnées
lon_x1 = patch_x1['lon_patch']
lon_x3 = patch_x3['lon_patch']
lon_x10 = patch_x10['lon_patch']

# Vérification : min(lon_x10) < min(lon_x3) < min(lon_x1)
assert lon_x10.min() <= lon_x3.min() <= lon_x1.min()
assert lon_x1.max() <= lon_x3.max() <= lon_x10.max()
```

### Résultats
- Toutes les assertions passent
- Coordonnées géographiques alignées
- Masques océaniques valides (>50% pixels océaniques dans les patches)

## Leçons Apprises

### Attention aux coordonnées locales vs globales
Lors de la création de datasets prétraités (zarr), **toujours conserver les coordonnées globales** pour éviter ce type de problème.

### Deux approches possibles
1. **Approche retenue** : Garder zarr avec coords locales, mais les remplacer par coords globales lors de l'extraction
2. **Approche alternative** : Modifier le preprocessing pour stocker directement les coords globales dans les zarr

--- 

**FIN DU MÉMO**

_Ce document doit être relu et mis à jour régulièrement au fur et à mesure de l'avancement du projet._

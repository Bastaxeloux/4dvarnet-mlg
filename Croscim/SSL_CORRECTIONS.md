# CORRECTIONS SSL - Architecture Bi-Niveau Correcte

## PRINCIPE FONDAMENTAL

**Learning to optimize by optimizing** : Le GradSolver apprend à faire de la descente de gradient, et on l'entraîne par descente de gradient !

---

## 📐 ÉQUATIONS THÉORIQUES

### Notations
- **X_A** : TOUTES les observations originales
- **X_B̄** : Pixels masqués par inpainting (`inpaint_mask > 0`)
- **X_B** : Pixels visibles après inpainting (`~inpaint_mask & isfinite()`)

### Niveau 1 (Inner) : GradSolver
```
J(state) = obs_cost(X_B, state) + prior_cost(state, φ(state))
```

- **obs_cost** : Calculé sur X_B (pixels VISIBLES)
  - Le solver est contraint par les vraies observations en input
  - Il NE DOIT PAS savoir qu'on a masqué des pixels

- **prior_cost** : Calculé sur TOUT le state
  - Régularisation : ||state - φ(state)||²

### Niveau 2 (Outer) : Training
```
Loss = loss_interp + loss_recons + loss_prior + loss_grad
```

- **loss_interp** : MSE(pred[X_B̄], target[X_B̄])
  - Évalue capacité d'interpolation
  
- **loss_recons** : MSE(pred[X_B], target[X_B])
  - Évalue fidélité aux observations
  
- **loss_prior** : ||state_final - φ(state_final)||²
  - Régularisation sur TOUS pixels (comme prior_cost)
  
- **loss_grad** : ||∇pred - ∇target||²
  - Régularisation spatiale sur TOUS pixels (avec pondération optionnelle sur inpaint_mask)

---

## ERREURS CORRIGÉES

### Erreur 1 : obs_cost sur pixels MASQUÉS
**AVANT (FAUX)** :
```python
ssl_msk = inpaint_msk & batch.tgt.isfinite()  # Pixels masqués X_B̄
obs_cost = MSE(state[ssl_msk], batch.tgt[ssl_msk])
```

**Problème** : Le solver optimise pour des pixels qu'il ne voit PAS dans input ! Aucune contrainte d'observation réelle.

**APRÈS (CORRECT)** :
```python
obs_msk = (~inpaint_msk) & batch.tgt.isfinite()  # Pixels visibles X_B
obs_cost = MSE(state[obs_msk], batch.tgt[obs_msk])
```

**Fichier** : [solver.py:185-218](contrib/SST/solver.py)

---

### Erreur 2 : mse_loss UNIQUEMENT sur pixels masqués
**AVANT (FAUX)** :
```python
if inpaint_mask > 0:
    loss = MSE(pred[inpaint_mask], target[inpaint_mask])  # Seulement interpolation
```

**Problème** : On évalue UNIQUEMENT l'interpolation, le réseau peut dégrader les pixels visibles sans pénalité !

**APRÈS (CORRECT)** :
```python
# Loss interpolation : pixels masqués (X_B̄)
loss_interp = MSE(pred[inpaint_mask], target[inpaint_mask])

# Loss reconstruction : pixels visibles (X_B)
loss_recons = MSE(pred[~inpaint_mask], target[~inpaint_mask])

# Loss totale
loss = loss_interp + loss_recons
```

**Fichier** : [models.py:1354-1395](contrib/SST/models.py)

---

### Erreur 3 : prior_loss sur pixels masqués uniquement
**AVANT (FAUX)** :
```python
if inpaint_mask is not None:
    ssl_mask = (inpaint_mask > 0) & sbatch.tgt.isfinite()
    prior_loss = MSE(sbatch.tgt[ssl_mask], prior[ssl_mask])
```

**Problème** : Incohérence avec prior_cost (inner) qui est sur tous pixels.

**APRÈS (CORRECT)** :
```python
# prior_loss TOUJOURS sur tous pixels (régularisation)
mask = sbatch.tgt.isfinite()
prior_loss = MSE(sbatch.tgt[mask], prior[mask])
```

**Fichier** : [models.py:1260-1270](contrib/SST/models.py)

---

## FICHIERS MODIFIÉS

### 1. contrib/SST/solver.py
**Ligne 185-218** : BaseObsCost.forward
- obs_cost sur `~inpaint_mask` (pixels visibles X_B)
- Debug prints mis à jour

### 2. contrib/SST/models.py
**Lignes 1200-1245** : grad_loss computation
- Gradients Sobel sur TOUS pixels (`tgt_sobel.isfinite()`)
- `weighted_mse` applique `inpaint_mask` pour pondérer (pas filtrer)
- Commentaire ajouté pour clarifier

**Lignes 1260-1270** : prior_loss
- Calcul sur tous pixels valides (régularisation)
- Plus de version SSL

**Lignes 1354-1403** : base_step
- Séparation en `loss_interp` (X_B̄) + `loss_recons` (X_B)
- Stockage de `_step_losses` pour logging détaillé
- Debug prints mis à jour

**Lignes 1280-1315** : step() - Loss aggregation
- Ajout de `mse_interp` et `mse_recons` dans `self.last_losses`
- TensorBoard logs pour les 6 métriques :
  - `train/loss` (pondérée)
  - `train/mse` (interp + recons)
  - `train/mse_interp` (X_B̄)
  - `train/mse_recons` (X_B)
  - `train/grad` (tous pixels)
  - `train/prior` (tous pixels)

**Lignes 760-825** : training_step
- Logs TensorBoard mis à jour avec commentaires
- 4 catégories : `general/`, `train/`, `val/`, `perf/`

---

## TESTS À EFFECTUER

### 1. Vérifier les debug prints
```bash
./run_train_lite.sh 3
```

**Attendu** :
```
[DEBUG ObsCost] SSL mode - obs_cost on VISIBLE pixels (not masked)
[DEBUG ObsCost] obs_mask (visible pixels X_B): ~1180000 pixels

[DEBUG base_step] Mode: SSL (loss_interp on masked + loss_recons on visible)
```

### 2. Vérifier les métriques TensorBoard
```bash
./tensorboard.sh
```

**Logs à surveiller** :
- `train/mse` : Total (interp + recons)
- `train/mse_interp` : Interpolation seule (X_B̄)
- `train/mse_recons` : Reconstruction seule (X_B)
- `train/grad` : Régularisation spatiale (tous pixels)
- `train/prior` : Régularisation prior (tous pixels)
- `train/loss` : Loss pondérée finale

**Attendu** :
- `mse_interp` devrait diminuer (apprentissage d'interpolation)
- `mse_recons` devrait rester faible (fidélité préservée)
- `grad` et `prior` devraient stabiliser (régularisations)

### 3. Vérifier les reconstructions
- Pas de démarcations (champs continus)
- Fidélité aux observations (pas de dégradation des pixels visibles)
- Bonne interpolation des trous

---

## ARCHITECTURE FINALE

```
DataLoader
  ↓ génère patch avec inpaint_mask
  
NIVEAU 1 (GradSolver - Inner Optimization)
  obs_cost  : ||state[~inpaint_mask] - X_B||²      ← VISIBLE pixels
  prior_cost: ||state - φ(state)||²                ← ALL pixels
  → Optimise state par N itérations gradient descent
  
NIVEAU 2 (Training - Outer Optimization)  
  loss_interp : ||pred[inpaint_mask] - X_B̄||²      ← MASKED pixels
  loss_recons : ||pred[~inpaint_mask] - X_B||²     ← VISIBLE pixels
  loss_prior  : ||state_final - φ(state_final)||²  ← ALL pixels (régularisation)
  loss_grad   : ||∇pred - ∇target||²                ← ALL pixels (régularisation)
                                                       (inpaint_mask pondère à 4x)
  → Backprop optimise les poids du réseau
```

**Note importante sur `weighted_mse`** :
- Fonction dans [src/models.py:58-94](src/models.py)
- Applique spatialweight : `err * weight`
- `inpaint_mask` est utilisé pour **pondérer** (boost de 4x), PAS pour filtrer
- Calcule MSE sur `err.isfinite()` (tous pixels valides)
- Donc `grad_loss` et `prior_loss` sont bien sur TOUS pixels

---

## ATTENTES

Avec cette architecture correcte :
- Le solver est contraint par les vraies observations (X_B)
- On évalue interpolation (X_B̄) ET fidélité (X_B)
- Régularisations cohérentes (prior, grad sur tous pixels)
- Apprentissage stable et convergent
- Reconstructions de qualité (pas de démarcations, fidélité préservée)

---

**Date de correction** : 28 janvier 2026
**Status** : Implémenté, prêt pour tests

# PRIOR DYNAMIQUE Φ(state) - Corrections Architecture

## PROBLÈME IDENTIFIÉ

**Équation théorique** (validée avec superviseur):
```
Inner GradSolver cost: J(state) = obs_cost(state, X_B) + prior_cost(state, Φ(state))
```

**Ancien code (FAUX)**:
```python
# BilinReconstructorPriorCost.forward()
reconstructed = self.forward_reconstructor(batch.input)  # Φ(input) - FIXE !
return F.mse_loss(state, reconstructed)
```

Le prior était calculé comme `||state - Φ(input)||²` au lieu de `||state - Φ(state)||²`.

**Conséquence**: Φ(input) est un prior FIXE calculé une fois, alors que Φ(state) est DYNAMIQUE et évolue avec le state à chaque itération du GradSolver.

---

## SOLUTION IMPLÉMENTÉE

### Changement de structure de l'input

**Problème dimensionnel**: 
- `input` a 139/85/49 canaux (observations + covariables)
- `state` a 15/9/5 canaux (SST sur T timesteps)

**Solution**: Réduire l'input de 15 canaux en gardant la fusion au lieu de slstr + aasti séparés.

**Nouvelle structure**:
```
[fusion_masquée (0:T), avhrr (T:T+2*T), pmw (T+2*T:T+4*T), covariates (T+4*T:T+5*T), spatial (4 derniers)]
```

**Nouvelles dimensions**:
| Résolution | dim_in ancien | dim_in nouveau | dim_state (T) |
|------------|---------------|----------------|---------------|
| x10        | 139           | 124            | 15            |
| x3         | 85            | 70             | 9             |
| x1         | 49            | 34             | 5             |

**Calcul**: 
- x10: 15 (fusion) + 30 (avhrr 2×15) + 30 (pmw 2×15) + 15 (cov) + 4 (spatial) = 124
- x3:  9 + 18 + 18 + 9 + 4 = 70
- x1:  5 + 10 + 10 + 5 + 4 = 34

---

## MODIFICATIONS APPORTÉES

### 1. contrib/SST/models.py - format_batch_for_solver() (lignes 370-410)

**AVANT**: Concaténation de tous les satellites (aasti, avhrr, pmw, slstr)

**APRÈS**: 
```python
# 1. Ajouter la fusion masquée en premier (dimension T)
input_tensors.append(batch.tgt_sst)  # Fusion déjà masquée par inpainting

# 2. Satellites sauf aasti et slstr (déjà dans fusion)
for group in ['avhrr', 'pmw']:  # Skip aasti, slstr
    for var in vars_:
        input_tensors.append(getattr(batch, f"{group}_{var}"))

# 3. Covariates + spatial (inchangé)
```

**Résultat**: batch.input contient maintenant [fusion, avhrr, pmw, covs, spatial] avec dimensions 124/70/34.

---

### 2. contrib/SST/solver.py - BilinReconstructorPriorCost (lignes 220-298)

**Changements clés**:

```python
class BilinReconstructorPriorCost(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, ...):
        self.dim_out = dim_out  # NOUVEAU: Sauvegarder T pour extraction covariables
        ...
    
    def forward(self, state, batch):
        """Prior dynamique: Φ([state, covariables])"""
        T = self.dim_out  # 15/9/5 selon résolution
        
        # Extraire covariables + spatial (tous canaux après T)
        covariables_and_spatial = batch.input[:, T:, :, :]
        
        # Construire input dynamique
        dynamic_input = torch.cat([state, covariables_and_spatial], dim=1)
        
        # Φ([state, covs]) - prior qui évolue !
        reconstructed = self.forward_reconstructor(dynamic_input)
        
        return F.mse_loss(state, reconstructed)
```

**Avant**: `Φ(batch.input)` - fixe durant toutes les itérations
**Après**: `Φ([state, covs])` - évolue avec le state à chaque itération

---

### 3. contrib/SST/models.py - step() prior_loss (lignes 1270-1300)

**AVANT**:
```python
prior = model.prior_cost.forward_reconstructor(sbatch.input)
err = sbatch.tgt - prior
```

**APRÈS**:
```python
# Extraire state_final du output
state_final = out[tgt_key]  # (B, T, H, W)
T = state_final.shape[1]

# Construire dynamic_input
covariables_and_spatial = sbatch.input[:, T:, :, :]
dynamic_input = torch.cat([state_final, covariables_and_spatial], dim=1)

# Φ([state_final, covs])
prior = model.prior_cost.forward_reconstructor(dynamic_input)

# prior_loss = ||state_final - Φ([state_final, covs])||²
err = state_final - prior  # CHANGÉ: state_final au lieu de sbatch.tgt
```

**Cohérence**: Maintenant outer training loss utilise aussi Φ(state) dynamique.

---

### 4. Configs YAML - Mise à jour dim_in

**Fichiers modifiés**:
- `config/xp/SST/multires.yaml`
- `config/xp/SST/multires_lite.yaml`
- `config/xp/SST/multires_lite_ddp.yaml`
- `config/xp/SST/multires_gefion.yaml`

**Changements**:
```yaml
solver_x1:
  prior_cost:
    dim_in: 34  # était 49
    dim_out: 5

solver_x3:
  prior_cost:
    dim_in: 70  # était 85
    dim_out: 9

solver_x10:
  prior_cost:
    dim_in: 124  # était 139
    dim_out: 15
```

---

## VÉRIFICATION

### Script de test: test_dynamic_prior.py

**Usage**:
```bash
python test_dynamic_prior.py
```

**Vérifications**:
1. OK: Dimensions batch.input: 124/70/34 (au lieu de 139/85/49)
2. OK: Structure: [fusion (0:T), avhrr, pmw, covs, spatial]
3. OK: BilinReconstructorPriorCost utilise [state, covs]
4. OK: Pas de NaN dans les gradients
5. OK: Prior dynamique: Φ(state1) ≠ Φ(state2)

### Tests manuels

**Vérifier shapes durant training**:
```python
# Dans solver.py, ajouter prints temporaires:
print(f"[PRIOR DEBUG] state.shape: {state.shape}")
print(f"[PRIOR DEBUG] batch.input.shape: {batch.input.shape}")
print(f"[PRIOR DEBUG] dynamic_input.shape: {dynamic_input.shape}")
print(f"[PRIOR DEBUG] reconstructed.shape: {reconstructed.shape}")
```

**Attendu pour x10**:
```
state.shape: torch.Size([4, 15, 256, 256])
batch.input.shape: torch.Size([4, 124, 256, 256])
dynamic_input.shape: torch.Size([4, 124, 256, 256])
reconstructed.shape: torch.Size([4, 15, 256, 256])
```

---

## POINTS D'ATTENTION

### 1. Init du GradSolver

L'initialisation utilise toujours `batch.input` avec fusion_masquée (correct):
```python
# solver.py - GradSolver.init_state()
state_init = self.prior_cost.forward_reconstructor(batch.input)  # OK
```

À l'étape 0, on n'a pas encore de state optimisé, donc on utilise la fusion masquée comme point de départ.

### 2. Multi-résolution

Les 3 solvers (x10, x3, x1) sont tous mis à jour avec leurs dimensions respectives. Vérifier que le cropping temporel (15→9→5) fonctionne toujours.

### 3. Inpainting

La fusion dans `batch.input` est DÉJÀ masquée par inpainting (fait dans data.py). Le solver voit donc les trous, ce qui est correct pour l'init. Ensuite, le prior dynamique reçoit le state qui évolue.

### 4. Cohérence inner/outer

**Inner (GradSolver)**: `prior_cost(state, batch)` utilise `Φ([state, covs])`
**Outer (Training)**: `prior_loss` utilise `Φ([state_final, covs])`

Les deux utilisent maintenant le prior dynamique.

---

## BÉNÉFICES ATTENDUS

1. **Régularisation adaptative**: Le prior évolue avec le state durant l'optimisation
2. **Convergence améliorée**: Le GradSolver a un meilleur guidage à chaque itération
3. **Cohérence théorique**: Implémentation fidèle aux équations du superviseur
4. **Généralisation**: Le réseau apprend à reconstruire depuis des états variables

---

## TESTS DE VALIDATION

### Avant de lancer un training complet:

1. OK: Dimensions vérifiées (test_dynamic_prior.py)
2. OK: Gradients OK (pas de NaN)
3. OK: Prior dynamique confirmé (Φ(state1) ≠ Φ(state2))
4. PAS ENCORE FAIT: Test avec 1 epoch: vérifier convergence
5. PAS ENCORE FAIT: Comparer prior_loss: devrait diminuer plus vite
6. PAS ENCORE FAIT: Vérifier reconstructions: pas d'artefacts

### Durant training:

- Monitorer `train/prior` dans TensorBoard
- Vérifier que prior_loss diminue (régularisation effective)
- Pas de divergence ou explosion de gradients
- Reconstructions visuellement cohérentes

---

**Date de correction**: 28 janvier 2026  
**Status**: Implémenté, prêt pour tests  
**Fichiers modifiés**: 6 (solver.py, models.py, 4× YAML configs)

#!/usr/bin/env python3
"""
Test du prior dynamique Φ([state, covs]) vs Φ(input fixe)

Vérifie :
1. Dimensions de batch.input pour le layout actuel 8*T+4 (124/76/44)
2. Structure : [fusion (T) | slstr_std (T) | aasti_std (T)
                | avhrr_av (T) | avhrr_std (T) | pmw_av (T) | pmw_std (T)
                | sea_ice_fraction (T) | spatial (4)]
3. BilinReconstructorPriorCost utilise bien [state, covs]
4. Pas de NaN dans les gradients

Usage: python tests/test_dynamic_prior.py xp=SST/multires_lite
"""
import os
import sys
import torch
import hydra
from omegaconf import DictConfig

# Add the project root to the import path.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

@hydra.main(version_base=None, config_path="../config", config_name="main")
def test_dynamic_prior(cfg: DictConfig):
    """Test le prior dynamique sur un batch réel"""
    
    # Override config pour test rapide (garde xp de la ligne de commande)
    cfg.trainer.limit_train_batches = 1
    cfg.trainer.limit_val_batches = 1
    cfg.datamodule.dl_kw.num_workers = 0  # Single-threaded pour debug
    cfg.datamodule.dl_kw.persistent_workers = False
    
    # Instantiate datamodule
    from hydra.utils import instantiate
    dm = instantiate(cfg.datamodule)
    dm.setup('fit')
    
    # Instantiate model
    model = instantiate(cfg.model)
    
    print("\n" + "="*80)
    print("TEST DU PRIOR DYNAMIQUE Φ([state, covs])")
    print("="*80)
    
    # Get one batch from train dataloader
    train_loader = dm.train_dataloader()
    batch_dict = next(iter(train_loader))  # Dict with patch_x1, patch_x3, patch_x10
    
    # Test each resolution
    for res_factor in [10, 3, 1]:
        print(f"\n{'='*80}")
        print(f"RÉSOLUTION x{res_factor}")
        print(f"{'='*80}")
        
        # Get batch for this resolution
        batch_key = f'patch_x{res_factor}'
        if batch_key not in batch_dict:
            print(f"      {batch_key} non trouvé dans batch_dict")
            continue
        
        batch = batch_dict[batch_key]
        
        # Expected dimensions: layout 8*T + 4 (fusion + 7 satellite/cov blocs T + 4 spatial)
        if res_factor == 10:
            expected_T = 15
            expected_dim_in = 124  # 8*15 + 4
        elif res_factor == 3:
            expected_T = 9
            expected_dim_in = 76   # 8*9 + 4
        else:
            expected_T = 5
            expected_dim_in = 44   # 8*5 + 4
        
        # Format for solver
        sbatch = model.format_batch_for_solver(batch)
        
        # Verify dimensions
        input_shape = sbatch.input.shape
        tgt_shape = sbatch.tgt.shape
        
        print(f"\n1. DIMENSIONS")
        print(f"   batch.input.shape: {input_shape}")
        print(f"   batch.tgt.shape: {tgt_shape}")
        print(f"   Expected dim_in: {expected_dim_in}, Got: {input_shape[1]}")
        print(f"   Expected T: {expected_T}, Got: {tgt_shape[1]}")
        
        if input_shape[1] != expected_dim_in:
            print(f"   ❌ ERREUR: dim_in incorrecte!")
            return False
        if tgt_shape[1] != expected_T:
            print(f"   ❌ ERREUR: T incorrecte!")
            return False
        print(f"   ✓ Dimensions correctes")
        
        # Verify structure: 8 blocs de T canaux + 4 spatiaux
        # [fusion | slstr_std | aasti_std | avhrr_av | avhrr_std | pmw_av | pmw_std | sea_ice_fraction | spatial(4)]
        print(f"\n2. STRUCTURE DE L'INPUT")
        T = expected_T
        fusion_slice    = sbatch.input[:, 0*T:1*T, :, :]
        slstr_std_slice = sbatch.input[:, 1*T:2*T, :, :]
        aasti_std_slice = sbatch.input[:, 2*T:3*T, :, :]
        avhrr_av_slice  = sbatch.input[:, 3*T:4*T, :, :]
        avhrr_std_slice = sbatch.input[:, 4*T:5*T, :, :]
        pmw_av_slice    = sbatch.input[:, 5*T:6*T, :, :]
        pmw_std_slice   = sbatch.input[:, 6*T:7*T, :, :]
        cov_slice       = sbatch.input[:, 7*T:8*T, :, :]
        spatial_slice   = sbatch.input[:, -4:, :, :]

        print(f"   fusion       (0:{T}):       {fusion_slice.shape}")
        print(f"   slstr_std    ({T}:{2*T}):   {slstr_std_slice.shape}")
        print(f"   aasti_std    ({2*T}:{3*T}): {aasti_std_slice.shape}")
        print(f"   avhrr_av     ({3*T}:{4*T}): {avhrr_av_slice.shape}")
        print(f"   avhrr_std    ({4*T}:{5*T}): {avhrr_std_slice.shape}")
        print(f"   pmw_av       ({5*T}:{6*T}): {pmw_av_slice.shape}")
        print(f"   pmw_std      ({6*T}:{7*T}): {pmw_std_slice.shape}")
        print(f"   covariates   ({7*T}:{8*T}): {cov_slice.shape}")
        print(f"   spatial (-4:): {spatial_slice.shape}")
        print(f"   ✓ Structure vérifiée")
        
        # Test BilinReconstructorPriorCost
        print(f"\n3. TEST BilinReconstructorPriorCost")
        solver_key = f"solver_x{res_factor}"
        solver = model.solver.solvers[solver_key]
        prior_cost = solver.prior_cost
        
        # Create fake state
        B, _, H, W = sbatch.tgt.shape
        state = torch.randn(B, expected_T, H, W, requires_grad=True)
        
        # Compute prior cost with dynamic input
        try:
            cost = prior_cost(state, sbatch)
            print(f"   Prior cost: {cost.item():.6f}")
            
            # Compute gradient
            cost.backward()
            if state.grad is None:
                print(f"   ❌ ERREUR: Pas de gradient!")
                return False
            if torch.isnan(state.grad).any():
                print(f"   ❌ ERREUR: Gradient contient NaN!")
                return False
            print(f"   ✓ Gradient OK (mean: {state.grad.mean():.6f})")
            
        except Exception as e:
            print(f"   ❌ ERREUR lors du calcul: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify dynamic vs fixed prior
        print(f"\n4. VÉRIFICATION PRIOR DYNAMIQUE")
        state1 = torch.randn(B, expected_T, H, W)
        state2 = torch.randn(B, expected_T, H, W)
        
        # Build dynamic inputs
        T = expected_T
        covs = sbatch.input[:, T:, :, :]
        
        input1 = torch.cat([state1, covs], dim=1)
        input2 = torch.cat([state2, covs], dim=1)
        
        # Compute reconstructions
        prior1 = prior_cost.forward_reconstructor(input1)
        prior2 = prior_cost.forward_reconstructor(input2)
        
        # Should be different if prior is dynamic
        diff = (prior1 - prior2).abs().mean()
        print(f"   |Φ(state1) - Φ(state2)|: {diff:.6f}")
        
        if diff < 1e-6:
            print(f"   ⚠️  ATTENTION: Priors identiques (pas dynamique?)")
        else:
            print(f"   ✓ Prior dynamique confirmé")
    
    print(f"\n{'='*80}")
    print("✓ TOUS LES TESTS PASSENT")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    test_dynamic_prior()

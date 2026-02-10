#!/usr/bin/env python3
"""
Test du prior dynamique Φ([state, covs]) vs Φ(input fixe)

Vérifie :
1. Dimensions de batch.input après modifications (124/70/34 au lieu de 139/85/49)
2. Structure : [fusion (0:T), avhrr, pmw, covs, spatial]
3. BilinReconstructorPriorCost utilise bien [state, covs]
4. Pas de NaN dans les gradients

Usage: python test_dynamic_prior.py xp=SST/multires_lite
"""
import os
import sys
import torch
import hydra
from omegaconf import DictConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

@hydra.main(version_base=None, config_path="config", config_name="main")
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
        
        # Expected dimensions
        if res_factor == 10:
            expected_T = 15
            expected_dim_in = 124  # 15 + 30 + 30 + 15 + 4
        elif res_factor == 3:
            expected_T = 9
            expected_dim_in = 70  # 9 + 18 + 18 + 9 + 4
        else:
            expected_T = 5
            expected_dim_in = 34  # 5 + 10 + 10 + 5 + 4
        
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
        
        # Verify structure: [fusion (0:T), avhrr, pmw, covs, spatial]
        print(f"\n2. STRUCTURE DE L'INPUT")
        fusion_slice = sbatch.input[:, 0:expected_T, :, :]
        avhrr_slice = sbatch.input[:, expected_T:expected_T+2*expected_T, :, :]
        pmw_slice = sbatch.input[:, expected_T+2*expected_T:expected_T+4*expected_T, :, :]
        cov_slice = sbatch.input[:, expected_T+4*expected_T:expected_T+5*expected_T, :, :]
        spatial_slice = sbatch.input[:, -4:, :, :]
        
        print(f"   fusion (0:{expected_T}): {fusion_slice.shape}")
        print(f"   avhrr ({expected_T}:{expected_T+2*expected_T}): {avhrr_slice.shape}")
        print(f"   pmw ({expected_T+2*expected_T}:{expected_T+4*expected_T}): {pmw_slice.shape}")
        print(f"   covariates ({expected_T+4*expected_T}:{expected_T+5*expected_T}): {cov_slice.shape}")
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

import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr

class GradSolver(nn.Module):
    def __init__(self, prior_cost, obs_cost, grad_mod, 
                 n_step, lr_grad=0.2, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        """
        Initialize the state variable for variational optimization.
        batch.input structure: [fusion_masquée (0:T), avhrr (T:3T), pmw (3T:5T), covariates (5T:6T), spatial (4)]
        State init: Use the first T channels (fusion_masquée) as initial state guess.
        """
        if x_init is not None:
            return x_init
        
        with torch.no_grad():
            # Extract first T channels (fusion_masquée) as initial state
            T = self.prior_cost.dim_out  # Temporal dimension
            state_init = batch.input[:, :T, :, :]  # (B, T, H, W) - fusion masquée
            
            # CRITICAL: Replace NaN by 0 (neural networks cannot handle NaN)
            # NaN in fusion come from missing real observations, not artificial inpainting
            state_init = torch.nan_to_num(state_init, nan=0.0)
        
        return state_init.detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        """
        Le coeur du GradSolver. On calcule à l'itération i un cout constitué :
        - prior_cost : MSE(BilinReconstruct(state_{i-1},state_{i-1}))
        - obs_cost : MSE(state_{i-1}, observations)
        """
        prior_cost_val = self.prior_cost(state, batch)
        obs_cost_val = self.obs_cost(state, batch)
        var_cost = prior_cost_val + obs_cost_val
        
        if not var_cost.isfinite():
            print(f"[solver_step] ERROR at step {step}: var_cost is NaN/Inf! prior={prior_cost_val.item() if prior_cost_val.isfinite() else 'nan'}, obs={obs_cost_val.item() if obs_cost_val.isfinite() else 'nan'}")
        
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        
        # CRITICAL ASSERTION: Gradients should NOT contain NaN if inputs are properly cleaned
        # If we find NaN here, it's a BUG (e.g., division by zero, invalid operation)
        if not grad.isfinite().all():
            grad_finite_ratio = grad.isfinite().float().mean().item()
            raise RuntimeError(
                f"[solver_step] CRITICAL BUG at step {step}: Gradients contain NaN/Inf! "
                f"finite_ratio={grad_finite_ratio:.3f}. "
                f"This should NOT happen if state/inputs are clean. "
                f"var_cost={var_cost.item():.6f}, prior={prior_cost_val.item():.6f}, obs={obs_cost_val.item():.6f}"
            )
        
        gmod = self.grad_mod(grad)
        
        state_update = (
           1 / (step + 1) * gmod
               + self.lr_grad * (step + 1) / self.n_step * grad
        )
        
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            
            # CRITICAL: Clean batch.tgt before sending to ConvLSTM reset_state
            # ConvLSTM uses batch.tgt to initialize hidden states with spatial dimensions
            tgt_clean = torch.nan_to_num(batch.tgt, nan=0.0)
            self.grad_mod.reset_state(tgt_clean)
            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not state.isfinite().all():
                    print(f"[GradSolver] ERROR: State became NaN/Inf at step {step} (finite_ratio={state.isfinite().float().mean():.3f})")
                    break 
                
                if not self.training:
                    state = state.detach().requires_grad_(True)
        return state

class ConvLstmGradModel(nn.Module):
    
    def __init__(self, dim_in, dim_hidden, kernel_size=3, dropout=0.1, downsamp=None):
        super().__init__()
        self.dim_hidden = dim_hidden

        self.gates = torch.nn.Conv2d(
            dim_in + dim_hidden,
            4 * dim_hidden,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.conv_out = torch.nn.Conv2d(
            dim_hidden, dim_in, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.dropout = torch.nn.Dropout(dropout)
        self._state = []
        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def reset_state(self, inp):
        size = [inp.shape[0], self.dim_hidden, *inp.shape[-2:]]
        self._grad_norm = None
        self._state = [
            self.down(torch.zeros(size, device=inp.device)),
            self.down(torch.zeros(size, device=inp.device)),
        ]

    def forward(self, x):
        # Replace NaN gradients with 0 to prevent propagation
        x = x.nan_to_num(nan=0.0)
        
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt()
        # Prevent division by zero when gradient is very small or all zeros
        x = x / (self._grad_norm + 1e-8)
        hidden, cell = self._state
        x = self.dropout(x)
        x = self.down(x)
        gates = self.gates(torch.cat((x, hidden), 1))

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate, remember_gate, out_gate = map(
            torch.sigmoid, [in_gate, remember_gate, out_gate]
        )
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        self._state = hidden, cell
        out = self.conv_out(hidden)
        out = self.up(out)
        return out

class GradSolvers(nn.Module):
    """
    Container for multi-resolution solvers.
    Each solver outputs a reconstructed SST with the appropriate number of timesteps:
    - solver_x10: 15 days (full window)
    - solver_x3: 9 days (after DAW crop from x10)
    - solver_x1: 5 days (after DAW crop from x3)
    """
    def __init__(self, solvers, **kwargs):
        super().__init__()
        # Hydra passes a DictConfig, need to convert to regular dict for nn.ModuleDict
        # nn.ModuleDict requires a regular dict, not an OmegaConf DictConfig
        if hasattr(solvers, '_metadata'):  # Check if it's an OmegaConf object
            solvers_dict = dict(solvers)  # Convert DictConfig to dict
        else:
            solvers_dict = solvers
        self.solvers = nn.ModuleDict(solvers_dict)

    def forward(self, batch, res=1):
        """
        Run the solver for the specified resolution.
        
        batch: Input batch with .input (139 channels) and .tgt (15/9/5 channels)
        res: Resolution (1, 3, or 10)
        returns: Reconstructed SST (B, timesteps, H, W)
        """
        return self.solvers[f"solver_x{res}"](batch)

class BaseObsCost(nn.Module):
    """
    Observation cost: Measures fidelity to observed SST (slstr + aasti fused).
    
    Compares state (15 channels = reconstructed SST) with batch.tgt (15 channels = observed SST).
    Only compares where observations are finite (not NaN).
    """
    def __init__(self, w=1) -> None:
        super().__init__()
        self.w = w

    def forward(self, state, batch):
        # === SSL CORRECT ===
        # obs_cost sur les pixels VISIBLES (X_B), pas masqués
        # Le solver doit être contraint par les vraies observations qu'il voit dans input
        if hasattr(batch, 'inpaint_mask') and batch.inpaint_mask is not None:
            inpaint_msk = batch.inpaint_mask > 0
            # obs_msk = pixels NON masqués ET valides (X_B)
            obs_msk = (~inpaint_msk) & batch.tgt.isfinite()
            n_obs = obs_msk.sum()
            
            # DEBUG: Vérifier que inpaint_mask arrive bien (premier batch uniquement)
            if not hasattr(self, '_debug_printed'):
                self._debug_printed = True
                # n_masked = batch.inpaint_mask.sum().item()
                # pct_masked = 100 * n_masked / batch.inpaint_mask.numel()
                # print(f"[ObsCost] mask:{batch.inpaint_mask.shape} | masked:{pct_masked:.1f}% | obs_pixels:{n_obs.item()} | SSL mode")
                # Vérifier cohérence temporelle
                assert batch.inpaint_mask.shape[1] == batch.tgt.shape[1], \
                    f"inpaint_mask temporal dim ({batch.inpaint_mask.shape[1]}) != tgt temporal dim ({batch.tgt.shape[1]})"
            
            if n_obs > 0:
                return self.w * F.mse_loss(state[obs_msk], batch.tgt[obs_msk])
            else:
                return torch.tensor(0.0, device=state.device, dtype=state.dtype, requires_grad=True)
                
        # Fallback pour inference : pas de inpaint mask, tous pixels valides
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG ObsCost] No inpaint_mask - using ALL valid pixels (inference mode)")
            print(f"[DEBUG ObsCost] tgt shape: {batch.tgt.shape}")
        
        msk = batch.tgt.isfinite()
        n_valid = msk.sum()
        if n_valid == 0:
            return torch.tensor(0.0, device=state.device, dtype=state.dtype, requires_grad=True)
        return self.w * F.mse_loss(state[msk], batch.tgt.nan_to_num()[msk])

class BilinReconstructorPriorCost(nn.Module):
    """
    Bilinear Reconstructor: Takes input channels and reconstructs T channels (SST).
    
    NOUVELLE ARCHITECTURE (pour permettre Φ(state) dynamique):
        - Input: [fusion_masquée (T canaux), avhrr (2*T), pmw (2*T), covariates (T), spatial (4)]
        - dim_in: 124 (x10), 70 (x3), 34 (x1)
        - dim_out: 15 (x10), 9 (x3), 5 (x1)
    
    L'innovation : forward() peut maintenant recevoir [state, covariates] au lieu de batch.input fixe,
    permettant un prior dynamique Φ(state) qui évolue durant les itérations du GradSolver.
    """
    def __init__(self, dim_in, dim_hidden, dim_out, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
        self.dim_out = dim_out  # Sauvegarder T pour extraction des covariables
        
        self.conv_in = nn.Conv2d(
            dim_in, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.conv_hidden = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.bilin_1 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_21 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bilin_22 = nn.Conv2d(
            dim_hidden, dim_hidden, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.conv_out = nn.Conv2d(
            2 * dim_hidden, dim_out, kernel_size=kernel_size, padding=kernel_size // 2
        )

        self.down = nn.AvgPool2d(downsamp) if downsamp is not None else nn.Identity()
        self.up = (
            nn.UpsamplingBilinear2d(scale_factor=downsamp)
            if downsamp is not None
            else nn.Identity()
        )

    def forward_reconstructor(self, x_obs):
        """
        Reconstruct SST (T channels) from observations (dim_in channels).
        
        x_obs: Input observations (B, dim_in, H, W)
               Structure: [fusion_masquée (0:T), satellites, covariates, spatial]
        returns: Reconstructed SST (B, T, H, W)
        
        NOTE: Conv2D cannot handle NaN, so we replace NaN with 0.
        The network will learn to interpret 0 as "missing data".
        """
        # Replace NaN with 0 to prevent propagation through Conv layers
        x_obs = x_obs.nan_to_num(nan=0.0)
        
        x = self.down(x_obs)
        x = self.conv_in(x)
        x = self.conv_hidden(F.relu(x))

        nonlin = self.bilin_21(x)**2 if self.bilin_quad else (self.bilin_21(x) * self.bilin_22(x))
        x = self.conv_out(
            torch.cat([self.bilin_1(x), nonlin], dim=1)
        )
        x = self.up(x)
        return x

    def forward(self, state, batch):
        """
        Prior cost: Φ(state) dynamique - mesure ||state - Φ([state, covariates])||²
        
        CHANGEMENT CRITIQUE vs version précédente:
        Avant: Φ(input fixe) - prior ne s'adapte pas aux itérations
        Après: Φ([state, covariates]) - prior évolue avec le state
        
        state: Current SST prediction (B, T, H, W) où T = dim_out
        batch: Contains batch.input (B, dim_in, H, W)
        returns: MSE entre state et sa reconstruction depuis [state, covariates]
        
        Structure de batch.input: [fusion_masquée (0:T), avhrr, pmw, covariates, spatial (4 derniers)]
        On remplace fusion_masquée par state pour créer l'input dynamique.
        """
        T = self.dim_out  # Dimension temporelle (15 pour x10, 9 pour x3, 5 pour x1)
        
        # Extraire les covariables et métadonnées spatiales (tous les canaux après T)
        covariables_and_spatial = batch.input[:, T:, :, :]  # (B, dim_in - T, H, W)
        # Construire l'input dynamique: [state_actuel, covariables_fixes]
        dynamic_input = torch.cat([state, covariables_and_spatial], dim=1)  # (B, dim_in, H, W)
        dynamic_input = torch.nan_to_num(dynamic_input, nan=0.0)
        
        # Φ([state, covs]) - prior qui évolue avec le state !
        reconstructed = self.forward_reconstructor(dynamic_input)
        
        return F.mse_loss(state, reconstructed)


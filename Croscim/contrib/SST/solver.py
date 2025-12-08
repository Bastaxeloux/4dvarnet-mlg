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
        """
        if x_init is not None:
            return x_init
        # Here we try to initialize with first-guess from BilinReconstructor
        with torch.no_grad():
            state_init = self.prior_cost.forward_reconstructor(batch.input)
        return state_init.detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        """
        Le coeur du GradSolver. On calcule à l'itération i un cout constitué :
        - prior_cost : MSE(BilinReconstruct(state_{i-1},state_{i-1}))
        - obs_cost : MSE(state_{i-1}, observations)
        """
        var_cost = self.prior_cost(state, batch) + self.obs_cost(state, batch)
        if not var_cost.isfinite():
            print(f"[solver_step] WARNING: var_cost is {var_cost.item():.4f} at step {step}")
            
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        if not grad.isfinite().all():
            print(f"[solver_step] WARNING: grad is {grad} at step {step}")
        
        gmod = self.grad_mod(grad)
        state_update = (
           1 / (step + 1) * gmod
               + self.lr_grad * (step + 1) / self.n_step * grad
        )
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            
            if not state.isfinite().all():
                print(f"[GradSolver] WARNING: init_state contains NaN/Inf!")
                print(f"  state finite ratio: {state.isfinite().float().mean():.3f}")
                print(f"  batch.tgt finite ratio: {batch.tgt.isfinite().float().mean():.3f}")
            
            self.grad_mod.reset_state(batch.tgt)
            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not state.isfinite().all() and step == 0:
                    print(f"[GradSolver] State became NaN/Inf at step {step}!")
                    print(f"  state finite ratio: {state.isfinite().float().mean():.3f}")
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
        x =  x / self._grad_norm
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
        """
        state: Predicted SST (B, 15, H, W)
        batch.tgt: Observed SST with gaps (B, 15, H, W)
        """
        msk = batch.tgt.isfinite()
        return self.w * F.mse_loss(state[msk], batch.tgt.nan_to_num()[msk])

class BilinReconstructorPriorCost(nn.Module):
    """
    Bilinear Reconstructor: Takes 139 input channels (all observations + auxiliaries)
    and reconstructs 15 channels (SST on 15 days).
    
    dim_in: Input channels (139 = satellites + covariates + spatial info)
    dim_out: Output channels (15 = SST on 15 days)
    dim_hidden: Hidden layer size
    """
    def __init__(self, dim_in, dim_hidden, dim_out, kernel_size=3, downsamp=None, bilin_quad=True, nt=None):
        super().__init__()
        self.nt = nt
        self.bilin_quad = bilin_quad
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
        Reconstruct SST (15 channels) from observations (139 channels).
        
        x_obs: Input observations (B, 139, H, W)
        returns: Reconstructed SST (B, 15, H, W)
        
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
        Prior cost: Measures how well state can be reconstructed from observations.
        
        state: Current SST prediction (B, 15, H, W)
        batch: Contains batch.input (B, 139, H, W) with all observations
        returns: MSE between state and reconstruction from observations
        """
        reconstructed = self.forward_reconstructor(batch.input)
        return F.mse_loss(state, reconstructed)


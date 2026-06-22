import os
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import kornia.filters as kfilts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import xarray as xr
from contrib.SST.model_components.grad_mods.convlstm import ConvLstmGradModel
from contrib.SST.model_components.priors.bilinear import BilinReconstructorPriorCost


_MEM_DEBUG_COUNT = 0


def _rank0():
    return os.environ.get("RANK", "0") == "0"


def _mem_debug_enabled():
    value = os.environ.get("CROSCIM_MEM_DEBUG", "0").lower()
    return value not in {"", "0", "false", "no", "off"}


def _mem_debug_trace_enabled():
    return os.environ.get("CROSCIM_MEM_DEBUG", "0").lower() == "trace"


def _cuda_snapshot():
    if not (_rank0() and _mem_debug_enabled() and torch.cuda.is_available()):
        return None
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
    }


def _cuda_mem(prefix):
    global _MEM_DEBUG_COUNT
    if not (_rank0() and _mem_debug_trace_enabled() and torch.cuda.is_available()):
        return
    max_logs = int(os.environ.get("CROSCIM_MEM_DEBUG_MAX_LINES", "300"))
    if _MEM_DEBUG_COUNT >= max_logs:
        return
    _MEM_DEBUG_COUNT += 1
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[CUDA MEM] {prefix} | "
        f"alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB max={max_allocated:.2f}GiB",
        flush=True,
    )


def _cuda_mem_summary(prefix, start_mem=None):
    mem = _cuda_snapshot()
    if mem is None:
        return
    start = ""
    if start_mem is not None:
        delta = mem["allocated"] - start_mem["allocated"]
        start = f" delta={delta:+.2f}GiB"
    print(
        f"[CUDA MEM SUMMARY] {prefix} | "
        f"alloc={mem['allocated']:.2f}GiB reserved={mem['reserved']:.2f}GiB "
        f"max={mem['max_allocated']:.2f}GiB{start}",
        flush=True,
    )


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
        _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} step={step} before prior")
        prior_cost_val = self.prior_cost(state, batch)
        _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} step={step} after prior")
        obs_cost_val = self.obs_cost(state, batch)
        var_cost = prior_cost_val + obs_cost_val
        
        if not var_cost.isfinite():
            print(f"[solver_step] ERROR at step {step}: var_cost is NaN/Inf! prior={prior_cost_val.item() if prior_cost_val.isfinite() else 'nan'}, obs={obs_cost_val.item() if obs_cost_val.isfinite() else 'nan'}")
        
        grad = torch.autograd.grad(var_cost, state, create_graph=self.training)[0]
        _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} step={step} after autograd")
        
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
        _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} step={step} after grad_mod")
        
        state_update = (
           1 / (step + 1) * gmod
               + self.lr_grad * (step + 1) / self.n_step * grad
        )
        
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            if _rank0() and _mem_debug_enabled() and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start_mem = _cuda_snapshot()
            state = self.init_state(batch)
            _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} forward start")
            
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
                    if hasattr(self.grad_mod, "detach_state"):
                        self.grad_mod.detach_state()
                    state = state.detach().requires_grad_(True)
                    _cuda_mem(f"GradSolver dim_out={self.prior_cost.dim_out} step={step} eval detached")
            mode = "train" if self.training else "eval"
            _cuda_mem_summary(
                f"GradSolver dim_out={self.prior_cost.dim_out} mode={mode} "
                f"steps={self.n_step} batch_shape={tuple(batch.input.shape)}",
                start_mem=start_mem,
            )
        return state

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

        batch: Input batch with .input (124 / 76 / 44 channels for x10 / x3 / x1)
               and .tgt (15 / 9 / 5 channels for x10 / x3 / x1).
        res: Resolution (1, 3, or 10)
        returns: Reconstructed SST (B, timesteps, H, W)
        """
        return self.solvers[f"solver_x{res}"](batch)

class BaseObsCost(nn.Module):
    """
    Observation cost: Measures fidelity to observed SST (slstr + aasti fused).

    Compares state (T channels = reconstructed SST, T = 15/9/5 selon résolution)
    avec batch.tgt (T channels = observed SST). Only compares where observations
    are finite (not NaN).
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

#!/usr/bin/env python3

"""
RÉFÉRENCE COMPLÈTE - Lit4dVarNet
================================

Ce fichier contient la version de référence de la classe Lit4dVarNet
pour la validation automatique des exercices.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

class Lit4dVarNet_Reference(pl.LightningModule):
    """Classe de référence pour la validation des exercices."""
    
    def __init__(self, solver, rec_weight, opt_fn, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        super().__init__()
        self.solver = solver
        
        if not isinstance(rec_weight, dict):
            self.register_buffer('rec_weight', torch.from_numpy(rec_weight), persistent=persist_rw)
        else:
            self.rec_weight = {}
            for key, weight_array in rec_weight.items():
                buffer_name = f"_rec_weight_{key}"
                weight_tensor = torch.from_numpy(weight_array).to("cuda")
                self.register_buffer(buffer_name, weight_tensor, persistent=persist_rw)
                self.rec_weight[key] = getattr(self, buffer_name)

        self.test_data = None
        self._norm_stats = norm_stats
        self.opt_fn = opt_fn
        self.metrics = test_metrics or {}
        self.pre_metric_fn = pre_metric_fn or (lambda x: x)

    @property
    def norm_stats(self):
        if self._norm_stats is not None:
            return self._norm_stats
        elif self.trainer.datamodule is not None:
            return self.trainer.datamodule.norm_stats()
        return (0., 1.)

    @staticmethod
    def weighted_mse(err, weight):
        err_w = err * weight[None, ...]
        non_zeros = (torch.ones_like(err) * weight[None, ...]) == 0.0
        err_num = err.isfinite() & ~non_zeros
        
        if err_num.sum() == 0:
            return torch.scalar_tensor(1000.0, device=err_num.device).requires_grad_()
        
        loss = F.mse_loss(err_w[err_num], torch.zeros_like(err_w[err_num]))
        return loss

    def forward(self, batch):
        return self.solver(batch)
    
    def base_step(self, batch, phase=""):
        out = self(batch=batch)
        loss = self.weighted_mse(out - batch.tgt, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")[0]

    def configure_optimizers(self):
        return self.opt_fn(self)

    def step(self, batch, phase=""):
        # Méthode simplifiée pour les exercices
        return self.base_step(batch, phase)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_data = []
        
        out = self(batch=batch)
        m, s = self.norm_stats

        self.test_data.append(torch.stack(
            [
                batch.input.cpu() * s + m,
                batch.tgt.cpu() * s + m,
                out.squeeze(dim=-1).detach().cpu() * s + m,
            ],
            dim=1,
        ))

    @property
    def test_quantities(self):
        return ['inp', 'tgt', 'out']

    def on_test_epoch_end(self):
        if isinstance(self.trainer.test_dataloaders, list):
            rec_da = self.trainer.test_dataloaders[0].dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )
        else:
            rec_da = self.trainer.test_dataloaders.dataset.reconstruct(
                self.test_data, self.rec_weight.cpu().numpy()
            )

        self.test_data = rec_da.assign_coords(
            dict(v0=self.test_quantities)
        ).to_dataset(dim='v0')

        if hasattr(self, 'metrics') and self.metrics:
            metric_data = self.test_data.pipe(self.pre_metric_fn)
            metrics = pd.Series({
                metric_n: metric_fn(metric_data) 
                for metric_n, metric_fn in self.metrics.items()
            })
            
            if self.logger:
                self.logger.log_metrics(metrics.to_dict())

        if self.logger:
            self.test_data.to_netcdf(Path(self.logger.log_dir) / 'test_data.nc')

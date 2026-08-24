#!/usr/bin/env python3
"""Regression tests for the optimized SST training hot path."""

import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from contrib.SST.model_components.grad_mods.convlstm import ConvLstmGradModel
from contrib.SST.model_components.priors.resunet import ResUNetPriorCost
from contrib.SST.models import Lit4dVarNet_SST, _masked_weighted_mse
from contrib.SST.solver import BaseObsCost, GradSolver
from src.models import Lit4dVarNet


RTOL = 1e-5
ATOL = 1e-7


class _CascadeHarness:
    multistep = Lit4dVarNet_SST.multistep

    def __init__(self, train_res):
        self.multires = [10, 3, 1]
        self.train_res = train_res
        self.calls = []
        self._step_losses = {}
        self.global_rank = 0

    def modify_multires_batch(self, batch):
        return batch

    def get_current_resolution_idx(self):
        return self.multires.index(self.train_res)

    def step(self, batch, res, phase=""):
        self.calls.append(("step", res, phase))
        return torch.tensor(float(res)), {"tgt_sst": torch.full((1, 1), float(res))}

    def _forward_resolution(self, batch, res):
        self.calls.append(("predict", res))
        return {"tgt_sst": torch.full((1, 1), float(res))}

    def interpolate_torch(self, values, lon_coarse, lat_coarse, lon_target, lat_target):
        coarse = int(lon_coarse.flatten()[0].item())
        target = int(lon_target.flatten()[0].item())
        self.calls.append(("interp", coarse, target))
        return {name: value.clone() for name, value in values.items()}

    def crop_daw(self, values, res):
        return values

    def update_batch_as_anomaly(self, batch, coarse):
        return batch

    def _track_time(self, name):
        pass

    def _log_residual_diagnostic(self, *args):
        pass


def _cascade_batch():
    return {
        f"patch_x{res}": SimpleNamespace(
            lon_geo=torch.full((1, 1, 1), float(res)),
            lat_geo=torch.full((1, 1, 1), float(res)),
        )
        for res in (10, 3, 1)
    }


def test_cascade_calls():
    expected_train_calls = {
        10: [("step", 10, "train")],
        3: [
            ("predict", 10),
            ("interp", 10, 3),
            ("step", 3, "train"),
        ],
        1: [
            ("predict", 10),
            ("interp", 10, 3),
            ("predict", 3),
            ("interp", 3, 1),
            ("step", 1, "train"),
        ],
    }
    for train_res, expected in expected_train_calls.items():
        harness = _CascadeHarness(train_res)
        harness.multistep(_cascade_batch(), "train")
        assert harness.calls == expected, (train_res, harness.calls)

    harness = _CascadeHarness(1)
    harness.multistep(_cascade_batch(), "val")
    assert [call[:2] for call in harness.calls if call[0] == "step"] == [
        ("step", 10),
        ("step", 3),
        ("step", 1),
    ]
    assert [call for call in harness.calls if call[0] == "interp"] == [
        ("interp", 10, 3),
        ("interp", 3, 1),
        ("interp", 10, 1),
    ]


def _old_weighted_mse(err, weight, inpaint_mask=None, inpaint_weight_factor=4.0):
    err_w = err * weight[None, ...]
    if inpaint_mask is not None:
        boost = 1.0 + (inpaint_weight_factor - 1.0) * inpaint_mask
        err_w = err_w * boost
    valid = err.isfinite() & ((torch.ones_like(err) * weight[None, ...]) != 0.0)
    if valid.sum() == 0:
        return torch.scalar_tensor(1000.0, device=err.device).requires_grad_()
    return F.mse_loss(err_w[valid], torch.zeros_like(err_w[valid]))


def _old_masked_weighted_mse(pred, target, weight, mask):
    valid_count = mask.sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    weighted_error = (pred - target) * weight[None, ...]
    return weighted_error[mask].square().sum() / valid_count


def _assert_value_and_gradient_match(old_fn, new_fn, source):
    old_source = source.detach().clone().requires_grad_(True)
    new_source = source.detach().clone().requires_grad_(True)
    old_loss = old_fn(old_source)
    new_loss = new_fn(new_source)
    torch.testing.assert_close(new_loss, old_loss, rtol=RTOL, atol=ATOL)
    old_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(new_source.grad, old_source.grad, rtol=RTOL, atol=ATOL)


def test_masked_losses():
    torch.manual_seed(20260821)
    source = torch.randn(2, 3, 4, 5)
    source[0, 0, 0, 0] = float("nan")
    weight = torch.rand(3, 4, 5)
    weight[:, 0, 0] = 0.0
    inpaint_mask = torch.randint(0, 2, source.shape, dtype=torch.float32)

    _assert_value_and_gradient_match(
        lambda value: _old_weighted_mse(value, weight, inpaint_mask, 4.0),
        lambda value: Lit4dVarNet.weighted_mse(value, weight, inpaint_mask, 4.0),
        source,
    )

    empty_source = torch.full_like(source, float("nan"), requires_grad=True)
    empty_loss = Lit4dVarNet.weighted_mse(empty_source, weight)
    torch.testing.assert_close(empty_loss, torch.tensor(1000.0))
    empty_loss.backward()
    assert torch.count_nonzero(empty_source.grad) == 0

    target = torch.randn_like(source)
    target[0, 0, 0, 0] = float("nan")
    mask = target.isfinite() & (torch.rand_like(target) > 0.35)
    _assert_value_and_gradient_match(
        lambda value: _old_masked_weighted_mse(value, target, weight, mask),
        lambda value: _masked_weighted_mse(value, target, weight, mask),
        source.nan_to_num(),
    )

    empty_mask = torch.zeros_like(mask)
    pred = source.nan_to_num().requires_grad_(True)
    empty_masked_loss = _masked_weighted_mse(pred, target, weight, empty_mask)
    torch.testing.assert_close(empty_masked_loss, torch.tensor(0.0))
    empty_masked_loss.backward()
    assert torch.count_nonzero(pred.grad) == 0

    obs_target = torch.randn_like(source)
    obs_target[0, 0, 0, 0] = float("nan")
    obs_inpaint = torch.randint(0, 2, source.shape)
    obs_batch = SimpleNamespace(tgt=obs_target, inpaint_mask=obs_inpaint)
    obs_mask = (obs_inpaint == 0) & obs_target.isfinite()
    _assert_value_and_gradient_match(
        lambda value: F.mse_loss(value[obs_mask], obs_target[obs_mask]),
        lambda value: BaseObsCost()(value, obs_batch),
        source.nan_to_num(),
    )

    empty_obs_batch = SimpleNamespace(
        tgt=obs_target,
        inpaint_mask=torch.ones_like(obs_target),
    )
    obs_state = source.nan_to_num().requires_grad_(True)
    empty_obs_loss = BaseObsCost()(obs_state, empty_obs_batch)
    torch.testing.assert_close(empty_obs_loss, torch.tensor(0.0))
    empty_obs_loss.backward()
    assert torch.count_nonzero(obs_state.grad) == 0


def test_resunet_numerics_debug():
    previous = os.environ.get("CROSCIM_NUMERICS_DEBUG")
    os.environ["CROSCIM_NUMERICS_DEBUG"] = "1"
    try:
        torch.manual_seed(20260821)
        prior = ResUNetPriorCost(
            dim_in=6,
            dim_hidden=4,
            dim_out=2,
            depth=2,
            norm_groups=2,
        )
        solver = GradSolver(
            prior_cost=prior,
            obs_cost=BaseObsCost(),
            grad_mod=ConvLstmGradModel(dim_in=2, dim_hidden=4, dropout=0.0),
            n_step=2,
        )
        solver.train()
        batch = SimpleNamespace(
            input=torch.randn(1, 6, 8, 8),
            tgt=torch.randn(1, 2, 8, 8),
            inpaint_mask=torch.zeros(1, 2, 8, 8),
        )
        state = solver(batch)
        state.retain_grad()
        state.square().mean().backward()
        assert torch.isfinite(state).all()
        assert state.grad is not None and torch.isfinite(state.grad).all()
        prior_grads = [p.grad for p in prior.parameters() if p.grad is not None]
        assert prior_grads
        assert all(torch.isfinite(grad).all() for grad in prior_grads)
    finally:
        if previous is None:
            os.environ.pop("CROSCIM_NUMERICS_DEBUG", None)
        else:
            os.environ["CROSCIM_NUMERICS_DEBUG"] = previous


def main():
    test_cascade_calls()
    test_masked_losses()
    test_resunet_numerics_debug()
    print("Training hot-path tests passed")


if __name__ == "__main__":
    main()

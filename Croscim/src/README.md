# src — Base framework

This is the general-purpose 4D-VarNet framework layer. The SST-specific code in `contrib/SST/` extends these base classes.

## Files

### `train.py`
Training entrypoints called by Hydra:
- `base_training()` — standard train then test
- `multi_dm_training()` — train with one datamodule, test with another

### `test.py`
Test entrypoint:
- `base_test()` — loads a checkpoint and runs the test loop. Used by the `run_test_checkpoint.sh` scripts.

### `models.py`
Base Lightning module and model components:
- **`Lit4dVarNet`** — base training loop (training_step, validation_step, test_step). Handles loss computation (MSE + Sobel gradient loss + prior loss), denormalization, metric logging, and test-time patch stitching. Extended by `contrib/SST/models.py`.
- **`ConvLstmGradModel`** — ConvLSTM network that takes raw gradients as input and outputs modulated gradients. Used by the GradSolver to learn a better descent direction than vanilla gradient descent.
- **`BaseObsCost`** — observation cost: MSE between the state and observed (non-NaN) pixels.

### `utils.py`
Utility functions used across the project:
- `encompassing_patch()` — finds the coarse-resolution patch that geographically contains a given fine-resolution patch. Handles pole wrapping and dateline crossings.
- `extract_encompassing_patch()` — higher-level wrapper that loads data for the encompassing patch.
- `get_linear_time_wei()` — creates temporal weighting for the optimization loss (used in `optim_weight`).
- `rmse_based_scores_from_ds()`, `psd_based_scores_from_ds()` — test metrics (RMSE, PSD-based spatial/temporal scales).

### `ConvLSTM.py`
Convolutional LSTM implementation:
- `ConvLSTMCell` — single cell with 4-gate LSTM using convolutions instead of dense layers.
- `ConvLSTM` — unrolls the cell over multiple time steps.

### `versioning_cb.py`
Lightning callback for tracking code version and git state during training.

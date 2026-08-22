"""Checkpoint callbacks for chained Slurm training jobs."""

from pathlib import Path
import signal

import pytorch_lightning as pl


class SignalCheckpoint(pl.Callback):
    """Checkpoint and stop at a batch boundary after receiving SIGUSR1."""

    def __init__(self, checkpoint_path, completion_path):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.completion_path = Path(completion_path)
        self.requested = False
        self.triggered = False
        self.previous_handler = None

    def _request(self, signum, frame):
        self.requested = True

    def on_fit_start(self, trainer, pl_module):
        self.previous_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self._request)

    def _save_if_requested(self, trainer):
        if not self.requested:
            return
        if trainer.is_global_zero:
            print(
                f"[SLURM SIGNAL] Saving checkpoint to {self.checkpoint_path} "
                "and stopping cleanly"
            )
        trainer.save_checkpoint(str(self.checkpoint_path))
        trainer.should_stop = True
        self.triggered = True
        self.requested = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        accumulation = max(1, int(trainer.accumulate_grad_batches))
        if (batch_idx + 1) % accumulation != 0:
            return
        self._save_if_requested(trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._save_if_requested(trainer)

    def on_fit_end(self, trainer, pl_module):
        completed_epochs = int(trainer.fit_loop.epoch_progress.current.completed)
        if self.triggered or completed_epochs < int(trainer.max_epochs):
            return
        if trainer.is_global_zero:
            self.completion_path.parent.mkdir(parents=True, exist_ok=True)
            self.completion_path.touch()
            print(f"[TRAIN COMPLETE] Marker written to {self.completion_path}")

    def teardown(self, trainer, pl_module, stage):
        if self.previous_handler is not None:
            signal.signal(signal.SIGUSR1, self.previous_handler)

from pytorch_lightning.callbacks import Callback

class FilterMetricsCallback(Callback):
    def __init__(self, metrics_to_filter=None):
        super().__init__()
        self.metrics_to_filter = metrics_to_filter or ['epoch', 'hp_metric']
    def on_train_epoch_end(self, trainer, pl_module):
        self._filter_logged_metrics(trainer)
    def on_validation_epoch_end(self, trainer, pl_module):
        self._filter_logged_metrics(trainer)
    def _filter_logged_metrics(self, trainer):
        if trainer.logger is not None and hasattr(trainer.logger, 'experiment'):
            pass

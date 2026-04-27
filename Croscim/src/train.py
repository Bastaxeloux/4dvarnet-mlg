import torch
import os
import random
import logging
import warnings
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers

warnings.filterwarnings("ignore", message=".*num_workers*")

def base_training(trainer, dm, lit_mod,
                  save_dir="/dmidata/users/malegu/results",
                  ckpt=None,test=True):

    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    if test:
        trainer.test(lit_mod, datamodule=dm, ckpt_path='best')

def multi_dm_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        lit_mod._norm_stats = test_dm.norm_stats()

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")

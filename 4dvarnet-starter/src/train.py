import torch
import os
import random
torch.set_float32_matmul_precision('high')
from pytorch_lightning import loggers
from pytorch_lightning.loggers import WandbLogger

def base_training(trainer, dm, lit_mod, 
                  save_dir="/Odyssey/private/m19beauc/DMI/results",
                  ckpt=None,test=True):

    # Garder le logger Wandb s'il existe, sinon utiliser TensorBoard
    if not isinstance(trainer.logger, loggers.WandbLogger):
        version = 'version_' + str(random.randint(0, 100000))
        logger_name = "lightning_logs"
        print(os.path.join(save_dir, logger_name, version))
        tb_logger = loggers.TensorBoardLogger(save_dir=save_dir,
                                                       name=logger_name,
                                                       version=version)
        trainer.logger = tb_logger

    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    if test:
        trainer.test(lit_mod, datamodule=dm, ckpt_path='best')
    
    # Finir le run Wandb proprement pour éviter les conflits dans multirun
    try:
        import wandb
        if wandb.run is not None:
            print("Fermeture du run Wandb...")
            wandb.finish()
    except Exception as e:
        print(f"Erreur lors de la fermeture de wandb: {e}")
        pass

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

import sys
import os
sys.path.append(os.path.dirname(__file__))
import hydra
import config
from omegaconf import OmegaConf

def main():
    # Configuration par défaut
    config_path = 'config/xp/BASE/test_base.yaml'
    overrides = [
        'xpd=outputs/2025-09-22/12-05-28',
        'trainer.accelerator=gpu',
        '+trainer.logger._target_=pytorch_lightning.loggers.CSVLogger',
        '+trainer.logger.save_dir=test_output',
        '+trainer.logger.name=test_logs',
        '+trainer.logger.version=""',
        'ckpt="results/lightning_logs/version_72723/checkpoints/val_mse=7.23063-epoch=077.ckpt"']
    with hydra.initialize(config_path="config", version_base='1.3'):
        cfg = hydra.compose(config_name="main", overrides=[f"xp=BASE/test_base"] + overrides)
        hydra.utils.call(cfg.entrypoints)
        
        
if __name__ == '__main__':
    main()
    
import hydra
from hydra import compose, initialize_config_dir
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf

def test_dataloader_with_filtering():
    """Test du DataLoader avec le filtrage activé"""

    print("TEST 1: Chargement de quelques batches avec filtrage")
    with initialize_config_dir(config_dir=str(Path.cwd() / 'config'), version_base=None):
        cfg = compose(config_name='main', overrides=['xp=SST/multires'])
    print("Instantiation du DataModule...")
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.setup('fit')
    
    train_dl = dm.train_dataloader()
    print(f"DataLoader créé: {len(train_dl)} batches par epoch\n")
    print("Chargement de 5 batches (avec filtrage activé)...")
    for i, batch in enumerate(train_dl):
        if i >= 5:
            break
        print(f"  Batch {i+1}/5: shape = {batch['tgt_sst'].shape}, "
              f"valid_ratio = {(~torch.isnan(batch['tgt_sst'])).float().mean():.2%}")
    
    dataset = train_dl.dataset
    if hasattr(dataset, '_rejection_count'):
        print(f"\nStatistiques de filtrage:")
        print(f"  Total rejets: {dataset._rejection_count}")
        print(f"  Taux de rejet: {dataset._rejection_count / (5 * cfg.datamodule.dl_kw.batch_size):.1%}")
    else:
        print("\nAucun patch rejeté (filtrage désactivé ou tous valides)")
    
    print("\nTest 1 réussi !\n")


def test_tensorboard_setup():
    """Test de la configuration TensorBoard"""
    print("TEST 2: Configuration TensorBoard")
    with initialize_config_dir(config_dir=str(Path.cwd() / 'config'), version_base=None):
        cfg = compose(config_name='main', overrides=['xp=SST/multires'])
    print("Configuration du logger:")
    print(f"  Type: {cfg.trainer.logger._target_}")
    print(f"  Save dir: {cfg.trainer.logger.save_dir}")
    print(f"  Name: {cfg.trainer.logger.name}")
    
    # Créer le logger
    logger = hydra.utils.instantiate(cfg.trainer.logger)
    print(f"\nLogger créé: {type(logger).__name__}")
    print(f"Les logs seront dans: {Path(cfg.trainer.logger.save_dir).absolute() / cfg.trainer.logger.name}")
    logger.log_metrics({'test_loss': 0.5, 'test_acc': 0.9}, step=0)
    print("Log de test écrit")
    
    print("\nTest 2 réussi !\n")


def print_quick_start_guide():
    print("Lancer l'entraînement:")
    print("   python main.py xp=SST/multires\n")
    print(" Pendant l'entraînement, dans un autre terminal:")
    print("   tensorboard --logdir=logs/sst_multires --host=0.0.0.0 --port=6006\n")
    print(" Ouvrir dans ton navigateur: http://localhost:6006\n")

if __name__ == "__main__":
    import sys
    import os
    os.chdir('/home/malegu/4D-MLG/Croscim')
    try:
        test_dataloader_with_filtering()
        test_tensorboard_setup()
        print_quick_start_guide()
        print("Tous les tests ont réussi !")
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import hydra
from hydra import compose, initialize_config_dir
from pathlib import Path
import os

os.chdir('/home/malegu/4D-MLG/Croscim')
with initialize_config_dir(config_dir=str(Path.cwd() / 'config'), version_base=None):
    cfg = compose(config_name='main', overrides=['xp=SST/multires_lite'])
    
    print("Config chargée avec succès!\n")
    
    print("Paramètres d'entraînement:")
    print(f"  - GPUs: {cfg.trainer.devices}")
    print(f"  - Epochs: {cfg.trainer.max_epochs}")
    print(f"  - Train batches/epoch: {cfg.trainer.limit_train_batches}")
    print(f"  - Val batches/epoch: {cfg.trainer.limit_val_batches}")
    print(f"  - Batch size: {cfg.datamodule.dl_kw.batch_size}")
    print(f"  - Precision: {cfg.trainer.precision}")
    
    print(f"\nDonnées:")
    print(f"  - Train: {cfg.datamodule.domains.train.time._args_}")
    print(f"  - Val: {cfg.datamodule.domains.val.time._args_}")
    print(f"  - Patch dims: {cfg.datamodule.xrds_kw.patch_dims}")
    print(f"  - Strides: {cfg.datamodule.xrds_kw.strides}")
    
    print(f"\nModèle (version LITE):")
    print(f"  - Résolutions: {cfg.model.multires}")
    solvers = cfg.model.solver.solvers
    for name, solver_cfg in solvers.items():
        print(f"  - {name}: {solver_cfg.n_step} steps, hidden={solver_cfg.grad_mod.dim_hidden}")
    
    print(f"\nLogger:")
    print(f"  - Type: TensorBoard")
    print(f"  - Directory: logs/{cfg.trainer.logger.name}")
    
    # Calculs approximatifs
    n_batches_train = cfg.trainer.limit_train_batches
    n_batches_val = cfg.trainer.limit_val_batches
    batch_size = cfg.datamodule.dl_kw.batch_size
    n_epochs = cfg.trainer.max_epochs
    
    # Temps estimé par batch (très approximatif)
    time_per_batch = 2  # secondes (avec solver lite)
    
    time_train_epoch = n_batches_train * time_per_batch
    time_val_epoch = n_batches_val * time_per_batch
    total_time_per_epoch = time_train_epoch + time_val_epoch
    total_time = total_time_per_epoch * n_epochs
    
    print(f"\nEstimation:")
    print(f"  - Temps/epoch: ~{total_time_per_epoch/60:.1f} min")
    print(f"  - Temps total: ~{total_time/60:.1f} min")
    print(f"  - Images traitées/epoch: {n_batches_train * batch_size} (train) + {n_batches_val * batch_size} (val)")
    
    print("\n" + "="*70)
    print("\nTerminal 1 - Lancer l'entraînement:")
    print("   python main.py xp=SST/multires_lite")
    print("\nTerminal 2 - Lancer TensorBoard (pendant l'entraînement):")
    print("   tensorboard --logdir=logs/sst_multires_lite --host=0.0.0.0 --port=6006")
    print("\nNavigateur - Ouvrir:")
    print("   http://localhost:6006")

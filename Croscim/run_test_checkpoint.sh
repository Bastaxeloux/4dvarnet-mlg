#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_checkpoint>"
    echo "Example: $0 /dmidata/projects/4dvarnet/checkpoints_sst_multires/last.ckpt"
    exit 1
fi

CKPT_PATH=$1

# Configuration pour le test
# On utilise src.test.base_test comme point d'entrée
# On surcharge entrypoints pour utiliser le script de test
# On spécifie le checkpoint
# On redirige les logs vers outputs/test_results
# On force l'utilisation du GPU 3 via CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1

# Filtrer les warnings "Device or resource busy" qui polluent le terminal
# On supprime les warnings Python avec PYTHONWARNINGS et on filtre stderr avec grep
python3 main.py xp=SST/multires \
    "entrypoints=[{_target_: src.test.base_test, trainer: \${trainer}, lit_mod: \${model}, dm: \${datamodule}, ckpt_path: '$CKPT_PATH', save_dir: './outputs/test_results'}]" \
    trainer.logger.save_dir=./outputs/test_results \
    trainer.devices=1 \
    trainer.precision=32 \
    datamodule.domains.test.time._args_=['2024-12-01','2024-12-31'] 2>&1

#!/bin/bash
cd /home/malegu/4D-MLG/Croscim

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_checkpoint>"
    echo "Example: $0 /dmidata/projects/4dvarnet/checkpoints_sst_multires/last.ckpt"
    exit 1
fi

CKPT_PATH=$1

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

python3 main.py xp=SST/multires \
    "entrypoints=[{_target_: src.test.base_test, trainer: \${trainer}, lit_mod: \${model}, dm: \${datamodule}, ckpt_path: '$CKPT_PATH', save_dir: './outputs/test_results'}]" \
    trainer.logger.save_dir=./outputs/test_results \
    trainer.devices=1 \
    trainer.precision=32 \
    datamodule.domains.test.time._args_=['2024-12-01','2024-12-31'] \
    datamodule.dl_kw.num_workers=4 \
    datamodule.dl_kw.batch_size=3 2>&1

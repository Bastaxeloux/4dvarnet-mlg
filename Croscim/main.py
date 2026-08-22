import hydra
import warnings
import os
import sys

# Suppress multiprocessing temp directory cleanup warnings
# These are harmless - temp dirs will be cleaned up by OS eventually
warnings.filterwarnings('ignore', message='.*Device or resource busy.*')
warnings.filterwarnings('ignore', category=ResourceWarning, message='.*subprocess.*')
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

# Suppress external library warnings
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Attribute .* is an instance of .* and is already saved during checkpointing.*')

# These distributed-training warnings are benign for the active Jean Zay run:
# Lightning has already assigned one CUDA device per rank, and train metrics
# emitted only on rank 0 intentionally avoid distributed collectives.
warnings.filterwarnings(
    'ignore',
    message=r'Precision bf16-mixed is not supported by the model summary\..*',
    category=UserWarning,
    module=r'pytorch_lightning\.utilities\.model_summary\.model_summary',
)
warnings.filterwarnings(
    'ignore',
    message=r'No device id is provided via `init_process_group` or `barrier `\..*',
    category=UserWarning,
    module=r'torch\.distributed\.distributed_c10d',
)
warnings.filterwarnings(
    'ignore',
    message=r'It is recommended to use `self\.log\(.*sync_dist=True.*',
    category=UserWarning,
    module=r'pytorch_lightning\.trainer\.connectors\.logger_connector\.result',
)

import multiprocessing.util
_original_rmtree_finalizer = None
def _silent_remove_temp_dir(rmtree_func, tempdir):
    """Silently remove temp directory, ignoring resource busy errors.
    
    Args:
        rmtree_func: The rmtree function to use (passed by multiprocessing)
        tempdir: The temporary directory path to remove
    """
    try:
        # Call rmtree without onerror/onexc - let our try/except handle it
        rmtree_func(tempdir)
    except OSError as e:
        if e.errno != 16:  # Not "Device or resource busy"
            raise
        # Silently ignore cleanup errors - OS will clean up later

try:
    import multiprocessing.util as mp_util
    _original_rmtree_finalizer = mp_util._remove_temp_dir
    mp_util._remove_temp_dir = _silent_remove_temp_dir
except Exception:
    pass

@hydra.main(config_path='config', config_name='main', version_base='1.3')
def main(cfg):
    seed = cfg.get('seed')
    if seed is not None:
        from pytorch_lightning import seed_everything
        seed_everything(int(seed), workers=True)
    hydra.utils.call(cfg.entrypoints)

if __name__ == '__main__':
    main()

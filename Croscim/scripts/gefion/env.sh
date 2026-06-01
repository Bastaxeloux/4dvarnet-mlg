#!/bin/bash
# Gefion runtime environment for Croscim.
#
# Load all modules before activating the venv. Loading SciPy-bundle after the
# venv can hide venv packages such as hydra; not loading it leaves modules such
# as pandas/mpmath unavailable.

module load GCC/12.3.0 Boost/1.82.0 snappy/1.1.10 GSL/2.7 Eigen/3.4.0 CUDA/12.8.0 Python/3.11.3 SciPy-bundle/2023.07
source /dcai/projects/cu_0026/croscim_env/bin/activate

export CROSCIM_ROOT="${CROSCIM_ROOT:-/dcai/users/guimae/4dvarnet-mlg/Croscim}"
export PYTHONPATH="$CROSCIM_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export DASK_SCHEDULER=synchronous

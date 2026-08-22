#!/bin/bash

# Load the same module stack used to create the Jean Zay virtual environment.
_CROSCIM_ARCH_MODULE="${1:-}"
module purge
if [ -n "$_CROSCIM_ARCH_MODULE" ]; then
    module load "$_CROSCIM_ARCH_MODULE"
    export CROSCIM_ARCH_MODULE="$_CROSCIM_ARCH_MODULE"
else
    unset CROSCIM_ARCH_MODULE
fi
module load pytorch-gpu/py3/2.8.0
unset _CROSCIM_ARCH_MODULE

VENV_PATH="${CROSCIM_VENV:-${WORK:?WORK is not defined}/venvs/venvai}"
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "ERROR: Jean Zay virtual environment not found: $VENV_PATH" >&2
    return 1 2>/dev/null || exit 1
fi

source "$VENV_PATH/bin/activate"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CROSCIM_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Each preprocessing worker is already a separate process.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

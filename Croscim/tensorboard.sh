#!/bin/bash
# Lance TensorBoard pour visualiser les résultats d'entraînement
# Usage: ./launch_tensorboard.sh [port]

PORT=${1:-6006}

LOGDIR="/dmidata/projects/4dvarnet/results_sst_multires/run"

echo "Lancement de TensorBoard..."
echo "  - Logdir: $LOGDIR"
echo "  - URL: http://ohm.dmi.dk:$PORT/"
echo ""

#  Si vous executer ce .sh sur une machine en ssh (ex : ohm au dmi), il faut forward le port avec (executer dans un terminal séparé):"
# ssh -L $PORT:localhost:$PORT user@ohm
#Si le port 6006 est occupé, on peut forward vers un autre qui soit libre (modifier le premier)"


export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::UserWarning"
tensorboard --logdir=$LOGDIR --port=$PORT --bind_all

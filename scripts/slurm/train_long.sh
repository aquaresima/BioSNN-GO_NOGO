#!/bin/bash
#SBATCH --job-name=bioinfo_long
#SBATCH --output=scripts/slurm/logs/long_%j.out
#SBATCH --error=scripts/slurm/logs/long_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
# NOTE: needs a GPU with >= 48 GB (measured peak 38.8 GB at 1000 units, batch 150, window 4.7 s)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
# Long training run. Checkpoints (last_model/last_optim, results.json) are written at every evaluation
# (every log_every_n_steps steps). A job that hits the wall time is continued with:
#   RESUME=<log_path of the run> sbatch --export=ALL scripts/slurm/train_long.sh
# or chained: sbatch --dependency=afterany:<jobid> --export=ALL,RESUME=<log_path> scripts/slurm/train_long.sh
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
export PYTHONPATH=$ROOT:$ROOT/.pylibs:$PYTHONPATH
export BIOINFO_DATAPATH=${BIOINFO_DATAPATH:?set BIOINFO_DATAPATH to the converted dataset folder}
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -n "$RESUME" ]; then
  $PY infopath/train.py --config long --resume "$RESUME"
else
  $PY infopath/config.py --config long --n_units 1000 --batch_size 150 --n_steps 20000 --log_every_n_steps 100 --eval_batch_size 200 || exit 1
  $PY infopath/train.py --config long
fi

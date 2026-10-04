#!/bin/bash
#SBATCH --job-name=bioinfo_smoke
#SBATCH --output=scripts/slurm/logs/smoke_%j.out
#SBATCH --error=scripts/slurm/logs/smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
# Submit from the project root: sbatch scripts/slurm/smoke.sh   (partition/qos names below are cluster specific: adapt)
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
export PYTHONPATH=$ROOT:$ROOT/.pylibs:$PYTHONPATH
export BIOINFO_DATAPATH=${BIOINFO_DATAPATH:?set BIOINFO_DATAPATH to the converted dataset folder}
nvidia-smi -L
$PY infopath/config.py --config smoke --n_units 600 --n_steps 60 --batch_size 50 || exit 1
$PY infopath/train.py --config smoke

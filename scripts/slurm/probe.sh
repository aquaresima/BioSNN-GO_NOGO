#!/bin/bash
#SBATCH --job-name=bioinfo_probe
#SBATCH --output=scripts/slurm/logs/probe_%j.out
#SBATCH --error=scripts/slurm/logs/probe_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# Memory and resume probe at long-run size on a 48 GB A40: 2 steps with an evaluation at
# every step, then a resume for 2 more.
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
export PYTHONPATH=$ROOT:$ROOT/.pylibs:$PYTHONPATH
export BIOINFO_DATAPATH=${BIOINFO_DATAPATH:?set BIOINFO_DATAPATH to the converted dataset folder}
nvidia-smi -L
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY infopath/config.py --config probe --n_units 1000 --batch_size 150 --n_steps 2 --log_every_n_steps 1 --eval_batch_size 200 || exit 1
$PY infopath/train.py --config probe || exit 1
LP=$(ls -dt log_dir/*/*_probe | head -1)
echo "RESUMING $LP"
$PY infopath/config.py --config probe --n_units 1000 --batch_size 150 --n_steps 4 --log_every_n_steps 1 --eval_batch_size 200 || exit 1
$PY infopath/train.py --config probe --resume "$LP"
nvidia-smi --query-gpu=memory.used --format=csv

#!/bin/bash
#SBATCH --job-name=bioinfo_stimcheck
#SBATCH --output=scripts/slurm/logs/stimcheck_%j.out
#SBATCH --error=scripts/slurm/logs/stimcheck_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=fast
#SBATCH --gres=gpu:1
# NOTE: needs a GPU with >= 48 GB (measured peak 38.8 GB at 1000 units, batch 150, window 4.7 s)
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# Stimulus dependence check on a saved checkpoint:
#   RUN=log_dir/master/<run> OUT=analysis/out/<name> [SCRIPT=lick_decoder_test] sbatch --export=ALL scripts/slurm/analysis.sh
# SCRIPT is stimulus_check (default) or lick_decoder_test
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
export PYTHONPATH=$ROOT:$ROOT/.pylibs:$PYTHONPATH
export BIOINFO_DATAPATH=${BIOINFO_DATAPATH:?set BIOINFO_DATAPATH to the converted dataset folder}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -m analysis.${SCRIPT:-stimulus_check} --run "$RUN" --out "$OUT"

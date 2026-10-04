#!/bin/bash
#SBATCH --job-name=pierre_convert
#SBATCH --output=scripts/slurm/logs/convert_%j.out
#SBATCH --error=scripts/slurm/logs/convert_%j.err
#SBATCH --partition=common
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
# Re-convert the NWB sessions with the corrected alignment (pierre/process.py).
# Submit from the project root: PIERRE_NWB=<folder with .nwb> PIERRE_OUT=<output folder> sbatch --export=ALL scripts/slurm/convert.sh
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
mkdir -p "${PIERRE_OUT:?set PIERRE_OUT to the output folder}"
$PY pierre/process.py

#!/bin/bash
#SBATCH --job-name=bioinfo_repertoire
#SBATCH --output=scripts/slurm/logs/repertoire_%j.out
#SBATCH --error=scripts/slurm/logs/repertoire_%j.err
#SBATCH --partition=common
#SBATCH --qos=fast
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
ROOT=${ROOT:-$SLURM_SUBMIT_DIR}   # project root (default: where sbatch was called)
PY=${PY:-python}                    # python of the environment with the dependencies
cd $ROOT || exit 1
export PYTHONPATH=$ROOT:$PYTHONPATH
$PY -m analysis.session_repertoire --data "$BIOINFO_DATAPATH" --out analysis/out/session_repertoire

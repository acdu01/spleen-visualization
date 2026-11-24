#!/usr/bin/env bash
#SBATCH --job-name=spleen-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prepare-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
PYTHON="${SPLEEN_PYTHON:-.venv/bin/python}"
exec "$PYTHON" -m ct_poc.cli prepare --cases all --workers "${SLURM_CPUS_PER_TASK:-1}"

#!/usr/bin/env bash
#SBATCH --job-name=spleen-viewer
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/viewer-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
PYTHON="${SPLEEN_PYTHON:-.venv/bin/python}"
VIEWER_PORT="${SPLEEN_PORT:-8501}"
echo "Viewer node: $(hostname -f)"
echo "Viewer port: ${VIEWER_PORT}"
echo "Use the SSH tunnel steps in docs/REMOTE.md. The viewer binds to loopback only."
exec "$PYTHON" -m streamlit run app.py --server.address 127.0.0.1 --server.port "${VIEWER_PORT}"

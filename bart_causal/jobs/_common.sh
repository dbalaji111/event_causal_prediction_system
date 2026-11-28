# Sourced by every job: settings, modules, the environment, and offline model loading.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
source jobs/config.sh
[ -n "$MODULES" ] && module load $MODULES
source .venv/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p logs "$RUN"
echo "== $(date) on $(hostname) | job ${SLURM_JOB_ID:-local} | run $RUN"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
PIPE="python bart_causal_pipeline.py"

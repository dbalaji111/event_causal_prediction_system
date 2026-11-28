#!/usr/bin/env bash
# Run ONCE on a LOGIN node (compute nodes often have no internet): creates .venv and downloads the models.
#   cd bart_causal && bash jobs/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
[ -n "$MODULES" ] && module load $MODULES
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p "$MODELS"
[ -d "$BART" ] || huggingface-cli download facebook/bart-large-mnli --local-dir "$BART"
if [ ! -d "$LLM" ]; then
  echo "Judge model not found at $LLM. Download one, e.g.:"
  echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir \"$LLM\""
  echo "or set LLM in jobs/config.sh to the Qwen you already have."
fi
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Environment ready. Submit the jobs with: bash jobs/run_all.sh"

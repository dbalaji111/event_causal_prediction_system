#!/bin/bash
#SBATCH --job-name=bart-label
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
source jobs/_common.sh
# Qwen judges N_LABELS pairs. A 7B model on one A100 takes roughly 15-30 min for 6000 pairs.
$PIPE label --out "$RUN" --labeler qwen --llm "$LLM" --n "$N_LABELS" --batch 16

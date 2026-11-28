#!/bin/bash
#SBATCH --job-name=bart-score
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
source jobs/_common.sh
$PIPE score --out "$RUN" --zero-shot --model "$BART"     # baseline: the base MNLI model, no fine-tuning
$PIPE score --out "$RUN"                                   # the fine-tuned model

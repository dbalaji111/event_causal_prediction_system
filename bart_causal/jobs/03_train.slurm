#!/bin/bash
#SBATCH --job-name=bart-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source jobs/_common.sh
$PIPE train --out "$RUN" --model "$BART" --epochs "$EPOCHS" --batch 16 --accum 2 --lr 1e-5

#!/bin/bash
#SBATCH --job-name=bart-eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
source jobs/_common.sh
$PIPE eval --out "$RUN"

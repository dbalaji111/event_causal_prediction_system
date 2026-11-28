#!/bin/bash
#SBATCH --job-name=bart-pairs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source jobs/_common.sh
$PIPE pairs --out "$RUN"

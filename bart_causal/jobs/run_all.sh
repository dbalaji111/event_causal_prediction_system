#!/usr/bin/env bash
# Submits the five jobs in order; each starts only if the one before it succeeded.
#   cd bart_causal && bash jobs/run_all.sh            # all stages
#   bash jobs/run_all.sh train                        # from a stage onwards: pairs | label | train | score | eval
set -euo pipefail
cd "$(dirname "$0")/.."
source jobs/config.sh
mkdir -p logs
opt() { local p="$1"; local a=(); [ -n "$ACCOUNT" ] && a+=(--account="$ACCOUNT"); [ -n "$p" ] && a+=(--partition="$p"); echo "${a[@]:-}"; }
CPU="$(opt "$CPU_PARTITION")"; GPU="$(opt "$GPU_PARTITION") --gres=$GPU_GRES"
stages=(pairs label train score eval); files=(01_pairs 02_label 03_train 04_score 05_eval); kinds=(cpu gpu gpu gpu cpu)
from="${1:-pairs}"; started=0; dep=""
for i in "${!stages[@]}"; do
  [ "${stages[$i]}" = "$from" ] && started=1
  [ $started = 1 ] || continue
  res=$CPU; [ "${kinds[$i]}" = gpu ] && res=$GPU
  id=$(sbatch --parsable $res ${dep:+--dependency=afterok:$dep} "jobs/${files[$i]}.slurm")
  echo "submitted ${stages[$i]}: job $id${dep:+ (after $dep)}"
  dep=$id
done
[ $started = 1 ] || { echo "unknown stage '$from' (use pairs, label, train, score or eval)"; exit 1; }
echo "Follow with: squeue -u \$USER    Logs: bart_causal/logs/"

#!/usr/bin/env bash
# Double-click in Finder to start the gold dashboard with its Scenario tests.
# Set your gold data folder here (the one with memory_train/, memory_test/, data/, kde_output/).
# Leave it empty to use demo/data_root_gold_real inside this folder.
DATA_ROOT=""

cd "$(dirname "$0")"
if [ -n "$DATA_ROOT" ]; then ./start_gold_dashboard.sh "$DATA_ROOT"; else ./start_gold_dashboard.sh; fi
echo; read -r -p "Press Enter to close this window…"

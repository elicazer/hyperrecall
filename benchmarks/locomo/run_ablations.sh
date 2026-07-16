#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python
declare -A CFG
CFG[no_dates]="MESH_NO_DATES=1"
CFG[no_rerank]="MESH_SIM_RERANK=0"
CFG[baretext]="MESH_EMBED_DB=$(pwd)/runs/phase1/conv-26.baretext.sqlite"
for name in no_dates no_rerank baretext; do
  echo "=== ABLATION: $name (${CFG[$name]}) ==="
  env ${CFG[$name]} $PY run_mesh_phase2.py > runs/phase2/ablation_${name}.log 2>&1
  env ${CFG[$name]} $PY judge_mesh_only.py 2>/dev/null | grep "SUMMARY_JSON:" | sed "s/^SUMMARY_JSON://" > runs/phase3/ablation_${name}.json
  echo "  done $name"
done
echo "ALL ABLATIONS DONE"

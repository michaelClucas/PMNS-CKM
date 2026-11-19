#!/usr/bin/env bash
# Full pipeline of exp 014 (<= 5 cores at a time). Run from this folder: bash run_all.sh >> run.log 2>&1
set -e
PY=../../.venv/Scripts/python
log() { echo "=== $(date -Iseconds) $*"; }
log dr_point;      $PY dr_point.py
log check_reparam; $PY check_reparam.py
log "global_fit MN DR";            [ -f fit_MN_DR.json ] || $PY global_fit.py MN DR 3000 1 5 MN_DR
log "global_fit MN ours";          $PY global_fit.py MN ours 3000 2 5 MN_ours
log "global_fit MN ours drrel";    $PY global_fit.py MN ours 3000 3 5 MN_ours_drrel all drrel
log "PT MN ql (5 independent ladders)"
for s in 1 2 3 4 5; do $PY pt_sample.py MN ql 100000 $s MN_ql_$s 32 24 & done; wait
log "global_fit FN DR";            $PY global_fit.py FN DR 3000 4 5 FN_DR
log "global_fit FN ours";          $PY global_fit.py FN ours 3000 5 5 FN_ours
log "PT FN ql (5 independent ladders)"
for s in 1 2 3 4 5; do $PY pt_sample.py FN ql 100000 $s FN_ql_$s 32 24 & done; wait
log analyze
$PY analyze_heldout.py MN MN_ql_1,MN_ql_2,MN_ql_3,MN_ql_4,MN_ql_5 heldout_MN
$PY analyze_heldout.py FN FN_ql_1,FN_ql_2,FN_ql_3,FN_ql_4,FN_ql_5 heldout_FN
log done

#!/usr/bin/env bash
# Second half of the pipeline as actually run (run_all.sh's 100k-iteration PT ladders were too slow on the shared
# machine, 31 ms/iteration, and were stopped at 5k iterations; PT runs here use 25k iterations per ladder).
# Expects fit_MN_DR, fit_MN_ours, fit_MN_ours_drrel (run_all.sh) and waits for fit_FN_DR.json.
PY=../../.venv/Scripts/python
log() { echo "=== $(date -Iseconds) $*"; }
until [ -f fit_FN_DR.json ]; do sleep 10; done
log "PT MN ql (5 independent ladders, 25k it)"
for s in 1 2 3 4 5; do $PY pt_sample.py MN ql 25000 $s MN_ql_$s 32 24 & done; wait
log "global_fit FN ours";          $PY global_fit.py FN ours 3000 5 5 FN_ours
log "PT FN ql (5 independent ladders, 25k it)"
for s in 1 2 3 4 5; do $PY pt_sample.py FN ql 25000 $s FN_ql_$s 32 24 & done; wait
log "global_fit MN ours nodl";     $PY global_fit.py MN ours 3000 6 5 MN_ours_nodl nodl
log "PT MN ql drrel (5 independent ladders, 25k it)"
for s in 1 2 3 4 5; do $PY pt_sample.py MN ql 25000 $s MN_qldr_$s 32 24 drrel & done; wait
log analyze
$PY analyze_heldout.py MN MN_ql_1,MN_ql_2,MN_ql_3,MN_ql_4,MN_ql_5 heldout_MN
$PY analyze_heldout.py FN FN_ql_1,FN_ql_2,FN_ql_3,FN_ql_4,FN_ql_5 heldout_FN
$PY analyze_heldout.py MN MN_qldr_1,MN_qldr_2,MN_qldr_3,MN_qldr_4,MN_qldr_5 heldout_MN_drrel
log done rest

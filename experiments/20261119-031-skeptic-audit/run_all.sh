#!/bin/sh
# Exp 031 reruns. Run from the repo root: sh experiments/20261119-031-skeptic-audit/run_all.sh
# Four streams in parallel (4 cores, ~25 min). Logs: experiments/20261119-031-skeptic-audit/run_*.log
# Per-eval cost of the exp 015 objective is ~1 ms at random points, so DE budgets are kept small
# (popsize 15, maxiter 400 for R0/R0eta; 600 for C); every DE result gets a least-squares polish.
D=experiments/20261119-031-skeptic-audit
PY=${PY:-.venv/Scripts/python}
export PYTHONWARNINGS=ignore
( for sc in SM 10 55; do $PY $D/rerun_bsv_de.py R0eta $sc 1 400 15; done ) > $D/run_bsv_R0eta.log 2>&1 &
( for sc in SM 10 55; do $PY $D/rerun_bsv_de.py R0 $sc 1 400 15; done ) > $D/run_bsv_R0.log 2>&1 &
( $PY $D/rerun_bsv_de.py C SM 1 600 15; $PY $D/rerun_bsv_de.py C 10 1 600 15 ) > $D/run_bsv_C.log 2>&1 &
( $PY $D/rerun_2o_de.py 1 3000 0; $PY $D/rerun_2o_de.py 2 3000 0; $PY $D/rerun_2o_de.py 3 3000 1 ) > $D/run_2o.log 2>&1 &
wait

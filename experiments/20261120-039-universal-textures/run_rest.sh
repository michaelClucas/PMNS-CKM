#!/usr/bin/env bash
# Exp 039 jobs after the CPU rule change (2 threads/workers, pinned via tools/pin.py), run from the repo root.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
PY="$(pwd)/.venv/Scripts/python.exe"
P="$PY tools/pin.py $PY"
D=experiments/20261120-039-universal-textures
# run from the repo root
echo "== train" >> $D/run_grammar.log
OMP_NUM_THREADS=2 $P $D/grammar.py train >> $D/run_grammar.log 2>&1
echo "== search" >> $D/run_grammar.log
OMP_NUM_THREADS=2 $P $D/grammar.py search >> $D/run_grammar.log 2>&1
echo "== verify top" >> $D/run_grammar.log
$P $D/grammar.py verify top >> $D/run_grammar.log 2>&1
# batch 4 (FZ4E) dropped: no CPU budget left

echo RESTDONE >> $D/run.log

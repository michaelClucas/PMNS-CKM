#!/usr/bin/env bash
# Exp 035 (b) lower bound as run: N = 12 finite subset (no PMNS), paper pair (-3,-3) seeds 1-2, and (1,1)
# (Y_e = Y_d, Y_nu = Y_u inside the two-operator model) seed 1. 3 processes, separate output files.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=../../.venv/Scripts/python
$PY dirac_staged.py lower -3 -3 1 > ../../artifacts/035/lower_a.json 2>&1 &
$PY dirac_staged.py lower -3 -3 2 > ../../artifacts/035/lower_b.json 2>&1 &
$PY dirac_staged.py lower 1 1 1 > ../../artifacts/035/lower_c.json 2>&1 &
wait
cat ../../artifacts/035/lower_?.json > dirac_lower.jsonl
echo finished

#!/usr/bin/env bash
# Exp 035 (b) as run (second attempt). The first attempt (36 Clebsch pairs, popsize 8, maxiter 100, NO)
# was stopped after 1 pair: 141 s per fit and the DE never entered the NuFIT tables
# (dirac_grid_partial.jsonl). Run instead: the paper's pair (-3, -3), NO, 3 seeds, popsize 8, maxiter 400.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
for s in 1 2 3; do
  ../../.venv/Scripts/python dirac.py -3 -3 NO $s 400 >> dirac_fits.jsonl 2>> dirac_fits.err &
done
wait
echo finished

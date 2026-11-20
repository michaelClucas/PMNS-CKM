#!/usr/bin/env bash
# Exp 035 (b) staged fit as run: stage 1 quark-only DE, seeds 1-3 in parallel; stage 2 all 36 Clebsch pairs
# (NO), then (-3, -3) in IO, 3 processes, each writing its own file (no interleaving). Log: run_dirac_staged.log
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=../../.venv/Scripts/python
for s in 1 2 3; do $PY dirac_staged.py stage1 $s > stage1_$s.out 2>&1 & done
wait
cat stage1_*.out; rm -f stage1_*.out
mkdir -p ../../artifacts/035/stage2
{
  for ce in 1 -3 1/3 9/2 4/9 6; do for cn in 1 -3 1/3 9/2 4/9 6; do echo "$ce $cn NO"; done; done
  echo "-3 -3 IO"
} | xargs -P 3 -L 1 sh -c 'f=../../artifacts/035/stage2/$(echo "$0_$1_$2" | tr "/" "o").json; ../../.venv/Scripts/python dirac_staged.py stage2 $0 $1 $2 > $f 2>&1; echo "done $0 $1 $2"'
cat ../../artifacts/035/stage2/*.json > dirac_staged.jsonl
echo finished

"""Exp 032 driver: split the pre-registered injection tasks over N single-thread workers.

Usage: python run_all.py [n_workers] [set]   set = main (default) | extra8
Logs to run.log (driver) and artifacts/032/logs/worker<k>.log.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE.parents[1] / "artifacts" / "032"
N_PER = 20


def tasks(which):
    if which == "extra8":
        return [f"OOM:8:{i}" for i in range(N_PER)]
    t = [f"IM:{f}:{i}" for f in ("MNu", "MN", "F2") for i in range(N_PER)]
    t += [f"OOM:{lv}:{i}" for lv in ("12", "8", "4", "2", "1", "0.5", "0") for i in range(N_PER)]
    return t


def main(n=5, which="main"):
    T = tasks(which)
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = dict(os.environ, EXP032_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = []
    with open(HERE / "run.log", "a") as lg:
        lg.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} start {which}: {len(T)} tasks, {n} workers\n")
    for k in range(n):
        f = open(ART / "logs" / f"{which}_worker{k}.log", "a")
        procs.append(subprocess.Popen([py, str(HERE / "inject.py"), "run", *T[k::n]], stdout=f, stderr=subprocess.STDOUT,
                                      env=env, cwd=HERE))
    for p in procs:
        p.wait()
    with open(HERE / "run.log", "a") as lg:
        lg.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} done {which}: exit codes "
                 f"{[p.returncode for p in procs]}\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 5, a[1] if len(a) > 1 else "main")

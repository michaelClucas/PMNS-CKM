"""Exp 034 driver: pre-registered task lists, 2 pipeline workers (2 threads each) + 1 oracle worker (1 thread).

Usage: python run_all.py <deadline_utc HH:MM>   workers are stopped at the deadline (resumable: done worlds skipped).
Logs: run.log (driver), artifacts/034/logs/<mode>_worker<k>.log.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE.parents[1] / "artifacts" / "034"
LEVELS = ("8", "4", "2", "1", "0.5")


def pipe_tasks():
    t = []
    for i in range(20):  # i-major: a deadline stop leaves every set with about the same count
        t += [f"{s}:{lv}:{i}" for s in ("L2", "L4") for lv in LEVELS]
        t += [f"K13:0:{i}", f"KQLC:0:{i}"]
    t += [f"S0:0:{i}" for i in range(10)]
    return t


def orc_tasks():
    t = [f"KR:{lb}:{i}" for i in range(100) for lb in ("13", "QLC")]
    for i in range(40):
        t += [f"{s}:12:{i}" for s in ("L2", "L4")]
        if i >= 20:
            t += [f"{s}:{lv}:{i}" for s in ("L2", "L4") for lv in LEVELS]
            t += [f"K13:0:{i}", f"KQLC:0:{i}"]
    return t


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(deadline):
    hh, mm = map(int, deadline.split(":"))
    now = time.gmtime()
    dl = time.time() + ((hh * 60 + mm) - (now.tm_hour * 60 + now.tm_min)) * 60 - now.tm_sec
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    py = sys.executable
    P, O = pipe_tasks(), orc_tasks()
    jobs = []
    for k in range(2):
        env = dict(os.environ, EXP034_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
        jobs.append(("pipe", k, env, P[k::2]))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs.append(("orc", 0, env, O))
    procs = []
    with open(HERE / "run.log", "a") as lg:
        lg.write(f"{stamp()} start: {len(P)} pipeline tasks (2 workers x 2 threads), {len(O)} oracle tasks (1 worker), "
                 f"deadline {deadline}Z\n")
    for mode, k, env, T in jobs:
        f = open(ART / "logs" / f"{mode}_worker{k}.log", "a")
        procs.append(subprocess.Popen([py, str(HERE / "ceiling.py"), mode, *T], stdout=f, stderr=subprocess.STDOUT,
                                      env=env, cwd=HERE))
    while any(p.poll() is None for p in procs):
        if time.time() > dl:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            with open(HERE / "run.log", "a") as lg:
                lg.write(f"{stamp()} deadline reached, workers stopped\n")
            break
        time.sleep(20)
    n_p = len(list((ART / "pipe").glob("*.json")))
    n_o = len(list((ART / "orc").glob("*.json")))
    with open(HERE / "run.log", "a") as lg:
        lg.write(f"{stamp()} end: exit codes {[p.returncode for p in procs]}; done pipeline {n_p}/{len(P)}, "
                 f"oracle-only {n_o}/{len(O)}\n")


if __name__ == "__main__":
    main(sys.argv[1])

"""Exp 034: relaunch the 2 pipeline workers (Deviation 1: stopping check every 25 fakes instead of 100).
Usage: python restart_pipe.py <deadline_utc HH:MM>. Same task lists and seeds as run_all.py; done worlds are skipped."""
import os
import subprocess
import sys
import time

import run_all as R

hh, mm = map(int, sys.argv[1].split(":"))
now = time.gmtime()
dl = time.time() + ((hh * 60 + mm) - (now.tm_hour * 60 + now.tm_min)) * 60 - now.tm_sec
P = R.pipe_tasks()
env = dict(os.environ, EXP034_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
           EXP034_BLOCK="25")
with open(R.HERE / "run.log", "a") as lg:
    lg.write(f"{R.stamp()} restart pipeline workers (block 25), deadline {sys.argv[1]}Z\n")
procs = [subprocess.Popen([sys.executable, str(R.HERE / "ceiling.py"), "pipe", *P[k::2]],
                          stdout=open(R.ART / "logs" / f"pipe_worker{k}.log", "a"), stderr=subprocess.STDOUT,
                          env=env, cwd=R.HERE) for k in range(2)]
while any(p.poll() is None for p in procs):
    if time.time() > dl:
        for p in procs:
            if p.poll() is None:
                p.kill()
        with open(R.HERE / "run.log", "a") as lg:
            lg.write(f"{R.stamp()} deadline reached, pipeline workers stopped\n")
        break
    time.sleep(20)
with open(R.HERE / "run.log", "a") as lg:
    lg.write(f"{R.stamp()} pipeline end: exit codes {[p.returncode for p in procs]}; "
             f"done {len(list((R.ART / 'pipe').glob('*.json')))}/{len(P)}\n")

"""Exp 034: third pipeline worker (1 thread) on the freed oracle core, tasks in reverse order (Deviation 2).
Usage: python extra_pipe.py <deadline_utc HH:MM>. Same worlds, seeds and rule; done worlds are skipped."""
import os
import subprocess
import sys
import time

import run_all as R

hh, mm = map(int, sys.argv[1].split(":"))
now = time.gmtime()
dl = time.time() + ((hh * 60 + mm) - (now.tm_hour * 60 + now.tm_min)) * 60 - now.tm_sec
P = R.pipe_tasks()[::-1]
env = dict(os.environ, EXP034_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           EXP034_BLOCK="25")
with open(R.HERE / "run.log", "a") as lg:
    lg.write(f"{R.stamp()} oracle-only worker done; extra pipeline worker (1 thread, reverse order), deadline {sys.argv[1]}Z\n")
p = subprocess.Popen([sys.executable, str(R.HERE / "ceiling.py"), "pipe", *P],
                     stdout=open(R.ART / "logs" / "pipe_worker2.log", "a"), stderr=subprocess.STDOUT, env=env, cwd=R.HERE)
while p.poll() is None:
    if time.time() > dl:
        p.kill()
        with open(R.HERE / "run.log", "a") as lg:
            lg.write(f"{R.stamp()} deadline reached, extra pipeline worker stopped\n")
        break
    time.sleep(20)

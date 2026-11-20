"""Exp 036 job queue: python run_train.py <slots>. Jobs q1/q0 x seeds 0-2, 1 torch thread each.
A job is claimed by creating its log in artifacts/036/logs/, so several queues can run side by side."""
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from common import ART, HERE, SEEDS

LOGS = ART / "logs"


def run(job):
    k, s = job
    if (ART / "models" / f"{k}_s{s}.pt").exists():
        return
    try:
        f = open(LOGS / f"{k}_s{s}.log", "x")
    except FileExistsError:
        return
    with f:
        r = subprocess.run([sys.executable, str(HERE / "train.py"), k, str(s)], stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    with open(HERE / "run.log", "a") as g:
        g.write(f"finished train {k} s{s} rc {r.returncode}\n")


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for k in ("q1", "q0") for s in SEEDS]
    with ThreadPoolExecutor(int(sys.argv[1])) as ex:
        list(ex.map(run, jobs))

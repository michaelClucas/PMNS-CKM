"""Job queue for exp 028 training: <slots> concurrent jobs, 2 torch threads each (logs in logs/).
Usage: python run_train.py <ordering> <slots> <seeds, e.g. 0,1,2> [resume]
Each job is claimed by creating its log file, so several queues can run side by side.
With "resume", flows of the listed seeds that already exist are continued (train.py EXP028_RESUME)."""
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ART = HERE.parents[1] / "artifacts" / "028" / "models"


def run(job):
    o, k, s, tag = job
    env = dict(os.environ, EXP028_THREADS="2")
    if tag == "resume":
        env["EXP028_RESUME"] = "1"
    elif (ART / f"{o}_{k}_s{s}.pt").exists():
        return
    try:
        f = open(HERE / "logs" / f"{o}_{k}_s{s}{'_resume' if tag == 'resume' else ''}.log", "x")
    except FileExistsError:
        return
    with f:
        r = subprocess.run([sys.executable, str(HERE / "train.py"), k, o, str(s)], stdout=f, stderr=subprocess.STDOUT, env=env)
    with open(HERE / "run.log", "a") as g:
        g.write(f"finished {o} {k} s{s} {tag} rc {r.returncode}\n")


if __name__ == "__main__":
    (HERE / "logs").mkdir(exist_ok=True)
    o, slots, seeds = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
    tag = sys.argv[4] if len(sys.argv) > 4 else "new"
    kinds = ("q1", "q0") if tag == "resume" else ("q1", "q0", "c1", "c0")
    jobs = [(o, k, s, tag) for k in kinds for s in seeds]
    jobs.sort(key=lambda j: j[1].startswith("c"))  # flows first
    with ThreadPoolExecutor(slots) as ex:
        list(ex.map(run, jobs))

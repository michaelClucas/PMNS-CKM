"""Run every flow training of exp 025 with 4 single-thread workers (logs appended to run.log)."""
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen import FAMILIES  # noqa: E402

TAGS = ["main"] + [f"drop_{f}" for f in FAMILIES] + ["seed1"]
JOBS = [(t, w) for t in TAGS for w in ("q1", "q0")]


def run(job):
    t, w = job
    if (HERE.parents[1] / "artifacts" / "025" / "flows" / f"{t}_{w}.pt").exists():
        return
    env = dict(os.environ, EXP025_THREADS="1")
    with open(HERE / "logs" / f"{t}_{w}.log", "w") as f:
        subprocess.run([sys.executable, str(HERE / "train.py"), t, w], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"finished {t} {w}", flush=True)


if __name__ == "__main__":
    (HERE / "logs").mkdir(exist_ok=True)
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        list(ex.map(run, JOBS))

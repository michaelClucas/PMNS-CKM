"""Step 1: run both pipelines on the real universe and compare with the original searches."""
import json
import time

import numpy as np

import pipelines as PL
from src.nulls import Universe


def main():
    out = {}
    u = Universe.real("NO")
    t0 = time.time()
    r = PL.pipeline_f2lo(u)
    out["F2LO"] = {"result": r, "sec": time.time() - t0}
    print("F2LO", r, time.time() - t0, flush=True)
    t0 = time.time()
    r = PL.pipeline_t1(u)
    out["T1"] = {"result": r, "sec": time.time() - t0, "expr": PL.t1_expr(r["cell"]["f"]),
                 "bits": PL.t1_bits()}
    print("T1", r, out["T1"]["expr"], time.time() - t0, flush=True)
    t0 = time.time()
    r2 = PL.pipeline_t1(u)
    print("T1 cached", time.time() - t0)
    cells = PL.t1_cells(u)
    out["T1_cells"] = cells
    (PL.HERE / "repro.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

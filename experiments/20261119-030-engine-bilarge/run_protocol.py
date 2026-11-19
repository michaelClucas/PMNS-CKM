"""Criterion-4 protocol for the frozen bi-large grammar (bilarge.pipeline), fit-only hit rule (informational).

usage: python run_protocol.py <task> [<task> ...]
  power:<null>:<ord>    src.nulls.power_check, 20 injections x 300 fakes, FIT_ONLY, early stop (seed [30, 1])
  null:<null>:<ord>     src.nulls.run_null, 1000 fakes, FIT_ONLY, early stop at > 4 hits (seed [30, 0])
  rate:<null>:<ord>:<n> the same null to a fixed length without early stop (seed [30, 2]); estimates
                        the hit rate, cannot change a verdict
Injection (power): make_injector(pred) with pred = the real-data best result OF THAT ORDERING
(PMNS prediction) plus ckm_lambda, ckm_A at their real values (so lambda and A get PDG noise).
Outputs <task>.json (colons -> _) and lines in run.log. 2 worker processes.
"""
import json
import sys
import time

import numpy as np

import bilarge as B
from src import nulls as N

WORKERS = 2
SEED = 30
LOG = B.HERE / "run.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def nr_dict(nr):
    if nr is None:
        return None
    d = nr.row()
    d["ref"] = nr.ref
    d["hit_rule"] = nr.hit_rule
    d["strict_hits"] = int(sum(N.is_hit(f["result"], nr.ref) for f in nr.fakes))
    d["cp_two_sided95"] = N.decide(nr.n_hits, max(nr.n_run, 1))
    d["info"] = nr.info
    d["fakes"] = nr.fakes
    return d


def best_of_ordering(o):
    allr = B.search(N.Universe.real("NO"), return_all=True)
    res = [r for r in allr if r[0].ordering == o][0][0]
    ck = N.Universe.real("NO").load_ckm()
    pred = dict(res.pred)
    pred["ckm_lambda"] = ck["lambda"]["value"]
    pred["ckm_A"] = ck["A"]["value"]
    return pred, res


def main():
    for task in sys.argv[1:]:
        kind, null, o, *rest = task.split(":")
        t0 = time.time()
        if kind == "power":
            pred, res = best_of_ordering(o)
            log(f"power {null} {o}: injecting {o} best (chi2 {res.chi2:.3f}, p_global {res.p_global:.4f}) pred {pred}")
            pr = N.power_check(B.pipeline, N.make_injector(pred), n_inject=20, n_fakes=300, null=null, ordering=o,
                               seed=[SEED, 1], workers=WORKERS, **N.FIT_ONLY)
            d = dict(null=null, ordering=o, n_inject=pr.n_inject, n_pass=pr.n_pass, fraction=pr.fraction,
                     adequate=pr.adequate, n_candidate_ok=pr.n_candidate_ok, hit_rule=pr.hit_rule,
                     injected_pred=pred, refs=pr.refs,
                     nulls=[None if x is None else {k: v for k, v in nr_dict(x).items() if k != "fakes"} |
                            {"hits_at": [f["index"] for f in x.fakes if f["hit"]]} for x in pr.nulls])
            log(f"power {null} {o}: {pr.n_pass}/{pr.n_inject} pass (fraction {pr.fraction:.2f}), "
                f"{pr.n_candidate_ok} injections fit-ok; {time.time() - t0:.0f} s")
        else:
            n = int(rest[0]) if kind == "rate" else 1000
            nr = N.run_null(B.pipeline, null=null, ordering=o, n_fakes=n,
                            seed=[SEED, 2 if kind == "rate" else 0], workers=WORKERS,
                            early_stop=(kind == "null"), hit_fn=N.is_hit_fit_only)
            d = nr_dict(nr)
            log(f"{kind} {null} {o}: hits {nr.n_hits}/{nr.n_run} (planned {n}), rate {nr.rate:.3f}, "
                f"CP95 two-sided [{d['cp_two_sided95']['lo']:.3f}, {d['cp_two_sided95']['hi']:.3f}], one-sided upper {nr.upper:.3f}, "
                f"passed {nr.passed}, early {nr.stopped_early}; ref p_global {nr.ref['p_global']:.4f}; "
                f"B_ckm {(nr.info or {}).get('B_ckm', float('nan')):.2f} bits; {time.time() - t0:.0f} s")
        json.dump(d, open(B.HERE / (task.replace(":", "_") + ".json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

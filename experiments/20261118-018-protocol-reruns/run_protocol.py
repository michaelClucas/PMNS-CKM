"""Steps 3-4: power checks and N1/N2 nulls for T1 and F2-LO with src.nulls.

Hit rules
  strict  src.nulls.is_hit (k <= N-2, max|pull| < 2, p_global >= ref). Both leads have k > N-2
          by construction (same k on every universe), so no fake can ever be a strict hit and no
          injection can pass candidate_ok: the strict protocol is uninformative for them. Strict
          counts are still recomputed from the stored per-fake summaries.
  fit     hit_fit: the strict rule without the compression clause (max|pull| < 2 and
          p_global >= ref p_global). k is identical on every universe, so dropping k <= N-2
          changes nothing else. This rule drives early stopping and the reported verdicts.
  Power   src.nulls.power_check with candidate_ok replaced by the same fit-only rule
          (max|pull| < 2, p_global > 0.05): monkeypatched in this process only, src untouched.

usage: python run_protocol.py <lead> <task> [<task> ...]
  lead: T1 | F2LO ; task: power:N1:NO | null:N2:IO | rate:N1:NO:300 | ...
  rate: the same null run to a fixed length without early stop (seed [18, 2]), only to
        estimate the hit rate; it cannot change a failed verdict.
"""
import json
import sys
import time

import numpy as np

import pipelines as PL
from src import nulls as N

WORKERS = 3
SEED = 18
LOG = PL.HERE / "run.log"


def hit_fit(fake, ref):
    if fake is None:
        return False
    pf, pr = float(fake["p_global"]), float(ref["p_global"])
    if not np.isfinite(pf):
        return False
    return bool(fake["max_abs_pull"] < 2 and (pf >= pr or not np.isfinite(pr)))


def ok_fit(res, p_min=0.05):
    if res is None:
        return False
    p = float(N._get(res, "p_global"))
    return bool(N._get(res, "max_abs_pull") < 2 and np.isfinite(p) and p > p_min)


PIPES = {"T1": PL.pipeline_t1, "F2LO": PL.pipeline_f2lo}


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
    d["strict_hits"] = int(sum(N.is_hit(f["result"], nr.ref) for f in nr.fakes))
    d["fakes"] = nr.fakes
    d["cp"] = N.decide(nr.n_hits, max(nr.n_run, 1))
    return d


def main():
    lead, tasks = sys.argv[1], sys.argv[2:]
    pipe = PIPES[lead]
    for task in tasks:
        kind, null, o, *rest = task.split(":")
        t0 = time.time()
        if kind == "rate":
            n = int(rest[0])
            nr = N.run_null(pipe, null=null, ordering=o, n_fakes=n, seed=[SEED, 2], workers=WORKERS,
                            early_stop=False, hit_fn=hit_fit)
            d = nr_dict(nr)
            d["sec"] = time.time() - t0
            (PL.HERE / f"rate_{lead}_{null}_{o}.json").write_text(json.dumps(d, indent=1, default=float))
            log(f"{lead} RATE {null} {o}: hits {nr.n_hits}/{nr.n_run} rate {nr.rate:.4f} "
                f"CP95 two-sided [{d['cp']['lo']:.4f}, {d['cp']['hi']:.4f}] one-sided upper {d['cp']['upper']:.4f}; strict hits {d['strict_hits']} {d['sec']:.0f}s")
        elif kind == "null":
            nr = N.run_null(pipe, null=null, ordering=o, n_fakes=1000, seed=[SEED, 0], workers=WORKERS,
                            early_stop=True, hit_fn=hit_fit)
            d = nr_dict(nr)
            d["sec"] = time.time() - t0
            (PL.HERE / f"null_{lead}_{null}_{o}.json").write_text(json.dumps(d, indent=1, default=float))
            log(f"{lead} {null} {o}: hits {nr.n_hits}/{nr.n_run} (planned {nr.n_fakes}, stopped_early "
                f"{nr.stopped_early}) rate {nr.rate:.4f} CP95 upper {nr.upper:.4f} passed {nr.passed}; "
                f"strict hits {d['strict_hits']}; ref p_global {nr.ref['p_global']:.4g} [{nr.ref['label']}] "
                f"{d['sec']:.0f}s")
        else:
            ref = pipe(N.Universe.real(o))
            inj = N.make_injector(ref["pred"])
            orig = N.candidate_ok
            N.candidate_ok = ok_fit
            try:
                pr = N.power_check(pipe, inj, n_inject=20, n_fakes=300, null=null, ordering=o,
                                   seed=[SEED, 1], workers=WORKERS, hit_fn=hit_fit)
            finally:
                N.candidate_ok = orig
            d = {"null": pr.null, "ordering": pr.ordering, "n_inject": pr.n_inject, "n_pass": pr.n_pass,
                 "fraction": pr.fraction, "adequate": pr.adequate, "n_candidate_ok_fit": pr.n_candidate_ok,
                 "n_candidate_ok_strict": int(sum(N.candidate_ok(r) for r in pr.refs)),
                 "injected_pred": ref["pred"], "refs": pr.refs,
                 "nulls": [None if x is None else {**x.row(), "ref_p_global": x.ref["p_global"],
                                                    "first_fakes": x.fakes[:10]} for x in pr.nulls],
                 "sec": time.time() - t0}
            (PL.HERE / f"power_{lead}_{null}_{o}.json").write_text(json.dumps(d, indent=1, default=float))
            runs = [x.n_run for x in pr.nulls if x is not None]
            log(f"{lead} power {null} {o}: {pr.n_pass}/{pr.n_inject} pass (fraction {pr.fraction:.2f}, "
                f"adequate {pr.adequate}); injected fits ok (fit-only) {pr.n_candidate_ok}/{pr.n_inject}; "
                f"fakes run per injection {runs}; {d['sec']:.0f}s")


if __name__ == "__main__":
    main()

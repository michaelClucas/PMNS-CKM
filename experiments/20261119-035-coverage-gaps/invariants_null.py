"""Exp 035 (a), descriptive look-elsewhere check (not the criterion-4 protocol, which is not run
because no relation is compressive): how many of the 1428 grammar relations pass the stage-1
screen (chi2_screen < 4) when the CKM is replaced by N1-type hierarchical fakes
(lambda U[0.05, 0.5], A U[0.3, 2], rhobar, etabar U[-1, 1]), keeping the real relative CKM errors.
Real PMNS tables. 300 fakes per ordering. Usage: python invariants_null.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import invariants as I  # noqa: E402
from src import data as D  # noqa: E402
from src.mixing import jarlskog, wolfenstein_matrix  # noqa: E402

IDX = {"Vus": (0, 1), "Vub": (0, 2), "Vcd": (1, 0), "Vcb": (1, 2), "Vtd": (2, 0), "Vts": (2, 1)}


def fake_ckm(rng, rel):
    lam, A = rng.uniform(0.05, 0.5), rng.uniform(0.3, 2.0)
    rb, eb = rng.uniform(-1, 1, 2)
    V = wolfenstein_matrix(lam, A, rb, eb)
    out = {k: (abs(V[i, j]), abs(V[i, j]) * rel[k]) for k, (i, j) in IDX.items()}
    J = abs(float(jarlskog(V)))
    out["J"] = (J, J * rel["J"])
    out["lambda"] = (lam, lam * rel["lambda"])
    return out


def main():
    rels = I.grammar()
    ck = D.load_ckm()
    rel = {k: 0.5 * (ck[k]["err_plus"] + ck[k]["err_minus"]) / ck[k]["value"] for k in list(IDX) + ["J", "lambda"]}
    rng = np.random.default_rng(3501)
    out = {}
    for o in D.ORDERINGS:
        prof = I.profiles(o, I.chi2_grid(o))

        def count(ckm):
            n = 0
            for r in rels:
                T, sT = I.target(r, ckm)
                key = r["el"] if r["kind"] == "mod" else "J"
                cen, m = prof[key]
                n += float(np.min(m + ((cen - T) / max(sT, 1e-12)) ** 2)) < 4
            return n

        real = count(None) if False else sum(I.screen_chi2(prof, r)[0] < 4 for r in rels)
        fakes = [count(fake_ckm(rng, rel)) for _ in range(300)]
        fr = np.array(fakes)
        out[o] = dict(real=int(real), fake_mean=float(fr.mean()), fake_q=[float(np.quantile(fr, q)) for q in
                                                                          (0.05, 0.5, 0.95)],
                      frac_fake_ge_real=float(np.mean(fr >= real)), frac_fake_any=float(np.mean(fr > 0)))
        print(f"[{o}] relations with chi2_screen < 4: real CKM {real}; N1 fakes mean {fr.mean():.1f}, "
              f"5/50/95% {out[o]['fake_q']}; fraction of fakes with >= as many {out[o]['frac_fake_ge_real']:.3f}; "
              f"fakes with >= 1 passing relation {out[o]['frac_fake_any']:.3f}", flush=True)
    (HERE / "invariants_null.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

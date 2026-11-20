"""Exp 034 aggregation -> results.json, analyze.log (README pre-registration, "Outputs")."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.infogain import cp_interval  # noqa: E402

ART = ROOT / "artifacts" / "034"
LEVELS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
OV = ("today_phys", "today_gauss", "nominal", "optimistic")
LOG = []


def out(s):
    print(s)
    LOG.append(s)


def load(sub):
    R = {}
    for p in (ART / sub).glob("*.json"):
        d = json.loads(p.read_text())
        d.pop("T_fakes", None)
        R[(d["set"], str(d["label"]), d["i"])] = d
    return R


def rate(flags):
    n, k = len(flags), int(sum(flags))
    lo, hi = cp_interval(k, n) if n else (float("nan"), float("nan"))
    return dict(n=n, k=k, rate=k / n if n else None, cp95=[lo, hi])


def fmt(r):
    return f"{r['k']}/{r['n']} [{r['cp95'][0]:.2f}, {r['cp95'][1]:.2f}]" if r["n"] else "0/0"


def p50(levels, rates):
    """Linear interpolation of the rate in log2(I) at 0.5; None if never reached."""
    pts = [(math.log2(l), r) for l, r in zip(levels, rates) if r is not None]
    for (x0, r0), (x1, r1) in zip(pts, pts[1:]):
        if r0 < 0.5 <= r1:
            return 2 ** (x0 + (0.5 - r0) * (x1 - x0) / (r1 - r0))
    if pts and pts[0][1] >= 0.5:
        return f"<= {2 ** pts[0][0]}"
    return None


def lab(lv):
    return f"{lv:g}"


def main():
    P, O = load("pipe"), load("orc")
    allw = {**O, **P}  # oracle from either source (same world, same seeds)
    res = dict(n_pipe=len(P), n_orc_only=len(O), sets={})
    for s in ("L2", "L4"):
        rows = []
        for lv in LEVELS:
            pw = [d for (ss, l, i), d in P.items() if ss == s and l == lab(lv)]
            ow = [d for (ss, l, i), d in allw.items() if ss == s and l == lab(lv)]
            row = dict(level=lv, pipe=rate([d["detected"] for d in pw]),
                       pipe_median_dT=float(np.median([d["T_world"] - d["T_fake_median"] for d in pw])) if pw else None,
                       pipe_median_rank=float(np.median([d["rank_eval"] for d in pw])) if pw else None,
                       I_link_median=float(np.median([d["I_link"] for d in ow])) if ow else None,
                       I_meas_median={sc: float(np.median([d["I_meas"][sc] for d in ow])) for sc in OV[1:]} if ow else None,
                       oracle={v: rate([d["oracle"][v]["detected"] for d in ow]) for v in OV},
                       oracle_median_rank=float(np.median([d["oracle"]["today_phys"]["rank"] for d in ow])) if ow else None)
            rows.append(row)
            im = row["I_meas_median"] or {}
            out(f"{s} {lv:>4} bits: pipeline {fmt(row['pipe'])} median dT {row['pipe_median_dT']} median rank "
                f"{row['pipe_median_rank']} | oracle today {fmt(row['oracle']['today_phys'])} gauss "
                f"{fmt(row['oracle']['today_gauss'])} nominal {fmt(row['oracle']['nominal'])} optimistic "
                f"{fmt(row['oracle']['optimistic'])} | I_meas today/nom/opt "
                f"{', '.join(f'{im.get(k, float('nan')):.2f}' for k in OV[1:])}")
        p50s = dict(pipe=p50([r["level"] for r in rows], [r["pipe"]["rate"] for r in rows]),
                    **{f"oracle_{v}": p50([r["level"] for r in rows], [r["oracle"][v]["rate"] for r in rows]) for v in OV})
        out(f"{s} 50%-power I_link (bits): {p50s}")
        res["sets"][s] = dict(rows=rows, p50=p50s)
    for s, lb in (("K13", "0"), ("KQLC", "0"), ("KR", "13"), ("KR", "QLC")):
        pw = [d for (ss, l, i), d in P.items() if ss == s and l == lb]
        ow = [d for (ss, l, i), d in allw.items() if ss == s and l == lb]
        row = dict(pipe=rate([d["detected"] for d in pw]),
                   pipe_median_dT=float(np.median([d["T_world"] - d["T_fake_median"] for d in pw])) if pw else None,
                   oracle={v: rate([d["oracle"][v]["detected"] for d in ow]) for v in OV},
                   oracle_median_rank={v: float(np.median([d["oracle"][v]["rank"] for d in ow])) for v in OV} if ow else None,
                   I_meas_median={sc: float(np.median([d["I_meas"][sc] for d in ow])) for sc in OV[1:]} if ow else None)
        res["sets"][f"{s}_{lb}"] = row
        out(f"{s} {lb}: pipeline {fmt(row['pipe'])} median dT {row['pipe_median_dT']} | oracle "
            + " ".join(f"{v} {fmt(row['oracle'][v])}" for v in OV)
            + f" | median oracle rank {row['oracle_median_rank']} | I_meas {row['I_meas_median']}")
    s0 = [d for (ss, l, i), d in P.items() if ss == "S0"]
    res["sets"]["S0"] = dict(pipe=rate([d["detected"] for d in s0]),
                             median_rank=float(np.median([d["rank_eval"] for d in s0])) if s0 else None)
    out(f"S0 size check: pipeline {fmt(res['sets']['S0']['pipe'])}")
    # descriptive (not pre-registered): pipeline rank vs world lambda over all pipeline worlds
    lam = np.array([d["w_meas"][0] for d in P.values()])
    rk = np.array([d["rank_eval"] for d in P.values()])
    full = np.array([not d["stopped_early"] for d in P.values()])
    if len(lam) > 2:
        c = float(np.corrcoef(lam, rk)[0, 1])
        res["descriptive"] = dict(corr_rank_lambda=c, n=len(lam), n_full_2000=int(full.sum()),
                                  min_rank=float(rk.min()), lambda_of_min_rank=float(lam[rk.argmin()]))
        out(f"descriptive: corr(pipeline rank, world lambda) = {c:.2f} over {len(lam)} worlds; "
            f"{int(full.sum())} needed all 2000 fakes; min rank {rk.min():.4f} at lambda {lam[rk.argmin()]:.3f}")
    res["per_world"] = [{k: d[k] for k in ("set", "label", "i", "I_link", "I_meas", "w_true", "w_meas", "oracle")
                         if k in d} | {k: d.get(k) for k in ("T_world", "T_fake_median", "k_ge", "n_fakes_eval",
                                                              "detected", "sigma")}
                        for d in sorted(allw.values(), key=lambda d: (d["set"], str(d["label"]), d["i"]))]
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (HERE / "analyze.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()

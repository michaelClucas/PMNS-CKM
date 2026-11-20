"""Exp 032 aggregation: detection rates with CP95 intervals, minimum detectable information at 50% power.

Usage: python analyze.py   -> results.json, analyze.log
"""
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

ART = ROOT / "artifacts" / "032" / "inj"


def rate(x):
    x = [bool(v) for v in x]
    k, n = sum(x), len(x)
    lo, hi = cp_interval(k, n) if n else (None, None)
    return dict(k=k, n=n, rate=k / n if n else None, cp95=[lo, hi])


def min_detectable(levels, rates):
    """Linear interpolation of the rate in log2(I) between adjacent levels (> 0); None if 0.5 never reached."""
    pts = [(l, r) for l, r in zip(levels, rates) if l > 0]
    for (l0, r0), (l1, r1) in zip(pts, pts[1:]):
        if r0 >= 0.5:
            return l0
        if r0 < 0.5 <= r1:
            a, b = math.log2(l0), math.log2(l1)
            return 2 ** (a + (0.5 - r0) / (r1 - r0) * (b - a))
    if pts and pts[0][1] >= 0.5:
        return pts[0][0]
    return None


def main():
    R = [json.load(open(p)) for p in sorted(ART.glob("*.json"))]
    out = dict(OOM={}, IM={})
    lines = []
    for kind in ("OOM", "IM"):
        labels = sorted({r["label"] for r in R if r["kind"] == kind}, key=lambda x: (isinstance(x, str), x))
        for lab in labels:
            S = [r for r in R if r["kind"] == kind and r["label"] == lab]
            row = dict(n=len(S), detected=rate([r["detected"] for r in S]),
                       detected_300=rate([r["detected_300"] for r in S]),
                       median_rank=float(np.median([r["rank_eval"] for r in S])),
                       n_rank_le_0p05=int(sum(r["rank_eval"] <= 0.05 for r in S)),
                       median_dT=float(np.median([r["T_real"] - r["T_fake_median"] for r in S])),
                       median_T=float(np.median([r["T_real"] for r in S])),
                       fakes_evaluated=int(sum(r["n_fakes_eval"] for r in S)),
                       per_world=[dict(i=r["i"], T=round(r["T_real"], 3), k=r["k_ge"], n=r["n_fakes_eval"],
                                       det=r["detected"], w_true=np.round(r["w_true"], 4).tolist(),
                                       **({"oracle_rank": r["oracle_rank"]} if "oracle_rank" in r else {}))
                                  for r in sorted(S, key=lambda r: r["i"])])
            if kind == "OOM":
                row["I_link_mean"] = float(np.mean([r["I_link"] for r in S]))
                row["I_meas_gauss_median"] = float(np.median([r["I_meas_gauss"] for r in S]))
                row["I_meas_gauss_range"] = [float(np.min([r["I_meas_gauss"] for r in S])),
                                             float(np.max([r["I_meas_gauss"] for r in S]))]
                if any("oracle_detected" in r for r in S):
                    row["oracle_detected"] = rate([r["oracle_detected"] for r in S])
                    row["oracle_median_rank"] = float(np.median([r["oracle_rank"] for r in S]))
                sig = [r["sigma"] for r in S if r["sigma"] is not None]
                row["sigma_median"] = float(np.median(sig)) if sig else None
            else:
                rho_eta = np.array([r["w_true"][2:] for r in S])
                row["n_rho_eta_in_box"] = int(np.sum(np.all(np.abs(rho_eta) <= 1, 1)))
                row["det_in_box"] = rate([r["detected"] for r, re in zip(S, rho_eta) if np.all(np.abs(re) <= 1)])
            out[kind][str(lab)] = row
            d = row["detected"]
            lines.append(f"{kind} {lab}: detected {d['k']}/{d['n']} = {d['rate']:.2f} CP95 [{d['cp95'][0]:.3f}, "
                         f"{d['cp95'][1]:.3f}]; 300-rule {row['detected_300']['k']}/{d['n']}; median rank "
                         f"{row['median_rank']:.3f}; rank<=0.05 {row['n_rank_le_0p05']}; median dT {row['median_dT']:.2f} bits"
                         + (f"; I_meas median {row['I_meas_gauss_median']:.2f}; oracle "
                            f"{row.get('oracle_detected', {}).get('k')}/{d['n']}, oracle median rank "
                            f"{row.get('oracle_median_rank')}" if kind == "OOM" else
                            f"; rho,eta in box {row['n_rho_eta_in_box']}/{d['n']}"))
    lv = sorted(float(k) for k in out["OOM"])
    rates = [out["OOM"][str(k) if str(k) in out["OOM"] else str(int(k))]["detected"]["rate"] for k in lv]
    out["min_detectable_bits_50pct"] = min_detectable(lv, rates)
    orates = [out["OOM"].get(str(k), out["OOM"].get(str(int(k)), {})).get("oracle_detected", {}).get("rate") for k in lv]
    out["oracle_min_detectable_bits_50pct"] = min_detectable([l for l, r in zip(lv, orates) if r is not None],
                                                             [r for r in orates if r is not None])
    lines.append(f"min detectable I_link at 50% power (pipeline): {out['min_detectable_bits_50pct']}")
    lines.append(f"min detectable I_link at 50% power (oracle): {out['oracle_min_detectable_bits_50pct']}")
    json.dump(out, open(HERE / "results.json", "w"), indent=1)
    open(HERE / "analyze.log", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

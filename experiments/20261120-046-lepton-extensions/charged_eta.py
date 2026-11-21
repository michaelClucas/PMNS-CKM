"""Exp 046: RRR5 charged-lepton stage beyond charged46's eta_l grid edge (eta_l 0.30, 0.45, 0.60; seed 1)."""
import json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import charged46 as C
out = []
for el in (0.30, 0.45, 0.60):
    r = C.run((("RRR5", "GJ", "G30", -0.25, el), 1))
    out.append(r)
    top = sorted(r["pulls"].items(), key=lambda t: -abs(t[1]))[:3]
    print(f"RRR5 eta_l {el}: N10 chi2 {r['chi2']:.4g} max pull {r['max_pull']:.3g} {[(k, round(v, 1)) for k, v in top]}", flush=True)
json.dump(out, open(HERE / "charged_eta.json", "w"), indent=1, default=float)

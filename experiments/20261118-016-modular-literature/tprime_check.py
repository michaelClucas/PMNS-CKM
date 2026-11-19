"""Score the published best-fit predictions of the 13/14-parameter T' non-holomorphic model
(arXiv:2506.19822, combined table) against our data, without refitting (no reimplementation of the
polyharmonic Maass forms). Predicted values below are the paper's numbers.

PMNS -> src.evaluate (NO, NuFIT 6.1 tables; SM running of PMNS to M_GUT is < 0.1 sigma, ignored).
CKM and mass ratios -> our GUT-scale (SM, 2e16 GeV) vector and noise model from exp 014's obs_gut.json
(2% mass floor). Writes tprime_check.json.
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import evaluate  # noqa: E402

PRED = {  # 2506.19822 combined best fit: (no gCP, gCP)
    "s12sq": (0.308, 0.291), "s13sq": (0.02205, 0.02158), "s23sq": (0.443, 0.462), "dcp_deg": (1.09 * 180, 0.99 * 180),
    "s12q": (np.sin(0.2256), np.sin(0.2263)), "s13q": (np.sin(0.0037), np.sin(0.0037)), "s23q": (np.sin(0.0406), np.sin(0.0374)),
    "dq": (np.radians(74.33), np.radians(70.52)), "mu_mc": (0.00204, 0.00204), "mc_mt": (0.00318, 0.00318),
    "md_ms": (0.1066, 0.0588), "ms_mb": (0.019, 0.018), "dm_ratio": (7.57e-5 / 2.51e-3, 7.11e-5 / 2.53e-3),
    "me_mmu": (0.004737, 0.004737), "mmu_mtau": (0.05882, 0.05882),
}
OBS = json.loads((ROOT / "experiments/20261118-014-so10-fits/obs_gut.json").read_text())
IDX = {"mu_mc": "lg_mu_mc", "mc_mt": "lg_mc_mt", "md_ms": "lg_md_ms", "ms_mb": "lg_ms_mb", "me_mmu": "lg_me_mmu",
       "mmu_mtau": "lg_mmu_mtau", "s12q": "s12q", "s13q": "lg_s13q", "s23q": "lg_s23q", "dq": "dq",
       "dm_ratio": "lg_dm21_dm3l"}


def main():
    names = OBS["names"]
    out = {}
    for j, tag in enumerate(("no_gCP_k14", "gCP_k13")):
        pmns = {k: PRED[k][j] for k in ("s12sq", "s13sq", "s23sq", "dcp_deg")}
        ev = evaluate(pmns, "NO", use=tuple(pmns), k=0)
        pulls = {}
        for k, n in IDX.items():
            i = names.index(n)
            v = PRED[k][j]
            x = np.log10(v) if n.startswith("lg_") else v
            pulls[k] = (x - OBS["z"][i]) / OBS["sigma"][i]
        chi2 = ev.chi2 + sum(p ** 2 for p in pulls.values())
        out[tag] = {"pmns_chi2_evaluate": ev.chi2, "pmns_pulls": ev.pulls, "gut_pulls": pulls, "chi2_total": chi2,
                    "N": 15}
        print(tag, f"chi2 (their point, our data) = {chi2:.1f}; PMNS part {ev.chi2:.2f}")
        print("   ", {k: round(v, 1) for k, v in {**ev.pulls, **pulls}.items()})
    (HERE / "tprime_check.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

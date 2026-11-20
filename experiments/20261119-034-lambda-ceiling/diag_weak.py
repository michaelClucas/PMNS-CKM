"""Exp 034 post-hoc diagnostic (not pre-registered): expected oracle detection rate at weak links.
For the oracle-only worlds i = 20-39 of L2 0.5, L2 2, L4 1, L4 2 bits (same g, sigma, world CKM, fakes), redraw the
measurement 50 times (Gaussian, today's covariance) and apply the oracle rule. Also a size check: same worlds with
y-hat generated at an independent CKM point. -> diag_weak.log"""
import json
import pathlib
import sys

import numpy as np

import ceiling as c

HERE = pathlib.Path(__file__).resolve().parent
best = c.I32.ev.pmns_best("NO")
y_ref = c.I32.target4(best)
cm = c.covariances(best)["today_gauss"]
lines = []
for s, lv in (("L2", "0.5"), ("L2", "2"), ("L4", "1"), ("L4", "2")):
    for null in (False, True):
        k_tot, per = 0, []
        for i in range(20, 40):
            d = json.load(open(c.ART / "orc" / f"{s}_{lv}_{i}.json"))
            code = c.code_of(s, lv)
            r = np.random.default_rng([34, code, i])
            u0 = r.uniform(-1, 1, (1, 4))
            L = c.Link(r, [0, 1] if s == "L2" else [0, 1, 2, 3], y_ref)
            assert np.allclose(c.from_u(u0)[0], d["w_true"])
            C = d["sigma"] ** 2 * np.eye(4) + cm
            P = np.linalg.inv(C)
            mf, mw = L.mean(c.to_u(c.fakes_w(code, i))), L.mean(u0)[0]
            rr = np.random.default_rng([5, i, int(null)])
            k = 0
            for _ in range(50):
                centre = L.mean(rr.uniform(-1, 1, (1, 4)))[0] if null else mw
                yh = centre + rr.multivariate_normal(np.zeros(4), C)
                df, dw = yh - mf, yh - mw
                k += int(np.sum(-0.5 * np.einsum("na,ab,nb->n", df, P, df) >= -0.5 * dw @ P @ dw) <= c.MAX_HITS)
            per.append(k)
            k_tot += k
        lines.append(f"{s} {lv} bits {'size (independent CKM)' if null else 'link'}: rate {k_tot / 1000:.3f} "
                     f"over 20 worlds x 50 draws; per-world detections of 50: {per}")
        print(lines[-1], flush=True)
(HERE / "diag_weak.log").write_text("\n".join(lines) + "\n")

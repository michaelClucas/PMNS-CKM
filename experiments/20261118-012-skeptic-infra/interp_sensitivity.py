"""How much does the interpolation scheme of the NuFIT tables move fitted chi^2 minima?

Refits TM1 (spot_check.py model) with cubic interpolation (1D CubicSpline, 2D bicubic
RectBivariateSpline) instead of the linear/bilinear scheme used by src.data, and measures
the largest linear-vs-cubic difference on random points inside 3 sigma.

    .venv/Scripts/python experiments/20261118-012-skeptic-infra/interp_sensitivity.py
"""
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize

from spot_check import Tables, pdg_angles, tm1_U


class CubicTables(Tables):
    def __init__(self, o):
        super().__init__(o)
        self.cs = {t: CubicSpline(*self.p[t]) for t in self.p}
        self.bs = RectBivariateSpline(self.xs, self.ys, self.z, kx=3, ky=3)

    def one(self, t, v):
        x, _ = self.p[t]
        return np.inf if (v < x[0] or v > x[-1]) else float(self.cs[t](v))

    def two(self, s23, d):
        d = ((d + 180.0) % 360.0) - 180.0
        if not (self.xs[0] <= s23 <= self.xs[-1]):
            return np.inf
        return float(self.bs(s23, d)[0, 0])


def fit(tab, x0):
    f = lambda x: tab.pmns4(*pdg_angles(tm1_U(*x)))
    return min((minimize(f, np.array(x0) + dx, method="Nelder-Mead",
                         options=dict(xatol=1e-9, fatol=1e-10, maxiter=8000))
                for dx in [(0, 0), (0.005, 0.05), (-0.005, -0.05)]), key=lambda r: r.fun)


rng = np.random.default_rng(12)
for o, x0, ctr in (("NO", (0.2628, 4.520), (0.3088, 0.02249, 0.47, 207.0)),
                   ("IO", (0.2634, 4.977), (0.3088, 0.02261, 0.555, 283.0))):
    lin, cub = Tables(o), CubicTables(o)
    rl, rc = fit(lin, x0), fit(cub, x0)
    print(f"TM1 {o}: linear chi2 {rl.fun:.4f}  cubic chi2 {rc.fun:.4f}  (diff {rl.fun - rc.fun:+.4f})")
    sig = np.array([0.0067, 0.00057, 0.016, 25.0])
    pts = np.array(ctr) + rng.uniform(-3, 3, (4000, 4)) * sig
    d = np.array([lin.pmns4(*p) - cub.pmns4(*p) for p in pts])
    d = d[np.isfinite(d)]
    print(f"   random points within 3 sigma: linear - cubic: median {np.median(d):+.3f}, "
          f"95th pct {np.percentile(np.abs(d), 95):.3f}, max |diff| {np.abs(d).max():.3f}")

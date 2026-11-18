"""Independent recomputation of three exp 006 baseline scores (TM1, GST, QLC solar).

Shares no code with src/evaluate.py, src/baselines.py, src/mixing.py or src/stats.py:
own matrix construction, own angle extraction, own NuFIT CSV reader and interpolation,
own chi^2 and p-value. Measured central values and errors come from src.data.load_*
(plain JSON readers); the NuFIT Delta chi^2 tables are parsed here from data/nufit/*.csv.

    .venv/Scripts/python experiments/20261118-012-skeptic-infra/spot_check.py
"""
import csv
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2dist

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import load_ckm, load_masses  # noqa: E402  (JSON readers only)

NUFIT = ROOT / "data" / "nufit"


def read_csv(name):
    with open(NUFIT / name) as fh:
        rows = list(csv.reader(fh))
    return np.array(rows[1:], dtype=float)


class Tables:
    """NuFIT 6.1 TBoff tables, each shifted to its own minimum (within-ordering)."""

    def __init__(self, o):
        self.p = {}
        for t in ("T12", "T13"):
            a = read_csv(f"v61_TBoff_{o}_{t}.csv")
            a = a[np.argsort(a[:, 0])]
            self.p[t] = (a[:, 0], a[:, 1] - a[:, 1].min())
        a = read_csv(f"v61_TBoff_{o}_T23-DCP.csv")
        xs, ys = np.unique(a[:, 0]), np.unique(a[:, 1])
        z = np.empty((len(xs), len(ys)))
        for s, d, c in a:
            z[np.where(xs == s)[0][0], np.where(ys == d)[0][0]] = c
        self.xs, self.ys, self.z = xs, ys, z - z.min()
        # offset of this ordering above the global minimum, from the raw 2D table
        self.offset = float(a[:, 2].min())

    def one(self, t, v):
        x, y = self.p[t]
        return np.inf if (v < x[0] or v > x[-1]) else float(np.interp(v, x, y))

    def two(self, s23, d):
        d = ((d + 180.0) % 360.0) - 180.0
        xs, ys, z = self.xs, self.ys, self.z
        if not (xs[0] <= s23 <= xs[-1]):
            return np.inf
        i = int(np.clip(np.searchsorted(xs, s23) - 1, 0, len(xs) - 2))
        j = int(np.clip(np.searchsorted(ys, d) - 1, 0, len(ys) - 2))
        u = (s23 - xs[i]) / (xs[i + 1] - xs[i])
        v = (d - ys[j]) / (ys[j + 1] - ys[j])
        return float(z[i, j] * (1 - u) * (1 - v) + z[i + 1, j] * u * (1 - v)
                     + z[i, j + 1] * (1 - u) * v + z[i + 1, j + 1] * u * v)

    def pmns4(self, s12, s13, s23, d):
        return self.one("T12", s12) + self.one("T13", s13) + self.two(s23, d)


def pdg_angles(U):
    """sin^2 of the three angles and delta (deg, [0,360)) of a 3x3 unitary, PDG convention."""
    A2 = np.abs(U) ** 2
    s13 = A2[0, 2]
    s12 = A2[0, 1] / (1 - s13)
    s23 = A2[1, 2] / (1 - s13)
    c12, c23, c13 = 1 - s12, 1 - s23, 1 - s13
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    r = np.sqrt(s12 * c12 * s23 * c23 * s13)  # s12 c12 s23 c23 s13 (sines, not squares)
    sind = J / (r * c13)
    cosd = (A2[1, 0] - s12 * c23 - c12 * s23 * s13) / (2 * r)
    return s12, s13, s23, np.degrees(np.arctan2(sind, cosd)) % 360


def pdg_matrix(s12, s13, s23, d_deg):
    """Build U from sin^2 values, to self-test pdg_angles."""
    t12, t13, t23 = (np.arcsin(np.sqrt(v)) for v in (s12, s13, s23))
    c1, s1, c2, s2, c3, s3 = np.cos(t12), np.sin(t12), np.cos(t13), np.sin(t13), np.cos(t23), np.sin(t23)
    e = np.exp(1j * np.radians(d_deg))
    return np.array([[c1 * c2, s1 * c2, s2 / e],
                     [-s1 * c3 - c1 * s3 * s2 * e, c1 * c3 - s1 * s3 * s2 * e, s3 * c2],
                     [s1 * s3 - c1 * c3 * s2 * e, -c1 * s3 - s1 * c3 * s2 * e, c3 * c2]])


def tm1_U(theta, phi):
    tbm = np.array([[2, np.sqrt(2), 0], [-1, np.sqrt(2), np.sqrt(3)], [1, -np.sqrt(2), np.sqrt(3)]]) / np.sqrt(6)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1, 0, 0], [0, c, s * np.exp(-1j * phi)], [0, -s * np.exp(1j * phi), c]])
    return tbm @ R


def fit_tm1(tab):
    def f(x):
        return tab.pmns4(*pdg_angles(tm1_U(x[0], x[1])))
    best = None
    for th in np.linspace(0.05, 0.55, 26):
        for ph in np.linspace(0, 2 * np.pi, 49)[:-1]:
            v = f((th, ph))
            if best is None or v < best[0]:
                best = (v, th, ph)
    r = min((minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-10, maxiter=8000))
             for x0 in [(best[1], best[2]), (best[1] + 0.01, best[2] - 0.1), (best[1] - 0.01, best[2] + 0.1)]),
            key=lambda r: r.fun)
    return r.fun, r.x, pdg_angles(tm1_U(*r.x))


def gauss(pred, rec):
    d = pred - rec["value"]
    return (d / (rec["err_plus"] if d > 0 else abs(rec["err_minus"]))) ** 2


def main():
    # self-test of the angle extraction
    for args in [(0.31, 0.022, 0.47, 207.0), (0.3, 0.02, 0.56, 283.0), (0.25, 0.05, 0.4, 40.0)]:
        got = pdg_angles(pdg_matrix(*args))
        assert np.allclose(got, args, atol=1e-9), (args, got)

    out = []
    for o in ("NO", "IO"):
        tab = Tables(o)
        c2, x, a = fit_tm1(tab)
        out.append(("B2-TM1", o, c2, c2 + tab.offset, chi2dist.sf(c2, 1),
                    f"theta={x[0]:.5f} phi={x[1] % (2*np.pi):.4f} pred s12={a[0]:.4f} s13={a[1]:.5f} s23={a[2]:.4f} d={a[3]:.1f}"))
        # QLC solar: s12^2 = sin^2(pi/4 - theta_C), theta_C = asin(s12 CKM)
        thc = np.arcsin(load_ckm()["s12"]["value"])
        s12p = np.sin(np.pi / 4 - thc) ** 2
        c2 = tab.one("T12", s12p)
        out.append(("A1s", o, c2, c2 + tab.offset, chi2dist.sf(c2, 1), f"pred s12sq={s12p:.5f}"))

    # GST: |Vus| = sqrt(m_d/m_s), m_d, m_s nuisances
    m, ck = load_masses(), load_ckm()

    def g(x):
        md, ms = x
        return gauss(np.sqrt(md / ms), ck["Vus"]) + gauss(md, m["m_d"]) + gauss(ms, m["m_s"])
    r = minimize(g, [m["m_d"]["value"], m["m_s"]["value"]], method="Nelder-Mead",
                 options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    # analytic linearized check: one effective Gaussian for sqrt(md/ms) vs Vus
    md, ms = m["m_d"]["value"], m["m_s"]["value"]
    sr = 0.5 * np.sqrt(md / ms) * np.hypot(m["m_d"]["err_plus"] / md, m["m_s"]["err_plus"] / ms)
    lin = (np.sqrt(md / ms) - ck["Vus"]["value"]) ** 2 / (sr ** 2 + ck["Vus"]["err_plus"] ** 2)
    out.append(("C1", "-", r.fun, r.fun, chi2dist.sf(r.fun, 1),
                f"m_d={r.x[0]:.7f} m_s={r.x[1]:.6f}; linearized chi2 {lin:.4f}"))

    print("id      ord  chi2      chi2_glob  p(dof)      detail")
    for row in out:
        print(f"{row[0]:7s} {row[1]:3s}  {row[2]:8.4f}  {row[3]:8.4f}  {row[4]:.3e}  {row[5]}")


if __name__ == "__main__":
    main()

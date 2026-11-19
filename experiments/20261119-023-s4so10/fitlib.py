"""Shared fitting code: official scorer at the GUT scale, residual vector, parameter packing.

Scoring: src.evaluate.Scorer with scale = {mu 2e16, MSSM, tan beta 5, M_SUSY 1 TeV} (the paper's
SUSY setting), mass_floor 0.02 on every mr_* key, NO, analysis IC23_woSKatm.

eta_b (tan-beta-enhanced SUSY threshold, 1705.01555 App. D, Antusch-Maurer parametrisation) is not in
src.rg. When used, it is applied to the model side to linear order: the data's GUT values with eta_b
differ from src.rg's (eta_b = 0) by theta_i3 -> (1 + eta_b) theta_i3, y_b -> y_b / (1 + eta_b), so the
model's s13^q, s23^q and m_s/m_b are divided by (1 + eta_b) and m_b/m_tau multiplied by it before
scoring. Neglected: the eta_b dependence of the MSSM running itself (y_b(GUT) ~0.14 instead of ~0.03
changes y_b and CKM running by O(1-2%)).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src import data as _data  # noqa: E402
from src.evaluate import Scorer  # noqa: E402
from model import PUBLISHED, REAL, PHASES, observables, pred_dimless  # noqa: E402

SCALE = {"mu": 2e16, "model": "MSSM", "tan_beta": 5.0, "m_susy": 1000.0}
FLOOR = 0.02
PMNS_USE = ("s12sq", "s13sq", "s23sq", "dcp_deg")
CKM_USE = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")
MR_Q = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")
MR_L = ("mr_e_mu", "mr_mu_tau", "mr_b_tau")
USE_ALL = PMNS_USE + ("dm_ratio",) + CKM_USE + MR_Q + MR_L          # N = 16
USE_QL = CKM_USE + MR_Q + MR_L                                       # N = 11 (held-out fit)
SCALES = {k: abs(PUBLISHED[k]) for k in REAL}

_SC = {}


def scorer(use=USE_ALL):
    use = tuple(use)
    if use not in _SC:
        _SC[use] = Scorer(use, "NO", mass_floor=FLOOR, scale=SCALE)
    return _SC[use]


def apply_eta_b(pred, eb):
    if eb == 0.0:
        return pred
    q = dict(pred)
    q["ckm_s13"] = pred["ckm_s13"] / (1 + eb)
    q["ckm_s23"] = pred["ckm_s23"] / (1 + eb)
    q["mr_s_b"] = pred["mr_s_b"] / (1 + eb)
    q["mr_b_tau"] = pred["mr_b_tau"] * (1 + eb)
    return q


class Param:
    """x = [reals / published magnitude (13, signed), free phases (len(free_phases)), eta_b (opt)]."""

    def __init__(self, free_phases=PHASES, fixed_phases=None, eta_b="fixed0"):
        self.free = tuple(free_phases)
        self.fixed = dict(fixed_phases or {})
        self.eb_mode = eta_b  # "fixed0", "free", or a float
        self.n = len(REAL) + len(self.free) + (1 if eta_b == "free" else 0)

    def unpack(self, x):
        p = {k: x[i] * SCALES[k] for i, k in enumerate(REAL)}
        p.update(self.fixed)
        for i, k in enumerate(self.free):
            p[k] = x[len(REAL) + i]
        if self.eb_mode == "free":
            eb = float(x[-1])
        elif self.eb_mode == "fixed0":
            eb = 0.0
        else:
            eb = float(self.eb_mode)
        return p, eb

    def pack(self, p, eb=-0.8):
        x = [p[k] / SCALES[k] for k in REAL] + [p[k] for k in self.free]
        if self.eb_mode == "free":
            x.append(eb)
        return np.array(x, float)


def predict(p, eb):
    o = observables(p)
    return apply_eta_b(pred_dimless(o), eb), o


def residuals(pred, sc):
    pp = sc._prep(pred)
    chi2, terms, pulls = sc._score(pp)
    r = [pulls[k] for k in sc.use if k.startswith("ckm_")]
    if sc.pmns_use:
        t = terms.get("pmns_joint", np.inf)
        if not np.isfinite(t):
            recs = _data.load_pmns("NO")
            g = 0.0
            for n in sc.pmns_use:
                d = pp[n] - sc.best[n]
                g += (d / (recs[n]["err_plus"] if d > 0 else recs[n]["err_minus"])) ** 2
            t = 1e4 + g
        r.append(np.sqrt(max(t, 0.0)))
    if "dm_ratio" in sc.use:
        t = terms["dm_ratio"]
        r.append(np.sqrt(t) if np.isfinite(t) else 100.0 + abs(np.log(max(pp["dm_ratio"], 1e-9) / 0.0295)) * 100)
    if sc.mr_use:
        res = np.array([pp[n] for n in sc.mr_use]) - sc._mr_value
        r.extend(np.linalg.solve(sc._mr_L, res))
    return np.array(r, float)


def make_resid(par, use=USE_ALL):
    sc = scorer(use)
    n_res = len(residuals(predict(PUBLISHED, 0.0)[0], sc))

    def f(x):
        p, eb = par.unpack(x)
        if par.eb_mode == "free" and not -0.95 < eb < 1.0:
            return np.full(n_res, 1e3)
        try:
            pred, _ = predict(p, eb)
            return residuals(pred, sc)
        except (ValueError, np.linalg.LinAlgError):
            return np.full(n_res, 1e3)
    return f

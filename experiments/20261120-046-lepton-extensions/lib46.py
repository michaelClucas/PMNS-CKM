"""Exp 046 library: lepton extensions of exp 045's quark survivors (extensions.md).

ext(model, assign, ynu, mr) -> (model_fn(theta) -> pred, bounds, periodic, quark_param_names)
  model  "RRR5" | "BB" | "HS"
  assign lepton assignment: RRR5 "GJ"; BB "XE"; HS "SU5" | "PS"
  ynu    "u" | "d"   (Y_nu = Y_u or Y_d)
  mr     "I" (M_R = 1) | "D" (M_R = diag(1, x, y))
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E45 = ROOT / "experiments" / "20261120-045-quark-literature"
for p in (str(ROOT), str(E45)):
    if p not in sys.path:
        sys.path.insert(0, p)

import structures as S  # noqa: E402
from lib45 import Q8, reduce_params  # noqa: E402
from src.yukawa import observables  # noqa: E402

CL = ("mr_e_mu", "mr_mu_tau")
PM = ("s12sq", "s13sq", "s23sq", "dcp_deg")
N15 = tuple(Q8) + CL + PM + ("dm_ratio",)
LNMR = (-35.0, 35.0)
LNXE = (-6.0, 6.0)

SCALES = {
    "G30": {"mu": 2e16, "model": "MSSM", "tan_beta": 30.0},
    "SM3": {"mu": 1e3, "model": "SM"}, "SM6": {"mu": 1e6, "model": "SM"}, "SM16": {"mu": 1e16, "model": "SM"},
    "SM5": {"mu": 1e5, "model": "SM"}, "SM7": {"mu": 1e7, "model": "SM"}, "SM9": {"mu": 1e9, "model": "SM"},
}


def _texture(name):
    tex = S.TEXTURES[name][0]()
    reduce_params(tex)
    return tex


def quark_part(model):
    """-> (fn(theta) -> (Yu, Yd), bounds, periodic)."""
    if model in ("RRR5", "BB"):
        tex = _texture(model)
        b = tex.bounds()

        def fn(th):
            t = tex.theta(th)
            return tex.up.matrix(t), tex.down.matrix(t)
        return fn, b, [k for k in b if k.startswith("f")]
    if model == "HS":
        return (lambda th: tuple(S.hs_matrices(th, "mod"))), dict(S.HS_BOUNDS), []
    raise KeyError(model)


def lepton_Ye(model, assign, Yd, th):
    if model == "RRR5" and assign == "GJ":
        Ye = Yd.copy()
        Ye[1, 1] *= -3.0
        return Ye
    if model == "BB" and assign == "XE":
        # Yd = [[0,A,0],[A,0,B x_d],[0,B/x_d,C]] -> same A, B, C with x_e
        B = np.sqrt(abs(Yd[1, 2] * Yd[2, 1]))
        xe = np.exp(th["lxe"])
        Ye = Yd.copy()
        Ye[1, 2], Ye[2, 1] = B * xe, B / xe
        return Ye
    if model == "HS" and assign == "SU5":
        return Yd.T.copy()
    if model == "HS" and assign == "PS":
        return Yd.copy()
    raise KeyError((model, assign))


def ext(model, assign, ynu, mr):
    qfn, b, per = quark_part(model)
    qnames = list(b)
    b = dict(b)
    if model == "BB":
        b["lxe"] = LNXE
    if mr == "D":
        b["lx"], b["ly"] = LNMR, LNMR
    if model == "HS" and assign == "PS" and ynu != "u":
        raise ValueError("HS-PS fixes Y_nu = Y_u")

    def fn(th):
        Yu, Yd = qfn(th)
        Ye = lepton_Ye(model, assign, Yd, th)
        Yn = Yu if ynu == "u" else Yd
        MR = np.eye(3, dtype=complex)
        if mr == "D":
            MR[1, 1], MR[2, 2] = np.exp(th["lx"]), np.exp(th["ly"])
        return observables(Yu, Yd, Ye, Y_nu=Yn, M_R=MR, ordering="NO")
    return fn, b, per, qnames


OPTIONS = {  # model -> list of (assign, ynu, mr, scales tried)
    "RRR5": [("GJ", y, m) for y in "ud" for m in "ID"],
    "BB": [("XE", y, m) for y in "ud" for m in "ID"],
    "HS": [("SU5", y, m) for y in "ud" for m in "ID"] + [("PS", "u", m) for m in "ID"],
}
BITS = {"RRR5": 3, "BB": 4, "HS": 3}
EXTRA_CONT = {"RRR5": 2, "BB": 0, "HS": 1}  # eta_b + eta_l; none; fitted scale (x_e is in the bounds)

"""Exp 045 library: published predictive quark structures -> the 8 quark observables.

Two kinds of structure:
  * textures: Y_u, Y_d given by a zero pattern (Hermitian, complex symmetric or general), every
    non-zero entry a free modulus (log) and, where the paper allows it, a phase; one entry per sector
    fixed to 1 (the overall scale drops out of the 8 observables). Phases that are removable by
    rephasing are dropped by a numerical Jacobian-rank test (`reduce_params`), so k_cont is the
    number of independent continuous directions, not the number of symbols written in the paper.
  * relations: closed-form CKM / mass relations (GST, Hall-Rasin, right unitarity triangle, ...),
    written as a model whose free parameters are the observables the relation does not fix.

Outputs use src.evaluate keys; the 8 quark observables are Q8.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mixing import extract_angles, standard_matrix  # noqa: E402

Q8 = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad", "mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b")
LOGMOD = (-16.0, 6.0)  # ln |entry| bounds relative to the fixed entry
TWO_PI = 2 * np.pi


# ------------------------------------------------------------------ observables from Yukawas

def left_svd(Y):
    """(U, s): columns of U are left singular vectors, s ascending."""
    U, s, _ = np.linalg.svd(Y)
    return U[:, ::-1], s[::-1]


def quark_obs(Yu, Yd):
    Uu, yu = left_svd(Yu)
    Ud, yd = left_svd(Yd)
    V = Uu.conj().T @ Ud
    t12, t13, t23, d = (float(x) for x in extract_angles(V))
    return {"ckm_s12": np.sin(t12), "ckm_s13": np.sin(t13), "ckm_s23": np.sin(t23), "ckm_delta_rad": d,
            "mr_u_c": yu[0] / yu[1], "mr_c_t": yu[1] / yu[2], "mr_d_s": yd[0] / yd[1], "mr_s_b": yd[1] / yd[2]}


# ------------------------------------------------------------------ textures

class Sector:
    """One Yukawa matrix.

    kind: "herm" (entries given on/above the diagonal, lower = conj), "sym" (complex symmetric,
          lower = upper), "gen" (every entry independent).
    entries: {(i, j): code} 0-based, code in
        "1"  fixed to +1 (normalisation; one per sector)
        "p"  real positive modulus
        "r"  real, either sign (Hermitian diagonal)
        "c"  complex: modulus + phase
        "c1" modulus fixed to 1 (normalisation), phase free
        ("tie", (k, l)) same value as entry (k, l) (general kind)
        ("k", value) fixed real constant times the normalisation (a published Clebsch-like number)
    extra: optional {(i, j): callable(theta) -> complex} for entries tied to other parameters.
    """

    def __init__(self, name, kind, entries):
        self.name, self.kind, self.entries = name, kind, dict(entries)

    def params(self):
        out = {}
        for (i, j), c in self.entries.items():
            tag = f"{self.name}{i + 1}{j + 1}"
            if c in ("p", "c"):
                out["l" + tag] = LOGMOD
            if c == "r":
                out["s" + tag] = (-1.0, 1.0)  # signed: sign(s) * exp(LOGMOD lo + |s| (hi - lo))
            if c in ("c", "c1"):
                out["f" + tag] = (0.0, TWO_PI)
        return out

    def matrix(self, th):
        Y = np.zeros((3, 3), complex)
        for (i, j), c in self.entries.items():
            tag = f"{self.name}{i + 1}{j + 1}"
            if c == "1":
                v = 1.0
            elif isinstance(c, tuple) and c[0] == "k":
                v = c[1]
            elif c == "p":
                v = np.exp(th["l" + tag])
            elif c == "r":
                s = th["s" + tag]
                v = np.sign(s) * np.exp(LOGMOD[0] + abs(s) * (LOGMOD[1] - LOGMOD[0])) if s != 0 else 0.0
            elif c == "c":
                v = np.exp(th["l" + tag] + 1j * th.get("f" + tag, 0.0))
            elif c == "c1":
                v = np.exp(1j * th.get("f" + tag, 0.0))
            elif isinstance(c, tuple) and c[0] == "tie":
                continue
            else:
                raise ValueError(c)
            Y[i, j] = v
            if i != j:
                if self.kind == "herm":
                    Y[j, i] = np.conj(v)
                elif self.kind == "sym":
                    Y[j, i] = v
        for (i, j), c in self.entries.items():  # ("tie", (k, l)): same value as entry (k, l)
            if isinstance(c, tuple) and c[0] == "tie":
                Y[i, j] = Y[c[1]]
        return Y


class Texture:
    def __init__(self, name, up: Sector, down: Sector, fixed=None):
        self.name, self.up, self.down = name, up, down
        self.fixed = dict(fixed or {})  # params pinned (removed from the fit) by reduce_params

    def all_params(self):
        p = dict(self.up.params())
        p.update(self.down.params())
        return p

    def bounds(self):
        return {k: v for k, v in self.all_params().items() if k not in self.fixed}

    def theta(self, th):
        t = dict(self.fixed)
        t.update(th)
        return t

    def model(self, th):
        t = self.theta(th)
        return quark_obs(self.up.matrix(t), self.down.matrix(t))


def obs_vector(pred):
    """Coordinates in which the Jacobian rank is measured (logs of ratios / angles, cos/sin delta)."""
    return np.array([np.log(pred["ckm_s12"]), np.log(pred["ckm_s13"]), np.log(pred["ckm_s23"]),
                     np.cos(pred["ckm_delta_rad"]), np.sin(pred["ckm_delta_rad"]),
                     np.log(pred["mr_u_c"]), np.log(pred["mr_c_t"]), np.log(pred["mr_d_s"]), np.log(pred["mr_s_b"])])


def jac_rank(fn, names, th0, h=1e-5, tol=1e-7):
    """Rank of d obs_vector / d theta (central differences; noise floor ~1e-10 relative)."""
    J = []
    for n in names:
        a, b = dict(th0), dict(th0)
        a[n] += h
        b[n] -= h
        J.append((obs_vector(fn(a)) - obs_vector(fn(b))) / (2 * h))
    J = np.array(J).T
    s = np.linalg.svd(J, compute_uv=False)
    return int(np.sum(s > tol * s[0])), s


def random_point(bounds, rng, near=None):
    th = {}
    for k, (lo, hi) in bounds.items():
        if near is not None and k in near:
            th[k] = near[k]
        elif k.startswith("l"):
            th[k] = rng.uniform(-5.0, -0.3)
        elif k.startswith("s"):
            th[k] = rng.choice([-1, 1]) * rng.uniform(0.5, 0.95)
        else:
            th[k] = rng.uniform(lo, hi)
    return th


def reduce_params(tex: Texture, n_points=6, seed=0):
    """Pin phases that do not change the observables' Jacobian rank (rephasing redundancies).

    Returns (rank, pinned names). Rank = max over n_points random points of the Jacobian rank of
    obs_vector with respect to all parameters. Greedy: drop each phase in turn if the max rank
    with it pinned to 0 is unchanged.
    """
    rng = np.random.default_rng(seed)
    pts = [random_point(tex.all_params(), rng) for _ in range(n_points)]

    def rank_with(active):
        r = 0
        for p in pts:
            def fn(t):
                full = dict(p)
                full.update({k: 0.0 for k in tex.all_params() if k not in active and k.startswith("f")})
                full.update(t)
                return tex.model(full)
            rr, _ = jac_rank(fn, active, {k: p[k] for k in active})
            r = max(r, rr)
        return r

    active = list(tex.all_params())
    full_rank = rank_with(active)
    pinned = []
    for n in [a for a in active if a.startswith("f")]:
        trial = [a for a in active if a != n]
        if rank_with(trial) == full_rank:
            active = trial
            pinned.append(n)
    tex.fixed.update({n: 0.0 for n in pinned})
    return full_rank, pinned, len(active)


# ------------------------------------------------------------------ helpers for relation models

def ckm_from_std(s12, s13, s23, d):
    return {"ckm_s12": s12, "ckm_s13": s13, "ckm_s23": s23, "ckm_delta_rad": d % TWO_PI}


def std_from_V(V):
    t12, t13, t23, d = (float(x) for x in extract_angles(V))
    return ckm_from_std(np.sin(t12), np.sin(t13), np.sin(t23), d)


def V_std(s12, s13, s23, d):
    return standard_matrix(np.arcsin(s12), np.arcsin(s13), np.arcsin(s23), d)


def alpha_of(V):
    """Unitarity-triangle alpha = arg(-V_td V_tb^* / (V_ud V_ub^*))."""
    return float(np.angle(-V[2, 0] * np.conj(V[2, 2]) / (V[0, 0] * np.conj(V[0, 2]))))


def fx_matrix(tu, td, t, phi):
    """Fritzsch-Xing parametrisation (hep-ph/9708366 Eq. 2 form), V = R12(u) (phase) R23 R12(d)^T."""
    su, cu, sd, cd, s, c = np.sin(tu), np.cos(tu), np.sin(td), np.cos(td), np.sin(t), np.cos(t)
    e = np.exp(-1j * phi)
    return np.array([[su * sd * c + cu * cd * e, su * cd * c - cu * sd * e, su * s],
                     [cu * sd * c - su * cd * e, cu * cd * c + su * sd * e, cu * s],
                     [-sd * s, -cd * s, c]], complex)

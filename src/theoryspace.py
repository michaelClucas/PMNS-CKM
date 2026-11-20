"""Shared synthetic theory-space generators (exp 033), built on src.yukawa.

    families()                          -> tuple of registered family names
    get(name)                           -> Family
    sample(name, n, rng)                -> dict of (n,) arrays (raw draws, src.evaluate keys)
    accept(obs, window=FILTER_025, ordering="NO") -> bool mask (exp 025 physical filter)
    sample_accepted(name, n, rng, ...)  -> (dict of (n,) arrays, stats)
    features_025(obs)                   -> (ctx_ckm, ctx_m, y) as exp 025 / 028 feed the flows
    register(Family(...))               -> add a family (e.g. from an experiment)

A family's sample(n, rng) returns n raw parameter draws as a dict of float arrays with the
src.evaluate observable keys (src.yukawa.OUTPUT_KEYS: ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad,
s12sq, s13sq, s23sq, dcp_deg, mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, mr_b_tau,
R_clebsch, dm_ratio, alpha21_deg, alpha31_deg) plus "io" (1.0 = inverted ordering). Draws a
family cannot produce (singular seesaw, ...) are nan rows; keys a family does not model are nan
(mr_b_tau, alpha*, for the exp 025 adapters). No measured number enters any generator.

Native families (ported from experiments/20261119-025-nn-ckm-info/gen.py, rng-call for rng-call,
so the same Generator state gives the same draws as exp 025; tests/test_theoryspace.py checks this
and the exp 025 gen_stats.json acceptance rates):
  GEN   generic shared-parameter textures: K = 2-3 Froggatt-Nielsen-suppressed complex operator
        matrices shared by quarks and leptons, GUT-like Clebsch coefficients, complex vev ratios
        for up-type and Dirac, type-I seesaw. One discrete structure per batch of 64.
  CTRL  the same generator with independent quark and lepton operator sets (no sharing).
Exp 025's matrices are RL (left rotation from M^dag M); the engine is LR, so Y_f = M_f^dag and
m_nu = M_nu = M_D^T M_R^-1 M_D as given (src.yukawa module docstring, exp 014 mapping). Unlike
exp 025 the native families also return mr_b_tau, R_clebsch and the Majorana phases.

Adapter families (not ported; F1, F2, MN, FN, MOD, SU5, S4SO): exp025_adapter(name) loads exp 025's
gen.py lazily on the first call, calls its GENS[name](rng, n) and renames its keys
(ckm_delta -> ckm_delta_rad, dcp [rad] -> dcp_deg, adds R_clebsch; mr_b_tau and Majorana phases
are nan). Those generators import their own experiments' code (f1lib, f2lib, so10fit, the
modular catalog, the SU(5) and S4xSO(10) models), so they stay where they are. Exp 025 gen.py's
rng stream is used unchanged, so adapter draws equal exp 025's for the same Generator state.
To add a heavy family from another experiment, write a function sample(n, rng) -> dict with
the keys above and register(Family(name, sample, ...)).

Filter (FILTER_025, accept): the exp 025 physical windows on log10 of the mass ratios, CKM sines
and dm_ratio, finite PMNS and CKM phase, 1e-6 < sin^2 < 1 - 1e-6, and an ordering cut ("NO" as
exp 025, "IO" as exp 028 gen_io.py, None for both).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .yukawa import OUTPUT_KEYS, observables_batch

__all__ = ["Family", "register", "families", "get", "sample", "accept", "sample_accepted", "features_025",
           "exp025_adapter", "FILTER_025", "MR_KEYS", "OBS_KEYS", "EXP025_GEN"]

OBS_KEYS = OUTPUT_KEYS + ("io",)
MR_KEYS = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau")
FILTER_025 = {
    "mr_u_c": (-5.0, -1.0), "mr_c_t": (-4.0, -1.0), "mr_d_s": (-3.0, -0.5), "mr_s_b": (-3.0, -0.5),
    "mr_e_mu": (-4.0, -1.0), "mr_mu_tau": (-3.0, -0.5),
    "ckm_s12": (float(np.log10(0.03)), float(np.log10(0.6))), "ckm_s23": (float(np.log10(3e-4)), float(np.log10(0.6))),
    "ckm_s13": (-6.0, float(np.log10(0.4))), "dm_ratio": (-3.0, float(np.log10(0.5))),
}
EXP025_GEN = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "20261119-025-nn-ckm-info" / "gen.py"


@dataclass(frozen=True)
class Family:
    name: str
    sample: Callable  # (n, rng) -> dict of (n,) arrays
    description: str = ""
    source: str = ""
    shared: bool | None = None  # quark and lepton sectors share parameters by construction
    native: bool = True         # implemented in src (False: adapter to experiment code)


_REGISTRY: dict[str, Family] = {}


def register(family: Family, overwrite: bool = False) -> Family:
    if family.name in _REGISTRY and not overwrite:
        raise ValueError(f"family {family.name!r} already registered")
    _REGISTRY[family.name] = family
    return family


def families() -> tuple:
    return tuple(_REGISTRY)


def get(name: str) -> Family:
    if name not in _REGISTRY:
        raise KeyError(f"unknown family {name!r}; registered: {families()}")
    return _REGISTRY[name]


def sample(name: str, n: int, rng) -> dict:
    """n raw draws of family `name` (module docstring)."""
    return get(name).sample(int(n), rng)


# ---------------------------------------------------------------- GEN / CTRL (port of exp 025)

CLEBSCH = np.array([1, -1, 3, -3, 1 / 3, -1 / 3, 2, 1 / 2, -1 / 2, 9 / 2, 4 / 9, 6, 0], float)
TEXTURE_BATCH = 64


def _basis(rng, B, K):
    """K shared complex 3x3 operator matrices per sample: Froggatt-Nielsen suppression
    eps^(qL_i + qR_j), eps ~ U[0.05, 0.3]; O(1) complex coefficients (|z| log-uniform [1/3, 3],
    uniform phase); a sparsity mask (zero probability 0 or 0.3); per operator symmetric (p 0.4,
    10/126-like), antisymmetric (p 0.2, 120-like) or general (p 0.4). Charges: qL = (a, b, 0),
    a ~ {1..4}, b ~ {0..a}; qR = qL (p 0.5, SO(10)-like) or drawn the same way with a ~ {0..4}.
    One discrete structure per batch. (exp 025 gen._basis, same rng calls.)"""
    eps = rng.uniform(0.05, 0.3, B)[:, None, None]
    a = int(rng.integers(1, 5))
    b = int(rng.integers(0, a + 1))
    qL = np.array([a, b, 0])
    if rng.uniform() < 0.5:
        qR = qL
    else:
        a2 = int(rng.integers(0, 5))
        qR = np.array([a2, int(rng.integers(0, a2 + 1)), 0])
    p0 = float(rng.choice([0.0, 0.3]))
    Ys = []
    for _ in range(K):
        z = 10 ** rng.uniform(-0.477, 0.477, (B, 3, 3)) * np.exp(1j * rng.uniform(0, 2 * np.pi, (B, 3, 3)))
        z = z * (rng.uniform(size=(1, 3, 3)) >= p0)
        kind = rng.choice(["sym", "anti", "gen"], p=[0.4, 0.2, 0.4])
        q = qL[:, None] + qR[None, :]
        if kind != "gen":
            q = qL[:, None] + qL[None, :]
            z = (z + np.swapaxes(z, 1, 2)) / 2 if kind == "sym" else (z - np.swapaxes(z, 1, 2)) / 2
        Ys.append(z * eps ** q[None])
    return Ys


def _combine(rng, Ys, B, vev_up):
    """sum_k c_k r_k Y_k, Clebsch c_k from CLEBSCH (not all zero); for up-type and Dirac a
    complex vev ratio r_k (|r| log-uniform 0.1-10, uniform phase). (exp 025 gen._combine.)"""
    K = len(Ys)
    while True:
        c = rng.choice(CLEBSCH, K)
        if np.any(c != 0):
            break
    M = 0
    for k in range(K):
        w = c[k]
        if vev_up:
            w = w * 10 ** rng.uniform(-1, 1, B)[:, None, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, B))[:, None, None]
        M = M + w * Ys[k]
    return M


def _sectors(rng, Yq, Yl, B):
    """(M_u, M_d, M_e, M_nu) in exp 025's RL convention; M_nu = M_D^T M_R^-1 M_D."""
    Mu = _combine(rng, Yq, B, True)
    Md = _combine(rng, Yq, B, False)
    Me = _combine(rng, Yl, B, False)
    MD = _combine(rng, Yl, B, True)
    MR = _combine(rng, [(Y + np.swapaxes(Y, 1, 2)) / 2 for Y in Yl], B, False)
    with np.errstate(all="ignore"):
        try:
            Mnu = np.swapaxes(MD, 1, 2) @ np.linalg.solve(MR, MD)
        except np.linalg.LinAlgError:
            Mnu = np.full_like(MD, np.nan)
    return Mu, Md, Me, Mnu


def _dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


def _obs_rl(Mu, Md, Me, Mnu):
    """src.yukawa engine on exp 025's RL matrices (Y_f = M_f^dag, m_nu = M_nu, symmetrised)."""
    Mnu = 0.5 * (Mnu + np.swapaxes(Mnu, 1, 2))
    return observables_batch(_dag(Mu), _dag(Md), _dag(Me), Mnu, sym_tol=np.inf)


def _gen_textures(rng, n, shared, batch=TEXTURE_BATCH):
    """exp 025 gen._gen_textures: batches of `batch` draws, rows with non-finite matrices dropped,
    truncated to n."""
    outs, got = [], 0
    while got < n:
        K = int(rng.choice([2, 3]))
        Yq = _basis(rng, batch, K)
        Yl = Yq if shared else _basis(rng, batch, int(rng.choice([2, 3])))
        mats = _sectors(rng, Yq, Yl, batch)
        bad = ~np.all(np.isfinite(np.stack(mats)), axis=(0, 2, 3))
        if bad.all():
            continue
        mats = [m[~bad] for m in mats]
        outs.append(_obs_rl(*mats))
        got += len(mats[0])
    return {k: np.concatenate([o[k] for o in outs])[:n] for k in outs[0]}


def gen_textures(n, rng):
    return _gen_textures(rng, n, True)


def ctrl_textures(n, rng):
    return _gen_textures(rng, n, False)


register(Family("GEN", gen_textures, "generic shared textures (exp 025 GEN)",
                "experiments/20261119-025-nn-ckm-info/gen.py gen_gen", shared=True, native=True))
register(Family("CTRL", ctrl_textures, "GEN with independent quark and lepton draws (exp 025 CTRL)",
                "experiments/20261119-025-nn-ckm-info/gen.py gen_ctrl", shared=False, native=True))


# ---------------------------------------------------------------- exp 025 adapters

_EXP025 = None


def _exp025_module():
    """exp 025 gen.py, loaded once under a private module name (its own imports are src + numpy;
    each GENS[...] loads its experiment's code on demand)."""
    global _EXP025
    if _EXP025 is None:
        if not EXP025_GEN.exists():
            raise FileNotFoundError(f"exp 025 generator not found at {EXP025_GEN}")
        spec = importlib.util.spec_from_file_location("_exp025_gen", EXP025_GEN)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_exp025_gen"] = mod
        spec.loader.exec_module(mod)
        _EXP025 = mod
    return _EXP025


def from_exp025(o: dict) -> dict:
    """Rename an exp 025 observables dict to src.evaluate keys (module docstring)."""
    n = len(o["ckm_s12"])
    out = {k: np.full(n, np.nan) for k in OBS_KEYS}
    for k in OBS_KEYS:
        if k in o:
            out[k] = np.asarray(o[k], float)
    out["ckm_delta_rad"] = np.asarray(o["ckm_delta"], float)
    out["dcp_deg"] = np.degrees(np.asarray(o["dcp"], float))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["R_clebsch"] = out["mr_e_mu"] / out["mr_d_s"]
    return out


def exp025_adapter(name: str) -> Callable:
    """sample(n, rng) for exp 025 family `name` (F1, F2, MN, FN, MOD, SU5, S4SO, GEN, CTRL)."""
    def sample_fn(n, rng):
        return from_exp025(_exp025_module().GENS[name](rng, n))
    sample_fn.__name__ = f"exp025_{name}"
    return sample_fn


_ADAPTED = {
    "F1": ("shared rotation U_PMNS = A Phi B of exp 007; masses log-uniform in the filter windows", True),
    "F2": ("Clebsch charged-lepton corrections of exp 009; masses log-uniform in the filter windows", True),
    "MN": ("minimal SO(10) 10+126bar type I (exp 014), broad box", True),
    "FN": ("real SO(10) 10+126bar+120 (exp 014)", True),
    "MOD": ("A4/S4/A5 common-tau modular catalog (exp 017)", True),
    "SU5": ("SU(5) sparse Clebsch + CSD3 (exp 022)", False),
    "S4SO": ("S4 x SO(10) (exp 023), +-2 decades around the published point", True),
}
for _n, (_d, _s) in _ADAPTED.items():
    register(Family(_n, exp025_adapter(_n), _d + " [adapter: exp 025 gen.py]",
                    f"experiments/20261119-025-nn-ckm-info/gen.py GENS['{_n}']", shared=_s, native=False))


# ---------------------------------------------------------------- filter and features

def accept(obs: dict, window: dict | None = None, ordering: str | None = "NO") -> np.ndarray:
    """exp 025 physical filter (gen.features) on a src.evaluate-keyed dict; ordering "NO" (exp 025),
    "IO" (exp 028 gen_io.py) or None (no ordering cut)."""
    window = FILTER_025 if window is None else window
    n = len(obs["ckm_s12"])
    ok = np.ones(n, bool)
    with np.errstate(all="ignore"):
        for k, (lo, hi) in window.items():
            lg = np.log10(np.asarray(obs[k], float))
            ok &= np.isfinite(lg) & (lg >= lo) & (lg <= hi)
        for k in ("s12sq", "s13sq", "s23sq", "dcp_deg", "ckm_delta_rad"):
            ok &= np.isfinite(obs[k])
        for k in ("s12sq", "s13sq", "s23sq"):
            ok &= (obs[k] > 1e-6) & (obs[k] < 1 - 1e-6)
        io = np.asarray(obs["io"], float)
        if ordering == "NO":
            ok &= io < 0.5
        elif ordering == "IO":
            ok &= io > 0.5
        elif ordering is not None:
            raise ValueError("ordering must be 'NO', 'IO' or None")
    return ok


def sample_accepted(name: str, n: int, rng, window: dict | None = None, ordering: str | None = "NO",
                    chunk: int | None = None, max_raw: int | None = None):
    """First n accepted draws of a family, generated in chunks as exp 025 gen._task did (first chunk
    max(2000, n // 4), later chunks sized from the running acceptance, capped at 200000).
    Returns (obs dict of (<= n,) arrays, {"n_raw", "n_pass", "acceptance", "kept"}). Stops early
    (fewer than n rows) once max_raw raw draws (default 2000 n) have been made."""
    max_raw = 2000 * n if max_raw is None else max_raw
    chunk = max(2000, n // 4) if chunk is None else int(chunk)
    parts, n_raw, n_ok = [], 0, 0
    while n_ok < n and n_raw < max_raw:
        o = sample(name, chunk, rng)
        ok = accept(o, window, ordering)
        parts.append({k: v[ok] for k, v in o.items()})
        n_raw += len(ok)
        n_ok += int(ok.sum())
        if n_ok < n and n_ok > 0:
            chunk = int(min(200000, max(2000, (n - n_ok) * n_raw / n_ok * 1.1)))
    obs = {k: np.concatenate([p[k] for p in parts])[:n] for k in parts[0]} if parts else {}
    kept = len(obs.get("ckm_s12", ()))
    return obs, {"n_raw": n_raw, "n_pass": n_ok, "acceptance": n_ok / max(n_raw, 1), "kept": kept}


def features_025(obs: dict):
    """(ctx_ckm (B, 5), ctx_m (B, 6), y (B, 5)) exactly as exp 025 gen.features builds them:
    ctx_ckm = log10 of the three CKM sines, cos and sin of the CKM phase; ctx_m = log10 of MR_KEYS;
    y = s12sq, s13sq, s23sq, dcp in rad mod 2 pi, log10 dm_ratio."""
    with np.errstate(all="ignore"):
        d = np.asarray(obs["ckm_delta_rad"], float)
        ctx_ckm = np.stack([np.log10(obs["ckm_s12"]), np.log10(obs["ckm_s13"]), np.log10(obs["ckm_s23"]),
                            np.cos(d), np.sin(d)], 1)
        ctx_m = np.stack([np.log10(obs[k]) for k in MR_KEYS], 1)
        y = np.stack([obs["s12sq"], obs["s13sq"], obs["s23sq"], np.mod(np.radians(obs["dcp_deg"]), 2 * np.pi),
                      np.log10(obs["dm_ratio"])], 1)
    return ctx_ckm, ctx_m, y

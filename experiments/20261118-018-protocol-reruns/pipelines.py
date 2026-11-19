"""Frozen pipelines for the criterion-4 protocol re-runs of leads T1 (exp 010) and F2-LO (exp 009).

Each pipeline(universe) reruns every data-dependent choice of the original search on one
src.nulls.Universe and returns a mapping with the src.nulls summary fields (N, k, k_discrete,
max_abs_pull, p, p_global, chi2, chi2_global, ordering, pred) plus a label. Measured numbers
come only from the universe (context(), load_masses(), load_pmns(), map_to_real()); the final
result is scored by universe.evaluate (src.evaluate). The vectorized screens below apply the
universe's own table translation (map_to_real) to the shifted NuFIT tables that
universe.pmns_delta_chi2 uses, and the winner is rescored through universe.evaluate.

T1   exp 010 multi-observable template search. One f(x) from the C <= 5 one-variable grammar
     (29,704 distinct f), 4 sector mappings (ckm, down, lepton, up) for (theta12, theta13,
     theta23), 4 angle subsets (12+13+23, 12+13, 12+23, 13+23), 2 scales (M_Z, 2e16 GeV SM
     running via src.rg.ckm_at with the universe's CKM and masses), 2 orderings. Validity rules
     as exp 010 (angle in [0, pi/2]; propagated input error <= NuFIT NO 1-sigma width, with the
     exp 010 relative input-error model scaled to the universe's inputs, as exp 010 did for its
     fakes). Cell score = sum of the 1D NuFIT profiles of the subset's angles (what
     src.evaluate uses for s_ij^2 without dcp). Change vs exp 010: input errors are not
     profiled in the score (src.evaluate cannot), see README. Selection: best template per
     (scale, subset, ordering) by chi2, then the cell with the largest p_global.
     k = ceil(log2(118,816 templates x 4 subsets x 2 scales)) = 20 bits + 1 ordering bit.
F2LO exp 009 LO tiers A+B+C (17,220 Clebsch configs) with fitted phi, M_Z CKM from the
     universe. Per ordering: 180-point phi grid, golden refinement of the 100 best rows
     (f2lib.search, unchanged), best config. Then the ordering with the lower chi2_global.
     k = 1 (phi) + ceil(log2 17,220) = 15 bits + 1 ordering bit = 17.
"""
from __future__ import annotations

import functools
import math
import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments" / "20261118-010-symbolic-masses",
          ROOT / "experiments" / "20261118-009-f2-clebsch"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
from scipy.stats import chi2 as _chi2  # noqa: E402

from src import data as _data  # noqa: E402
from src.stats import discrete_bits  # noqa: E402

ART = ROOT / "artifacts" / "protocol"
PMNS4 = ("s12sq", "s13sq", "s23sq", "dcp_deg")


# ---------------------------------------------------------------- universe-consistent vector scoring

def vec_dchi2(u, o, s12=None, s13=None, s23=None, dcp=None):
    """Vectorized universe.pmns_delta_chi2(o, ...) (method auto, relative_to local). NaN -> inf."""
    m = u.map_to_real(o, s12sq=s12, s13sq=s13, s23sq=s23, dcp_deg=dcp)
    tot = 0.0
    if m["s23sq"] is not None and m["dcp_deg"] is not None:
        x, y, z = _data._shifted_grid_2d(u.analysis, o, "T23-DCP")
        sv = np.asarray(m["s23sq"], float)
        d = (np.asarray(m["dcp_deg"], float) + 180.0) % 360.0 - 180.0
        sv, d = np.broadcast_arrays(sv, d)
        i = np.clip(np.searchsorted(x, sv) - 1, 0, len(x) - 2)
        j = np.clip(np.searchsorted(y, d) - 1, 0, len(y) - 2)
        tx = (sv - x[i]) / (x[i + 1] - x[i])
        ty = (d - y[j]) / (y[j + 1] - y[j])
        v2 = ((1 - tx) * (1 - ty) * z[i, j] + tx * (1 - ty) * z[i + 1, j]
              + (1 - tx) * ty * z[i, j + 1] + tx * ty * z[i + 1, j + 1])
        bad = (sv < x[0]) | (sv > x[-1]) | (d < y[0]) | (d > y[-1]) | ~np.isfinite(sv) | ~np.isfinite(d)
        tot = tot + np.where(bad, np.inf, v2)
        rest = {"s12sq": m["s12sq"], "s13sq": m["s13sq"]}
    else:
        rest = {k: m[k] for k in PMNS4}
    for k, v in rest.items():
        if v is None:
            continue
        r = _data.pmns_profile_1d(o, k, v, analysis=u.analysis)
        tot = tot + np.where(np.isfinite(r), r, np.inf)
    return tot


def _summary(res, label, extra=None):
    out = {f: getattr(res, f) for f in ("N", "k", "k_discrete", "max_abs_pull", "p", "p_global",
                                         "chi2", "chi2_global", "ordering")}
    out["pred"] = dict(res.pred)
    out["pulls"] = dict(res.pulls)
    out["label"] = label
    if extra:
        out.update(extra)
    return out


# ---------------------------------------------------------------- T1: template search (exp 010)

T1_CT = 5
T1_MAPS = ("ckm", "down", "lepton", "up")
T1_ANG = ("s12sq", "s13sq", "s23sq")
T1_SUBSETS = {"12+13+23": (0, 1, 2), "12+13": (0, 1), "12+23": (0, 2), "13+23": (1, 2)}
T1_SCALES = {"MZ": None, "GUT": 2e16}


@functools.lru_cache(maxsize=None)
def t1_templates():
    """The fixed C <= 5 one-variable grammar of exp 010 templates.py (built once, cached)."""
    f = ART / f"t1_templates_C{T1_CT}.npz"
    if f.exists():
        z = np.load(f)
        return z["op"], z["a"], z["b"], z["comp"]
    import symlib as S
    E = S.Enum(T1_CT, leaf_set=[0] + list(range(S.N_VAR, S.N_LEAF)))
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, op=E.op, a=E.a, b=E.b, comp=E.comp)
    return E.op, E.a, E.b, E.comp


T1_N_F = None  # filled lazily


def t1_bits():
    n = len(t1_templates()[0])
    return discrete_bits(4 * n * len(T1_SUBSETS) * len(T1_SCALES)), n


@functools.lru_cache(maxsize=None)
def _t1_rel_dirs(scale):
    """exp 010 relative input-error model: real 1-sigma directions / real base (absolute for delta).
    A fixed property of the search (the Universe keeps the real errors), computed from the
    exp 010 cache artifacts/exp010/real_inputs_<scale>.json (rebuilt by symlib if missing)."""
    import symlib as S
    real = S.real_inputs(scale)
    rel = real["base"].copy()
    rel[3] = 1.0
    return real["dirs"] / rel[None, :]


def _mz():
    from src import rg
    return rg.load_sm_inputs()["pdg2026"]["M_Z"]["value"]


def t1_base(u, scale):
    """[t12q, t13q, t23q, dq, r_uc, r_ct, r_ds, r_sb, r_em, r_mt] at the scale, from the universe."""
    from src import rg
    ctx = u.context()
    ckm = (ctx["theta_C"], ctx["theta_ub"], ctx["theta_cb"], ctx["delta_q"])
    mu = _mz() if T1_SCALES[scale] is None else T1_SCALES[scale]
    r = rg.ckm_at(mu, ckm=ckm, masses=u.masses)
    rr = r["ratios"]
    return np.array([r["theta12"], r["theta13"], r["theta23"], r["delta"] % (2 * np.pi),
                     rr["mu_over_mc"], rr["mc_over_mt"], rr["md_over_ms"], rr["ms_over_mb"],
                     rr["me_over_mmu"], rr["mmu_over_mtau"]])


def _t1_mapped(base, dirs):
    """exp 010 templates.mapped: map -> [(x, dx)] for (12, 13, 23)."""
    q = lambda i: np.sqrt(base[i])
    dq = lambda i: 0.5 / np.sqrt(base[i]) * dirs[:, i]
    out = {"ckm": [(base[0], dirs[:, 0]), (base[1], dirs[:, 1]), (base[2], dirs[:, 2])]}
    for name, (i12, i23) in {"down": (6, 7), "lepton": (8, 9), "up": (4, 5)}.items():
        x12, x23 = q(i12), q(i23)
        d12, d23 = dq(i12), dq(i23)
        out[name] = [(x12, d12), (x12 * x23, x23 * d12 + x12 * d23), (x23, d23)]
    return out


def _t1_eval(x, dx):
    import symlib as S
    op, a, b, comp = t1_templates()
    base = np.zeros(10)
    base[0] = x
    base[4:] = 1.0
    dirs = np.zeros((len(dx), 10))
    dirs[:, 0] = dx
    return S.evaluate(op, a, b, comp, base, dirs)


def _t1_predictions(v, T, sig_max):
    """exp 010 symlib.predictions for a mixing angle: p = sin^2 v, nan where invalid."""
    sig_v = np.sqrt((T ** 2).sum(axis=1))
    with np.errstate(all="ignore"):
        ok = (v >= 0) & (v <= np.pi / 2)
        p = np.sin(v) ** 2
        s = np.abs(np.sin(2 * v)) * sig_v
        s = np.where(np.isfinite(s), s, np.inf)
        ok &= s <= sig_max
    return np.where(ok, p, np.nan)


_T1_PRED_CACHE = {}


def t1_preds(u):
    """scale -> map -> (3, n_f) predicted s_ij^2 (nan = invalid). Cached on the universe's inputs."""
    key = (tuple(sorted((k, v["value"]) for k, v in u.load_ckm().items())),
           None if u.masses is None else tuple(sorted((k, v["value"]) for k, v in u.load_masses().items())))
    if key in _T1_PRED_CACHE:
        return _T1_PRED_CACHE[key]
    pn = u.load_pmns("NO")
    sig = {t: 0.5 * (pn[t]["err_plus"] + pn[t]["err_minus"]) for t in T1_ANG}
    out = {}
    for scale in T1_SCALES:
        base = t1_base(u, scale)
        rel = base.copy()
        rel[3] = 1.0
        dirs = _t1_rel_dirs(scale) * rel[None, :]
        out[scale] = {}
        for m, lst in _t1_mapped(base, dirs).items():
            rows = []
            for k, (x, dx) in enumerate(lst):
                v, T = _t1_eval(x, dx)
                rows.append(_t1_predictions(v, T, sig[T1_ANG[k]]))
            out[scale][m] = np.stack(rows)
    if len(_T1_PRED_CACHE) > 8:
        _T1_PRED_CACHE.clear()
    _T1_PRED_CACHE[key] = out
    return out


def t1_cells(u):
    """All (scale, subset, ordering) cells: best template and its chi2 (vector screen)."""
    preds = t1_preds(u)
    cells = []
    for scale in T1_SCALES:
        for o in _data.ORDERINGS:
            d = {}
            for m in T1_MAPS:
                P = preds[scale][m]
                d[m] = np.stack([vec_dchi2(u, o, **{("s12", "s13", "s23")[k]: P[k]}) for k in range(3)])
            for sname, idx in T1_SUBSETS.items():
                best = (np.inf, None, None)
                for m in T1_MAPS:
                    tot = d[m][list(idx)].sum(axis=0)
                    tot = np.where(np.isfinite(tot), tot, np.inf)
                    j = int(np.argmin(tot))
                    if tot[j] < best[0]:
                        best = (float(tot[j]), m, j)
                cells.append({"scale": scale, "subset": sname, "ordering": o, "chi2": best[0],
                              "map": best[1], "f": best[2]})
    return cells


T1_INPUTS = {"ckm": ("ckm_s12", "ckm_s13", "ckm_s23"), "down": ("m_d", "m_s", "m_b"),
             "lepton": ("m_e", "m_mu", "m_tau"), "up": ("m_u", "m_c", "m_t_direct")}


def t1_select(cells):
    def pg(c):
        if not np.isfinite(c["chi2"]):
            return -1.0
        off = _data.ordering_offset(c["ordering"]) if c["ordering"] == "IO" else 0.0
        return float(_chi2.sf(c["chi2"] + off, len(T1_SUBSETS[c["subset"]])))
    return max(cells, key=lambda c: (pg(c), -c["chi2"]))


def pipeline_t1(u):
    cells = t1_cells(u)
    c = t1_select(cells)
    if not np.isfinite(c["chi2"]):
        return None
    preds = t1_preds(u)
    idx = T1_SUBSETS[c["subset"]]
    pred = {T1_ANG[k]: float(preds[c["scale"]][c["map"]][k, c["f"]]) for k in idx}
    bits, _ = t1_bits()
    res = u.evaluate(pred, use=tuple(pred), ordering=c["ordering"], k=bits, k_discrete=bits,
                     inputs=T1_INPUTS[c["map"]], ordering_selected=True)
    return _summary(res, f"{c['scale']}|{c['subset']}|{c['map']}|f#{c['f']}",
                    {"cell": {k: c[k] for k in ("scale", "subset", "ordering", "map", "f")},
                     "screen_chi2": c["chi2"]})


def t1_expr(j):
    import symlib as S
    op, a, b, comp = t1_templates()
    return S.expr_str(op, a, b, j).replace("t12q", "x")


# ---------------------------------------------------------------- F2-LO: Clebsch enumeration (exp 009)

@functools.lru_cache(maxsize=None)
def f2_cfgs():
    import f2lib as L
    return tuple(c for c in L.configs() if c["tier"] in ("A", "B", "C"))


_F2_CFG_LIST = None


def _f2_cfg_list():
    global _F2_CFG_LIST
    if _F2_CFG_LIST is None:
        _F2_CFG_LIST = list(f2_cfgs())  # one object, so f2lib._structure caches it
    return _F2_CFG_LIST


F2_NPHI, F2_NREF = 180, 100
_F2_OBS_CACHE = {}


def f2_obs(ckm):
    """Grid observables for all LO rows at one CKM: (P, Q, phis, (s12, s13, s23, dcp) each (R, nphi))."""
    if ckm in _F2_OBS_CACHE:
        return _F2_OBS_CACHE[ckm]
    import f2lib as L
    P, Q, rc, _ = L.build_PQ(_f2_cfg_list(), ckm)
    phis = np.arange(F2_NPHI) * (2 * np.pi / F2_NPHI)
    ph = np.exp(1j * phis)[None, :, None]
    R = len(P)
    G = [np.empty((R, F2_NPHI)) for _ in range(4)]
    for a in range(0, R, 500):
        b = min(a + 500, R)
        o = L.fast_obs(P[a:b, None, :] + ph * Q[a:b, None, :])
        for k in range(4):
            G[k][a:b] = o[k]
    out = (P, Q, rc, phis, G)
    _F2_OBS_CACHE.clear()
    _F2_OBS_CACHE[ckm] = out
    return out


def f2_search(u, o, ckm):
    """f2lib.search (fit variant) for one ordering with the universe's tables -> (chi2, row, phi)."""
    import f2lib as L
    P, Q, rc, phis, G = f2_obs(ckm)
    scorer = lambda s12, s13, s23, d: vec_dchi2(u, o, s12, s13, s23, d)
    R = len(P)
    best, arg = np.empty(R), np.empty(R)
    for a in range(0, R, 2000):
        b = min(a + 2000, R)
        c = scorer(*(g[a:b] for g in G))
        i = np.argmin(c, axis=1)
        best[a:b] = c[np.arange(b - a), i]
        arg[a:b] = phis[i]
    top = np.argsort(best)[:F2_NREF]
    phi_r, fr = L.golden_refine(P[top], Q[top], scorer, arg[top], half=2 * np.pi / F2_NPHI)
    phi_best = arg.copy()
    better = fr < best[top]
    best[top] = np.where(better, fr, best[top])
    phi_best[top] = np.where(better, phi_r, arg[top])
    r = int(np.argmin(best))
    return float(best[r]), r, float(phi_best[r])


def f2_pred(ckm, row, phi):
    import f2lib as L
    P, Q, rc, phis, G = f2_obs(ckm)
    s12, s13, s23, d = L.fast_obs(P[row] + np.exp(1j * phi) * Q[row])
    return {"s12sq": float(s12), "s13sq": float(s13), "s23sq": float(s23), "dcp_deg": float(d)}


F2_INPUTS = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta_rad")


def pipeline_f2lo(u):
    ctx = u.context()
    ckm = (ctx["theta_C"], ctx["theta_ub"], ctx["theta_cb"], ctx["delta_q"])
    bits = discrete_bits(len(f2_cfgs()))
    per = {}
    for o in _data.ORDERINGS:
        c2, row, phi = f2_search(u, o, ckm)
        if not np.isfinite(c2):
            continue
        pred = f2_pred(ckm, row, phi)
        per[o] = (u.evaluate(pred, use=PMNS4, ordering=o, k=1 + bits, k_discrete=bits, inputs=F2_INPUTS),
                  row, phi, c2)
    if not per:
        return None
    o = min(per, key=lambda k: per[k][0].chi2_global)
    r0, row, phi, c2 = per[o]
    res = u.evaluate(r0.pred, use=PMNS4, ordering=o, k=1 + bits, k_discrete=bits, inputs=F2_INPUTS,
                     ordering_selected=True)
    P, Q, rc, _, _ = f2_obs(ckm)
    name = _f2_cfg_list()[int(np.nonzero(np.array([c["id"] for c in _f2_cfg_list()]) == rc[row])[0][0])]["name"]
    return _summary(res, f"{name}|phi={math.degrees(phi):.3f}",
                    {"screen_chi2": c2, "phi_deg": math.degrees(phi), "config": name,
                     "per_ordering_chi2": {k: v[0].chi2 for k, v in per.items()}})

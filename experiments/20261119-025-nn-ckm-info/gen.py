"""Synthetic theory-space mixture for the NN CKM -> PMNS information test (exp 025).

Each family generator returns raw observables for a batch of parameter draws. No measured
number enters any generator (exp 023's prior is centred on the paper's published point, not on
our data). The physical filter (FILTER below) is applied after generation.

Families (code reused with attribution; see README "Families and priors"):
  F1    U_PMNS = A Phi B shared-rotation forms of exp 007 (f1lib.x_catalog/_forms/PERMS)
  F2    U_PMNS = V_e(Clebsch x CKM angles)^dag D(phi) U_nu of exp 009 (f2lib.configs/full_matrix)
  MN    minimal SO(10) 10+126bar type I, exp 014 so10fit.mn_mats (quark-anchored parametrisation)
  FN    real SO(10) 10+126bar+120, exp 014 so10fit.fn_mats / fn_box
  MOD   A4/S4/A5 common-tau catalog of exp 017 (models.charged_matrix/nu_matrix, search box)
  SU5   SU(5) sparse Clebsch + CSD3 of exp 022 (model.predict, fit.BOUNDS)
  S4SO  S4 x SO(10) of exp 023 (model.observables; heldout.py prior: +-2 decades log-uniform
        around the paper's Table 4 point, phases uniform)
  GEN   generic shared-parameter textures (new, this file)
  CTRL  same generator as GEN with independent quark and lepton draws (no sharing)

Usage: python gen.py [n_per_family] [seed]   -> artifacts/025/data.npz
"""
from __future__ import annotations

import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import importlib.util
import json
import multiprocessing as mp
import pathlib
import sys
import time
from fractions import Fraction

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / "experiments"
ART = ROOT / "artifacts" / "025"
sys.path.insert(0, str(ROOT))

from src.mixing import extract_angles, standard_matrix  # noqa: E402
from src.nulls import N1_PRIOR  # noqa: E402
from src.mixing import wolfenstein_matrix  # noqa: E402

FAMILIES = ("F1", "F2", "MN", "FN", "MOD", "SU5", "S4SO", "GEN", "CTRL")

# ---------------------------------------------------------------- physical filter (log10 windows)
FILTER = {
    "mr_u_c": (-5.0, -1.0), "mr_c_t": (-4.0, -1.0), "mr_d_s": (-3.0, -0.5), "mr_s_b": (-3.0, -0.5),
    "mr_e_mu": (-4.0, -1.0), "mr_mu_tau": (-3.0, -0.5),
    "ckm_s12": (np.log10(0.03), np.log10(0.6)), "ckm_s23": (np.log10(3e-4), np.log10(0.6)),
    "ckm_s13": (-6.0, np.log10(0.4)), "dm_ratio": (-3.0, np.log10(0.5)),
}
MR_KEYS = ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau")


def _load(name, path, extra_path=None):
    if extra_path is not None and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def dag(M):
    return np.conj(np.swapaxes(M, -1, -2))


# ---------------------------------------------------------------- common observable engine
def obs_from_mats(Mu, Md, Me, Mnu):
    """RL convention (exp 014 so10fit.raw_obs, exp 017 models): left rotation from M^dag M,
    CKM = Uu^dag Ud, PMNS = Ue^dag Unu, Unu from Mnu^dag Mnu, ordering from the spectrum
    (solar pair = the closer pair). Returns a dict of arrays."""
    def lb(M):
        w, U = np.linalg.eigh(dag(M) @ M)
        return np.sqrt(np.clip(w, 0, None)), U
    mu, Uu = lb(Mu)
    md, Ud = lb(Md)
    me, Ue = lb(Me)
    mn, Un = lb(Mnu)
    q12, q13, q23, dq = extract_angles(dag(Uu) @ Ud)
    a2, b2, c2 = mn[:, 0] ** 2, mn[:, 1] ** 2, mn[:, 2] ** 2
    io = (b2 - a2) > (c2 - b2)
    Un = np.where(io[:, None, None], Un[:, :, [1, 2, 0]], Un)
    l12, l13, l23, dl = extract_angles(dag(Ue) @ Un)
    dm21 = np.where(io, c2 - b2, b2 - a2)
    dm3l = c2 - a2
    with np.errstate(all="ignore"):
        return dict(ckm_s12=np.sin(q12), ckm_s13=np.sin(q13), ckm_s23=np.sin(q23), ckm_delta=dq,
                    mr_u_c=mu[:, 0] / mu[:, 1], mr_c_t=mu[:, 1] / mu[:, 2],
                    mr_d_s=md[:, 0] / md[:, 1], mr_s_b=md[:, 1] / md[:, 2],
                    mr_e_mu=me[:, 0] / me[:, 1], mr_mu_tau=me[:, 1] / me[:, 2],
                    s12sq=np.sin(l12) ** 2, s13sq=np.sin(l13) ** 2, s23sq=np.sin(l23) ** 2, dcp=dl,
                    dm_ratio=dm21 / dm3l, io=io.astype(float))


def _mix_only(rng, V_list, U_list):
    """Families without a mass model: CKM/PMNS from matrices, masses and dm ratio drawn
    log-uniform inside the filter windows, independently of everything (both orderings)."""
    V, U = np.stack(V_list), np.stack(U_list)
    q12, q13, q23, dq = extract_angles(V)
    l12, l13, l23, dl = extract_angles(U)
    n = len(V)
    out = dict(ckm_s12=np.sin(q12), ckm_s13=np.sin(q13), ckm_s23=np.sin(q23), ckm_delta=dq,
               s12sq=np.sin(l12) ** 2, s13sq=np.sin(l13) ** 2, s23sq=np.sin(l23) ** 2, dcp=dl,
               dm_ratio=10 ** rng.uniform(*FILTER["dm_ratio"], n), io=(rng.uniform(size=n) < 0.5).astype(float))
    for k in MR_KEYS:
        out[k] = 10 ** rng.uniform(*FILTER[k], n)
    return out


def _n1_ckm(rng):
    w = [rng.uniform(*N1_PRIOR[k]) for k in ("lambda", "A", "rhobar", "etabar")]
    return wolfenstein_matrix(*w)


# ---------------------------------------------------------------- F1 (exp 007)
def gen_f1(rng, n):
    f1 = _load("f1lib", EXP / "20261118-007-f1-shared-rotation" / "f1lib.py")
    cat = f1.x_catalog()
    names = list(cat)
    P = [f1._pmat(p) for p in f1.PERMS]
    Vs, Us = [], []
    for _ in range(n):
        V = _n1_ckm(rng)
        X = P[rng.integers(6)] @ cat[names[rng.integers(len(names))]] @ P[rng.integers(6)]
        forms = f1._forms(V, X)
        A, B = forms[f1.FORM_NAMES[rng.integers(len(f1.FORM_NAMES))]]
        a, b = rng.uniform(0, 2 * np.pi, 2)
        Us.append(A @ np.diag([1, np.exp(1j * a), np.exp(1j * b)]) @ B)
        Vs.append(V)
    return _mix_only(rng, Vs, Us)


# ---------------------------------------------------------------- F2 (exp 009)
def gen_f2(rng, n):
    f2 = _load("f2lib", EXP / "20261118-009-f2-clebsch" / "f2lib.py")
    cfgs = f2.configs()
    Vs, Us = [], []
    for _ in range(n):
        V = _n1_ckm(rng)
        t12, t13, t23, d = (float(x) for x in extract_angles(V))
        cfg = cfgs[rng.integers(len(cfgs))]
        th_nu = rng.uniform(0, 0.5) if cfg["tier"] == "TM" else None
        Us.append(f2.full_matrix(cfg, (t12, t13, t23, d), rng.uniform(0, 2 * np.pi), th_nu))
        Vs.append(V)
    return _mix_only(rng, Vs, Us)


# ---------------------------------------------------------------- SO(10) MN / FN (exp 014)
def _so10():
    return _load("so10fit", EXP / "20261118-014-so10-fits" / "so10fit.py")


def mn_box_broad():
    """exp 014 mn_box with the data-centred windows replaced by fixed broad ones (no measured
    number): quark mass log-ratios +-1 decade around round hierarchical values, CKM inputs over
    the N1 range."""
    T = 2 * np.pi
    b = [(-3.5, -1.5), (-3.5, -1.5), (1.0, 3.0), (-2.3, -0.5), (-2.7, -0.7),
         (0.05, 0.5), (-4.0, -0.5), (-3.0, -0.3), (0, T)] + [(0, T)] * 5 + [(0.0, 4.0), (-2.0, 2.0), (0, T)]
    b = np.array(b, float)
    return b[:, 0], b[:, 1]


def gen_mn(rng, n):
    so = _so10()
    lo, hi = mn_box_broad()
    th = rng.uniform(lo, hi, size=(n, len(lo)))
    return obs_from_mats(*so.mn_mats(th))


def gen_fn(rng, n):
    so = _so10()
    lo, hi = so.fn_box("ours")
    th = rng.uniform(lo, hi, size=(n, len(lo)))
    return obs_from_mats(*so.fn_mats(th))


# ---------------------------------------------------------------- modular common tau (exp 017)
def _mod_charged(md, N, qrep, rowset, tau1, logalpha, extra):
    """models.charged_matrix (exp 017) with the forms evaluated at one tau and broadcast over
    the coupling draws (logalpha (B, 2), extra (B, n_extra))."""
    mf = sys.modules["modforms"]
    B = logalpha.shape[0]
    la = np.concatenate([np.zeros((B, 1)), logalpha], axis=1)
    rows, ie = [], 0
    for i, (w, r, n) in enumerate(rowset):
        v = mf.eval_series(mf.yukawa_row_structures(N, w, r, qrep), tau1, N)  # (1, n, 3)
        row = np.broadcast_to(v[:, 0], (B, 3))
        if n == 2:
            row = row + extra[:, ie: ie + 1] * v[:, 1]
            ie += 1
        rows.append(np.exp(la[:, i: i + 1]) * row)
    return np.stack(rows, axis=1)


def _mod_nu(md, N, lrep, nu, tau1, g, B):
    mf = sys.modules["modforms"]
    w, n = nu
    v = mf.eval_series(mf.weinberg_structures(N, w, lrep), tau1, N)  # (1, n, 3, 3)
    M = np.broadcast_to(v[:, 0], (B, 3, 3)).copy()
    for a in range(1, n):
        M = M + g[:, a - 1, None, None] * v[:, a]
    return M


def gen_mod(rng, n, batch=2000):
    """One (group, quark assignment, lepton assignment, tau) per batch, `batch` coupling draws at
    that tau (exact prior sampling: tau, assignments and couplings independent uniform as in exp 017;
    the modular forms are evaluated once per batch and broadcast)."""
    d = EXP / "20261118-017-modular-search"
    md = _load("models", d / "models.py", extra_path=d)
    LA, G = (-14.0, 4.0), (-6.0, 6.0)  # search.py LA_RANGE, G_RANGE
    outs, got = [], 0
    while got < n:
        N = int(rng.choice([3, 4, 5]))
        qa = md.quark_catalog(N)[rng.integers(len(md.quark_catalog(N)))]
        la = md.lepton_catalog(N)[rng.integers(len(md.lepton_catalog(N)))]
        B = batch
        tau = md.tau_from_unit(rng.uniform(size=(1, 2)))
        nu_, nd_, ne_ = md.n_extra(qa.u), md.n_extra(qa.d), md.n_extra(la.e)
        Mu = _mod_charged(md, N, qa.qrep, qa.u, tau, rng.uniform(*LA, (B, 2)), rng.uniform(*G, (B, nu_)))
        Md = _mod_charged(md, N, qa.qrep, qa.d, tau, rng.uniform(*LA, (B, 2)), rng.uniform(*G, (B, nd_)))
        Me = _mod_charged(md, N, la.lrep, la.e, tau, rng.uniform(*LA, (B, 2)), rng.uniform(*G, (B, ne_)))
        Mn = _mod_nu(md, N, la.lrep, la.nu, tau, rng.uniform(*G, (B, la.nu[1] - 1)), B)
        outs.append(obs_from_mats(Mu, Md, Me, Mn))
        got += B
    return {k: np.concatenate([o[k] for o in outs])[:n] for k in outs[0]}


# ---------------------------------------------------------------- SU(5) + CSD3 (exp 022)
def gen_su5(rng, n):
    d = EXP / "20261119-022-su5-csd3"
    M = _load("su5model", d / "model.py")
    rows = []
    for _ in range(n):
        u = tuple(rng.uniform(-10, 10, 5)) + (1.0,)
        dd = tuple(rng.uniform(-10, 10, 3)) + (1.0,)
        p = M.predict(u, dd, rng.uniform(0, 1))
        rows.append((p["ckm_s12"], p["ckm_s13"], p["ckm_s23"], p["ckm_delta_rad"], p["mr_u_c"], p["mr_c_t"],
                     p["mr_d_s"], p["mr_s_b"], p["mr_e_mu"], p["mr_mu_tau"], p["s12sq"], p["s13sq"], p["s23sq"],
                     np.radians(p["dcp_deg"]), p["dm_ratio"], 0.0))
    a = np.array(rows, float).T
    keys = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta") + MR_KEYS + ("s12sq", "s13sq", "s23sq", "dcp", "dm_ratio", "io")
    return dict(zip(keys, a))


# ---------------------------------------------------------------- S4 x SO(10) (exp 023)
def gen_s4so(rng, n):
    d = EXP / "20261119-023-s4so10"
    M = _load("s4so10model", d / "model.py")
    pub = M.PUBLISHED
    rows = []
    for _ in range(n):
        p = {k: np.sign(pub[k]) * abs(pub[k]) * 10 ** rng.uniform(-2, 2) for k in M.REAL}
        p.update({k: rng.uniform(0, 2 * np.pi) for k in M.PHASES})
        o = M.observables(p)
        yu, yd, ye, m = o["yu"], o["yd"], o["ye"], o["mnu_meV"]
        a2, b2, c2 = m ** 2
        io = (b2 - a2) > (c2 - b2)
        # model.observables labels the spectrum ascending (NO); relabel IO like obs_from_mats
        l12, l13, l23, ld = o["pmns"]
        if io:
            Yu, Yd, Ye, mnu = M.matrices(p)
            _, Ue = M.left(Ye)
            kk = Ue.T @ mnu.conj() @ Ue
            mm, Un = M.takagi_masses_vectors(kk)
            l12, l13, l23, ld = (float(x) for x in extract_angles(Un[:, [1, 2, 0]]))
            r = (c2 - b2) / (c2 - a2)
        else:
            r = (b2 - a2) / (c2 - a2)
        q12, q13, q23, qd = o["ckm"]
        rows.append((np.sin(q12), np.sin(q13), np.sin(q23), qd, yu[0] / yu[1], yu[1] / yu[2], yd[0] / yd[1],
                     yd[1] / yd[2], ye[0] / ye[1], ye[1] / ye[2], np.sin(l12) ** 2, np.sin(l13) ** 2,
                     np.sin(l23) ** 2, ld, r, float(io)))
    a = np.array(rows, float).T
    keys = ("ckm_s12", "ckm_s13", "ckm_s23", "ckm_delta") + MR_KEYS + ("s12sq", "s13sq", "s23sq", "dcp", "dm_ratio", "io")
    return dict(zip(keys, a))


# ---------------------------------------------------------------- generic shared textures (new)
CLEBSCH = np.array([1, -1, 3, -3, 1 / 3, -1 / 3, 2, 1 / 2, -1 / 2, 9 / 2, 4 / 9, 6, 0], float)


def _basis(rng, B, K):
    """K shared complex 3x3 'operator' matrices per sample with Froggatt-Nielsen suppression
    eps^(qL_i + qR_j), O(1) complex coefficients (|z| log-uniform [1/3, 3], uniform phase), a
    random sparsity mask (zero probability 0 or 0.3) and, per operator, symmetric (10/126-like)
    or antisymmetric (120-like, prob 0.2) or general. One discrete structure per batch."""
    eps = rng.uniform(0.05, 0.3, B)[:, None, None]
    a = int(rng.integers(1, 5))
    b = int(rng.integers(0, a + 1))
    qL = np.array([a, b, 0])
    if rng.uniform() < 0.5:  # SO(10)-like: one charge vector for all fields
        qR = qL
    else:
        a2 = int(rng.integers(0, 5))
        qR = np.array([a2, int(rng.integers(0, a2 + 1)), 0])
    p0 = float(rng.choice([0.0, 0.3]))
    Ys = []
    for k in range(K):
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
    """Sector matrix sum_k c_k r_k Y_k with GUT-like Clebsch c_k (at least one non-zero) and,
    for up-type/Dirac, a complex vev ratio r_k (log-uniform 0.1-10, uniform phase)."""
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


def _gen_textures(rng, n, shared, batch=64):
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
        outs.append(obs_from_mats(*mats))
        got += len(mats[0])
    return {k: np.concatenate([o[k] for o in outs])[:n] for k in outs[0]}


def gen_gen(rng, n):
    return _gen_textures(rng, n, True)


def gen_ctrl(rng, n):
    return _gen_textures(rng, n, False)


GENS = {"F1": gen_f1, "F2": gen_f2, "MN": gen_mn, "FN": gen_fn, "MOD": gen_mod, "SU5": gen_su5,
        "S4SO": gen_s4so, "GEN": gen_gen, "CTRL": gen_ctrl}


# ---------------------------------------------------------------- features and filter
def features(o):
    """ctx_ckm (5), ctx_m (6), y (5) and a pass mask."""
    with np.errstate(all="ignore"):
        lg = {k: np.log10(o[k]) for k in ("ckm_s12", "ckm_s13", "ckm_s23", "dm_ratio") + MR_KEYS}
        ok = np.ones(len(o["ckm_s12"]), bool)
        for k, (lo, hi) in FILTER.items():
            ok &= np.isfinite(lg[k]) & (lg[k] >= lo) & (lg[k] <= hi)
        for k in ("s12sq", "s13sq", "s23sq", "dcp", "ckm_delta"):
            ok &= np.isfinite(o[k])
        for k in ("s12sq", "s13sq", "s23sq"):
            ok &= (o[k] > 1e-6) & (o[k] < 1 - 1e-6)
        ok &= o["io"] < 0.5  # normal ordering only (README)
        ctx_ckm = np.stack([lg["ckm_s12"], lg["ckm_s13"], lg["ckm_s23"], np.cos(o["ckm_delta"]),
                            np.sin(o["ckm_delta"])], 1)
        ctx_m = np.stack([lg[k] for k in MR_KEYS], 1)
        y = np.stack([o["s12sq"], o["s13sq"], o["s23sq"], np.mod(o["dcp"], 2 * np.pi), lg["dm_ratio"]], 1)
    return ctx_ckm, ctx_m, y, o["io"].astype(bool), ok


def _task(args):
    fam, n_target, seed = args
    rng = np.random.default_rng(seed)
    t0 = time.time()
    parts, n_raw, n_ok, chunk = [], 0, 0, max(2000, n_target // 4)
    while n_ok < n_target and n_raw < 2000 * n_target:
        o = GENS[fam](rng, chunk)
        c, m, y, io, ok = features(o)
        parts.append((c[ok], m[ok], y[ok], io[ok]))
        n_raw += len(ok)
        n_ok += int(ok.sum())
        if n_ok < n_target and n_ok > 0:
            chunk = int(min(200000, max(2000, (n_target - n_ok) * n_raw / n_ok * 1.1)))
    c, m, y, io = (np.concatenate([p[i] for p in parts])[:n_target] for i in range(4))
    return fam, c, m, y, io, n_raw, n_ok, time.time() - t0


def main(n_per=30000, seed=25, workers=5):
    ART.mkdir(parents=True, exist_ok=True)
    tasks = [(f, n_per, [seed, i]) for i, f in enumerate(FAMILIES)]
    res = {}
    with mp.Pool(workers) as pool:
        for fam, c, m, y, io, n_raw, n_ok, dt in pool.imap_unordered(_task, tasks):
            res[fam] = (c, m, y, io)
            print(f"{fam:5s} kept {len(c):6d} / raw {n_raw:8d} (acceptance {n_ok / n_raw:.4f}), "
                  f"IO frac {io.mean():.3f}, {dt:.0f} s", flush=True)
            res[fam + "_stats"] = dict(n_raw=n_raw, n_pass=n_ok, kept=len(c), acceptance=n_ok / n_raw,
                                       io_frac=float(io.mean()), seconds=dt)
    fam_id = np.concatenate([np.full(len(res[f][0]), i) for i, f in enumerate(FAMILIES)])
    arr = {k: np.concatenate([res[f][j] for f in FAMILIES]) for j, k in enumerate(("ctx_ckm", "ctx_m", "y", "io"))}
    np.savez_compressed(ART / "data.npz", fam=fam_id, families=np.array(FAMILIES), **arr)
    stats = {f: res[f + "_stats"] for f in FAMILIES}
    json.dump(dict(n_per=n_per, seed=seed, filter=FILTER, stats=stats), open(HERE / "gen_stats.json", "w"), indent=1)
    print("saved", len(fam_id), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 30000, int(a[1]) if len(a) > 1 else 25)

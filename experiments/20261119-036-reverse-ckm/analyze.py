"""Exp 036 evaluation (prereg.md). Usage:
  python analyze.py heldout            held-out information per family -> heldout.json
  python analyze.py real               real data vs 2000 N2 fakes -> real.json
  python analyze.py inject <family>    injection worlds (worlds_<family>.json) -> inject_<family>.json
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from common import ART, FAMS, HERE, SEEDS, Flow, ckm_target, load, pmns_ctx
from src import data as sd, evaluate as ev, nulls, theoryspace as ts

N_CKM, N_PMNS, N_FAKES, BLOCK = 4, 128, 2000, 100  # N_CKM 16 -> 4: deviation 2 (README)
F = len(FAMS)
MAXK = nulls.max_hits_to_pass(N_FAKES)


def log(msg):
    print(msg, flush=True)
    with open(HERE / "run.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def models():
    return [(Flow.load(ART / "models" / f"q1_s{s}.pt"), Flow.load(ART / "models" / f"q0_s{s}.pt")) for s in SEEDS]


def lme(a, axis):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis) + np.log(np.mean(np.exp(a - m), axis=axis))


# ---------------------------------------------------------------- held-out
def heldout():
    D = load()
    M = models()
    te = D["split"] == 2
    c, p, m, oh, fam = D["c"][te], D["p"][te], D["m"][te], D["onehot"][te], D["fam"][te]
    l1 = np.mean([q1.log_prob(c, np.concatenate([p, m], 1), oh) for q1, _ in M], 0)
    l0 = np.mean([q0.log_prob(c, m, oh) for _, q0 in M], 0)
    per_seed = [[q1.log_prob(c, np.concatenate([p, m], 1), oh) - q0.log_prob(c, m, oh) for q1, q0 in [mm]][0] for mm in M]
    out = {}
    for f, name in enumerate(FAMS):
        s = fam == f
        d = (l1[s] - l0[s]) / np.log(2)
        out[name] = {"n": int(s.sum()), "I_bits": float(d.mean()), "se": float(d.std() / np.sqrt(s.sum())),
                     "per_seed": [float(ps[s].mean() / np.log(2)) for ps in per_seed]}
        log(f"heldout {name}: I = {d.mean():.3f} +- {d.std() / np.sqrt(s.sum()):.3f} bits (per seed "
            f"{', '.join(f'{ps[s].mean() / np.log(2):.3f}' for ps in per_seed)}), n {s.sum()}")
    json.dump(out, open(HERE / "heldout.json", "w"), indent=1)


# ---------------------------------------------------------------- statistic
KEYS = ("s12sq", "s13sq", "s23sq", "dcp_deg")
OFF = None


def offsets():
    global OFF
    if OFF is None:
        rng = np.random.default_rng(3602)
        OFF = np.array([[o[k] for k in KEYS] for o in (nulls.pmns_noise("NO", rng) for _ in range(N_PMNS))])
    return OFF


def ckm_points(ckm):
    """ckm: {s12, s13, s23, delta_rad: (value, err_plus, err_minus)} -> (N_CKM, 4) targets."""
    rng = np.random.default_rng(3603)
    pts = [[ckm[k][0] for k in ("s12", "s13", "s23", "delta_rad")]]
    for _ in range(N_CKM - 1):
        row = []
        for k in ("s12", "s13", "s23", "delta_rad"):
            z = rng.standard_normal()
            v, ep, em = ckm[k]
            row.append(v + z * (ep if z > 0 else em))
        pts.append(row)
    pts = np.array(pts)
    return ckm_target(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3])


def pmns_points(centres):
    """centres (n, 4) physical -> (n, N_PMNS, 5) PMNS context at centre - o_j."""
    x = centres[:, None, :] - offsets()[None]
    return pmns_ctx(x[..., 0], x[..., 1], x[..., 2], np.radians(x[..., 3]) % (2 * np.pi))


class Stat:
    def __init__(self, M, cpts, mvec):
        self.M, self.c, self.m = M, cpts, mvec
        # Q0 term (independent of PMNS)
        oh = np.repeat(np.eye(F), N_CKM, 0)
        cc = np.tile(cpts, (F, 1))
        mm = np.tile(mvec, (F * N_CKM, 1))
        lq0 = []
        for _, q0 in M:
            L = q0.log_prob(cc, mm, oh).reshape(F, N_CKM)
            lq0.append(lme(L, 0))  # equal-weight family mixture
        self.lq0 = np.mean(lq0, 0)  # (N_CKM,), seed ensemble of log Q0
        self.T0 = lme(self.lq0, 0) / np.log(2)

    def T(self, centres):
        """T (bits) for each centre (n, 4); also the point statistic."""
        n = len(centres)
        P = pmns_points(centres).reshape(n * N_PMNS, 5)
        Pc = pmns_ctx(*(centres[:, k] for k in range(3)), np.radians(centres[:, 3]) % (2 * np.pi))
        P = np.concatenate([P, Pc], 0)  # last n rows: centres themselves (point statistic)
        R = len(P)
        # rows: (f, i_ckm, r)
        ctx = np.tile(np.concatenate([P, np.tile(self.m, (R, 1))], 1), (F * N_CKM, 1))
        cc = np.repeat(np.tile(self.c, (F, 1)), R, 0)
        oh = np.repeat(np.eye(F), N_CKM * R, 0)
        lq = []
        for q1, _ in self.M:
            L = q1.log_prob(cc, ctx, oh).reshape(F, N_CKM, R)
            lq.append(lme(L, 0))
        lq = np.mean(lq, 0)  # (N_CKM, R)
        conv = lq[:, : n * N_PMNS].reshape(N_CKM, n, N_PMNS)
        T1 = lme(conv.transpose(1, 0, 2).reshape(n, -1), 1) / np.log(2)
        Tpt = (lq[0, n * N_PMNS:] - self.lq0[0]) / np.log(2)
        return T1 - self.T0, Tpt


def fake_centres(idx, code):
    base = nulls.Universe.real("NO")
    return np.array([[nulls.n2_universe(base, np.random.default_rng([*code, i])).pmns_centre["NO"][k] for k in KEYS] for i in idx])


def real_inputs():
    c = sd.load_ckm()
    ckm = {k: (c[k]["value"], c[k]["err_plus"], c[k]["err_minus"]) for k in ("s12", "s13", "s23", "delta_rad")}
    mr, _, _ = ev.mass_ratio_targets()
    m = np.array([np.log10(mr[k]) for k in ts.MR_KEYS] + [np.log10(sd.load_dm_ratio("NO")["value"])])
    best = ev.pmns_best("NO")
    return ckm, m, np.array([[best[k] for k in KEYS]])


def run_world(S, centre, code, sequential):
    Tw, Tw_pt = S.T(centre)
    Tf, Tf_pt = [], []
    k = 0
    for b in range(0, N_FAKES, BLOCK):
        t, tp = S.T(fake_centres(range(b, b + BLOCK), code))
        Tf.append(t)
        Tf_pt.append(tp)
        k = int(np.sum(np.concatenate(Tf) >= Tw[0]))
        if sequential and k > MAXK:
            break
    Tf, Tf_pt = np.concatenate(Tf), np.concatenate(Tf_pt)
    return float(Tw[0]), float(Tw_pt[0]), Tf, Tf_pt, k


def real():
    torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    t0 = time.time()
    ckm, m, centre = real_inputs()
    log(f"real inputs: CKM {ckm}; m {np.round(m, 4).tolist()}; PMNS centre {centre[0].tolist()}")
    S = Stat(models(), ckm_points(ckm), m)
    Tw, Tw_pt, Tf, Tf_pt, k = run_world(S, centre, (36, 2), sequential=False)
    d = nulls.decide(k, N_FAKES)
    kp = int(np.sum(Tf_pt >= Tw_pt))
    from oracle import cp2
    out = {"T_real": Tw, "T_real_point": Tw_pt, "k": k, "rank": k / N_FAKES, "cp95_two_sided": cp2(k, N_FAKES),
           "cp95_upper_one_sided": d["upper"], "passed": d["passed"], "k_point": kp, "rank_point": kp / N_FAKES,
           "T_fake_q05_50_95": np.quantile(Tf, [0.05, 0.5, 0.95]).tolist(), "T0_bits": float(S.T0),
           "T_fakes": np.round(Tf, 4).tolist(), "seconds": time.time() - t0}
    json.dump(out, open(HERE / "real.json", "w"), indent=1)
    log(f"REAL: T = {Tw:.3f} bits (point {Tw_pt:.3f}); N2 fakes >= real: {k}/{N_FAKES} = {k / N_FAKES:.4f}, "
        f"one-sided CP95 upper {d['upper']:.4f}, passed {d['passed']}; point rank {kp / N_FAKES:.4f}; "
        f"fake T q05/50/95 {np.round(np.quantile(Tf, [0.05, 0.5, 0.95]), 3).tolist()} ({time.time() - t0:.0f}s)")


def inject(fam):
    torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    W = json.load(open(HERE / f"worlds_{fam}.json"))["worlds"]
    path = HERE / f"inject_{fam}.json"
    res = json.load(open(path)) if path.exists() else []
    done = {r["index"] for r in res}
    M = models()
    from oracle import cp2
    for w in W:
        if w["index"] in done:
            continue
        t0 = time.time()
        me = w["meas"]
        recs = sd.load_ckm()
        ckm = {}
        for k, rk in (("s12", "ckm_s12"), ("s13", "ckm_s13"), ("s23", "ckm_s23")):
            rel_p, rel_m = recs[k]["err_plus"] / recs[k]["value"], recs[k]["err_minus"] / recs[k]["value"]
            ckm[k] = (me[rk], me[rk] * rel_p, me[rk] * rel_m)
        ckm["delta_rad"] = (me["ckm_delta_rad"] % (2 * np.pi), recs["delta_rad"]["err_plus"], recs["delta_rad"]["err_minus"])
        m = np.array([np.log10(me[k]) for k in ts.MR_KEYS] + [np.log10(me["dm_ratio"])])
        S = Stat(M, ckm_points(ckm), m)
        centre = np.array([[me[k] for k in KEYS]])
        Tw, Tw_pt, Tf, Tf_pt, k = run_world(S, centre, (36, 3, w["index"]), sequential=True)
        n_ev = len(Tf)
        det = bool(n_ev == N_FAKES and k <= MAXK)
        res.append({"index": w["index"], "T": Tw, "T_point": Tw_pt, "k": k, "n_eval": n_ev, "detected": det,
                    "T_minus_median_fake": float(Tw - np.median(Tf)), "k300": int(np.sum(Tf[:300] >= Tw)) if n_ev >= 300 else None})
        json.dump(res, open(path, "w"), indent=1)
        log(f"inject {fam} world {w['index']}: T {Tw:.3f}, k {k}/{n_ev}, detected {det}, T - med fake "
            f"{Tw - np.median(Tf):+.3f} ({time.time() - t0:.0f}s)")
    det = sum(r["detected"] for r in res)
    lo, hi = cp2(det, len(res))
    log(f"inject {fam}: detected {det}/{len(res)} CP95 [{lo:.3f}, {hi:.3f}]; median T - med fake "
        f"{np.median([r['T_minus_median_fake'] for r in res]):+.3f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "heldout":
        torch.set_num_threads(3)
        heldout()
    elif cmd == "real":
        real()
    elif cmd == "inject":
        inject(sys.argv[2])

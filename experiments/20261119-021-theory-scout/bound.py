"""Information carried by the CKM (+ masses) about PMNS within the families searched so far (exps 007-018).

Writes bound.json. Run from this folder: ../../.venv/Scripts/python bound.py
Single process, < 2 min. Reads committed experiment outputs and the gitignored PT samples in artifacts/so10/
(exp 014); if those are missing, the SO(10) section falls back to the numbers in exp 014's heldout_*.json.

Quantities (all in bits, log base 2):
  data_bits   KL(NuFIT posterior || Haar) per PMNS observable, Gaussian approximation: the information the
              measurements hold relative to "no theory". A relation that predicts an observable must supply it.
  B_ckm       log2[ L(PMNS | family, real CKM) / E_fake L(PMNS | family, fake CKM) ] with L = exp(-chi2_best/2),
              fakes from the hierarchical null (b)/N1. Profile likelihood, so it is optimistic for the family.
              B_ckm > 0 means the real CKM fits better than a typical hierarchical CKM.
  S_rank      -log2 P(fake best chi2 <= real best chi2): how strongly the real CKM is singled out (rank-based).
  G_pred      log2[ p(x_obs | family, quark+charged-lepton data) / p(x_obs | family prior) ], p = predictive density
              convolved with the measurement error: does conditioning on quark data move PMNS toward the truth?
  KL_shift    KL(posterior predictive || prior predictive) per observable: how much quark data change the PMNS
              prediction at all (direction ignored). Upper bound on the CKM->PMNS channel actually used.
  I_gauss     -0.5 log2(1 - R2) from exp 014's cross-validated regression of PMNS on CKM + masses within the
              posterior (Gaussian-channel equivalent of the explained variance).
"""
import csv
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
sys.path.insert(0, str(ROOT))
from src.data import load_pmns  # noqa: E402

LN2 = math.log(2)
OUT = {}


def lse(a):
    a = np.asarray(a, float)
    m = np.max(a)
    return m + math.log(np.mean(np.exp(a - m)))


def cp_upper_rank(h, n):
    from scipy.stats import beta
    lo = beta.ppf(0.025, h, n - h + 1) if h > 0 else 0.0
    hi = beta.ppf(0.975, h + 1, n - h) if h < n else 1.0
    return lo, hi


def null_stats(real, fakes):
    fakes = np.asarray([f for f in fakes if np.isfinite(f)], float)
    n, h = len(fakes), int(np.sum(fakes <= real))
    lo, hi = cp_upper_rank(h, n)
    B = (-real / 2 - lse(-fakes / 2)) / LN2
    return dict(real_chi2=float(real), n=n, hits=h, rate=h / n, rate_ci95=[float(lo), float(hi)],
                S_rank=float(-math.log2(max(h, 0.5) / n)), S_rank_max_ci=float(-math.log2(max(lo, 0.5 / n))),
                B_ckm=float(B), fake_median=float(np.median(fakes)))


# ---------------------------------------------------------------- 0. data information budget (vs Haar)
def data_budget():
    res = {}
    for o in ("NO", "IO"):
        p = load_pmns(o)
        r = {}
        for key, haar in (("s12sq", lambda x: 1.0), ("s13sq", lambda x: 2 * (1 - x)), ("s23sq", lambda x: 1.0)):
            s = 0.5 * (p[key]["err_plus"] + p[key]["err_minus"])
            r[key] = -math.log2(haar(p[key]["value"]) * s * math.sqrt(2 * math.pi * math.e))
        s = 0.5 * (p["dcp_deg"]["err_plus"] + p["dcp_deg"]["err_minus"])
        r["dcp_deg"] = max(0.0, -math.log2(s / 360 * math.sqrt(2 * math.pi * math.e)))
        r["total_4"] = sum(r.values())
        res[o] = r
    return res


# ---------------------------------------------------------------- 1. F1 (exp 007) and F2 (exp 009) hierarchical nulls
def f1_f2():
    res = {}
    rp = json.load(open(EXP / "20261118-007-f1-shared-rotation/real_pipeline.json"))["pipeline"]
    rows = list(csv.DictReader(open(EXP / "20261118-007-f1-shared-rotation/null_b.csv")))
    for o in ("NO", "IO"):
        for v in ("cont", "struct"):
            res[f"F1_{o}_{v}"] = null_stats(rp[f"table_{o}"][f"best_{v}"], [float(r[f"best_{v}_{o}"]) for r in rows])
    pr = json.load(open(EXP / "20261118-009-f2-clebsch/pipeline_real.json"))
    rows = list(csv.DictReader(open(EXP / "20261118-009-f2-clebsch/null_b.csv")))
    for sc in ("MZ", "GUT"):
        for o in ("NO", "IO"):
            for v in ("fit", "struct"):
                for t in ("A", "ABC", "ALL"):
                    col = f"{sc}_tab_{o}_{v}_{t}"
                    res[f"F2_{col}"] = null_stats(pr[sc][f"tab_{o}"][v][t], [float(r[col]) for r in rows])
    return res


# ---------------------------------------------------------------- 2. symbolic search (exp 010) null b, chi2 = pull^2
def symbolic():
    res = {}
    summ = list(csv.DictReader(open(EXP / "20261118-010-symbolic-masses/null_summary.csv")))
    real = {(r["scale"], r["target"], r["ordering"], r["C"]): float(r["real_best"]) for r in summ if r["kind"] == "b" and r["real_best"] != ""}
    for sc in ("MZ", "GUT"):
        rows = list(csv.DictReader(open(EXP / f"20261118-010-symbolic-masses/null_b_{sc}.csv")))
        for (s, t, o, C), rb in real.items():
            if s != sc:
                continue
            fk = [float(r[f"best_C{C}"]) ** 2 for r in rows if r["target"] == t and r["ordering"] == o and r[f"best_C{C}"] != ""]
            if fk:
                res[f"{sc}_{t}_{o}_C{C}"] = null_stats(rb ** 2, fk)
    return res


# ---------------------------------------------------------------- 3. protocol reruns (exp 018), chi2_global of fakes
def protocol():
    res = {}
    for f in sorted((EXP / "20261118-018-protocol-reruns").glob("rate_*.json")):
        d = json.load(open(f))
        fk = [x["result"]["chi2_global"] for x in d["fakes"] if x.get("result")]
        res[f.stem] = null_stats(d["ref"]["chi2_global"], fk)
        res[f.stem]["note"] = "rank uses chi2_global; exp 018 hit rule used p_global and |pull| < 2, so rates differ"
    return res


# ---------------------------------------------------------------- 4. SO(10) held-out predictive gain (exp 014)
def so10():
    sys.path.insert(0, str(EXP / "20261118-014-so10-fits"))
    import so10fit as S
    res = {}
    cases = {"MN_ours": ("MN", "ours", [f"MN_ql_{i}" for i in range(1, 6)], "heldout_MN"),
             "MN_drrel": ("MN", "ours", [f"MN_qldr_{i}" for i in range(1, 6)], "heldout_MN_drrel"),
             "FN_ours": ("FN", "ours", [f"FN_ql_{i}" for i in range(1, 6)], "heldout_FN")}
    names = ["s12sq", "s13sq", "s23sq", "dl", "lg_dm_ratio"]
    rng = np.random.default_rng(5)
    for case, (model, tgt, tags, hj) in cases.items():
        r = {}
        try:
            z = np.concatenate([np.load(S.ART / f"pt_{t}.npz")["z"] for t in tags])
            io = np.concatenate([np.load(S.ART / f"pt_{t}.npz")["io"] for t in tags])
        except FileNotFoundError:
            r["error"] = "PT samples missing; see exp 014 heldout json"
            res[case] = r
            continue
        prob = S.Problem(model, tgt)
        val, err = prob.val, prob.err
        lo, hi = S.mn_box(tgt, val) if model == "MN" else S.fn_box(tgt)
        zp, op = prob.predict(lo + (hi - lo) * rng.random((200000, len(lo))))
        okp = np.all(np.isfinite(zp), 1)
        zp, iop = zp[okp], op["io"][okp]
        ok = np.all(np.isfinite(z), 1)
        z, io = z[ok], io[ok]
        z, zp = z[~io.astype(bool)], zp[~iop.astype(bool)]   # NO data: compare within NO-classified samples
        r["n_post_NO"], r["n_prior_NO"] = int(len(z)), int(len(zp))

        def chi(Z, js, scale=1.0):
            d = Z[:, js] - val[js]
            for c, j in enumerate(js):
                if j == 15:
                    d[:, c] = S.wrap(d[:, c])
            return ((d / (scale * err[js])) ** 2).sum(1)

        for c, j in enumerate(range(12, 17)):
            g = {}
            for scale in (1.0, 3.0):
                lp, lq = lse(-chi(z, [j], scale) / 2), lse(-chi(zp, [j], scale) / 2)
                g[f"G_pred_sigma_x{int(scale)}"] = float((lp - lq) / LN2)
                g[f"logp_post_x{int(scale)}"] = float(lp / LN2)
                g[f"logp_prior_x{int(scale)}"] = float(lq / LN2)
            a, b = z[:, j], zp[:, j]
            edges = np.quantile(b, np.linspace(0, 1, 41))
            edges[0], edges[-1] = min(edges[0], a.min()) - 1e-12, max(edges[-1], a.max()) + 1e-12
            P = np.histogram(a, edges)[0] + 0.5
            Q = np.histogram(b, edges)[0] + 0.5
            P, Q = P / P.sum(), Q / Q.sum()
            g["KL_shift_bits"] = float(np.sum(P * np.log2(P / Q)))
            r[names[c]] = g
        for lab, js in (("joint4", [12, 13, 14, 15]), ("joint5", [12, 13, 14, 15, 16])):
            for scale in (1.0, 3.0):
                lp, lq = lse(-chi(z, js, scale) / 2), lse(-chi(zp, js, scale) / 2)
                r[f"{lab}_G_pred_sigma_x{int(scale)}"] = float((lp - lq) / LN2)
                r[f"{lab}_min_chi2_post_prior_x{int(scale)}"] = [float(chi(z, js, scale).min()), float(chi(zp, js, scale).min())]
        hjd = json.load(open(EXP / f"20261118-014-so10-fits/{hj}.json"))
        ig = {}
        for t, v in hjd["regression_R2_post"]["gbm"].items():
            best = max(x for x in (hjd["regression_R2_post"][m][t] for m in ("linear", "quadratic", "gbm")) if x is not None)
            ig[t] = dict(R2_best=best, I_gauss_bits=float(-0.5 * math.log2(1 - max(best, 0.0))))
        r["I_gauss_post"] = ig
        res[case] = r
    return res


# ---------------------------------------------------------------- 5. shared-modulus cost (exps 016, 017)
def modular():
    occam = math.log2(3.5 / 0.04 ** 2)  # fundamental domain (Im tau <= 4) area / lepton-island area (exp 016 width)
    rows = {}
    # (joint chi2, own-tau lepton chi2, own-tau quark chi2); quark-only for 2O assumed 0 (k = N in that sector)
    for lab, j, l, q in (("2O_ours_NO", 68.8, 14.4, 0.0), ("A4_cat_IO", 446.5, 1.6, 254.8), ("A4_cat_NO", 468.4, 0.0, 254.8)):
        d = j - l - q
        rows[lab] = dict(joint=j, lepton_own=l, quark_own=q, delta_chi2_sharing=d,
                         occam_gain_max_bits=occam, net_bits=occam - d / (2 * LN2))
    return rows


if __name__ == "__main__":
    OUT["data_budget_bits_vs_haar"] = data_budget()
    OUT["F1_F2_hierarchical_null"] = f1_f2()
    OUT["symbolic_null_b"] = symbolic()
    OUT["protocol_rate_runs"] = protocol()
    OUT["so10_heldout"] = so10()
    OUT["modular_sharing"] = modular()
    (HERE / "bound.json").write_text(json.dumps(OUT, indent=1))

    print("data bits:", {o: {k: round(v, 2) for k, v in r.items()} for o, r in OUT["data_budget_bits_vs_haar"].items()})
    for sec in ("F1_F2_hierarchical_null", "protocol_rate_runs"):
        for k, v in OUT[sec].items():
            print(f"{sec[:8]} {k:32s} real {v['real_chi2']:8.3f} rate {v['rate']:.3f} S {v['S_rank']:5.2f} "
                  f"S_max {v['S_rank_max_ci']:5.2f} B {v['B_ckm']:+6.2f} fake_med {v['fake_median']:.2f}")
    sym = OUT["symbolic_null_b"]
    best = sorted(sym.items(), key=lambda kv: -kv[1]["B_ckm"])[:8]
    for k, v in best:
        print(f"symb {k:28s} real {v['real_chi2']:.4f} rate {v['rate']:.3f} S {v['S_rank']:.2f} B {v['B_ckm']:+.2f}")
    print("symbolic: max B", max(v["B_ckm"] for v in sym.values()), "max S", max(v["S_rank"] for v in sym.values()))
    for case, r in OUT["so10_heldout"].items():
        print(case, json.dumps({k: v for k, v in r.items() if k != "I_gauss_post"})[:1500])
        print(case, "I_gauss", {t: round(v["I_gauss_bits"], 3) for t, v in r.get("I_gauss_post", {}).items()})
    print(json.dumps(OUT["modular_sharing"], indent=1))

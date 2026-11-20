"""Exp 035 (c): the Delta m^2 ratio r = dm21 / |dm3l| as a relation target.

Part 1, pre-declared grammar (fixed before scoring; README):
  bases X = {mr_u_c, mr_c_t, mr_d_s, mr_s_b, mr_e_mu, mr_mu_tau, lambda}            (7)
  G1  r = c x^p        p in {1/2, 1, 3/2, 2}, c in {1, 2, 3, 1/2, 1/3}            7*4*5 = 140
  G2  r = x^a y^b      x != y in X (unordered), a, b in {1/2, 1}                    21*4  =  84
  scales: mass ratios at M_Z (scored at low scale), or at 2e16 GeV (SM running, relation imposed
  there, prediction scored with scale={"mu": 2e16, "model": "SM"})                       x2
  M = 448 relations -> 9 bits. (lambda is scale-independent here: |V_us| hardly runs; at GUT the
  CKM key is translated by src.evaluate's additive rule for ckm_s12 ... so lambda is used only
  through ckm_lambda at low scale; at GUT the lambda relations use ckm_s12 (= lambda to 0.03%,
  the same alias atom).)
Scoring (official): inputs are fitted nuisances and scored: use = ["dm_ratio"] + base keys,
N = 1 + n_b, k_cont = n_b, dof = 1, k = k_cont + 9 bits. Both orderings.
Look-elsewhere: (i) Sidak over the grammar with p_loc = P(a log-uniform target on [0.003, 0.3]
falls within |z_best| sigma_eff of the best relation), (ii) a fake-target null: 5000 targets
log-uniform on [0.003, 0.3] with the real relative error, fraction where the best grammar member
is at least as close (|z| <= |z_best|) as for the real target.

Part 2: does any src.theoryspace family predict r from quark data? For each family, accepted draws
(exp 025 filter, NO), regress log10 r on quark-only features (log10 CKM sines, cos/sin delta_q, log10
of 4 quark mass ratios) with a gradient-boosted regressor; held-out R^2, conditional width vs marginal
width (Gaussian bits), and the conditional prediction at the measured quark point (src loaders).
Usage: python dmratio.py grammar | python dmratio.py families [FAM ...]
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from src import data as D  # noqa: E402
from src import evaluate as E  # noqa: E402
from src import rg  # noqa: E402
from src.stats import discrete_bits, trials_factor_pvalue  # noqa: E402

MR = ["mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau"]
BASES = MR + ["lambda"]
P1 = [0.5, 1.0, 1.5, 2.0]
C1 = [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3)]
SCALES = ["MZ", "GUT"]
GUT = {"mu": 2e16, "model": "SM"}
PRIOR = (0.003, 0.3)


def grammar():
    rel = []
    for sc in SCALES:
        for x, p, c in itertools.product(BASES, P1, C1):
            rel.append(dict(scale=sc, terms=((x, p),), c=c))
        for (x, y) in itertools.combinations(BASES, 2):
            for a, b in itertools.product((0.5, 1.0), repeat=2):
                rel.append(dict(scale=sc, terms=((x, a), (y, b)), c=(1, 1)))
    return rel


def name(r):
    c = "" if r["c"] == (1, 1) else (f"{r['c'][0]}/{r['c'][1]} " if r["c"][1] != 1 else f"{r['c'][0]} ")
    t = " ".join(f"({x})^{p:g}" for x, p in r["terms"])
    return f"r = {c}{t}  [{r['scale']}]"


def key_of(x, sc):
    if x == "lambda":
        return "ckm_lambda" if sc == "MZ" else "ckm_s12"
    return x


def base_values(sc):
    """central values and log sigmas of the bases at the relation's scale (src loaders / src.rg)."""
    mr = rg.mass_ratios(mu=None if sc == "MZ" else 2e16, model="SM", errors=True)
    mz = rg.mass_ratios(errors=True)
    ck = D.load_ckm()
    v, sl = {}, {}
    for i, n in enumerate(mr["names"]):
        v[n] = mr["value"][n]
        sl[n] = float(np.sqrt(mz["cov_log"][i, i]))  # log errors: running is multiplicative
    lam = ck["lambda"] if sc == "MZ" else ck["s12"]
    v["lambda"] = lam["value"]
    sl["lambda"] = 0.5 * (lam["err_plus"] + lam["err_minus"]) / lam["value"]
    return v, sl


def predict(r, v):
    cf = r["c"][0] / r["c"][1]
    return cf * np.prod([v[x] ** p for x, p in r["terms"]])


def log_sigma_pred(r, sl):
    return float(np.sqrt(sum((p * sl[x]) ** 2 for x, p in r["terms"])))  # ignores mr correlations (screen only)


def refit(r, ordering):
    keys = [key_of(x, r["scale"]) for x, _ in r["terms"]]
    v, _ = base_values(r["scale"])
    cf = r["c"][0] / r["c"][1]
    bounds = {k: (np.log10(v[x]) - 0.5, np.log10(v[x]) + 0.5) for k, (x, _) in zip(keys, r["terms"])}

    def model(th):
        out = {k: 10 ** th[k] for k in keys}
        out["dm_ratio"] = cf * np.prod([10 ** (p * th[k]) for k, (_, p) in zip(keys, r["terms"])])
        return out
    fr = E.fit_and_evaluate(model, bounds, ordering, ["dm_ratio"] + keys, k_discrete=9, method="lbfgsb",
                            n_starts=4, scale=GUT if r["scale"] == "GUT" else None)
    res = fr.result
    return dict(name=name(r), ordering=ordering, chi2=res.chi2, chi2_global=res.chi2_global, N=res.N, k=res.k,
                k_cont=res.k_continuous, dof=res.dof, p=res.p, p_global=res.p_global, max_pull=res.max_abs_pull,
                pulls=res.pulls, pred_low=float(res.pred.get("dm_ratio", np.nan)),
                compression_ok=bool(res.compression_ok), fit_ok=bool(res.fit_ok))


def run_grammar():
    t0 = time.time()
    rels = grammar()
    M = len(rels)
    bits = discrete_bits(M)
    print(f"grammar: {M} relations, {bits} bits", flush=True)
    vals = {sc: base_values(sc) for sc in SCALES}
    out = {"M": M, "bits": bits}
    for o in D.ORDERINGS:
        rec = D.load_dm_ratio(o)
        r0 = rec["value"]
        s0 = 0.5 * (rec["err_plus"] + rec["err_minus"]) / r0
        # GUT predictions are compared after the multiplicative running of dm_ratio (src.evaluate factor)
        fac = E.Scorer(["dm_ratio"], o, scale=GUT).scale_shift["dm_ratio"]
        rows = []
        for i, r in enumerate(rels):
            v, sl = vals[r["scale"]]
            T = predict(r, v) * (fac if r["scale"] == "GUT" else 1.0)
            se = float(np.hypot(s0, log_sigma_pred(r, sl)))
            rows.append(dict(i=i, name=name(r), pred=T, z=float(np.log(T / r0) / se), sig_eff=se))
        rows.sort(key=lambda x: abs(x["z"]))
        print(f"[{o}] r = {r0:.5f} (rel err {s0:.3f}); GUT->low factor {fac:.4f}; screen best 8:", flush=True)
        for x in rows[:8]:
            print(f"    z {x['z']:+.3f}  pred {x['pred']:.5f}  {x['name']}", flush=True)
        n2 = sum(abs(x["z"]) < 2 for x in rows)
        print(f"[{o}] screen |z| < 2: {n2}; |z| < 1: {sum(abs(x['z']) < 1 for x in rows)}", flush=True)
        ref = [refit(rels[x["i"]], o) for x in rows if abs(x["z"]) < 3]
        ref.sort(key=lambda x: x["chi2"])
        kept = [x for x in ref if x["max_pull"] < 2]
        print(f"[{o}] official refits {len(ref)}, all pulls < 2: {len(kept)}, fit_ok: {sum(x['fit_ok'] for x in ref)}, "
              f"compression_ok: {sum(x['compression_ok'] for x in ref)}", flush=True)
        for x in ref[:10]:
            print(f"    chi2 {x['chi2']:.4f} p_g {x['p_global']:.3f} maxpull {x['max_pull']:.2f} N {x['N']} k {x['k']}  "
                  f"{x['name']}", flush=True)
        # look-elsewhere
        zb, seb = abs(rows[0]["z"]), rows[0]["sig_eff"]
        width = np.log(PRIOR[1] / PRIOR[0])
        p_loc = min(1.0, 2 * zb * seb / width)
        p_sid = float(trials_factor_pvalue(p_loc, M))
        rng = np.random.default_rng(350 + (o == "IO"))
        logT = np.array([np.log(x["pred"]) for x in rows])
        se_arr = np.array([x["sig_eff"] for x in rows])
        fakes = np.exp(rng.uniform(np.log(PRIOR[0]), np.log(PRIOR[1]), 5000))
        zf = np.min(np.abs((logT[None, :] - np.log(fakes)[:, None]) / se_arr[None, :]), axis=1)
        hits = int(np.sum(zf <= zb))
        print(f"[{o}] best |z| {zb:.3f}: p_loc {p_loc:.2e}, Sidak over {M}: {p_sid:.3f}; fake-target rate "
              f"{hits}/5000 = {hits / 5000:.3f}; fakes with any member |z| < 1: {np.mean(zf < 1):.3f}", flush=True)
        out[o] = dict(r=r0, rel_err=s0, gut_factor=fac, screen=rows[:60], refits=ref, n_screen_2sig=n2,
                      best_abs_z=zb, p_loc=p_loc, p_sidak=p_sid, fake_hits=hits, fake_n=5000,
                      fake_rate_any_1sig=float(np.mean(zf < 1)))
    out["elapsed_s"] = time.time() - t0
    (HERE / "dmratio_grammar.json").write_text(json.dumps(out, indent=1, default=float))


# ------------------------------------------------------------------ part 2: theory-space families

def real_quark_features():
    ck = D.load_ckm()
    mz = rg.mass_ratios(errors=False)["value"]
    d = ck["delta_rad"]["value"]
    return np.array([np.log10(ck["s12"]["value"]), np.log10(ck["s13"]["value"]), np.log10(ck["s23"]["value"]),
                     np.cos(d), np.sin(d)] + [np.log10(mz[k]) for k in MR[:4]])


def run_families(n=40000, budget_s=420, fams=None, tag=""):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from src import theoryspace as TS
    xr = real_quark_features()
    out = {}
    for fam in fams or ["GEN", "CTRL", "MN", "FN", "SU5", "S4SO", "MOD", "F2"]:
        t0 = time.time()
        rng = np.random.default_rng(3500)
        try:
            obs, st = TS.sample_accepted(fam, n, rng, ordering="NO", max_raw=400 * n)
        except Exception as e:  # noqa: BLE001
            print(f"{fam}: sampling failed: {e!r}", flush=True)
            out[fam] = {"error": repr(e)}
            continue
        if st["kept"] < 2000:
            print(f"{fam}: only {st['kept']} accepted, skipped", flush=True)
            out[fam] = {"kept": st["kept"]}
            continue
        ck, cm, y = TS.features_025(obs)
        X = np.concatenate([ck, cm[:, :4]], 1)
        t = y[:, 4]
        ok = np.all(np.isfinite(X), 1) & np.isfinite(t)
        X, t = X[ok], t[ok]
        m = len(t)
        idx = np.random.default_rng(1).permutation(m)
        tr, te = idx[: int(0.8 * m)], idx[int(0.8 * m):]
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr], t[tr])
        res = t[te] - reg.predict(X[te])
        r2 = 1 - np.var(res) / np.var(t[te])
        sm, sc = float(np.std(t[te])), float(np.std(res))
        pr = float(reg.predict(xr[None, :])[0])
        # local check: residual spread among the 300 test points nearest the real quark point
        zx = (X[te] - X[tr].mean(0)) / X[tr].std(0)
        zr = (xr - X[tr].mean(0)) / X[tr].std(0)
        near = np.argsort(np.sum((zx - zr) ** 2, 1))[:300]
        dist = float(np.sqrt(np.median(np.sum((zx[near] - zr) ** 2, 1))))
        sloc = float(np.std(res[near]))
        robs = np.log10(D.load_dm_ratio("NO")["value"])
        q16, q84 = pr + np.quantile(res[near], 0.16), pr + np.quantile(res[near], 0.84)
        bits = float(np.log2(sm / sc)) if sc > 0 else float("nan")
        row = dict(kept=int(m), acceptance=st["acceptance"], r2_test=float(r2), sd_marginal=sm, sd_conditional=sc,
                   gauss_bits=bits, pred_log10_at_real=pr, local_sd=sloc, local_median_dist=dist,
                   interval68_log10=[float(q16), float(q84)], obs_log10=float(robs),
                   obs_in_68=bool(q16 <= robs <= q84), marginal_q16_q84=[float(np.quantile(t, 0.16)),
                                                                         float(np.quantile(t, 0.84))],
                   secs=time.time() - t0)
        out[fam] = row
        print(f"{fam}: kept {m} (acc {st['acceptance']:.3g}) R2 {r2:.3f}  sd marg {sm:.3f} cond {sc:.3f} "
              f"({bits:.2f} bits)  at real quarks: log10 r = {pr:.3f} [{q16:.3f}, {q84:.3f}] vs obs {robs:.3f}"
              f"  (neighbour dist {dist:.2f})  {time.time() - t0:.0f}s", flush=True)
        (HERE / f"dmratio_families{tag}.json").write_text(json.dumps(out, indent=1, default=float))
        if time.time() - t0 > budget_s:
            print(f"{fam} took {time.time() - t0:.0f}s (budget {budget_s}s)", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "families" and len(sys.argv) > 2:  # e.g. families MOD  (one family, own json)
        run_families(fams=sys.argv[2:], tag="_" + "_".join(sys.argv[2:]))
    else:
        {"grammar": run_grammar, "families": run_families}[sys.argv[1]]()

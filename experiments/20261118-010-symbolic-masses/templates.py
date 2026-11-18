"""Multi-observable templates: one function f, applied to the sector-matched quark/lepton quantity
for each PMNS angle: theta_ij = f(x_ij). N = 3 angles (and each pair, N = 2), same f.

f: every distinct expression in one variable x plus the 10 grammar constants, all unary/binary ops,
complexity <= CT (same dedup and validity rules as the single-observable search).
Mappings x_ij (fixed before any score):
  ckm:    x12 = t12q, x13 = t13q, x23 = t23q
  down:   x12 = sqrt(md/ms), x13 = sqrt(md/mb), x23 = sqrt(ms/mb)
  lepton: x12 = sqrt(me/mmu), x13 = sqrt(me/mtau), x23 = sqrt(mmu/mtau)
  up:     x12 = sqrt(mu/mc), x13 = sqrt(mu/mt), x23 = sqrt(mc/mt)
Score: sum of the three 1D profile dchi2 (ProfileScorer, input errors profiled) for real targets;
Gaussian NuFIT widths around Haar targets for null (a); fake inputs (as run_null.py) for null (b).
Outputs: templates_real.csv (top templates), templates_null.csv, templates_summary.json.
"""
import csv
import json
import sys
import time

import numpy as np

import pipeline as P
import symlib as S

CT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NFAKE = int(sys.argv[2]) if len(sys.argv) > 2 else 300
ANG = ["s12sq", "s13sq", "s23sq"]
MAPS = ["ckm", "down", "lepton", "up"]
SUBSETS = {"12+13+23": [0, 1, 2], "12+13": [0, 1], "12+23": [0, 2], "13+23": [1, 2]}


def mapped(base, dirs):
    """-> dict map -> list over (12, 13, 23) of (x, dx directions)."""
    q = lambda i: np.sqrt(base[i])
    dq = lambda i: 0.5 / np.sqrt(base[i]) * dirs[:, i]
    out = {"ckm": [(base[0], dirs[:, 0]), (base[1], dirs[:, 1]), (base[2], dirs[:, 2])]}
    for name, (i12, i23) in {"down": (6, 7), "lepton": (8, 9), "up": (4, 5)}.items():
        x12, x23 = q(i12), q(i23)
        d12, d23 = dq(i12), dq(i23)
        out[name] = [(x12, d12), (x12 * x23, x23 * d12 + x12 * d23), (x23, d23)]
    return out


def eval_template(E, x, dx):
    base = np.zeros(10)
    base[0] = x
    base[4:] = 1.0  # unused leaves
    dirs = np.zeros((len(dx), 10))
    dirs[:, 0] = dx
    return E.evaluate(base, dirs)


def preds(E, base, dirs):
    """-> dict map -> list over the 3 angles of (p, sigma)."""
    out = {}
    for m, lst in mapped(base, dirs).items():
        out[m] = []
        for k, (x, dx) in enumerate(lst):
            v, T = eval_template(E, x, dx)
            out[m].append(S.predictions(v, T, ANG[k]))
    return out


def score(pr, scorers):
    """scorers: list over the 3 angles of callables (p, s) -> dchi2. Returns dict map -> (3, nE)."""
    res = {}
    for m, lst in pr.items():
        d = np.stack([scorers[k](*lst[k]) for k in range(3)])
        res[m] = np.where(np.isfinite(d), d, np.inf)
    return res


def best_per_subset(res):
    out = {}
    for name, idx in SUBSETS.items():
        b = np.inf
        for m in MAPS:
            b = min(b, float(res[m][idx].sum(axis=0).min()))
        out[name] = b
    return out


def main():
    t0 = time.time()
    E = S.Enum(CT, leaf_set=[0] + list(range(S.N_VAR, S.N_LEAF)))
    nf = len(E)
    log = open(S.HERE / "run.log", "a")
    msg = f"[templates] C<={CT}: {nf} distinct f(x), x 4 mappings = {4 * nf} templates ({np.log2(4 * nf):.1f} bits)"
    print(msg, flush=True); log.write(msg + "\n")
    from src import data
    summary = {"CT": CT, "n_f": nf, "n_templates": 4 * nf, "per_level": E.counts}
    rows = []
    from src.haar import sample_haar_angles
    for scale in ("MZ", "GUT"):
        real = S.real_inputs(scale)
        pr_real = preds(E, real["base"], real["dirs"])
        prof = {o: [P.profile_scorer(t, o).dchi2 for t in ANG] for o in P.ORDERINGS}
        br, bg = {}, {}
        for o in P.ORDERINGS:
            res = score(pr_real, prof[o])
            gs = [S.GaussScorer(t, o) for t in ANG]
            resg = score(pr_real, [lambda p, s, g=g: g.pull(p, s) ** 2 for g in gs])
            br[o], bg[o] = best_per_subset(res), best_per_subset(resg)
            summary[f"real_{scale}_{o}"] = {"profile": br[o], "gauss": bg[o]}
            for m in MAPS:
                tot = res[m].sum(axis=0)
                for jj in np.argsort(tot)[:5]:
                    rows.append([scale, o, m, int(E.comp[jj]), E.expr(jj).replace("t12q", "x"), float(tot[jj])]
                                + [float(res[m][k, jj]) for k in range(3)])
        nul_a = {o: [] for o in P.ORDERINGS}
        nul_b = {o: [] for o in P.ORDERINGS}
        for i in range(NFAKE):
            rng = np.random.default_rng([2002 if scale == "MZ" else 2003, i])
            bf = S.fake_base_inputs(rng, real["base"][4:], 1)[0]
            pf = preds(E, bf, S.fake_dirs(real, bf))
            t12, t13, t23, d = sample_haar_angles(np.random.default_rng([1001, i]))
            c = [np.sin(t12) ** 2, np.sin(t13) ** 2, np.sin(t23) ** 2]
            for o in P.ORDERINGS:
                ga = [lambda p, s, g=S.GaussScorer(t, o, ci): g.pull(p, s) ** 2 for t, ci in zip(ANG, c)]
                nul_a[o].append(best_per_subset(score(pr_real, ga)))
                nul_b[o].append(best_per_subset(score(pf, prof[o])))
        for o in P.ORDERINGS:
            for name in SUBSETS:
                for kind, nul, ref in (("a", nul_a[o], bg[o][name]), ("b", nul_b[o], br[o][name])):
                    hits = int(sum(x[name] <= ref for x in nul))
                    r = S_rate(hits, NFAKE)
                    summary.setdefault("null", []).append(
                        {"scale": scale, "ordering": o, "subset": name, "kind": kind, "real_best_chi2": ref,
                         "hits": hits, "n": NFAKE, **r,
                         "fake_median": float(np.median([x[name] for x in nul]))})
            m_ = f"[templates] {scale} {o}: real best {br[o]} (gauss {bg[o]}) {time.time() - t0:.0f}s"
            print(m_, flush=True); log.write(m_ + "\n"); log.flush()
    with open(S.HERE / "templates_real.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scale", "ordering", "mapping", "complexity", "f(x)", "chi2_3angles", "dchi2_s12", "dchi2_s13",
                    "dchi2_s23"])
        w.writerows(rows)
    (S.HERE / "templates_summary.json").write_text(json.dumps(summary, indent=1))


def S_rate(k, n):
    from src.stats import fake_universe_rate
    return fake_universe_rate(k, n)


if __name__ == "__main__":
    main()

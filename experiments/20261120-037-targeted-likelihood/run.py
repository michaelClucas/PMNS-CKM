"""Exp 037 runner (frozen by prereg.md). One relation id per task, both orderings, three nulls.

    ../../.venv/Scripts/python run.py banks                      # build fake banks (N1, N1lam, N2; 50,000 each)
    ../../.venv/Scripts/python run.py worker <deadline_utc> <rid> [<rid> ...]   # tasks in order
    ../../.venv/Scripts/python run.py timing <rid>               # worlds only, no real data (design check)

Data sets per (relation, ordering):
  real    NuFIT 6.1 tables (untranslated)                                        n_min 1000, cap N_MAX
  today   20 injected worlds: truth = frozen pred_*, data centre = truth + src.nulls.pmns_noise,
          translated tables without octant mirror (src.nulls.make_injector), CKM = real records
                                                                                 n_min 300, cap N_MAX (symbolic) / W_CAP (fitted)
  nom     20 worlds at exp 029 nominal future precision: centre = truth + N(0, sigma_fut), Gaussian chi2
  opt     20 worlds at optimistic precision (symbolic rows only)
Fake j of every null is shared by all data sets (rng [37, code, j]). Sequential rule: a data set stays
alive while n < n_min or (k < K_STOP and n < cap). Output: artifacts/037/rel_<id>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import numpy as np

import lib37 as L
from src import nulls as nl

W_CAP = 2000            # fitted rows: injected worlds are extended at most to 2,000 fakes
N_WORLDS = 20
VERSIONS_FIT = ("real", "today", "nom")
VERSIONS_SYM = ("real", "today", "nom", "opt")
VCODE = {"nom": 1, "opt": 2}
SCEN = {"nom": "nominal", "opt": "optimistic"}
RECORD_AT = (300, 1000, 2000, 5000)
OUT = L.BANK_DIR


def log(msg):
    print(f"{dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%SZ')} {msg}", flush=True)


def world_centre(truth, o, i, version, ridx):
    """Injected data centre for world i (truth = relation prediction at the real CKM)."""
    if version == "today":
        rng = np.random.default_rng([L.SEED, 7, ridx, L.ORDS.index(o), i])
        u = nl.make_injector(truth)(rng, nl.Universe.real(o))
        return u, u.pmns_data_centre(o)
    rng = np.random.default_rng([L.SEED, 8, ridx, L.ORDS.index(o), VCODE[version], i])
    sig = L.future_sigma(o, SCEN[version])
    base = nl.Universe.real(o).pmns_data_centre(o)
    c = {}
    for k in L.PMNS:
        z = float(rng.standard_normal())
        c[k] = truth.get(k, base[k]) + sig[k] * z
    c["dcp_deg"] %= 360.0
    for k in ("s12sq", "s13sq", "s23sq"):
        c[k] = float(np.clip(c[k], 1e-6, 1 - 1e-6))
    return None, c


def make_datasets(rows_r, ridx, versions, include_real=True):
    ds = []
    for r in rows_r:
        o = r["ordering"]
        truth = dict(r["pred"])
        for v in versions:
            if v == "real":
                if include_real:
                    ds.append(dict(name=f"{o}/real", o=o, version="real", i=-1,
                                   D=L.TableData(nl.Universe.real(o), o), n_min=L.N_MIN_REAL))
                continue
            for i in range(N_WORLDS):
                u, c = world_centre(truth, o, i, v, ridx)
                D = L.TableData(u, o) if v == "today" else L.GaussData(c, L.future_sigma(o, SCEN[v]), o)
                ds.append(dict(name=f"{o}/{v}/{i}", o=o, version=v, i=i, D=D, n_min=L.N_MIN_WORLD, centre=c))
    return ds


def n2_fake_D(bank, j, o, version):
    if version in ("real", "today"):
        return L.TableData(L.n2_fake_universe(bank, j), o)
    c = {k: float(bank[f"{o}_{k}"][j]) for k in L.PMNS}
    return L.GaussData(c, L.future_sigma(o, SCEN[version]), o)


# ---------------------------------------------------------------- sequential bookkeeping

class Seq:
    def __init__(self, ds, caps):
        self.nd = len(ds)
        self.k = np.zeros(self.nd, int)
        self.n = np.zeros(self.nd, int)
        self.nmin = np.array([d["n_min"] for d in ds])
        self.cap = np.array(caps)
        self.rec = {m: np.full(self.nd, -1) for m in RECORD_AT}
        self.first_hits = [[] for _ in range(self.nd)]

    def alive(self):
        return np.flatnonzero((self.n < self.nmin) | ((self.k < L.K_STOP) & (self.n < self.cap)))

    def add(self, idx, hits, j):
        self.n[idx] += 1
        self.k[idx] += hits.astype(int)
        for a, h in zip(idx, hits):
            if h and len(self.first_hits[a]) < 5:
                self.first_hits[a].append(int(j))
        for m in RECORD_AT:
            sel = idx[self.n[idx] == m]
            self.rec[m][sel] = self.k[sel]

    def result(self, ds, chi2_data, truncated):
        out = []
        for a, d in enumerate(ds):
            out.append(dict(name=d["name"], o=d["o"], version=d["version"], i=d["i"], chi2_data=float(chi2_data[a]),
                            k=int(self.k[a]), n=int(self.n[a]), **{f"k{m}": int(self.rec[m][a]) for m in RECORD_AT},
                            first_hits=self.first_hits[a], truncated=bool(truncated and self.n[a] < self.cap[a]
                                                                           and self.k[a] < L.K_STOP)))
        return out


def finish_vectorised(hits, nmin, cap):
    """Sequential rule applied to a full hit sequence (symbolic rows)."""
    ck = np.cumsum(hits)
    n_idx = np.arange(1, len(hits) + 1)
    stop = np.flatnonzero((n_idx >= nmin) & (ck >= L.K_STOP))
    n = int(n_idx[stop[0]]) if len(stop) else int(min(cap, len(hits)))
    k = int(ck[n - 1])
    rec = {m: (int(ck[m - 1]) if m <= n else -1) for m in RECORD_AT}
    return k, n, rec, [int(x) for x in np.flatnonzero(hits[:n])[:5]]


# ---------------------------------------------------------------- tasks

def run_symbolic(rid, rows_r, ridx, banks):
    rel = L.make_relation(rows_r[0])
    ctx0 = L.real_ctx()
    p0 = float(rel.predict_ctx(ctx0))
    ds = make_datasets(rows_r, ridx, VERSIONS_SYM)
    chi2_data = np.array([float(d["D"].chi2({rel.key: np.array([p0])}, rel.keys)[0]) for d in ds])
    res = {}
    for null in ("N1", "N1lam"):
        b = banks[null]
        p = rel.predict(b["t12q"], b["t13q"], b["t23q"], b["dq"])
        out = []
        for a, d in enumerate(ds):
            c = d["D"].chi2({rel.key: p}, rel.keys)
            h = L.is_hit(c, chi2_data[a])
            k, n, rec, fh = finish_vectorised(h, d["n_min"], L.N_MAX)
            out.append(dict(name=d["name"], o=d["o"], version=d["version"], i=d["i"], chi2_data=float(chi2_data[a]),
                            k=k, n=n, **{f"k{m}": rec[m] for m in RECORD_AT}, first_hits=fh, truncated=False))
        res[null] = dict(results=out, n_invalid=int(np.sum(~np.isfinite(p))), n_bank=len(p))
    b = banks["N2"]
    out = []
    cache = {}
    for a, d in enumerate(ds):
        o, v = d["o"], d["version"]
        key = (o, "today" if v in ("real", "today") else v)
        if key not in cache:
            cen = b[f"{o}_{rel.key}"]
            if key[1] == "today":
                cache[key] = L.translated_chi2_many(o, rel.key, p0, cen)
            else:
                cache[key] = L.GaussData({rel.key: cen}, L.future_sigma(o, SCEN[key[1]]), o).chi2(
                    {rel.key: np.full(len(cen), p0)}, rel.keys)
        h = L.is_hit(cache[key], chi2_data[a])
        k, n, rec, fh = finish_vectorised(h, d["n_min"], L.N_MAX)
        out.append(dict(name=d["name"], o=o, version=v, i=d["i"], chi2_data=float(chi2_data[a]), k=k, n=n,
                        **{f"k{m}": rec[m] for m in RECORD_AT}, first_hits=fh, truncated=False))
    res["N2"] = dict(results=out, n_invalid=0, n_bank=len(b[f"NO_{rel.key}"]))
    return dict(id=rid, kind="symbolic", expr=rel.expr, key=rel.key, pred_real=p0, nulls=res)


def run_fitted(rid, rows_r, ridx, banks, deadline, include_real=True, max_fakes=None):
    rel = L.make_relation(rows_r[0])
    ctx0 = L.real_ctx()
    ds = make_datasets(rows_r, ridx, VERSIONS_FIT, include_real)
    Ds = [d["D"] for d in ds]
    chi2_data, arg = rel.profile(ctx0, Ds)
    caps = [L.N_MAX if d["version"] == "real" else W_CAP for d in ds]
    if max_fakes:
        caps = [min(c, max_fakes) for c in caps]
    res = {}
    obs_real = rel.obs(rel.P0, ctx0)
    for null in ("N1", "N1lam", "N2"):
        b = banks[null]
        S = Seq(ds, caps)
        t0, j, trunc, n_inv = time.time(), 0, False, 0
        while True:
            al = S.alive()
            if not len(al):
                break
            if time.time() > deadline or j >= len(next(iter(b.values()))):
                trunc = True
                break
            if null == "N2":
                groups = {}
                for a in al:
                    groups.setdefault((ds[a]["o"], "today" if ds[a]["version"] in ("real", "today") else ds[a]["version"]), []).append(a)
                hits = np.zeros(len(al), bool)
                pos = {a: m for m, a in enumerate(al)}
                keys = list(groups)
                fakeD = [n2_fake_D(b, j, o, v) for o, v in keys]
                best, _ = rel.profile(ctx0, fakeD, coarse=obs_real)
                for g, key in enumerate(keys):
                    for a in groups[key]:
                        hits[pos[a]] = bool(L.is_hit(np.array([best[g]]), chi2_data[a])[0])
            else:
                c = L.ctx_from_bank(b, j)
                if not rel.valid(c):
                    n_inv += 1
                    hits = np.zeros(len(al), bool)
                else:
                    best, _ = rel.profile(c, [Ds[a] for a in al])
                    hits = L.is_hit(best, chi2_data[al])
            S.add(al, hits, j)
            j += 1
            if j in (300, 1000, 2000) or j % 5000 == 0:
                log(f"{rid} {null} fake {j}: alive {len(S.alive())} ({time.time() - t0:.0f} s)")
        res[null] = dict(results=S.result(ds, chi2_data, trunc), n_invalid=n_inv, n_used=j, truncated=trunc,
                         seconds=time.time() - t0)
        log(f"{rid} {null} done: {j} fakes, truncated {trunc}, {time.time() - t0:.0f} s")
    return dict(id=rid, kind="fitted", names=list(rel.names),
                theta_real={d["name"]: (None if arg[a] is None else [float(x) for x in arg[a]])
                            for a, d in enumerate(ds) if d["version"] == "real"}, nulls=res)


def main():
    rows = L.load_frozen()
    ids = list(dict.fromkeys(r["id"] for r in rows))
    by = {rid: [r for r in rows if r["id"] == rid] for rid in ids}
    cmd = sys.argv[1]
    if cmd == "banks":
        for null in ("N1", "N1lam", "N2"):
            t0 = time.time()
            L.build_bank(null)
            log(f"bank {null} built ({time.time() - t0:.0f} s)")
        return
    banks = {null: L.build_bank(null) for null in ("N1", "N1lam", "N2")}
    if cmd == "timing":
        rid = sys.argv[2]
        t0 = time.time()
        run_fitted(rid, by[rid], ids.index(rid), banks, time.time() + 600, include_real=False, max_fakes=20)
        log(f"timing {rid}: 20 fakes x 3 nulls on world data sets only: {time.time() - t0:.1f} s")
        return
    if cmd == "worker":
        hh, mm = (int(x) for x in sys.argv[2].split(":"))
        now = dt.datetime.now(dt.timezone.utc)
        dl = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if dl < now:
            dl += dt.timedelta(days=1)
        deadline = dl.timestamp()
        for rid in sys.argv[3:]:
            f = OUT / f"rel_{rid}.json"
            if f.exists():
                log(f"{rid} exists, skipped")
                continue
            t0 = time.time()
            log(f"{rid} start")
            if rid.startswith("S-"):
                out = run_symbolic(rid, by[rid], ids.index(rid), banks)
            else:
                out = run_fitted(rid, by[rid], ids.index(rid), banks, deadline)
            out["seconds"] = time.time() - t0
            f.write_text(json.dumps(out))
            log(f"{rid} written ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

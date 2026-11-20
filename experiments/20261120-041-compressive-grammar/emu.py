"""Exp 041 step 3: MLP emulator (entry, params) -> observables on synthetic theory-space draws.

  gen [n] [seed] [tag]   uniform entry, uniform params in grammar.BOUNDS -> artifacts/041/draws_<tag>.npz
                         (no measured number enters; observables from src.yukawa.observables_batch)
  refine [n] [seed]      round 2 draws: Gaussian clouds (sd 0.15 in scaled units) around the round-1
                         emulator optima of every entry (theory-space draws again; the optima are
                         where the search went, the training targets are still engine outputs)
  train [epochs] [tags]  MLP, 90/10 train/val split by draw; held-out MAE per feature -> emulator_train.json
  search [starts]        per entry, Adam on a Gaussian surrogate chi2 of the emulator output against the
                         measured values (target only) -> search.json (entry ranking + optima)
Threads: torch.set_num_threads(2) (CPU rule: 3 threads total for this agent).
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import grammar as g  # noqa: E402
from src.yukawa import observables_batch  # noqa: E402

ART = ROOT / "artifacts" / "041"
FEAT = ["l_ckm_s12", "l_ckm_s13", "l_ckm_s23", "c_dq", "s_dq", "s12sq", "l_s13sq", "s23sq", "c_dl", "s_dl",
        "l_mr_u_c", "l_mr_c_t", "l_mr_d_s", "l_mr_s_b", "l_mr_e_mu", "l_mr_mu_tau", "l_dm_ratio"]
LO = np.array([g.BOUNDS[p][0] for p in g.PARAMS])
HI = np.array([g.BOUNDS[p][1] for p in g.PARAMS])
REPS = np.array([e["reps"] for e in g.ENTRIES])
NUI = np.array([g.NU_OPTS.index(e["nu"]) for e in g.ENTRIES])
THREADS = 2


def features(o):
    L = lambda x: np.log10(np.clip(x, 1e-12, None))  # noqa: E731
    dq, dl = o["ckm_delta_rad"], np.radians(o["dcp_deg"])
    return np.stack([L(o["ckm_s12"]), L(o["ckm_s13"]), L(o["ckm_s23"]), np.cos(dq), np.sin(dq), o["s12sq"],
                     L(o["s13sq"]), o["s23sq"], np.cos(dl), np.sin(dl), L(o["mr_u_c"]), L(o["mr_c_t"]),
                     L(o["mr_d_s"]), L(o["mr_s_b"]), L(o["mr_e_mu"]), L(o["mr_mu_tau"]), L(o["dm_ratio"])], 1)


def inputs(eid, X):
    s = (X[:, :6] - LO[:6]) / (HI[:6] - LO[:6]) * 2 - 1
    nu1 = np.eye(4)[NUI[eid]]
    return np.concatenate([REPS[eid], nu1, s, np.cos(X[:, 6:]), np.sin(X[:, 6:])], 1).astype(np.float32)


def _simulate(eid, X):
    with np.errstate(all="ignore"):
        o = observables_batch(*g.build_batch(X, REPS[eid], NUI[eid]), ordering="NO")
        F = features(o)
    ok = np.all(np.isfinite(F), 1)
    return inputs(eid[ok], X[ok]), F[ok].astype(np.float32), ok


def gen(n=2_000_000, seed=41, tag="r1", chunk=50_000):
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    t0 = time.time()
    nok = 0
    for s in range(0, n, chunk):
        B = min(chunk, n - s)
        eid = rng.integers(0, g.G, B)
        X = LO + rng.uniform(size=(B, 8)) * (HI - LO)
        xi, y, ok = _simulate(eid, X)
        Xs.append(xi)
        Ys.append(y)
        nok += ok.sum()
        print(f"gen {s + B}/{n} finite {ok.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / f"draws_{tag}.npz", X=np.concatenate(Xs), Y=np.concatenate(Ys))
    return dict(n=n, finite=int(nok), seed=seed)


def refine(n=1_000_000, seed=42, sd=0.15):
    S = json.load(open(HERE / "search.json"))
    rng = np.random.default_rng(seed)
    centres = []
    for r in S:
        for x in r["top_x"]:
            centres.append((r["eid"], np.array(x)))
    Xs, Ys = [], []
    per = n // len(centres)
    for eid0, x in centres:
        z = (x - LO) / (HI - LO)
        Z = np.clip(z + sd * rng.standard_normal((per, 8)), 0, 1)
        Z[:, 6:] = np.mod(z[6:] + sd * rng.standard_normal((per, 2)), 1.0)
        X = LO + Z * (HI - LO)
        # half the cloud keeps the entry, half is relabelled uniformly (the net must still learn the entry effect)
        eid = np.where(rng.uniform(size=per) < 0.5, eid0, rng.integers(0, g.G, per))
        xi, y, _ = _simulate(eid, X)
        Xs.append(xi)
        Ys.append(y)
    np.savez_compressed(ART / "draws_r2.npz", X=np.concatenate(Xs), Y=np.concatenate(Ys))
    print("refine draws", sum(len(x) for x in Xs), flush=True)


def net(din, dout=len(FEAT), h=256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dout))


def train(epochs=20, tags=("r1",), seed=0, minutes=60.0, out="emulator"):
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    Xs, Ys = [], []
    for t in tags:
        d = np.load(ART / f"draws_{t}.npz")
        Xs.append(d["X"])
        Ys.append(d["Y"])
    X, Y = torch.tensor(np.concatenate(Xs)), torch.tensor(np.concatenate(Ys))
    mu, sd = Y.mean(0), Y.std(0)
    Yn = (Y - mu) / sd
    n = len(X)
    perm = torch.randperm(n)
    nv = n // 10
    iv, it = perm[:nv], perm[nv:]
    f = net(X.shape[1])
    bs = 2048
    nb = len(it) // bs
    opt = torch.optim.AdamW(f.parameters(), 1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * nb)
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        sh = it[torch.randperm(len(it))]
        f.train()
        for b in range(nb):
            idx = sh[b * bs:(b + 1) * bs]
            loss = torch.nn.functional.smooth_l1_loss(f(X[idx]), Yn[idx], beta=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        f.eval()
        with torch.no_grad():
            ev = torch.cat([(f(X[iv[i:i + 65536]]) - Yn[iv[i:i + 65536]]).abs() for i in range(0, nv, 65536)]).mean(0)
        hist.append(float(ev.mean()))
        print(f"epoch {ep} loss {loss.item():.4f} val MAE (std units) {ev.mean():.4f} {time.time() - t0:.0f}s",
              flush=True)
        torch.save(dict(state=f.state_dict(), mu=mu, sd=sd, din=X.shape[1]), ART / f"{out}.pt")
        if (time.time() - t0) / 60 > minutes:
            print("time budget reached, stopping", flush=True)
            break
    with torch.no_grad():
        r = torch.cat([f(X[iv[i:i + 65536]]) * sd + mu - Y[iv[i:i + 65536]] for i in range(0, nv, 65536)])
        dqf = [FEAT.index(k) for k in ("c_dq", "s_dq", "c_dl", "s_dl")]
        mae = r.abs().mean(0)
        med = r.abs().median(0).values
    res = dict(n_train=len(it), n_val=nv, epochs_run=len(hist), tags=list(tags), features=FEAT,
               val_mae_raw=dict(zip(FEAT, mae.tolist())), val_median_abs_raw=dict(zip(FEAT, med.tolist())),
               val_mae_std_units_by_epoch=hist, secs=time.time() - t0,
               note="raw units: log10 for l_* features, plain for s12sq/s23sq, cos/sin for phases")
    del dqf
    json.dump(res, open(HERE / f"{out}_train.json", "w"), indent=1)


def surrogate_target():
    """Gaussian surrogate (centre, sigma) in feature space from the measured data via src loaders
    (the search target only; the emulator never trains on these). Diagonal; phases circular."""
    from src import data as D
    from src import evaluate as E
    c, pm = D.load_ckm(), D.load_pmns("NO")
    vals, names, cov = E.mass_ratio_targets()
    sig_mr = {n: float(np.sqrt(cov[i, i] + g.FLOOR ** 2) / np.log(10)) for i, n in enumerate(names)}
    e = lambda r: 0.5 * (r["err_plus"] + r["err_minus"])  # noqa: E731
    lg = lambda r: (np.log10(r["value"]), e(r) / r["value"] / np.log(10))  # noqa: E731
    dmb, dms = E._dm_best("NO", D.DEFAULT_ANALYSIS), E._dm_sigma("NO", D.DEFAULT_ANALYSIS)
    t = {"l_ckm_s12": lg(c["s12"]), "l_ckm_s13": lg(c["s13"]), "l_ckm_s23": lg(c["s23"]),
         "dq": (c["delta_rad"]["value"], e(c["delta_rad"])),
         "s12sq": (pm["s12sq"]["value"], e(pm["s12sq"])), "l_s13sq": lg(pm["s13sq"]),
         "s23sq": (pm["s23sq"]["value"], e(pm["s23sq"])),
         "dl": (np.radians(pm["dcp_deg"]["value"]), np.radians(e(pm["dcp_deg"]))),
         "l_dm_ratio": (np.log10(dmb["dm_ratio"]), dms["dm_ratio"] / dmb["dm_ratio"] / np.log(10))}
    for n in ("mr_u_c", "mr_c_t", "mr_d_s", "mr_s_b", "mr_e_mu", "mr_mu_tau"):
        t["l_" + n] = (np.log10(vals[n]), sig_mr[n])
    return t


def load(name="emulator"):
    import torch
    ck = torch.load(ART / f"{name}.pt")
    f = net(ck["din"])
    f.load_state_dict(ck["state"])
    f.eval()
    for q in f.parameters():
        q.requires_grad_(False)
    return f, ck["mu"], ck["sd"]


def search(n_starts=512, steps=400, seed=1, name="emulator", top=4):
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    f, mu, sd = load(name)
    t = surrogate_target()
    idx = {n: i for i, n in enumerate(FEAT)}
    E_ = np.repeat(np.arange(g.G), n_starts)
    B = len(E_)
    head = torch.tensor(np.concatenate([REPS[E_], np.eye(4)[NUI[E_]]], 1), dtype=torch.float32)
    z = (torch.rand(B, 6) * 2 - 1).requires_grad_(True)  # scaled params, kept in [-1, 1] by tanh-free clamp
    ph = (torch.rand(B, 2) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([z, ph], 0.03)

    def chi2():
        zc = torch.clamp(z, -1, 1)
        x = torch.cat([head, zc, torch.cos(ph), torch.sin(ph)], 1)
        y = f(x) * sd + mu
        c2 = 0
        for n, (m0, s0) in t.items():
            if n in ("dq", "dl"):
                cn, sn = ("c_dq", "s_dq") if n == "dq" else ("c_dl", "s_dl")
                d = torch.atan2(y[:, idx[sn]], y[:, idx[cn]]) - m0
                d = torch.atan2(torch.sin(d), torch.cos(d))
                c2 = c2 + (d / s0) ** 2
            else:
                c2 = c2 + ((y[:, idx[n]] - m0) / s0) ** 2
        return c2

    t0 = time.time()
    for _ in range(steps):
        c2 = chi2()
        opt.zero_grad()
        torch.log1p(c2).sum().backward()
        opt.step()
    with torch.no_grad():
        c2 = chi2().numpy().reshape(g.G, n_starts)
        zc = torch.clamp(z, -1, 1).numpy()
        X6 = (zc + 1) / 2 * (HI[:6] - LO[:6]) + LO[:6]
        X = np.concatenate([X6, np.mod(ph.numpy(), 2 * np.pi)], 1).reshape(g.G, n_starts, 8)
    out = []
    for e in range(g.G):
        o = np.argsort(c2[e])
        out.append(dict(eid=e, entry=g.name(g.ENTRIES[e]), emu_chi2=float(c2[e, o[0]]), x=X[e, o[0]].tolist(),
                        top_x=[X[e, j].tolist() for j in o[:top]], top_chi2=[float(c2[e, j]) for j in o[:top]]))
    out.sort(key=lambda r: r["emu_chi2"])
    for rank, r in enumerate(out):
        r["rank"] = rank
    json.dump(out, open(HERE / "search.json", "w"), indent=0)
    print(f"search {time.time() - t0:.0f}s", flush=True)
    for r in out:
        print(r["rank"], r["eid"], r["entry"], round(r["emu_chi2"], 1), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == "gen":
        print(gen(*(int(x) for x in a[:2]), *a[2:3]))
    elif cmd == "refine":
        refine(*(int(x) for x in a))
    elif cmd == "train":
        train(int(a[0]), tuple(a[1].split(",")) if len(a) > 1 else ("r1",),
              minutes=float(a[2]) if len(a) > 2 else 60.0, out=a[3] if len(a) > 3 else "emulator")
    elif cmd == "search":
        search(int(a[0]) if a else 512, name=a[1] if len(a) > 1 else "emulator")
    else:
        raise SystemExit(__doc__)

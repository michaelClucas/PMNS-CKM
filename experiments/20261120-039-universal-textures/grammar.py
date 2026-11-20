"""Exp 039 part (c): neural-guided search over the GEO universal-texture grammar.

Grammar (model.py): M_f[ij] = c_ij sqrt(mu_i^f mu_j^f) e^{i s_f phi_ij}, c_ij in {0, 1} on the six
Hermitian positions (64 patterns), mu^f = (mu1, mu2, 1) per sector (8 mass parameters),
shared phases phi_ij on the nonzero off-diagonals (d, e only), neutrino option dirac | seesaw.
Grammar size 128 -> 7 bits.

Steps (subcommands):
  gen      synthetic theory-space data (no measured number enters): uniform pattern / option,
           log-uniform mu in the model.geo_bounds box, uniform phases -> src.yukawa.observables_batch.
           -> artifacts/exp039/geo_train.npz
  train    MLP emulator (pattern bits, option, masses, phases) -> 17 observable features.
           -> artifacts/exp039/emulator.pt, emulator_train.json (held-out errors)
  search   for each of the 128 (pattern, option) combos, minimise a Gaussian surrogate chi^2 of the
           emulator output against the data (measured values only as the target), 512 random
           starts with Adam. -> search.json (emulator ranking)
  verify   exact engine + official scorer fits (fit.fit, cls GEO) for ALL 128 combos with a cheap DE
           (ground truth for the emulator ranking) and full DE fits for the emulator top 10.
           -> verify_all.jsonl, verify_top.jsonl
"""
from __future__ import annotations

import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import model as M  # noqa: E402
from src.yukawa import observables_batch  # noqa: E402

ART = ROOT / "artifacts" / "exp039"
FEAT = ["l_ckm_s12", "l_ckm_s13", "l_ckm_s23", "c_dq", "s_dq", "s12sq", "l_s13sq", "s23sq", "c_dl", "s_dl",
        "l_mr_u_c", "l_mr_c_t", "l_mr_d_s", "l_mr_s_b", "l_mr_e_mu", "l_mr_mu_tau", "l_dm_ratio"]
MASKS = np.array([M.pattern_mask(p) for p in range(64)])
LO = np.array([-8.0, -6.0] * 4)
HI = np.array([1.0, 1.0] * 4)


def build_batch(p, nu, lm, ph):
    """p (B,), nu (B,) 0 dirac 1 seesaw, lm (B, 8) [lm1_u, lm2_u, lm1_d, ...], ph (B, 3) on OFF."""
    B = len(p)
    mask = MASKS[p]  # (B, 6)
    C = np.zeros((B, 3, 3))
    for k, (i, j) in enumerate(M.POS):
        C[:, i, j] = mask[:, k]
        C[:, j, i] = mask[:, k]
    P = np.ones((B, 3, 3), complex)
    for k, (i, j) in enumerate(M.OFF):
        P[:, i, j] = np.exp(1j * ph[:, k])
        P[:, j, i] = np.exp(-1j * ph[:, k])
    mats = []
    for s in range(4):
        mu = np.stack([10 ** lm[:, 2 * s], 10 ** lm[:, 2 * s + 1], np.ones(B)], 1)
        r = np.sqrt(mu)
        Mf = C * r[:, :, None] * r[:, None, :]
        if s in (1, 2):
            Mf = Mf * P
        mats.append(Mf.astype(complex))
    Mu, Md, Me, Mn = mats
    w, U = np.linalg.eigh(Mn @ np.conj(np.swapaxes(Mn, 1, 2)))
    y = np.sqrt(np.clip(w, 0, None))
    mdir = np.conj(U) @ (y[:, :, None] * np.conj(np.swapaxes(U, 1, 2)))
    mss = -np.conj(Mn) @ np.conj(np.swapaxes(Mn, 1, 2))
    mnu = np.where((nu == 0)[:, None, None], mdir, mss)
    mnu = 0.5 * (mnu + np.swapaxes(mnu, 1, 2))
    return Mu, Md, Me, mnu


def features(o):
    L = lambda x: np.log10(np.clip(x, 1e-12, None))  # noqa: E731
    dq, dl = o["ckm_delta_rad"], np.radians(o["dcp_deg"])
    return np.stack([L(o["ckm_s12"]), L(o["ckm_s13"]), L(o["ckm_s23"]), np.cos(dq), np.sin(dq), o["s12sq"],
                     L(o["s13sq"]), o["s23sq"], np.cos(dl), np.sin(dl), L(o["mr_u_c"]), L(o["mr_c_t"]),
                     L(o["mr_d_s"]), L(o["mr_s_b"]), L(o["mr_e_mu"]), L(o["mr_mu_tau"]), L(o["dm_ratio"])], 1)


def inputs(p, nu, lm, ph):
    mask = MASKS[p]
    offm = mask[:, 3:]
    return np.concatenate([mask, nu[:, None], (lm - LO) / (HI - LO) * 2 - 1,
                           offm * np.cos(ph), offm * np.sin(ph)], 1).astype(np.float32)


def gen(n=1_500_000, seed=39, chunk=50_000):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    t0 = time.time()
    for s in range(0, n, chunk):
        B = min(chunk, n - s)
        p = rng.integers(0, 64, B)
        nu = rng.integers(0, 2, B)
        lm = LO + rng.uniform(size=(B, 8)) * (HI - LO)
        ph = rng.uniform(0, 2 * np.pi, (B, 3))
        with np.errstate(all="ignore"):
            o = observables_batch(*build_batch(p, nu, lm, ph), ordering="NO")
            F = features(o)
        ok = np.all(np.isfinite(F), 1)
        X.append(inputs(p, nu, lm, ph)[ok])
        Y.append(F[ok].astype(np.float32))
        print(f"gen {s + B}/{n} finite {ok.mean():.3f} {time.time() - t0:.0f}s", flush=True)
    ART.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ART / "geo_train.npz", X=np.concatenate(X), Y=np.concatenate(Y))


def net(din=22, dout=len(FEAT), h=256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(), nn.Linear(h, dout))


def train(epochs=2, seed=0):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    d = np.load(ART / "geo_train.npz")
    X, Y = torch.tensor(d["X"]), torch.tensor(d["Y"])
    mu, sd = Y.mean(0), Y.std(0)
    Yn = (Y - mu) / sd
    n = len(X)
    perm = torch.randperm(n)
    nv = 50_000
    iv, it = perm[:nv], perm[nv:]
    f = net(X.shape[1])
    opt = torch.optim.Adam(f.parameters(), 1e-3)
    steps = epochs * (len(it) // 2048)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    t0 = time.time()
    for ep in range(epochs):
        sh = it[torch.randperm(len(it))]
        for b in range(len(it) // 2048):
            idx = sh[b * 2048:(b + 1) * 2048]
            loss = torch.nn.functional.smooth_l1_loss(f(X[idx]), Yn[idx], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
        with torch.no_grad():
            ev = (f(X[iv]) - Yn[iv]).abs().mean(0) * sd
        print(f"epoch {ep} loss {loss.item():.4f} val MAE (raw units) mean {ev.mean():.4f} {time.time() - t0:.0f}s",
              flush=True)
    with torch.no_grad():
        r = f(X[iv]) * sd + mu - Y[iv]
        mae = r.abs().mean(0)
        med = r.abs().median(0).values
    torch.save(dict(state=f.state_dict(), mu=mu, sd=sd, din=X.shape[1]), ART / "emulator.pt")
    json.dump(dict(n_train=len(it), n_val=nv, epochs=epochs, features=FEAT, val_mae=dict(zip(FEAT, mae.tolist())),
                   val_median_abs=dict(zip(FEAT, med.tolist()))), open(HERE / "emulator_train.json", "w"), indent=1)


def surrogate_target():
    """Gaussian surrogate (centre, sigma) in feature space from the measured data via src loaders
    (target only; the emulator never sees these). Diagonal; circular for the two phases."""
    from src import data as D
    from src import evaluate as E
    c, pm = D.load_ckm(), D.load_pmns("NO")
    vals, names, cov = E.mass_ratio_targets()
    sig_mr = {n: float(np.sqrt(cov[i, i] + M.FLOOR ** 2) / np.log(10)) for i, n in enumerate(names)}
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


def search(n_starts=128, steps=200, seed=1):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    ck = torch.load(ART / "emulator.pt")
    f = net(ck["din"])
    f.load_state_dict(ck["state"])
    for q in f.parameters():
        q.requires_grad_(False)
    mu, sd = ck["mu"], ck["sd"]
    t = surrogate_target()
    combos = [(p, nu) for p in range(64) for nu in (0, 1)]
    P = torch.tensor(np.repeat([c[0] for c in combos], n_starts))
    NU = torch.tensor(np.repeat([c[1] for c in combos], n_starts), dtype=torch.float32)
    B = len(P)
    mask = torch.tensor(MASKS, dtype=torch.float32)[P]
    z = torch.randn(B, 8, requires_grad=True)
    ph = (torch.rand(B, 3) * 2 * np.pi).requires_grad_(True)
    opt = torch.optim.Adam([z, ph], 0.05)
    idx = {n: i for i, n in enumerate(FEAT)}

    def chi2():
        lm01 = torch.sigmoid(z) * 2 - 1
        offm = mask[:, 3:]
        x = torch.cat([mask, NU[:, None], lm01, offm * torch.cos(ph), offm * torch.sin(ph)], 1)
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

    for it in range(steps):
        c2 = chi2()
        opt.zero_grad()
        torch.log1p(c2).sum().backward()
        opt.step()
    with torch.no_grad():
        c2 = chi2().numpy().reshape(len(combos), n_starts)
        lm = ((torch.sigmoid(z) * 2 - 1 + 1) / 2 * torch.tensor(HI - LO, dtype=torch.float32)
              + torch.tensor(LO, dtype=torch.float32)).numpy().reshape(len(combos), n_starts, 8)
        phn = np.mod(ph.numpy(), 2 * np.pi).reshape(len(combos), n_starts, 3)
    out = []
    for c, (p, nu) in enumerate(combos):
        j = int(np.argmin(c2[c]))
        out.append(dict(p=p, nu=["dirac", "seesaw"][nu], emu_chi2=float(c2[c, j]), lm=lm[c, j].tolist(),
                        ph=phn[c, j].tolist(), mask=MASKS[p].tolist()))
    out.sort(key=lambda r: r["emu_chi2"])
    json.dump(out, open(HERE / "search.json", "w"), indent=0)
    for r in out[:15]:
        print(r["p"], r["mask"], r["nu"], round(r["emu_chi2"], 1), flush=True)


def _start(r):
    th = {}
    for s, f in enumerate(M.SECTORS):
        th["lm1_" + f], th["lm2_" + f] = r["lm"][2 * s], r["lm"][2 * s + 1]
    for k, ij in enumerate(M.OFF):
        if ij in M.pattern_phases(r["p"]):
            th[f"ph{ij[0]}{ij[1]}"] = r["ph"][k]
    return th


def _verify(args):
    import fit as F
    r, popsize, maxiter = args
    job = dict(cls="GEO", opt=dict(p=r["p"], nu=r["nu"]), seed=1, scale="MZ")
    try:
        out = F.fit(job, popsize=popsize, maxiter=maxiter)
        # also polish from the emulator's own optimum (exact scorer), keep the better
        out2 = F.fit_from(job, _start(r), maxfun=2000)
        if out2 is not None and out2["chi2"] < out["chi2"]:
            out = out2
    except Exception as e:  # noqa: BLE001
        out = dict(job, error=repr(e))
    out["emu_chi2"] = r["emu_chi2"]
    return out


def verify(which="all"):
    S = json.load(open(HERE / "search.json"))
    if which == "all":
        args, fn = [(r, 5, 40) for r in S], "verify_all.jsonl"
    else:
        args, fn = [(r, 8, 100) for r in S[:3]], "verify_top.jsonl"
    with mp.Pool(2) as pool, open(HERE / fn, "w") as fh:
        for out in pool.imap_unordered(_verify, args):
            fh.write(json.dumps(out) + "\n")
            fh.flush()
            print(out.get("opt"), "chi2", round(out.get("chi2", np.nan), 2), "emu", round(out["emu_chi2"], 1),
                  out.get("error", ""), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"gen": gen, "train": train, "search": search, "verify": lambda: verify(sys.argv[2])}[cmd]()

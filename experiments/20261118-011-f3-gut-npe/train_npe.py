"""Train zuko neural spline flows on an F3 dataset.

  python train_npe.py <name> <dataset> <variant> <target> <cond> <minutes> [seed]
    target 'theta': NPE q(theta | x_cond)          (all theta dims)
    target 'pmns' : predictive q(y | x_cond), y = noisy PMNS block z[12:17] (s12^2, s13^2, s23^2, delta_l, lg dm ratio),
                    NO-classified samples only
    cond   'all' (19-dim x + IO flag), 'ql' (13-dim x: masses + CKM), 'mass' (8 mass ratios), 'none'
Noise (gen.add_noise with obs_gut.json sigma) is redrawn every epoch. 2% of samples are held out (fixed noise,
seed 12345) for validation; the held-out indices are stored with the model for calibration checks.
torch.set_num_threads(2). Model + normalisation saved to artifacts/f3/npe_<name>.pt, log to artifacts/f3/logs/.
"""
import json
import pathlib
import sys
import time

import numpy as np
import torch
import zuko

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gen  # noqa: E402

ART = HERE.parents[1] / "artifacts" / "f3"
torch.set_num_threads(2)
REF = gen.gut_reference()
SIG = np.array(REF["sigma"])
X_QL = list(range(13))          # x indices for masses + CKM (x has cos/sin for dq at 11, 12)
X_MASS = list(range(8))


def features(z, io, cond):
    x = gen.z_to_x(z)
    if cond == "all":
        return np.concatenate([x, io[:, None].astype(float)], 1)
    if cond == "ql":
        return x[:, X_QL]
    if cond == "mass":
        return x[:, X_MASS]
    return np.zeros((len(z), 0))


def build_flow(nf, nc, transforms=5, hidden=256):
    return zuko.flows.NSF(nf, nc, transforms=transforms, hidden_features=[hidden, hidden], bins=8, passes=2)


class Model:
    """Flow + standardisation wrapper (save/load and sampling in original units)."""

    def __init__(self, flow, stats, meta):
        self.flow, self.stats, self.meta = flow, stats, meta

    def ctx(self, c):
        s = self.stats
        return torch.as_tensor((c - s["c_mu"]) / s["c_sd"], dtype=torch.float32)

    def sample(self, c, n):
        with torch.no_grad():
            d = self.flow(self.ctx(c[None]) if c.size else None) if self.meta["nc"] else self.flow()
            t = d.sample((n,)).reshape(n, -1).numpy()
        return t * self.stats["t_sd"] + self.stats["t_mu"]

    def log_prob(self, t, c):
        tt = torch.as_tensor((t - self.stats["t_mu"]) / self.stats["t_sd"], dtype=torch.float32)
        with torch.no_grad():
            if self.meta["nc"]:
                lp = self.flow(self.ctx(c)).log_prob(tt).numpy()
            else:
                lp = self.flow().log_prob(tt).numpy()
        return lp - np.log(self.stats["t_sd"]).sum()

    @staticmethod
    def load(name):
        ck = torch.load(ART / f"npe_{name}.pt", weights_only=False)
        f = build_flow(ck["meta"]["nf"], ck["meta"]["nc"])
        f.load_state_dict(ck["state"])
        f.eval()
        return Model(f, ck["stats"], ck["meta"])


def main():
    name, ds, variant, target, cond, minutes = sys.argv[1:7]
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    minutes = float(minutes)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    (ART / "logs").mkdir(parents=True, exist_ok=True)
    logf = open(ART / "logs" / f"{name}.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()

    d = np.load(ART / ds)
    th, z, io, ok = d["theta"].astype(np.float64), d["z"].astype(np.float64), d["io"], d["ok"]
    keep = ok.copy()
    if target == "pmns":
        keep &= io == 0
    idx_all = np.flatnonzero(keep)
    rng.shuffle(idx_all)
    n_val = max(2000, len(idx_all) // 50)
    val_idx, tr_idx = idx_all[:n_val], idx_all[n_val:]
    vrng = np.random.default_rng(12345)
    zv = gen.add_noise(z[val_idx], SIG, vrng)

    def tgt(zn, i):
        return th[i] if target == "theta" else zn[:, gen.PMNS_IDX]

    tv = tgt(zv, val_idx)
    cv = features(zv, io[val_idx], cond)
    t_mu, t_sd = tv.mean(0), tv.std(0) + 1e-9
    c_mu = cv.mean(0) if cv.shape[1] else np.zeros(0)
    c_sd = (cv.std(0) + 1e-9) if cv.shape[1] else np.ones(0)
    if cond == "all":  # binary IO flag: no rescaling
        c_mu[-1], c_sd[-1] = 0.0, 1.0
    stats = dict(t_mu=t_mu, t_sd=t_sd, c_mu=c_mu, c_sd=c_sd)
    nf, nc = tv.shape[1], cv.shape[1]
    flow = build_flow(nf, nc)
    meta = dict(name=name, ds=ds, variant=variant, target=target, cond=cond, nf=nf, nc=nc, seed=seed,
                val_idx=val_idx.tolist() if len(val_idx) <= 50000 else None)
    opt = torch.optim.AdamW(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    T = lambda a: torch.as_tensor(a, dtype=torch.float32)
    tvT, cvT = T((tv - t_mu) / t_sd), T((cv - c_mu) / c_sd)

    def nll(tb, cb):
        return -(flow(cb) if nc else flow()).log_prob(tb).mean()

    log(f"{name}: ds={ds} variant={variant} target={target} cond={cond} n_train={len(tr_idx)} n_val={n_val} nf={nf} nc={nc}")
    t0, best, best_state, ep = time.time(), np.inf, None, 0
    bs = 2048
    while time.time() - t0 < 60 * minutes:
        zn = gen.add_noise(z[tr_idx], SIG, rng)
        tt, cc = T((tgt(zn, tr_idx) - t_mu) / t_sd), T((features(zn, io[tr_idx], cond) - c_mu) / c_sd)
        perm = torch.randperm(len(tr_idx))
        flow.train()
        tot = 0.0
        for k in range(0, len(perm) - bs + 1, bs):
            b = perm[k:k + bs]
            loss = nll(tt[b], cc[b] if nc else None)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            if time.time() - t0 > 60 * minutes:
                break
        flow.eval()
        with torch.no_grad():
            vl = np.mean([nll(tvT[k:k + 8192], cvT[k:k + 8192] if nc else None).item() for k in range(0, n_val, 8192)])
        sched.step(vl)
        ep += 1
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in flow.state_dict().items()}
        log(f"  epoch {ep} train {tot / max(1, len(perm) // bs):.4f} val {vl:.4f} lr {opt.param_groups[0]['lr']:.1e} t={time.time() - t0:.0f}s")
        if opt.param_groups[0]["lr"] < 3e-5:
            break
    flow.load_state_dict(best_state)
    meta["best_val_nll"] = float(best)
    meta["epochs"] = ep
    torch.save(dict(state=flow.state_dict(), stats=stats, meta=meta, val_idx=val_idx), ART / f"npe_{name}.pt")
    log(f"{name}: done, best val NLL {best:.4f} (standardised units), {ep} epochs, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

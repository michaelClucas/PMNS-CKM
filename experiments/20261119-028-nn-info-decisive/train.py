"""Family-conditional flows and family classifiers for exp 028.

Usage: python train.py <kind> <ordering> <seed>
  kind: q1 | q0 (flow) or c1 | c0 (family classifier)
  ordering: NO (artifacts/025/data.npz, exp 025 data) or IO (artifacts/028/data_io.npz)

q1(y | CKM ctx, mass ratios, family one-hot), q0(y | mass ratios, family one-hot): zuko NSF,
same architecture and target map as exp 025. c1(family | CKM ctx, mass ratios), c0(family |
mass ratios): MLP softmax classifiers. The mixture density for any family weights w is
  q_w(y | c) = sum_f pi_f(c; w) q(y | c, f),   pi_f(c; w) ∝ w_f p(f | c) / p_train(f)
(see README). Training runs to convergence: early stopping on the validation split
(patience PATIENCE epochs without improvement > MIN_DELTA), plateau LR decay, epoch cap.
Split 80/10/10 with exp 025's split seed (2501) for NO.
Writes artifacts/028/models/<ordering>_<kind>_s<seed>.pt and curves/<ordering>_<kind>_s<seed>.json.
"""
from __future__ import annotations

import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import json
import pathlib
import time

import numpy as np
import torch
import zuko

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "artifacts" / "028"
DATA = {"NO": ROOT / "artifacts" / "025" / "data.npz", "IO": ART / "data_io.npz"}
SPLIT_SEED = 2501
ARCH = dict(transforms=5, hidden_features=(128, 128), bins=8)  # exp 025 architecture
BATCH = 1024
LR = 1e-3
MAX_EPOCHS = int(os.environ.get("EXP028_MAX_EPOCHS", 80))
PATIENCE = 6
MIN_DELTA = 1e-3
MAX_SECONDS = float(os.environ.get("EXP028_MAX_SECONDS", 3600))
THREADS = int(os.environ.get("EXP028_THREADS", 2))


def to_target(y):
    """exp 025 target map: (s12sq, s13sq, s23sq, delta in [0, 2pi), log10 r) -> R^5."""
    lg = lambda p: np.log(p) - np.log1p(-p)
    d = np.clip(y[:, 3] / (2 * np.pi), 1e-6, 1 - 1e-6)
    return np.stack([lg(y[:, 0]), lg(y[:, 1]), lg(y[:, 2]), lg(d), y[:, 4]], 1)


def load(ordering="NO"):
    D = np.load(DATA[ordering])
    fams = [str(f) for f in D["families"]]
    n = len(D["fam"])
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    split = np.empty(n, int)
    split[perm[: int(0.8 * n)]] = 0
    split[perm[int(0.8 * n): int(0.9 * n)]] = 1
    split[perm[int(0.9 * n):]] = 2
    fam = D["fam"].astype(int)
    return dict(fam=fam, fams=fams, onehot=np.eye(len(fams))[fam], c1=np.concatenate([D["ctx_ckm"], D["ctx_m"]], 1),
                c0=D["ctx_m"], y=to_target(D["y"]), split=split)


class Flow:
    """Family-conditional flow plus the standardisation fitted on its training split.
    Context = standardised physical context ++ one-hot family."""

    def __init__(self, n_phys, n_fam, cm, cs, ym, ys):
        self.n_fam = n_fam
        self.flow = zuko.flows.NSF(features=5, context=n_phys + n_fam, **ARCH)
        self.cm, self.cs, self.ym, self.ys = (np.asarray(a, np.float32) for a in (cm, cs, ym, ys))

    def _ctx(self, c, onehot):
        return torch.as_tensor(np.concatenate([(c - self.cm) / self.cs, onehot], 1).astype(np.float32))

    def log_prob(self, y, c, onehot, chunk=50000):
        """log q(y_target | c, f) in nats (standardised target; Jacobian cancels in q1/q0)."""
        out = []
        self.flow.eval()
        with torch.no_grad():
            for i in range(0, len(y), chunk):
                yy = torch.as_tensor(((y[i:i + chunk] - self.ym) / self.ys).astype(np.float32))
                out.append(self.flow(self._ctx(c[i:i + chunk], onehot[i:i + chunk])).log_prob(yy).numpy())
        return np.concatenate(out).astype(float)

    def sample(self, c, onehot, m):
        self.flow.eval()
        with torch.no_grad():
            s = self.flow(self._ctx(c, onehot)).sample((m,)).numpy().astype(float)
        return s * self.ys + self.ym  # back to target space

    def save(self, path):
        torch.save(dict(state=self.flow.state_dict(), cm=self.cm, cs=self.cs, ym=self.ym, ys=self.ys,
                        n_phys=len(self.cm), n_fam=self.n_fam), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(d["n_phys"], d["n_fam"], d["cm"], d["cs"], d["ym"], d["ys"])
        m.flow.load_state_dict(d["state"])
        return m


class Clf:
    """MLP family classifier p(f | c) on the standardised physical context."""

    def __init__(self, n_phys, n_fam, cm, cs):
        self.net = torch.nn.Sequential(torch.nn.Linear(n_phys, 256), torch.nn.SiLU(), torch.nn.Linear(256, 256),
                                       torch.nn.SiLU(), torch.nn.Linear(256, 256), torch.nn.SiLU(),
                                       torch.nn.Linear(256, n_fam))
        self.n_fam = n_fam
        self.cm, self.cs = np.asarray(cm, np.float32), np.asarray(cs, np.float32)

    def log_proba(self, c, chunk=100000):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(c), chunk):
                x = torch.as_tensor(((c[i:i + chunk] - self.cm) / self.cs).astype(np.float32))
                out.append(torch.log_softmax(self.net(x), 1).numpy())
        return np.concatenate(out).astype(float)

    def save(self, path):
        torch.save(dict(state=self.net.state_dict(), cm=self.cm, cs=self.cs, n_phys=len(self.cm), n_fam=self.n_fam), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(d["n_phys"], d["n_fam"], d["cm"], d["cs"])
        m.net.load_state_dict(d["state"])
        return m


def train(kind, ordering, seed):
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    np.random.seed(seed)
    D = load(ordering)
    c = D["c1"] if kind in ("q1", "c1") else D["c0"]
    tr, va, te = (D["split"] == i for i in range(3))
    nf = len(D["fams"])
    cm, cs = c[tr].mean(0), c[tr].std(0)
    if kind.startswith("q"):
        y = D["y"]
        m = Flow(c.shape[1], nf, cm, cs, y[tr].mean(0), y[tr].std(0))
        params = m.flow.parameters()
        X = m._ctx(c[tr], D["onehot"][tr])
        Y = torch.as_tensor(((y[tr] - m.ym) / m.ys).astype(np.float32))
        lossf = lambda idx: -m.flow(X[idx]).log_prob(Y[idx]).mean()
        valf = lambda: float(-m.log_prob(y[va], c[va], D["onehot"][va]).mean())
        module = m.flow
    else:
        m = Clf(c.shape[1], nf, cm, cs)
        params = m.net.parameters()
        X = torch.as_tensor(((c[tr] - cm) / cs).astype(np.float32))
        Y = torch.as_tensor(D["fam"][tr])
        ce = torch.nn.CrossEntropyLoss()
        lossf = lambda idx: ce(m.net(X[idx]), Y[idx])
        valf = lambda: float(-m.log_proba(c[va])[np.arange(va.sum()), D["fam"][va]].mean())
        module = m.net
    lr0, curve, best, ep0, seconds0 = LR, [], np.inf, 0, 0.0
    mpath = ART / "models" / f"{ordering}_{kind}_s{seed}.pt"
    if os.environ.get("EXP028_RESUME") == "1" and mpath.exists():
        # continue from the saved best state; LR = last LR of the previous run; fresh Adam moments
        prev = json.load(open(HERE / "curves" / f"{ordering}_{kind}_s{seed}.json"))
        module.load_state_dict(torch.load(mpath, weights_only=False)["state"])
        curve, best, ep0, seconds0 = [tuple(r) for r in prev["curve"]], prev["best_val_nll"], prev["epochs"], prev["seconds"]
        lr0 = curve[-1][3]
        print(f"resuming from epoch {ep0}, best {best:.4f}, lr {lr0:.1e}", flush=True)
    opt = torch.optim.Adam(params, lr=lr0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best_state = {k: t.clone() for k, t in module.state_dict().items()} if ep0 else None
    t0, bad, reason = time.time() - seconds0, 0, "max_epochs"
    for ep in range(ep0, MAX_EPOCHS):
        module.train()
        perm = torch.randperm(len(Y))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = lossf(idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        v = valf()
        sched.step(v)
        lr = opt.param_groups[0]["lr"]
        curve.append((ep, tot / len(perm), v, lr, time.time() - t0))
        print(f"[{ordering} {kind} s{seed}] ep {ep} train {tot / len(perm):.4f} val {v:.4f} lr {lr:.1e} "
              f"{time.time() - t0:.0f}s", flush=True)
        if v < best - MIN_DELTA:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in module.state_dict().items()}
        else:
            bad += 1
        if bad >= PATIENCE:
            reason = "early_stop"
            break
        if time.time() - t0 - seconds0 > MAX_SECONDS:
            reason = "wall_cap"
            break
    module.load_state_dict(best_state)
    (ART / "models").mkdir(parents=True, exist_ok=True)
    m.save(mpath)
    out = dict(kind=kind, ordering=ordering, seed=seed, arch=ARCH if kind.startswith("q") else "mlp3x256",
               n_train=int(tr.sum()), n_val=int(va.sum()), best_val_nll=float(best),
               best_epoch=int(np.argmin([r[2] for r in curve])), epochs=len(curve), stop_reason=reason, resumed_at=ep0 or None,
               seconds=time.time() - t0, curve=curve)
    (HERE / "curves").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "curves" / f"{ordering}_{kind}_s{seed}.json", "w"), indent=1)
    print(f"[{ordering} {kind} s{seed}] done best val {best:.4f} ({reason}, {len(curve)} epochs)", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]))

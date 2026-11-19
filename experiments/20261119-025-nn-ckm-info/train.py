"""Train the conditional flows q1(PMNS | CKM, mass ratios) and q0(PMNS | mass ratios) (exp 025).

Usage: python train.py <tag> <which>      tag: "main" or "drop_<FAMILY>"; which: q1 | q0
Reads artifacts/025/data.npz, writes artifacts/025/flows/<tag>_<which>.pt and
experiments/.../flows/<tag>_<which>.json (training curve, test NLL per family).

Target y (5): logit s12^2, logit s13^2, logit s23^2, logit(delta / 2 pi), log10(dm21/dm3l), then
standardised. The transform's Jacobian is the same for q1 and q0, so it cancels in every log
ratio reported. Context: q1 = (log10 s12q, log10 s13q, log10 s23q, cos dq, sin dq, 6 log10 mass
ratios), q0 = the 6 mass ratios. Split (seeded, shared by every run): 80 / 10 / 10.
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
ART = ROOT / "artifacts" / "025"
SPLIT_SEED = 2501
MAX_SECONDS = float(os.environ.get("EXP025_MAX_SECONDS", 420))
EPOCHS = 60
BATCH = 1024
ARCH = dict(transforms=5, hidden_features=(128, 128), bins=8)


def to_target(y):
    """(s12sq, s13sq, s23sq, delta in [0, 2pi), log10 r) -> unbounded 5-vector."""
    lg = lambda p: np.log(p) - np.log1p(-p)
    d = np.clip(y[:, 3] / (2 * np.pi), 1e-6, 1 - 1e-6)
    return np.stack([lg(y[:, 0]), lg(y[:, 1]), lg(y[:, 2]), lg(d), y[:, 4]], 1)


def load(tag):
    D = np.load(ART / "data.npz")
    fams = [str(f) for f in D["families"]]
    n = len(D["fam"])
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    split = np.empty(n, int)
    split[perm[: int(0.8 * n)]] = 0
    split[perm[int(0.8 * n): int(0.9 * n)]] = 1
    split[perm[int(0.9 * n):]] = 2
    keep_train = np.ones(n, bool)
    if tag.startswith("drop_"):
        keep_train &= D["fam"] != fams.index(tag[5:])
    return dict(fam=D["fam"], fams=fams, c1=np.concatenate([D["ctx_ckm"], D["ctx_m"]], 1), c0=D["ctx_m"],
                y=to_target(D["y"]), split=split, keep_train=keep_train)


def make_flow(n_ctx):
    return zuko.flows.NSF(features=5, context=n_ctx, **ARCH)


class Model:
    """Flow plus the standardisation fitted on its training split."""

    def __init__(self, n_ctx, cm, cs, ym, ys):
        self.flow = make_flow(n_ctx)
        self.cm, self.cs, self.ym, self.ys = (np.asarray(a, np.float32) for a in (cm, cs, ym, ys))

    def log_prob(self, y, c, chunk=20000):
        """log q(y_target | c) in the standardised target space (nats), numpy in/out."""
        out = []
        self.flow.eval()
        with torch.no_grad():
            for i in range(0, len(y), chunk):
                yy = torch.as_tensor(((y[i:i + chunk] - self.ym) / self.ys).astype(np.float32))
                cc = torch.as_tensor(((c[i:i + chunk] - self.cm) / self.cs).astype(np.float32))
                out.append(self.flow(cc).log_prob(yy).numpy())
        return np.concatenate(out).astype(float)

    def sample(self, c, m):
        """m samples (m, n, 5) in the standardised target space for each context row."""
        self.flow.eval()
        with torch.no_grad():
            cc = torch.as_tensor(((c - self.cm) / self.cs).astype(np.float32))
            return self.flow(cc).sample((m,)).numpy().astype(float)

    def to_std(self, y):
        return (y - self.ym) / self.ys

    def save(self, path):
        torch.save(dict(state=self.flow.state_dict(), cm=self.cm, cs=self.cs, ym=self.ym, ys=self.ys,
                        n_ctx=len(self.cm)), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        m = cls(d["n_ctx"], d["cm"], d["cs"], d["ym"], d["ys"])
        m.flow.load_state_dict(d["state"])
        return m


def train(tag, which, seed=0):
    torch.set_num_threads(int(os.environ.get("EXP025_THREADS", 1)))
    torch.manual_seed(seed)
    D = load(tag)
    c = D["c1"] if which == "q1" else D["c0"]
    y = D["y"]
    tr = (D["split"] == 0) & D["keep_train"]
    va = (D["split"] == 1) & D["keep_train"]
    te = D["split"] == 2
    m = Model(c.shape[1], c[tr].mean(0), c[tr].std(0), y[tr].mean(0), y[tr].std(0))
    ctr = torch.as_tensor(((c[tr] - m.cm) / m.cs).astype(np.float32))
    ytr = torch.as_tensor(((y[tr] - m.ym) / m.ys).astype(np.float32))
    opt = torch.optim.Adam(m.flow.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, curve, t0, bad = np.inf, None, [], time.time(), 0
    for ep in range(EPOCHS):
        m.flow.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            loss = -m.flow(ctr[idx]).log_prob(ytr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.flow.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(idx)
        v = -m.log_prob(y[va], c[va]).mean()
        sched.step(v)
        curve.append((ep, tot / len(perm), float(v), time.time() - t0))
        print(f"[{tag} {which}] ep {ep} train {tot / len(perm):.4f} val {v:.4f} {time.time() - t0:.0f}s", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in m.flow.state_dict().items()}
        else:
            bad += 1
        if bad >= 6 or time.time() - t0 > MAX_SECONDS:
            break
    m.flow.load_state_dict(best_state)
    (ART / "flows").mkdir(parents=True, exist_ok=True)
    m.save(ART / "flows" / f"{tag}_{which}.pt")
    lp = m.log_prob(y[te], c[te])
    per_fam = {f: float(-lp[D["fam"][te] == i].mean()) for i, f in enumerate(D["fams"])}
    out = dict(tag=tag, which=which, arch=ARCH, n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()),
               best_val_nll=float(best), test_nll=float(-lp.mean()), test_nll_per_family=per_fam,
               epochs=len(curve), seconds=time.time() - t0, curve=curve)
    (HERE / "flows").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "flows" / f"{tag}_{which}.json", "w"), indent=1)
    print(f"[{tag} {which}] done test nll {out['test_nll']:.4f}", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], seed=1 if sys.argv[1] == "seed1" else 0)

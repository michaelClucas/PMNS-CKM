"""Map the published MN no-RGE point (1306.4468 Eq. 15) into the quark-anchored theta of so10fit.mn_mats and check
that the DR-target chi2 is unchanged (validates the reparametrisation and the global-fit objective).
Also writes the theta so it can seed / be compared with the optimizer. Output: check_reparam.json."""
import json

import numpy as np

import so10fit as S
from src.mixing import extract_angles, rephase_to_standard


def takagi(A):
    """Q unitary with Q^T A Q = diag(sigma) real >= 0, ascending."""
    w, V = np.linalg.eigh(A.conj().T @ A)
    d = np.diag(V.T @ A @ V)
    return V * np.exp(-0.5j * np.angle(d))[None, :]


H = np.diag([1.22387e-6, 5.92428e-5, 6.29473e-3]).astype(complex)
F = np.array([[-2.95102e-6 - 3.48291e-6j, 1.27484e-5 - 7.53714e-8j, 1.07772e-4 + 6.02931e-5j],
              [1.27484e-5 - 7.53714e-8j, -1.538e-4 + 6.75236e-5j, -2.67281e-4 + 2.48978e-4j],
              [1.07772e-4 + 6.02931e-5j, -2.67281e-4 + 2.48978e-4j, -7.38503e-4 - 1.44559e-3j]])
r, s, rR = 68.9624, 0.370726 + 0.063044j, 3.014e-16
Yu, Yd = r * (H + s * F), H + F
Qu = takagi(Yu)
u = np.real(np.diag(Qu.T @ Yu @ Qu))
Ydp = Qu.T @ Yd @ Qu
Qd = takagi(Ydp)
d = np.real(np.diag(Qd.T @ Ydp @ Qd))
Ustd, P, Qr = rephase_to_standard(Qd)
Pn = P / P[0, 0]
U = Pn @ Ustd                     # U = Qd Qr^-1 / P11, so U^T Yd' U is diagonal with phases
D = np.diag(U.T @ Ydp @ U)
t12, t13, t23, dq = extract_angles(Ustd)
yb = d[2]
th = np.array([np.log10(u[0] / u[1]), np.log10(u[1] / u[2]), np.log10(u[2] / yb), np.log10(d[0] / d[1]),
               np.log10(d[1] / d[2]), np.sin(t12), np.log10(np.sin(t13)), np.log10(np.sin(t23)), dq,
               np.angle(Pn[1, 1]) % (2 * np.pi), np.angle(Pn[2, 2]) % (2 * np.pi)]
              + list(np.angle(D) % (2 * np.pi)) + [np.log10(r), np.log10(abs(s)), np.angle(s) % (2 * np.pi)])
# absolute scales: masses = V * yb * sv; m_nu = V^2 rR r^2 yb * (MD F^-1 MD)[units of yb]  (F_true = yb F)
lg_y = np.log10(S.V_EW * yb)
lg_nu = np.log10(S.V_EW ** 2 * rR * r ** 2 * yb * 1e9)
theta = np.concatenate([th, [lg_y, lg_nu]])[None]
prob = S.Problem("MN", "DR")
p, o = prob.pulls(theta)
# sanity of the Takagi step
err_u = np.abs(Qu.T @ Yu @ Qu - np.diag(u)).max() / u.max()
res = dict(theta=dict(zip(prob.names, map(float, theta[0]))), chi2=float((p ** 2).sum()),
           pulls=dict(zip(prob.obs_names, map(float, p[0]))), takagi_rel_err=float(err_u))
(S.HERE / "check_reparam.json").write_text(json.dumps(res, indent=1))
print(f"quark-anchored chi2 at the published MN point: {res['chi2']:.4f} (DR-form eval 1.111); takagi err {err_u:.1e}")
print(" ".join(f"{k}={v:+.2f}" for k, v in res["pulls"].items()))
print(res["theta"])

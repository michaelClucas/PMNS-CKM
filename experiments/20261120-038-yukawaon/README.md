# 20261120-038 Koide-Nishiura yukawaon / democratic universal seesaw model

## Hypothesis

The Koide-Nishiura "yukawaon" class builds every Yukawa matrix from one shared flavon
Phi_0 = diag(sqrt m_e, sqrt m_mu, sqrt m_tau), and in its most compressive version
(arXiv:1510.05370, PRD 92 111301(R); theory in 1512.08386) claims 16 quark + lepton observables
from 6 family-independent parameters, with PMNS fixed once the quark sector is fitted. If that
holds with today's data it is a candidate with k << N in which the quark sector really does
determine PMNS.

## Method

1. Literature (arXiv API author listing, full PDF text kept in the session scratchpad, not
   committed). 1510.05370 is the most compressive unified version with a full fit. 1701.06287 has
   the same mass matrices and reuses that fit. 1805.07334 is quark-only and 1911.03411 is
   neutrino-only. Matrices, equation numbers and the paper-vs-text discrepancies are in `model.md`.
2. `reproduce.py` builds the paper's Eqs. (18)-(19) through `src.yukawa` from the paper's own
   inputs (its six parameters plus its M_Z charged-lepton masses). Of 16 convention variants,
   exactly one reproduces every published number (`reproduce.log`).
3. `fit.py` uses the official scorer, `fit_and_evaluate(method="de")`, popsize 20, maxiter 600,
   NO tables (`IC23_woSKatm`), with an extra start at the paper's point.
   - Use: ckm_s12, ckm_s13, ckm_s23, ckm_delta_rad, mr_u_c, mr_c_t, mr_d_s, mr_s_b, s12sq, s13sq,
     s23sq, dcp_deg, dm_ratio. N = 13.
   - Inputs (not scored): mr_e_mu, mr_mu_tau, which build Phi_0 and are read through
     `src.evaluate.mass_ratio_targets` on every call.
   - mass_floor 0.02 on every mass key.
   - Scale: M_Z, no running, as in the paper (Sec. 4).
   - Two DE seeds for the full fit, plus a quark-only fit.
4. `bound.py`: grid and local scans of the reachable quark mass ratios, independent of the
   parameters. `link.py`: shared-parameter sensitivities and a rescore with a 30% floor.
   `heldout.py`: held-out PMNS prediction.
5. Sensitivity: the full fit, the quark fit and the held-out test rerun with a 30% mass floor
   (`YK_FLOOR=0.3`), roughly the error budget the paper used.

## Parameter count

- k_cont = 6: b_u, b_d, beta_d, phi1, phi2, xi_R. This is the paper's own count. The phi3 phase is
  unobservable, as the paper says.
- k_bits = 4 (honest): which published yukawaon/seesaw variant to use. Koide-Nishiura published
  at least 9 unified quark + lepton versions (2010-2017) that differ in phase-flavon placement,
  the Y_R structure and S_f vs S_f^-1, and they chose among them by data, so ceil(log2 >= 9) = 4.
  The paper itself counts 0 bits.
- No ordering bit: the model gives NO by itself.
- Convention choices are fixed by reproducing the paper, not by our data, so they cost 0 bits.
- k = 10, N = 13, N - 2 = 11. **compression_ok = True**, even with the honest bits. This is the
  first CKM-linked family in the program that passes criterion 2 with an honest count.
- dof = N - k_cont = 7.

## Result

### Reproduction (their inputs, our code; `reproduce.log`)

| quantity | ours | paper |
|---|---|---|
| \|Vus\|, \|Vcb\|, \|Vub\|, \|Vtd\| | 0.22569, 0.039956, 0.0037003, 0.009172 | 0.2257, 0.03996, 0.003701, 0.009173 |
| delta_q | 80.993 deg | 80.99 deg |
| sqrt(mu/mc), sqrt(mc/mt), md/ms, ms/mb | 0.0610, 0.0598, 0.0494, 0.0273 | 0.061, 0.060, 0.049, 0.027 |
| sin^2 2theta12, 23, 13 | 0.8263, 0.9967, 0.1006 | 0.8254, 0.9967, 0.1007 |
| delta_l, R = Dm21/Dm32 | -67.88 deg, 0.03100 | -68.1 deg, 0.03118 |

Agreement: CKM to 2e-4 relative, PMNS to 0.1%, phases to 0.22 deg, R to 0.6%, mass ratios at the
paper's 2-digit precision. The residual is from xi_R and the lepton masses being printed to 5 digits.

The paper does not show that the benchmark's actual mixing angle lies on the "dark side":
s12^2 = 0.712 (theta12 = 57.6 deg with m1 < m2). sin^2 2theta12 is octant-blind, so its Table 1
hides this. Scored with our data, the published point gives chi^2 = 1426 (N 13).

### Fit to our data (`fit_full_s*.log`, `fit_quark_s1.log`)

| fit | floor | chi^2 | N | k_cont + bits | dof | p_global | max pull | compression_ok | fit_ok |
|---|---|---|---|---|---|---|---|---|---|
| full, seed 1 | 2% | 1027.18 | 13 | 6 + 4 | 7 | 1.6e-217 | mr_u_c +22.9 | yes | no |
| full, seed 2 | 2% | 1027.05 | 13 | 6 + 4 | 7 | 1.7e-217 | mr_u_c +22.9 | yes | no |
| quark only | 2% | 912.4 | 8 | 5 + 4 | 3 | 1.8e-197 | mr_u_c +23.7 | no | no |
| full (sensitivity) | 30% | 41.12 | 13 | 6 + 4 | 7 | 7.7e-7 | ckm_delta +2.6, dcp -2.7, mr_u_c +2.9 | yes | no |
| quark only (sensitivity) | 30% | 11.00 | 8 | 5 + 4 | 3 | 0.012 | mr_u_c +2.9 | no | no |

- The two full-fit seeds land on the same minimum: b_d -> -b_d, beta_d -> beta_d + 180 is an exact
  symmetry.
- Pulls at the 2% best fit: mr_u_c +22.9, mr_s_b +20.2, ckm_delta +5.3, ckm_s12 +3.9, s13sq +3.9,
  dcp -2.7, s23sq +1.4, s12sq +0.3.

**Why the fit fails, independent of parameters (`bound.py`, `bound.log`):**
- **Up sector.** Once m_c/m_t matches data, m_u/m_c is fixed at 0.00373-0.00379 for every
  (b_u, phi1, phi2). That is ~0.79 m_e/m_mu, against the data value 0.00198 (a factor 1.89). With
  b_u -> -1 the up-quark light block is Phi_0-dominated, so the phases do not move this ratio.
- **Down sector.** With m_d/m_s within +-5% of data, m_s/m_b >= 0.0225, against the data value
  0.0187. This is the "parameter-independent quark mass relation" Koide and Nishiura themselves
  derived later (1805.07334).
- Together these give a mass-block chi^2 >= ~620 at the 2% floor, before any mixing observable is
  counted. The paper fitted against Xing et al. 2008 ratios with 20-30% errors and says in the
  text that it misses sqrt(m_u/m_c) and m_s/m_b.
- Even with a 30% floor the full fit fails (p 7.7e-7). The leftover tension is in the mixing:
  delta_q, s13sq and delta_l pull against each other through the shared phases (phi1, phi2).

### Does the CKM enter PMNS? (`link.py`, `link.log`)

Yes, through shared parameters. Y_u sits inside the RH Majorana matrix Y_R, so b_u, phi1 and phi2
act on both sectors. b_d and beta_d act only on quarks (dPMNS/db_d = 0 exactly), and xi_R acts
only on leptons. At the best fit, in sigma units per degree:

| | CKM | PMNS |
|---|---|---|
| phi1 | delta_q +2.3 | s12sq -5.0 |
| phi2 | delta_q -4.9 | s13sq +5.4 |

Redrawing (phi1, phi2) uniformly spreads s12sq over [0.06, 0.47] and s13sq over [0.03, 0.49]
(5-95%). The link is real, but PMNS moves several sigma for every degree the CKM allows the
phases to move.

### Held-out test: quark sector -> PMNS (`heldout.py`, `heldout*.log`)

Method:
- Fit the quark sector only (N 8).
- Run a Metropolis chain over its 5 parameters (4000 steps, 400 kept).
- For each sample, fit xi_R to dm_ratio only; the four PMNS parameters are then predictions.

| floor | s12sq median [95%] | s13sq median [95%] | s23sq median | dcp median | in NuFIT 95% (4 params) |
|---|---|---|---|---|---|
| 2% | 0.225 [0.010, 0.227] | 0.0033 [0.0017, 0.77] | 0.515 | 297 deg | 0/400 |
| 30% | 0.240 [0.009, 0.246] | 0.0034 [0.0013, 0.76] | 0.524 | 218 deg | 0/400 |

NuFIT NO: 0.309 / 0.0225 / 0.470 / 207 deg. The quark sector predicts theta13 about 2.5x too
small and theta12 too small. The good PMNS of the full fit comes from moving (phi1, phi2) about
1 deg away from the quark-preferred point, which costs CKM chi^2 (ckm_delta +2.6 to 5.3 sigma). So
the PMNS is accommodated at the CKM's expense, not predicted.

### Leak probe

`leak.log`: undeclared probe over ckm.json, pmns.json and masses.json at the best fit flags
only m_e, m_mu and m_tau, which are the declared inputs (mr_e_mu, mr_mu_tau) that build Phi_0.
No CKM, PMNS or quark-mass record reaches the model.

## Verdict

- **Fit quality: fails.** At the program's error budget (PDG 2026, 2% floor), chi^2 = 1027 for
  N 13, p_global 1.7e-217. Two quark mass relations of the model are independent of its
  parameters and miss the data by about 20 sigma each (m_u/m_c ~ 0.79 m_e/m_mu; m_s/m_b >= 0.0225
  given m_d/m_s). With a 30% mass floor the fit still fails (p 7.7e-7).
- **Compression: passes.** k = 6 + 4 bits = 10 <= N - 2 = 11.
- **Held-out PMNS from the quark sector: fails.** 0/400 posterior samples fall inside NuFIT 95%
  under either floor.
- The class is "excluded" in the fit-quality sense (STATE protocol item 7). It is a known
  published model, so it is logged as a baseline, not a candidate. Criterion 4 was not run, since
  compression and fit must both pass first.
- Not a reduction to an already-ruled-out family. U_e = 1 (no F2 Clebsch rotation), and there is
  no fixed-pattern shared rotation (F1). The quark-lepton link runs through Y_u inside M_R,
  which differs from exp 035(b) (Y_nu = Y_u).
- The model has the structure the program has been looking for: few shared parameters, and PMNS
  depends on the quark phases. It fails for a reason unrelated to the CKM-PMNS link: Phi_0 built
  from charged-lepton masses forces quark mass ratios that 2026 light-quark data rule out.

## Caveats

- No running between M_Z and the flavon scale (the paper does the same; mass ratios are compared
  at M_Z).
- The 6x6 universal seesaw is used in the paper's approximate form, Eq. (17). The authors say this
  approximation is not valid at b_u ~ -1 (Sec. 3). The exact 6x6 diagonalisation was not tried,
  and could change m_u/m_c. A modified model would need its own parameter count.
- The bound scans are grids plus local polishing, not a proof. The up-sector ratio barely moves
  over the whole grid (0.00373-0.00379), which supports the relation.
- The held-out chain is short (4000 steps). The PMNS distribution has long tails because PMNS is
  so sensitive to (phi1, phi2).

## Files

- `model.md`: the paper's matrices with equation numbers, and the conventions.
- `model.py`: the model.
- `reproduce.py` / `.log` / `.json`: benchmark reproduction.
- `fit.py`, `fit_*.json` / `.log`: DE fits (`YK_FLOOR` env var for the sensitivity runs).
- `bound.py` / `.log` / `.json`: parameter-independent mass bounds.
- `link.py` / `.log` / `.json`: shared-parameter sensitivity.
- `heldout.py`, `heldout*.json` / `.log`: held-out PMNS test.
- `leak.log`: leak probe.
- `run.log`: background job log.

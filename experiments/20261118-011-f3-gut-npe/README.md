# 011: F3 groundwork, GUT Yukawa generator + simulation-based inference

## Hypothesis

In minimal renormalizable SO(10) (10_H + 126bar_H, type I or type II seesaw), every fermion mass matrix is built from the same two Yukawa matrices. If the measured quark and charged-lepton data pin the model down, the leftover freedom might force PMNS onto a low-dimensional set: a "ridge" tying lepton mixing to CKM quantities that could be distilled into a closed form. The model is not compressive by itself (17 dimensionless parameters for 17 dimensionless observables). Its job here is to generate hypotheses.

## Clebsch structure (verified)

Source: A. Dueck, W. Rodejohann, "Fits to SO(10) Grand Unified Models", arXiv:1306.4468, Eq. (3) (general VEV form) and Eq. (4):

    Y_u = r (H + s F + i t_u G),   Y_d = H + F + i G,   Y_D = r (H - 3 s F + i t_D G),   Y_l = H - 3 F + i t_l G,   M_R = r_R^-1 F

Type II: M_L = v_L^126 Y_126 (their Eq. (3)), so m_nu ∝ F = (Y_d - Y_l)/4. That is the Bajc-Senjanovic-Vissani relation (hep-ph/0210207, ID and abstract checked: type II plus large atmospheric mixing needs b-τ unification). Their Sec. 2.1 count for 10 + 126bar is 19 real parameters with type I (H 3, F 12, r, s complex, r_R), and r and r_R can be taken real WLOG. The exp 003 note was right on the -3 for leptons and the -3s for Y_D. Its "Y_e = H - 3F, Y_ν = H - 3 r F" mixed the r normalisation; Eq. (4) above is the correct form. Joshipura-Patel arXiv:1102.5148 (ID and abstract checked) report good fits under type-I dominance and failure of type II for lack of b-τ unification.

## Generator (`gen.py`)

All matrices are Weyl mass matrices (f^c M f). The `src/rg.py` Yukawa is Y = M^dagger, the left rotations come from M^dagger M, and kappa ∝ m_nu. Check: `check_gen.py` reproduces `src.rg.observables` angles to 2.4e-8 rad, and type-II m_nu equals (Y_d - Y_l)/4 to 3e-17 (run.log).

Quark-anchored parametrisation (datasets): the SO(10) flavour basis is chosen so that Y_u is real diagonal (Takagi) instead of H. Then Y_d = U^* diag(d_i e^{iφ_i}) U^dagger with U = diag(1, e^{ia2}, e^{ia3}) V_CKM, F = (Y_u/r - Y_d)/(s-1), H = Y_d - F, Y_l = Y_d - 4F, Y_D ∝ Y_d - (1+3s)F. This is an exact reparametrisation of (H, F, r, s). It puts the quark data directly in the parameters, so quark-sector acceptance is not wasted.

Observable vector z (17 dims; each phase enters the network as cos/sin, giving 19 dims plus an IO flag): log10 of mu/mc, mc/mt, md/ms, ms/mb, mb/mt, me/mμ, mμ/mτ, mτ/mb; sin θ12^q, log10 sin θ13^q, log10 sin θ23^q, δ_q; sin²θ12, sin²θ13, sin²θ23, δ_ℓ, log10(Δm²21/|Δm²3ℓ|). Each sample is classified NO or IO from its own light spectrum (IO when the closest pair is the upper one), and the PMNS columns are ordered to match.

### Variants and parameter counts (real, after removing unphysical phases)

| variant | content | k dimensionless (fits ratios) | k incl. absolute scales |
|---|---|---|---|
| `so10_I` | 10 + 126bar, type I | 17: Y_u 3 (u1/u2, u2/u3, u3/d3), Y_d 12 (2 ratios + CKM 4 + 5 phases + overall scale dropped), r, s (2) | 19 (+ y_b scale, + r_R) = 1306.4468 Sec. 2.1 |
| `so10_II` | 10 + 126bar, type II (m_nu ∝ F) | 17 | 19 (+ y_b, + v_L) |
| `so10_mixed` | type I + II, complex relative weight w | 19 | 21 |
| `so10_120` | + 120_H, all complex (H-diagonal basis, hierarchical F, G priors) | 29 | 31 |
| `su5_min` | SU(5) 5_H (Y_e = Y_d^T) + 15_H type II with independent Yukawa | 19 (quark 11 + ν 8) | 21 |
| `su5_gj` | + 45_H entry with Georgi-Jarlskog -3 | 21 | 23 |

Observables: 17 dimensionless (19 with the m_b and Δm²31 scales). So SO(10) 10 + 126bar has k = N and is not compressive.

### Priors (stated, partly data-informed)
Quark block: uniform windows of ±0.5 dex around the GUT-scale data for the mass ratios, s12^q ∈ [0.15, 0.30], log10 s13, s23 ±0.4 dex, δ_q ∈ [0, 2π). The 5 phases are uniform. log10 r ∈ [0, 4], log10|s| ∈ [-2, 2], arg s uniform. These windows are centred on the data (`obs_gut.json`), which is a deliberate choice: the question is the lepton sector given quark data.

### Data at M_GUT and noise model (`data_gut.py` → `obs_gut.json`)
M_Z data from `data/` are run to 2×10^16 GeV with `src/rg.py` (SM). Simulator noise and measurement error use the same model: each mass gets max(PDG error, 2%) (theory floor), CKM s12 1% floor, s13 and s23 3%, δ_q 0.05 rad, and PMNS NuFIT 6.1 symmetrised. GUT values used: mτ/mb = 1.59 (lg 0.201), sin²θ23 = 0.470, sin²θ13 = 0.0225, sin²θ12 = 0.309, δ_ℓ = 207° (NO).

### Running map (`rg_validate.py` → `rg_map_so10_I.json`)
I ran 200 data-like samples (ql posterior) from M_GUT down to M_Z with `src/rg.py`, at 0.11 s per sample. Shifts GUT→M_Z: lg(mτ/mb) -0.416, lg(mb/mt) +0.095, lg(mc/mt) +0.055, lg s13, s23 -0.055. PMNS shifts are 0 to 5 decimals (SM, hierarchical). A linear map leaves an RMS residual of ≤ 0.003σ on every observable, and a constant shift leaves ≤ 0.23σ. Full per-sample running is cheap enough for validation, but 1e6 samples would take about 30 core-hours. The datasets therefore stay at M_GUT, and the comparison is made at M_GUT against data run up with `src/rg.py`. This was only validated on near-data samples.

## Datasets (`make_dataset.py`, gitignored `artifacts/f3/`, 365 MB)
`so10_I` and `so10_II`: 1,000,000 samples each from the broad prior (seed 11, chunk seeds 11000-11009, 23 s each on 4 workers). `su5_min`, `su5_gj` and `so10_mixed`: 100,000 each (seed 21). Summary and quantiles are in `dataset_summary.json`. In the prior predictive, IO occurs in ≤ 1e-4 of so10 samples and in 8% of SU(5) samples.

## Method
1. **Amortised NPE** (`train_npe.py`): zuko neural spline flow (5 coupling transforms, 256x256), q(θ | x, IO flag), trained on 0.98e6 samples with fresh noise each epoch and torch 2 threads. Compute was shared with other agents and the machine sat at 100% CPU, so each model got 11 min, which is **one epoch**. Calibration (`analyze_npe.py`) used SBC ranks, marginal coverage and joint HPD coverage on 300 held-out synthetic samples with 300 posterior draws each.
2. **Real-data posterior without a network**: the data occupy a region that random prior draws essentially never reach. In 2e5 prior draws per variant, none had charged-lepton χ² < 9 or χ²(all 17) < 40, and the minimum χ²(all) was 6600 (so10_I) and 8400 (so10_II) (`check_gen.py`). I therefore used multi-start least squares (`fit_ms.py`) on the 12 quark + charged-lepton observables (32 starts per variant, seed 7, 18 converge to χ² ≈ 0; a later 100-start so10_I run, 67 converged, overwrote `fits_so10_I_ql.npz` after the MCMC had read the 32-start file), followed by vectorised differential-evolution MCMC (`mcmc.py`, 192 chains × 20000 iterations, exact likelihood under the uniform prior) for p(θ | quark + charged-lepton data) ("ql") and p(θ | all 17) ("all").
3. **Held-out PMNS test**: PMNS predictive = the PMNS outputs of the ql posterior. PMNS never enters the ql likelihood. The ql likelihood is identical for type I and type II (seesaw type only changes m_nu), so both variants share the same ql θ-posterior and differ only in their PMNS readout.
4. Predictive flows q(PMNS | masses + CKM) and q(PMNS | masses) for I(PMNS; CKM | masses) (`mi_flows.py`, so10_II only), plus a b-τ / θ23 check (`btau_ridge.py`) and an SU(5) control (`su5_check.py`).

## Results

**NPE calibration (synthetic, broad prior; `broad_so10_*_calib.json/.png`).** The joint HPD coverage is conservative. At nominal 0.5 / 0.7 / 0.9 it reads 0.69 / 0.84 / 0.95 for so10_I and 0.65 / 0.81 / 0.97 for so10_II. The quark-sector dims are under-resolved: median posterior sd / prior sd is 0.07-0.14, against a true value of about 0.01-0.02, and their SBC rank tests reject uniformity (p < 0.01). r and |s| are partly learned (sd ratio 0.5-0.8). The 5 phases and arg s stay at the prior (ratio 1.0, ranks uniform). Verdict: after 1 epoch the network is calibrated in the conservative direction but far from converged.

**Real data vs the amortised NPE.** The real data are outside the NPE's effective support. Posterior-predictive χ²(all 17) at the NO data vector has median 8.0e4 (so10_I) and 4.4e4 (so10_II), and only 48-53% of NPE draws fall inside the prior box. **The amortised NPE posterior at the real data is not usable.** This is a prior-volume problem: the data region is a tiny, curved, fine-tuned subset of the prior.

**Quark + charged-lepton posterior (DE-MC; `local_so10_*.json`).** The 12 ql observables are fitted at χ² ≈ 0 by many solutions. They fix r ≈ 0.9 m_t/m_b (lg r = 1.82 [1.80, 1.88] at 90%), |s| = 0.89 [0.72, 1.04], and s ≈ 0.75 ± 0.4i, close to 1, which is a near-cancellation of the third-generation Y_u/r against Y_d. The 5 phases are essentially unconstrained (circular sd 1.4-2.2 rad).

**Held-out PMNS prediction vs measurement (the key test; `local_so10_I.png`, `local_so10_II.png`).** Within the modes found, conditioning on quark + charged-lepton data predicts small lepton mixing:

| | so10_I (type I) 95% interval | so10_II (type II) 95% interval | measured (GUT) |
|---|---|---|---|
| sin²θ12 | [0.0002, 0.0039] | [0.009, 0.466] | 0.309 |
| sin²θ13 | [0, 0.0003] | [0.0009, 0.0112] | 0.0225 |
| sin²θ23 | [0.0002, 0.0023] | [0.0069, 0.095] | 0.470 |
| δ_ℓ | flat | flat | 207° |
| lg Δm²21/Δm²3ℓ | [-2.70, -2.33] | [-2.98, -1.32] | -1.52 |

PIT of the measured values: θ13 and θ23 at 1.000 for both variants. No posterior sample has a 4-parameter NuFIT Δχ² below 9.49. **The held-out PMNS is not reproduced: in these modes the model predicts quark-like small lepton angles.** The prior predictive is similar (sin²θ23 97.5% quantile 0.022 and 0.009), so the ql data barely move the PMNS prediction toward the measurement.

**Full fit (all 17, NO).** The DE-MC was started from the ql modes. Best χ² is 131.8 (type I) and 462.6 (type II). Type I reaches the measured θ12 and θ13 by giving up quark data (md/ms -4.2σ, s12^q +3.5σ, δ_q +4.2σ, sin²θ23 -5.3σ, δ_ℓ +6.5σ). The type-II best point pulls mτ/mb at -10σ toward b-τ unification and still has sin²θ23 at -16σ. No converged least-squares full fit is available (see limitations). This is **not** evidence that minimal SO(10) fails: Dueck-Rodejohann report χ² ≈ 1 for the non-SUSY 10 + 126bar type-I model (no RGE, NO, 5% charged-lepton errors, older data). My search did not reach that region.

**Ridge / mutual information.** The KSG MI between PMNS outputs and θ or ql observables across the ql posterior comes out 0.7-1.7 nats for every pair. That value reflects chain identity: 192 slowly mixing chains in separate modes, so any coordinate labels the chain. It does **not** measure a CKM-PMNS ridge, and I do not report it as one. The amortised estimate I(PMNS; CKM | masses) from the predictive flows came out -0.22 ± 0.01 nats (so10_II). A negative value means the 1-epoch flows are not converged, so the result is inconclusive. Its predictive HPD coverage at 0.5 / 0.68 / 0.95 is 0.40 / 0.58 / 0.91.

**b-τ / θ23 (type II, `btau_ridge.json/.png`).** Among 686 broad-prior samples with me/mμ and mμ/mτ within 0.3 dex of the data, sin²θ23 > 0.3 occurs only for |log10 mτ/mb| < 0.1 (3-6% of those bins, 0 elsewhere). That agrees with BSV, but the statistics are low. The data have log10(mτ/mb) = +0.20 at M_GUT in the SM, which lies in the region with zero large-θ23 samples.

**SU(5) control (`su5_check.json`).** In `su5_min`, U_e = 1 exactly in the Y_u-diagonal basis, so PMNS = U_ν, independent of the quark sector. KSG MI(PMNS angles; CKM angles) is ≤ 0.011 nats (estimator noise) for both su5_min and su5_gj. mτ/mb = 1 at GUT exactly, against 1.59 in the data. Minimal SU(5) has no CKM-PMNS link at all unless the neutrino sector is tied to Y_d by extra structure.

## Verdict
No candidate. The generator, the Clebsch verification, the running map and the 2×10^6-sample SO(10) datasets are in place. The amortised NPE is calibrated (conservatively) on synthetic data but cannot be conditioned on the real data, because the data sit in a fine-tuned corner of the prior. Exact-likelihood MCMC conditioned on quark + charged-lepton data predicts small lepton mixing in every mode found (type I and type II), which fails the held-out PMNS test there. The realistic-PMNS solutions known from the literature lie in a region this search did not reach.

## Hypotheses suggested (for F1/F2/F6, lower k)
1. **Type II: θ23^ℓ large only near b-τ unification** (m_ν ∝ Y_d - Y_l). This is BSV, already known, so it is a baseline and not new. With SM running, mτ/mb(GUT) = 1.59 disfavours it. With MSSM running (tanβ-dependent mb/mτ → 1) it would be a testable one-parameter relation: θ23 as a function of (mb - mτ)/mb at M_GUT. It should be scored with MSSM `src/rg.py`, keeping in mind the 2% θ23^q systematic.
2. **Quark-anchored SO(10) generically yields quark-like PMNS** (s23² ~ 1e-3 type I). Large lepton mixing needs a cancellation in F^-1 (type I) or in Y_d - Y_l (type II). A low-k family worth testing is "large mixing from a near-singular F": M_R ∝ F with one small eigenvalue, where PMNS is then controlled by the F eigenvector, itself a function of the quark data and s. That is close to F6 (texture + seesaw tied to quark Yukawas).
3. The ql data constrain only r and s (s ≈ 1, r ≈ m_t/m_b) and leave the 5 Y_d phases free. Any SO(10)-derived PMNS-CKM relation therefore has to come from a mechanism that fixes those phases (spontaneous CP or a real-parameter model as in 1306.4468 Sec. 2.3, "FN" with 18 parameters). Next F3 step: a real-parameter 10 + 126bar + 120 variant, where the phases cannot absorb anything (the lesson from exp 007).

## Limitations
- Compute: the machine was saturated by parallel jobs. The NPEs got 1 epoch, and the predictive flows for so10_I were not trained (lane stopped for time, see run.log). The 200-start random full fit and the 16-start from-ql full fit were killed unfinished, so full-fit numbers come from DE-MC.
- The DE-MC does not hop between distant modes. Posterior mode weights are set by the multi-start solutions, which came from 32 random starts per variant on the 12 ql observables (plus 100 more for so10_I, which landed in the same family), and all of them landed in the small-mixing family. The ql posteriors for I and II are identical by construction (same seeds and likelihood). Thinned samples are strongly autocorrelated (acceptance 1.2-1.8%).
- The noise model uses a 2% floor per mass and symmetrised Gaussian PMNS errors at M_GUT (NuFIT 2D tables are used only in the held-out NuFIT Δχ² check, which treats GUT PMNS ≈ M_Z PMNS: SM shift ≤ 0.12°, exp 008). Majorana phases and m_lightest are 0 in the data running.
- No seesaw thresholds or RH-neutrino decoupling in the running (Dueck-Rodejohann show this matters). SM only, no MSSM. The 10_H is effectively PQ-restricted (no 10_H^* coupling).
- The quark-sector prior windows are centred on the data. The `so10_120` and SU(5) variants have generators and counts but no inference.
- No `src/` bugs found. Note: `src.rg.observables` takes the neutrino ordering from `state["ordering"]` rather than from the spectrum, so callers must set it (done here).

## Rerun / resume
```
cd experiments/20261118-011-f3-gut-npe ; PY=../../.venv/Scripts/python
$PY data_gut.py ; $PY check_gen.py
$PY make_dataset.py broad so10_I 1000000 11 ; $PY make_dataset.py broad so10_II 1000000 11
for v in su5_min su5_gj so10_mixed; do $PY make_dataset.py broad $v 100000 21; done
$PY fit_ms.py so10_I ql 32 7 ; $PY fit_ms.py so10_II ql 32 7
bash run_stage2.sh            # broad NPEs (11 min each), predictive flows, DE-MC ql + all
$PY analyze_npe.py broad_so10_I 300 ; $PY analyze_npe.py broad_so10_II 300
$PY analyze_local.py so10_I ; $PY analyze_local.py so10_II ; $PY mi_flows.py so10_II
$PY rg_validate.py so10_I mcmc_so10_I_ql.npz 200 ; $PY btau_ridge.py ; $PY su5_check.py
```
To resume: train the NPEs for more epochs (raise the minutes argument, ~11 min per epoch on 2 threads under load), train `pred_*_so10_I`, and run `fit_ms.py so10_I all 64 9 fromql 4` on idle cores. The most useful next step is a sequential / truncated NPE round around the full-fit modes (a `make_dataset.py local` KDE-proposal mode with log p~ for importance reweighting is implemented but was not run).

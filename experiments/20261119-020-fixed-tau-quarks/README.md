# 020: quark sector at the lepton-fixed τ (S4 τ-only neutrino lead, 2O island)

## Hypothesis

In a common-τ modular model, τ is fixed by the lepton fit. If the S4 τ-only neutrino lead (exp 017) is the
lepton half of a unified model, some S4 quark assignment of the exp 017 catalog should fit CKM + quark mass
ratios with τ restricted to the lepton-allowed region and only coupling ratios free. This is the first test in
the program where the lepton sector predicts something on the quark side (two real numbers, τ).

## Method

Scripts: `lepton_region.py` (+ `region_json.py`), `quark_search.py`, `analyze_null.py`, `twoO_region.py`,
core `ftcore.py`. `modforms.py`, `models.py`, `search.py`, `lead_scan.py` are verbatim copies from exp 017;
`fit_ours.py`, `model_2o.py`, `modforms_2o.py` verbatim from exp 016. Logs: `run.log`.

### 1. Lepton τ region (S4 lead)

- Model as exp 017 `lead_scan.py`: S4, L ~ 3', type-I seesaw, constant M_D, M_N from the single weight-2
  form, e^c rows (4·1', 6·1 [2 couplings], 6·1'), NO. χ² identical to exp 017 (e log mass ratios 1% floor,
  Δm²21/|Δm²3ℓ|, PMNS 4 through `src.evaluate` NuFIT tables). N = 7.
- Global search over the fundamental domain (|Re τ| ≤ 1/2, |τ| ≥ 1, Im τ ≤ 4): 200k prior samples → 80
  distinct-τ starts + the exp 017 optimum and its gCP image, Nelder-Mead with τ free. Only one basin within
  Δχ² 10: the lead. Then a grid flood fill (step 0.0005) around it with the 3 charged-lepton couplings
  profiled at each τ (998 grid points evaluated).
- The second-best e-set variant (e-set 20, χ² 6.68 in exp 017) was started and aborted for time; not reported.

### 2. Quark fits at fixed τ (`quark_search.py real`)

- Observables N = 8: CKM s12, s13, s23, δ (PDG, asymmetric Gaussian; the CKM part was cross-checked against
  the `src.evaluate` Scorer: agrees to < 0.1 in χ² at the best points, column `chi2_ckm_src_evaluate`) and
  ln(m_u/m_c), ln(m_c/m_t), ln(m_d/m_s), ln(m_s/m_b) at M_Z (SM, `src.rg.derived_yukawas_mz`, as exp 017).
  Errors: PDG relative errors in quadrature with a **2% floor on m_u/m_c, m_d/m_s, m_s/m_b** and exp 017's 1%
  floor on m_c/m_t (actual: 3.3%, 2%, 2%, 1%). Scored as exp 017 did, outside `src.evaluate` (the new
  same-scale mass-ratio/floor features were not in `src` when this ran).
- Couplings only (2 log ratios per sector + ≤ 1 extra real coupling per sector, gCP); τ only on grid points of
  the Δχ²_lep ≤ 6.18 region (514 points). Modular forms precomputed at each τ; batched projected
  Levenberg-Marquardt (`ftcore.lm_batch`). Residuals reproduce exp 017's `exact_chi2_fn` exactly (checked).
- Stage A: every assignment, 1500 prior samples → 12 LM starts × 40 iterations. S4: all 968 at 5 region points
  (best + farthest-point spread). A4 control: all 12,100 at the best lepton τ. No emulator was needed: the
  exact cheap fit covered the whole catalog.
- Stage B: top 40 per group, 8000 samples → 64 starts × 120 iterations at the 5 points. Stage C: top 5 per group
  at 30 region points (heavy). Cheap vs heavy fits differ by up to ~15% in χ² (checked on 12 S4 assignments),
  so stage A rankings are approximate; this cannot change the conclusion (best χ² > 2000).

### 3. Random-τ comparison (`quark_search.py null`, `analyze_null.py`)

Nothing fits (and nothing can pass compression, below), so the reduced version was run: 100 random τ uniform
in area over the fundamental domain (Im τ ≤ 4), ≥ 0.05 from the lepton region and its gCP image, seed 2020.
At each τ (single point) the same reduced S4 search: all 968 assignments with 800 samples → 6 LM starts × 30
iterations, then the top 10 heavy. The same reduced search was run at the lepton best τ for a like-for-like
reference. Statistic: fraction of random τ whose best quark χ² ≤ the lepton τ's. Clopper-Pearson 95%.

### 4. 2O lepton island (`twoO_region.py`, exp 016 conventions: GUT scale, MSSM tanβ 5, 2% mass floor)

Lepton-only region (gE2/gE1, gE3/gE1 profiled; grid step 0.001 flood fill from exp 016's lepton optimum and its
gCP image), then the 2O model's own quark sector (6 coupling ratios, its single published assignment) fitted
at every Δχ²_lep ≤ 6.18 grid point. N_q = 8 (4 GUT mass ratios + CKM 4 via `src.evaluate` after exp 016's
running shift).

## Results

### Lepton τ region (S4 lead, `lepton_region.json`)

| | value |
|---|---|
| lepton best fit | χ² 5.84 (N 7, k_cont 5, p 0.054) at τ = −0.1797 + 1.0980i |
| Δχ² ≤ 2.30 | one island, Re τ ∈ [−0.1865, −0.1725], Im τ ∈ [1.091, 1.105], area 4.8e-5 (190 grid points) |
| Δχ² ≤ 6.18 | one island, Re τ ∈ [−0.1915, −0.1675], Im τ ∈ [1.086, 1.1095], area 1.3e-4 (514 points) |
| gCP image τ → −τ* | 0.1797 + 1.0980i, χ² 15.92 (Δχ² 10.1; δ_ℓ → −δ_ℓ): outside both contours |
| other basins | none within Δχ² 10 anywhere in the fundamental domain (Im τ ≤ 4) |

τ is fixed by leptons to about ±0.01 in each component.

### Quark sector at the lepton τ (`results_fixed_tau.csv`, `real_summary.json`)

| group | assignments fitted exactly | best χ² (N 8) | k_cont | bits | k | p | assignment |
|---|---|---|---|---|---|---|---|
| S4 | 968 / 968 (stage A at 5 τ points; 40 in B, 5 in C) | **3167.5** | 6 | 10 | 16 | 0 | Q~3', u[4·1, 6·1', 8·1'*], d[4·1, 4·1', 8·1*], τ = −0.1805 + 1.1005i |
| A4 (control) | 12,100 / 12,100 (stage A at the best τ; 40 in B, 5 in C) | **2178.1** | 6 | 14 | 20 | 0 | Q~3, u[2·1'', 4·1', 8·1''*], d[4·1, 4·1'', 6·1*], τ = −0.1915 + 1.0860i |

Pulls at the best points:

| | ln m_u/m_c | ln m_c/m_t | ln m_d/m_s | ln m_s/m_b | s12 | s13 | s23 | δ_q |
|---|---|---|---|---|---|---|---|---|
| S4 best | −6.0 | 0.0 | +0.2 | 0.0 | −3.0 | +15.7 | −28.1 | −46.1 |
| A4 best | +0.1 | 0.0 | +0.1 | +0.7 | +0.2 | −45.0 | +0.1 | +12.5 |

The top ~10 S4 assignments all sit at χ² ≈ 3167-3168 with the same CKM pulls: at this τ the S4 quark textures
cannot generate a CKM phase near 66° (δ_q pull −46) or the s23/s13 pattern. Masses always fit. The best quark
fits sit at the edge of the Δχ²_lep ≤ 6.18 region, not inside the 2.30 one.

Counting: N = 8, k_cont = 4-6, assignment bits 10 (S4, 968) or 14 (A4, 12,100), τ not counted (fixed by
leptons). k = 14-20 > N − 2 = 6 for every assignment: criterion 2 fails before any fit. Even with zero bits
(an assignment fixed in advance) k_cont 4-6 leaves 2-4 dof, and the best χ² has p = 0.

### Random-τ comparison (`null_random_tau.csv`, `null_summary.json`)

| | value |
|---|---|
| lepton τ, reduced S4 search | best χ² 3167.5 (same as the full search) |
| 100 random τ: best χ² quantiles | min 196, 10% 1108, median 2088, 90% 3116, max 3171 |
| fraction of random τ doing as well or better | **98/100 = 0.98, CP95 [0.930, 0.998]** |
| restricted to random Im τ ≤ 2 | 30/32 = 0.94, CP95 [0.79, 0.99] |
| rank of the lepton τ | 99th of 101 (only 2 random τ are worse) |

The lepton-selected τ is not special for quarks; it is among the worst points in the fundamental domain for
the S4 quark catalog. Best random point: τ = 0.314 + 1.806i, χ² 196 (S4 Q~3', u[4·1, 6·1*, 6·1'],
d[4·1, 6·1', 8·1'*]); that is still p ≈ 0, but lower than exp 017's best S4 quark fit with free τ (2163), so exp
017's S4 quark optimum was search-limited, as its README said.

### 2O island (`twoO_region.json`)

| | value |
|---|---|
| 2O lepton-only best | χ² 14.45 (N 7, k 4, p 0.0023) at τ ≈ −0.191 + 1.087i (grid min 14.454 used as reference) |
| Δχ² ≤ 2.30 | Re τ ∈ [−0.195, −0.187], Im τ ∈ [1.082, 1.092] (28 points, step 0.001) |
| Δχ² ≤ 6.18 | Re τ ∈ [−0.198, −0.184], Im τ ∈ [1.079, 1.095] (79 points) |
| gCP image | +0.191 + 1.087i, χ² 24.2: outside |
| 2O quarks at fixed τ in the 6.18 region | best χ² 71.5 (N 8, k_cont 6, dof 2, p 3e-16) at τ = −0.198 + 1.079i (region edge); 88.9 inside the 2.30 region; median over the region 121 |
| pulls at that point | m_s/m_b +6.7, s23^q −4.4, m_u/m_c +2.3, s13^q −1.0, δ_q +1.2, rest < 0.1 |

The 2O lepton sector already fails criterion 3 on its own (p 0.0023, exp 016), and its own quark sector at the
lepton τ fails with p ~ 1e-16: the same m_s/m_b vs θ23^q tension exp 016 found.

## Parameter count

Fixed-τ quark claim: N = 8; k = k_cont (4-6 coupling ratios) + ceil(log2 #assignments) (S4 10, A4 14) + τ
(0, fixed by leptons; profiling over the 6.18 region gives a small extra freedom, not counted). Lepton side
(not part of this claim): 5 continuous + 8 bits (e-set) + 1 ordering bit for N = 7.

## Verdict

Negative. With τ fixed by the S4 τ-only lepton lead (τ = −0.180 + 1.098i ± 0.01), no S4 quark assignment
(968, all fitted) fits: best χ² 3167 for N 8 (CKM δ −46σ, s23 −28σ, s13 +16σ). The A4 control (12,100, all
fitted) is no better (2178). Compression fails by count (k 14-20 > 6). The lepton τ is worse for quarks than
98% of random τ (CP95 [0.93, 0.998]), so the leptons select a τ that is anti-correlated, if anything, with a good
quark sector in this catalog. The 2O island gives the same answer with the 2O model's own quarks (χ² 71.5,
p 3e-16). No candidate. The common-τ S4 lead cannot be extended to quarks within the exp 017 catalog
(holomorphic forms up to weight 8, gCP, one τ, no running).

Caveats: no running of the model (a GUT/MSSM treatment shifts quark ratios by tens of percent but not
δ_q/s23 by tens of σ); catalog limited to exp 017's rule (singlet RH fields, ≤ 1 extra coupling per sector);
stage A fits are approximate (≤ ~15% χ²); the null uses a single τ per draw and the reduced pipeline; lepton
region for e-set 20 not computed.

## src gaps

- `src.evaluate` still cannot score same-scale in-sector quark mass ratios or apply a mass floor (no new
  features were present at run time), so the mass terms are scored here with exp 017's code (2% floor added).
- No `src` support for a lepton-fixed τ region as a reusable constraint; done here in experiment code.

## Reproduce (from this folder, ≤ 5 cores)

```
FT_WORKERS=5 ../../.venv/Scripts/python lepton_region.py      # ~20 min; e-set 19 (FT_ESETS) -> artifacts/fixedtau/
../../.venv/Scripts/python region_json.py                       # lepton_region.json
FT_WORKERS=5 ../../.venv/Scripts/python quark_search.py real    # ~17 min -> results_fixed_tau.csv, real_summary.json
FT_WORKERS=4 FT_NRAND=100 ../../.venv/Scripts/python quark_search.py null   # ~55 min -> null_random_tau.csv
../../.venv/Scripts/python analyze_null.py                      # null_summary.json
FT_WORKERS=1 ../../.venv/Scripts/python twoO_region.py          # ~85 min on 1 core -> twoO_region.json
```
Seeds fixed in the scripts. The original `lepton_region.py` run also started e-set 20 and was stopped; the json
was then written by `region_json.py` from the saved e-set 19 grid.

## Storage note (overseer)
The full `results_fixed_tau.csv` (13,068 rows, 1.5 MB) is kept locally in the gitignored `artifacts/fixedtau/`; rerunning `quark_search.py real` regenerates it. `results_fixed_tau_top.csv` holds the best 200 rows per group.

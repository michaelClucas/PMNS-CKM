# 017: modular flavour symmetry with one common modulus τ (Γ3 ≅ A4, Γ4 ≅ S4, Γ5 ≅ A5)

## Hypothesis

A single complex modulus τ, shared by quarks, charged leptons and neutrinos, with the Yukawa
couplings and the Weinberg operator built from level-N modular forms, could fix CKM and PMNS
together with few real couplings (generalized CP makes them real). If some assignment of
representations and weights fits 8 mixing observables + mass ratios + Δm² ratio with
k ≤ N − 2 (bits included), it is a candidate CKM-PMNS link through τ.

## Method

### 1. Modular forms (`modforms.py`, tests `test_modforms.py`)

- Weight 2: M_2(Γ(N)) spanned by log-derivatives D log g_a of Siegel functions,
  a ∈ (1/N)Z², from the product formula (Kubert-Lang). ρ(T) exact from the q^(1/N) phases;
  ρ(S) exact from g_a(γτ) ∝ g_{aγ}(τ) (a → (a2, −a1)), solved in coefficient space.
- Weight 4, 6, 8: products of weight-2 forms, independent subset by pivoted QR on q-coefficients;
  ρ(S) from the weight-2 ρ(S) applied factorwise (exact). q-series to q^20 in q^(1/N).
- Field irreps: explicit symmetric matrices (A4 Feruglio basis; S4 Penedo-Petcov basis up to a
  global sign; A5 Novichkov et al. basis for 3, 3'); characters of S4 2 and A5 4, 5 from tensor
  products. Couplings = equivariant maps (null space of the S, T conditions), real form chosen
  (real q-coefficients ⇔ real couplings under gCP).
- Verification (121 tests pass, `python -m pytest -q experiments/20261118-017-modular-search`):
  - dim M_w = w+1, 2w+1, 5w+1 for N = 3, 4, 5, w = 2..8;
  - Y(−1/τ) = τ^w ρ(S) Y(τ) and Y(τ+1) = ρ(T) Y(τ) at random τ to ≤ 1e-9 (relative);
  - ρ(S)² = (ρ(S)ρ(T))³ = ρ(T)^N = 1 for every weight space and irrep; |G| = 12, 24, 60; character orthogonality;
  - irrep content per weight equals the published tables: A4 w2 3; w4 1+1'+3; w6 1+3+3; w8 1+1'+1''+3+3
    (Feruglio arXiv:1706.08749); S4 w2 2+3'; w4 1+2+3+3'; w6 1+1'+2+3+3'+3'; w8 1+2+2+3+3+3'+3'
    (Penedo-Petcov arXiv:1806.11040, Novichkov et al. arXiv:1811.04933); A5 w2 3+3'+5; w4 1+3+3'+4+5+5;
    w6 1+3+3+3'+3'+4+4+5+5; w8 1+3+3+3'+3'+4+4+5+5+5+5 (Novichkov et al. arXiv:1812.02158);
  - the A4 weight-2 triplet equals Feruglio's η-quotient (E2) formula to 1e-10 at test points, and its
    q-expansion is Y1 = 1+12q+36q²+12q³, Y2 = −6q^{1/3}(1+7q+8q²), Y3 = −18q^{2/3}(1+2q+5q²);
  - every Yukawa-row and Weinberg coupling structure transforms correctly under S and T at random τ,
    has real q-coefficients, and the Weinberg ones are symmetric; A4 counts match the literature
    (weight-2 Weinberg: one coupling, Feruglio's model; weight 4: three).
  - Not checked: paper equation numbers and the S4/A5 q-expansion normalisations against the PDFs
    (no PDF access in this run). Exp 016 can cross-check; normalisations are absorbed by couplings.

### 2. Catalog (`models.py`, rule fixed before any fit)

- Q, L: any triplet of the group (A4: 3; S4: 3, 3'; A5: 3, 3'), independently. Each conjugate RH field
  f^c_i is a singlet with Yukawa form weight w_i ∈ {2,4,6,8}; a row type (w, r) is allowed if it has
  1 or 2 invariant couplings. A sector (u, d or e) = 3 distinct row types with at most one 2-coupling row.
- Neutrinos: Weinberg operator, weight 2, 4 or 6 (1, 3, 3 couplings). Seesaw and T' not done (time).
- Couplings real (gCP), so τ is the only CP source. Continuous parameters: τ (2) + per charged sector
  2 coupling ratios (overall scale unobservable) + ≤ 1 extra ratio + (n_ν − 1) neutrino ratios.

### 3. Scale, observables, counting

- Scale: M_Z, SM, no running of the model. Model Yukawa ratios are compared with M_Z Yukawa ratios from
  `src.rg.derived_yukawas_mz` (read through the Universe's mass records); CKM and NuFIT data as given.
- Observables (N_ext = 15): CKM s12, s13, s23, δ and PMNS s12², s13², s23², δ (8, scored by `src.evaluate`,
  explicit `use=`, NuFIT tables); 6 charged-fermion log mass ratios (m_u/m_c, m_c/m_t, m_d/m_s, m_s/m_b,
  m_e/m_μ, m_μ/m_τ; error = PDG relative errors in quadrature with a 1% floor); Δm²21/|Δm²3ℓ|. The last 7
  are scored outside `src.evaluate` (gap, below).
- k_cont = 2 (τ) + 4 quark ratios + ≤ 2 quark extras + 2 lepton ratios + ≤ 1 extra + 0-2 ν ratios = 8-13.
- Trials (whole catalog, all groups): A4 3,993,000 (12,100 quark × 330 lepton), S4 127,776 (968 × 132),
  A5 96 (8 × 12); total 4,120,872 → 22 bits, + 1 ordering bit. Even a group fixed in advance costs
  22 / 17 / 7 bits. So k ≥ 8 + 23 = 31 against N − 2 = 13: **every catalog entry fails compression by
  counting, before any fit** (`catalog.json`).
- `src.evaluate` core (N = 8 mixing, masses declared as inputs, k_cont − 7 + 22 bits + ordering bit) is also
  reported in `results.csv` (`eval_*` columns); it fails compression the same way.

### 4. Search pipeline (`search.py`, `score.py`)

1. `gen`: synthetic samples, uniform priors (τ uniform in the fundamental domain with Im τ ≤ 4, ln coupling
   ratios U[−14, 4], extra/ν ratios U[−6, 6]): 2.42M (A4 quarks; 600k used for training), 0.3M quark and
   0.15M lepton samples for the other groups.
2. `train`: one MLP per group and sector (4×384 SiLU, 8 epochs, torch 2 threads, 2 jobs at a time):
   (τ features, padded params, assignment multi-hot) → quark: 4 log mass ratios, log s_ij, cos/sin δ;
   lepton: 2 log mass ratios + per ordering s²_ij, cos/sin δ, log r. Test RMSE per output in `run.log`
   (e.g. A4 lepton s12²/s13²/s23²: 0.20/0.15/0.21; quark log s_ij: 1.0-1.5). Poor: mixing angles over a
   broad prior are hard to emulate.
3. `rank`: per assignment, 4 Adam starts × 150 steps on the emulator χ², sector by sector.
4. `exact`: exact multi-start fits (4000 prior samples, 3 Nelder-Mead starts + the emulator optimum; mixing
   χ² through the `src.evaluate` Scorer) on the emulator top 15 and 15 random assignments per group, sector
   and ordering (all assignments for A5): 212 fits (`exact_sector_fits.json`).
5. `joint`: for the 6 best quark and 6 best lepton assignments per group/ordering, τ profiles on a 32-point
   Sobol grid (sector couplings re-optimised at each τ); the 4 best quark × lepton combinations per
   group/ordering refined jointly with one τ and scored (`score.py`, `results.csv`, 24 rows). The first
   scoring pass had a bug (each sector's own best τ, appended to the grid, was summed as if shared); fixed
   and rerun. `results.csv` is from the rerun; both passes are in `run.log`.
6. First training launch used 5 workers × 2 threads (oversubscribed); killed after ~1 min and restarted
   with 2 × 2 (logged).

## Results

### Emulator vs exact (`emulator_vs_exact.csv`)

| group, sector | Spearman (emu vs exact), random subset | median exact χ², emu top 15 / random 15 | exact best-10 from emu top |
|---|---|---|---|
| A4 quarks | 0.15 | 3186 / 10276 | 6/10 |
| A4 leptons NO / IO | 0.51 / 0.24 | 535 / 10804, 11062 / 11147 | 6/10, 5/10 |
| S4 quarks | 0.77 | 3745 / 7170 | 7/10 (best exact fit, 2163, was a random pick) |
| S4 leptons NO / IO | 0.61 / 0.75 | 5.5 / 804, 97 / 316 | 8/10, 7/10 |
| A5 (every assignment fitted) | 0.98 q; 0.90 / 0.87 l (all) | | |

The emulator enriches the top of the list (median exact χ² of its top 15 is 1-150× lower than random) but its
ranking is weak (Spearman 0.15-0.77 on random subsets) and it misses good assignments: the second-best A4 quark
fit (465) and the best S4 quark fit (2163) came from the random subsets. It works as a coarse filter, not as a
ranker. With 12,100 A4 quark assignments and 30 exact fits, the quark optimum is not established; the quark
numbers below are upper bounds on min χ².

### Sector fits (exact, best found)

| | quarks (8 obs; 6-8 couplings + τ) | leptons NO (7 obs) | leptons IO |
|---|---|---|---|
| A4 | 254.8 (Q~3, u[2·1, 2·1', 8·1''*], d[2·1'', 4·1'', 8·1'*]) | 0.0 | 1.6 |
| S4 | 2163 | 0.1 | 0.4 |
| A5 | 2501 | 273 | 16.0 |

(* = row with two couplings.) Leptons with their own τ fit easily (5 parameters for 7 observables, 2 of which
the e couplings absorb). Quarks do not fit in this catalog: best quark-only χ² 255 for 8 observables.

### Joint common-τ fits (`results.csv`, top rows)

| group | ord | χ²_ext | N | k_cont | bits | k | p (dof 2) | `src.evaluate` p_global (N 8) | flags | τ |
|---|---|---|---|---|---|---|---|---|---|---|
| A4 | IO | 446.5 | 15 | 13 | 23 | 36 | 1e-97 | 2e-94 | compression ✗, fit ✗ | 0.180 + 2.431i |
| A4 | NO | 468.4 | 15 | 13 | 23 | 36 | 2e-102 | 6e-10 | ✗, ✗ | −0.061 + 1.548i |
| S4 | IO | 2139.8 | 15 | 13 | 23 | 36 | 0 | 0 | ✗, ✗ | −0.029 + 2.214i |
| A5 | NO/IO | ≥ 4043 | 15 | 12-13 | 23 | 35-36 | 0 | 0 | ✗, ✗ | |

A4 IO: quarks u[2·1, 2·1', 8·1''*] d[2·1'', 4·1'', 8·1'*], leptons L~3 e[4·1, 4·1', 6·1*], Weinberg weight 4.
Nothing has `compression_ok` or `fit_ok`. The joint χ² is quark-dominated. No power check or null was run:
the protocol runs them only for results passing criteria 2-3.

### Exp 016 lead: τ-only neutrino sector (`lead_scan.py`, `lead_tauonly_nu.json`, `lead.log`)

Type-I seesaw with N^c in the same triplet as L, constant Dirac matrix (weight-0 invariant), M_N from the single
weight-2 invariant, so m_ν is fixed by τ alone. Scanned against every charged-lepton set of the catalog
(158 e-sets × 2 orderings, exact fits). Best: **S4, L ~ 3', e^c rows (4·1', 6·1*, 6·1'), NO,
τ = −0.180 + 1.098i: χ² 5.84 for 7 lepton observables (k_cont 5 incl. τ, dof 2, p 0.054)**. s12², s13², the
Δm² ratio and the e masses are fitted exactly; pulls s23² +1.52 (0.503), δ +1.26 (238°). Only 2 of 316 fits
have χ² < 10 (both S4, NO; the second has p 0.035). The τ is on the narrow island exp 016 reported
(Re τ ≈ ±0.2, Im τ ≈ 1.1). It is lepton-only and says nothing about the CKM; with 8 bits for the e-set choice
plus the ordering bit, k = 14 > N − 2 = 5. Very likely the known S4 gCP seesaw family (Novichkov, Penedo,
Petcov, Titov arXiv:1811.04933); exp 016 should confirm.

## Parameter count (summary)

k = 2 (τ) + 6 charged-fermion coupling ratios + 0-3 extra couplings + 0-2 ν ratios (8-13 continuous)
+ 22 bits (assignment among 4.1M) + 1 bit (ordering) = 31-36, against N = 15.

## Verdict

Fails compression by counting (k ≥ 31 vs N − 2 = 13) for every model in the catalog, and fails fit quality:
the best common-τ model found (A4, IO) has χ² 446.5 for 15 observables, quark-dominated (best quark-only
fit 255). No candidate. Caveats: (1) the quark optimum is search-limited (weak emulator ranking; 30 of 12,100
A4 and 30 of 968 S4 quark assignments fitted exactly), so a common-τ fit in this catalog was not found, not
excluded; (2) no running (a GUT/MSSM treatment shifts mass ratios by tens of percent); (3) 1% mass-ratio floor
(exp 016 used 2% for light quarks; that would lower the mass terms, not the CKM terms that dominate);
(4) no seesaw in the main catalog, no T', no multiple moduli, no Kähler corrections. Likely-known: A4 rows with
Weinberg weight 2/4 are the Feruglio / Okada-Tanimoto (arXiv:1905.13421, common τ, quark-sector failure)
structures; the S4 lepton lead is the NPPT family.

## src gaps

- `src.evaluate` cannot score fermion mass ratios at a common scale (ratio records share flavour atoms, so at
  most m_u/m_d and m_c/m_s can be used together) or the Δm² ratio. Both are scored outside it here, with
  `src.rg` M_Z ratios and `Universe.load_pmns` splittings; the `eval_*` columns cover only the 8 mixing
  observables.
- `src.rg.derived_yukawas_mz` has no uncertainty output; ratio errors are propagated by hand.

## Reproduce

```
.venv/Scripts/python -m pytest -q experiments/20261118-017-modular-search     # 121 tests
cd experiments/20261118-017-modular-search
MODSEARCH_WORKERS=5 ../../.venv/Scripts/python search.py gen
MODSEARCH_WORKERS=2 ../../.venv/Scripts/python search.py train
MODSEARCH_WORKERS=2 ../../.venv/Scripts/python search.py rank
MODSEARCH_WORKERS=5 ../../.venv/Scripts/python search.py exact
MODSEARCH_WORKERS=3 ../../.venv/Scripts/python search.py joint
MODSEARCH_WORKERS=5 ../../.venv/Scripts/python score.py
MODSEARCH_WORKERS=2 ../../.venv/Scripts/python lead_scan.py
```
Seeds are fixed in the scripts; samples, emulators and τ profiles go to `artifacts/modular/` (gitignored).
Wall time about 2 h on ≤ 5 cores (training dominates).

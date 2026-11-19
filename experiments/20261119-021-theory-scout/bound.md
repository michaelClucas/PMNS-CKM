# How much information about PMNS does the CKM (+ masses) carry, within the families searched so far?

Script: `bound.py` → `bound.json`, `bound.log`. Single process, about 1 min. It reads the committed outputs of exps 007, 009, 010, 014 (+ the gitignored PT samples in `artifacts/so10/`), 016, 017 and 018. It runs no new fits, except for a fresh 200k-sample SO(10) prior predictive drawn with exp 014's box and seed.

## 1. What has to be explained: the PMNS information budget

Take the Haar measure on U(3) as the "no theory" reference. In it s12², s23² and δ are uniform and p(s13²) = 2(1 − s13²). The NuFIT 6.1 measurements then hold (Gaussian approximation, KL from Haar to the measurement):

| | s12² | s13² | s23² | δ | total (4) |
|---|---|---|---|---|---|
| NO | 5.2 | 7.8 | 4.0 | 2.0 | **18.9 bits** |
| IO | 5.2 | 7.8 | 4.1 | 1.7 | 18.8 bits |

A relation that predicts all four PMNS observables from quark data, with no lepton-side freedom, has to supply about 19 bits. Most of these bits are not CKM-specific. "Small θ13 plus a nearly symmetric pattern" is supplied by any TBM/BM/GR anchor plus an O(λ) rotation, and any hierarchical V does as well as the real one (section 2). The question below is how many bits can be attributed to the actual CKM.

## 2. CKM specificity from the hierarchical nulls (F1, F2, T1, F2-LO, symbolic)

For each search I compare the real data with fakes that keep the real PMNS and swap the CKM (and, in exp 010, the mass ratios) for random hierarchical ones. All fakes go through the same pipeline: null (b) in exps 007/009/010, N1/N2 in exp 018. Two numbers per cell:

- **B_ckm** = log2 [ e^{−χ²_real/2} / mean_fakes e^{−χ²_fake/2} ], with χ² the best profile χ² of the search. This is the log Bayes factor of "this family with the real CKM" against "this family with a random hierarchical CKM". It is optimistic for the family, because it profiles over every discrete choice and phase. Positive means the real CKM helps.
- **S_rank** = −log2 P(fake ≤ real): how far the real CKM is singled out, by rank only.

| search (exp) | cells | B_ckm range (bits) | best single-cell S_rank (bits) | cells × trials penalty |
|---|---|---|---|---|
| F1 shared rotation (007), NO/IO × cont/struct | 4 | −0.7 to **+1.2** | 2.7 (rate 0.157; CI-limited max 3.1) | log2 4 = 2 bits |
| F2 Clebsch (009), 2 scales × 2 ord × fit/struct × tiers A/ABC/ALL | 24 | **−14.8 to +0.2** | 3.9 (IO struct ALL, rate 0.067) | log2 24 = 4.6 bits |
| F2-LO protocol rerun (018), N2 NO | 1 | −3.1 | 0.6 | |
| T1 protocol rerun (018), N1/N2 | 3 | 0.0 to +0.25 | 1.8 | |
| symbolic single-observable (010), null b, C ≤ 1..5 | 80 | −0.0 to +0.09 | 8.2 (M_Z s23² NO C ≤ 4: 1/300 fakes) | log2 80 = 6.3 bits |

Reading:

- **Largest likelihood gain from using the real CKM instead of a random hierarchical one: +1.2 bits** (F1, NO, structured phases). That is a likelihood ratio of 2.3, against the about 11-17 bits of discrete choices that search used.
- In F2 the real CKM does worse than typical fakes in every LO tier. For tier A, B_ckm is −12 to −15 bits. A random λ ∈ [0.10, 0.35] lands a lattice point θ12e = c·λ on the ≈ 0.21 rad that θ13 needs more easily than the real λ = 0.225 does.
- The one striking single cell is exp 010's s23² at C ≤ 4, with 1/300 fakes (CI upper 1.6%). It sits next to C ≤ 3 (rate 5%) and C ≤ 5 (6.7%), one cell of 160. It is also a mass-ratio formula, not a CKM one. After the 6-7 bit cell penalty it carries ≲ 2 bits.
- Pooled statement: **within F1, F2, T1 and the exp 010 symbolic grammar, the real CKM + mass ratios carry at most about 1-2 bits of PMNS information beyond what a random hierarchical CKM carries. That is ≲ 10% of the 19-bit budget, and it is consistent with 0 after cell selection.** "At most" here is the profile-likelihood bound B_ckm. S_rank before the trials penalty is ≤ 4 bits in every multi-observable search.

Not covered: the null prior is hierarchical (λ ∈ [0.10, 0.35] or [0.05, 0.5], A ∈ [0.5, 1.2] or [0.3, 2]). Information carried by the hierarchy itself, which all fakes share, is not measured here. Exps 007/009 null (a) (Haar PMNS) puts that at a few bits: rates of 1-10% correspond to 3-7 bits. That is the "PMNS looks like a symmetric pattern plus a small correction" fact, not a quark-lepton link.

## 3. Held-out PMNS predictive gain in SO(10) (exp 014 PT samples)

The posterior is conditioned on the 12 quark + charged-lepton observables only (75k cold samples, 5 ladders). The prior predictive is a fresh 200k-sample draw from exp 014's box, seed 5, with NO-classified samples only. **G_pred** = log2 [ p(x_obs | quark data) / p(x_obs | prior) ]. Here p is the predictive density convolved with the measurement error σ, and also with 3σ as a robustness check against tail-sample noise. **KL_shift** = KL(posterior predictive ‖ prior predictive), the amount the quark data move the PMNS prediction in any direction.

| model (exp 014) | s12² | s13² | s23² | δ | lg Δm² ratio | joint 4 (σ×1 / σ×3) | joint min χ²₄ post / prior |
|---|---|---|---|---|---|---|---|
| MN, our errors | −0.6 | +4.2 | +3.1 | −1.1 | +3.2 | **−76 / −6.6** | 297 / 190 |
| MN, DR-size errors | −1.6 | −13.3 | −3.4 | −1.2 | −0.2 | −857 / −93 | 1481 / 289 |
| FN (real 10+126+120) | −1.3 | −301 | −4.4 | +0.8 | −0.9 | −1132 / −127 | 1590 / 21 |
| KL_shift (MN ours; DR; FN), bits | 1.1; 0.8; 1.6 | 2.5; 2.8; 1.7 | 1.8; 3.3; 2.1 | 0.2; 0.5; 0.7 | 3.2; 2.3; 0.6 | | |

- The quark data change the PMNS prediction by 1-3 bits per observable (KL_shift). The model does transmit quark information to the lepton side.
- That information points away from the data. Jointly, G_pred is −7 to −127 bits even with 3σ smoothing. For MN with our errors the marginal s13² and s23² gains are positive (+3 to +4 bits). The cause is the prior's extreme small-angle tail: the posterior is a bit less wrong than the prior, one observable at a time, and much more wrong jointly.
- Within the posterior, the CKM + mass observables explain ≤ 12% of PMNS variance. The Gaussian-channel equivalent is I_gauss = −½ log2(1 − R²) ≤ 0.09 bits per observable (0.28 for the Δm² ratio under DR errors). PMNS is controlled by the free phases and s, not by the CKM.
- Caveats: the MN tail weight is uncertain by about 2× (±1 bit; exp 014 ladders). FN is not converged and is prior-dominated. My fresh FN prior predictive has min χ²₅ = 79, against 216 in exp 014's `heldout_FN.json` with the same seed and box. I did not resolve this, and it does not change any sign.

## 4. Shared-modulus families (exps 016, 017): the cost of sharing

When quarks and leptons share τ, the best case is that sharing removes one modulus's Occam factor. That is at most log2(fundamental-domain area / lepton-island area) ≈ log2(3.5 / 0.04²) ≈ **11 bits**. The price is the fit-quality loss Δχ² = χ²_joint − χ²_lepton(own τ) − χ²_quark(own τ). The net gain is 11 − Δχ²/(2 ln 2):

| model | χ²_joint | χ²_lepton (own τ) | χ²_quark (own τ) | Δχ²_sharing | net bits |
|---|---|---|---|---|---|
| 2O (exp 016, our data) | 68.8 | 14.4 | 0 (assumed: k = N in that sector) | 54.4 | **−28** |
| A4 catalog best, IO (exp 017) | 446.5 | 1.6 | 254.8 | 190 | −126 |
| A4 catalog best, NO (exp 017) | 468.4 | 0.0 | 254.8 | 214 | −143 |

Sharing τ is disfavoured by ≥ 28 bits. For it to pay off, Δχ²_sharing would have to stay below about 15 (2 ln 2 × 11).

## 5. Summary statement (for the write-up)

Within every family searched (shared rotation F1, Clebsch-corrected patterns F2, symbolic formulas up to complexity 6, minimal and real SO(10), BSV type-II, 2O and A4/S4/A5 common-τ modular models):

1. The real CKM (+ mass ratios) carries **at most about 1-2 bits** of PMNS information beyond a random hierarchical CKM (profile-likelihood bound, F1 best +1.2 bits). The PMNS data hold about 19 bits relative to Haar.
2. Where quark data do propagate to the lepton sector (SO(10) MN/FN, common τ), they move the PMNS prediction by 1-3 bits per observable, but **away** from the measurement: joint predictive gain −7 to −1100 bits, and the shared-τ net gain is ≤ −28 bits.
3. So no family tried has PMNS information that comes from the CKM. Every good PMNS fit is paid for by lepton-only parameters (phases, θν, τ, couplings), and every forced sharing fails at PDG 2026 light-quark precision.

Not covered by this bound:
- Families outside this list, in particular those with fixed Clebsch textures that also predict mass ratios (proposals P1-P3).
- Information in the hierarchy itself (shared by all fakes).
- RG/threshold effects beyond the SM/MSSM one-loop treatment used.
- IO for the SO(10) held-out test (NO only).
- B_ckm being a profile quantity. A fully marginal Bayes factor over the discrete choices would be lower, and the bound is therefore conservative in the family's favour.

"""How much PMNS information a family's quark-side input carries (exp 021 bound.py, made reusable in exp 024).

All quantities are in bits (log base 2).

    pmns_data_bits(ordering)                  information the PMNS measurements hold vs Haar
    null_info(real_chi2, fake_chi2s)          B_ckm, S_rank and the null rate with a CP interval
    log_ratio_rank(real_T, fake_Ts, unit)     the same for a log-likelihood-ratio statistic T (exp 030)
    predictive_gain(post, prior, x_obs, sigma, periodic=None)   G_pred
    kl_shift(post, prior, n_bins=40)          KL(posterior predictive || prior predictive)
    i_gauss(r2)                               Gaussian-channel bits of an explained variance
    sharing_net_bits(joint, own_a, own_b, occam_bits)           cost of sharing a parameter

Definitions
-----------
pmns_data_bits  -log2[p_Haar(x) * sigma * sqrt(2 pi e)] per observable: KL(N(x, sigma) || Haar
                marginal) in the Gaussian approximation. Haar marginals: s12sq, s23sq uniform on
                [0, 1], s13sq density 2(1 - x), delta uniform on 360 deg (clipped at 0 bits).
                sigma = mean of the NuFIT err_plus / err_minus. A relation that predicts an
                observable must supply these bits.
B_ckm           log2[ L(real) / E_fake L(fake) ], L = exp(-chi2_best / 2): profile log Bayes
                factor of the real CKM against the CKM-side null (hierarchical / N1 fakes run
                through the same search). Optimistic for the family (profile, not marginal).
                > 0: the real CKM fits better than a typical fake.
S_rank          -log2 P(fake chi2 <= real chi2), with hits floored at 0.5: how strongly the real
                CKM is singled out by rank. S_rank_max_ci uses the lower 95% CP limit.
G_pred          log2[ p(x_obs | conditioned) / p(x_obs | prior) ], each p a sample mean of the
                Gaussian likelihood exp(-chi2/2) (predictive density convolved with the
                measurement error, up to a common constant that cancels). Does conditioning on
                quark data move the PMNS prediction toward the truth?
KL_shift        histogram KL of posterior vs prior predictive on n_bins prior-quantile bins
                (+0.5 pseudo-count per bin): how much conditioning moves the prediction at all.
I_gauss         -0.5 log2(1 - R^2), R^2 floored at 0.
sharing_net     occam_bits - (chi2_joint - chi2_own_a - chi2_own_b) / (2 ln 2).

The numbers of exp 021 (bound.json) are reproduced by tests/test_infogain.py.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from . import data as _data

LN2 = math.log(2)

__all__ = ["pmns_data_bits", "null_info", "log_ratio_rank", "predictive_gain", "kl_shift", "i_gauss",
           "sharing_net_bits", "logmeanexp", "cp_interval"]


def logmeanexp(a) -> float:
    """log(mean(exp(a))), stable. Natural log."""
    a = np.asarray(a, float)
    m = np.max(a)
    if not np.isfinite(m):
        return float(m)
    return float(m + math.log(np.mean(np.exp(a - m))))


def cp_interval(hits: int, n: int, cl: float = 0.95) -> tuple:
    """Two-sided Clopper-Pearson interval of a binomial rate (each tail (1 - cl) / 2)."""
    from scipy.stats import beta
    a = (1 - cl) / 2
    lo = beta.ppf(a, hits, n - hits + 1) if hits > 0 else 0.0
    hi = beta.ppf(1 - a, hits + 1, n - hits) if hits < n else 1.0
    return float(lo), float(hi)


def _haar_density(key, x):
    return 2 * (1 - x) if key == "s13sq" else 1.0


def pmns_data_bits(ordering: str, analysis: str = _data.DEFAULT_ANALYSIS) -> dict:
    """{s12sq, s13sq, s23sq, dcp_deg, total_4}: bits the NuFIT measurement holds vs Haar."""
    p = _data.load_pmns(ordering, analysis)
    r = {}
    for key in ("s12sq", "s13sq", "s23sq"):
        s = 0.5 * (p[key]["err_plus"] + p[key]["err_minus"])
        r[key] = -math.log2(_haar_density(key, p[key]["value"]) * s * math.sqrt(2 * math.pi * math.e))
    s = 0.5 * (p["dcp_deg"]["err_plus"] + p["dcp_deg"]["err_minus"])
    r["dcp_deg"] = max(0.0, -math.log2(s / 360 * math.sqrt(2 * math.pi * math.e)))
    r["total_4"] = sum(r.values())
    return r


def null_info(real_chi2: float, fake_chi2s: Iterable[float]) -> dict:
    """B_ckm, S_rank and the null hit rate (fake best chi2 <= real best chi2) of one search.

    real_chi2: best chi2 of the search on real data. fake_chi2s: best chi2 of the same search
    on each fake universe (non-finite entries are dropped). Returns real_chi2, n, hits, rate,
    rate_ci95 (two-sided CP), S_rank, S_rank_max_ci, B_ckm, fake_median.
    """
    fakes = np.asarray([f for f in fake_chi2s if np.isfinite(f)], float)
    n = len(fakes)
    if n == 0:
        raise ValueError("no finite fake chi2 values")
    h = int(np.sum(fakes <= real_chi2))
    lo, hi = cp_interval(h, n)
    B = (-real_chi2 / 2 - logmeanexp(-fakes / 2)) / LN2
    return dict(real_chi2=float(real_chi2), n=n, hits=h, rate=h / n, rate_ci95=[lo, hi],
                S_rank=float(-math.log2(max(h, 0.5) / n)), S_rank_max_ci=float(-math.log2(max(lo, 0.5 / n))),
                B_ckm=float(B), fake_median=float(np.median(fakes)))


def log_ratio_rank(real_T: float, fake_Ts: Iterable[float], unit: str = "nats") -> dict:
    """Rank of a real log-ratio statistic among fakes, with the CP interval and B_ckm (exp 030).

    T is a log likelihood ratio where larger means the real CKM explains the PMNS data better,
    e.g. exp 025's flow statistic T = log q1(PMNS | CKM, m) - log q0(PMNS | m) evaluated with the
    real CKM (real_T) and with each N1 fake CKM (fake_Ts). unit: "nats" or "bits" (both inputs).
    Non-finite fakes are dropped. Returns (bits throughout):
      n, hits = #(fake T >= real T), rate = hits / n, rate_ci95 (two-sided CP),
      S_rank = -log2(max(hits, 0.5) / n), S_rank_max_ci = -log2(max(CP lower, 0.5 / n)),
      B_ckm = log2[ exp(T_real) / mean exp(T_fake) ]   (T in nats inside the exponentials),
      real_T_bits, fake_T_bits_median, fake_T_bits_q05_q95.
    This is null_info with chi2 = -2 T (identical hits, rate, S_rank, B_ckm); it replaces the
    chi2 = -2 T workaround of exp 025 analyze.py."""
    if unit not in ("nats", "bits"):
        raise ValueError("unit must be 'nats' or 'bits'")
    f = LN2 if unit == "bits" else 1.0  # -> nats
    fakes = np.asarray([t for t in fake_Ts if np.isfinite(t)], float) * f
    n = len(fakes)
    if n == 0:
        raise ValueError("no finite fake statistics")
    r = float(real_T) * f
    h = int(np.sum(fakes >= r))
    lo, hi = cp_interval(h, n)
    B = (r - logmeanexp(fakes)) / LN2
    fb = fakes / LN2
    return dict(real_T_bits=r / LN2, n=n, hits=h, rate=h / n, rate_ci95=[lo, hi],
                S_rank=float(-math.log2(max(h, 0.5) / n)), S_rank_max_ci=float(-math.log2(max(lo, 0.5 / n))),
                B_ckm=float(B), fake_T_bits_median=float(np.median(fb)),
                fake_T_bits_q05_q95=[float(np.quantile(fb, 0.05)), float(np.quantile(fb, 0.95))])


def _chi2(Z, x_obs, sigma, periodic):
    Z = np.atleast_2d(np.asarray(Z, float).T).T  # (n,) -> (n, 1)
    d = Z - np.asarray(x_obs, float)
    if periodic is not None:
        for j, per in enumerate(np.atleast_1d(periodic)):
            if per:
                d[:, j] = (d[:, j] + per / 2) % per - per / 2
    return ((d / np.asarray(sigma, float)) ** 2).sum(1)


def predictive_gain(post, prior, x_obs, sigma, periodic: Sequence[float | None] | None = None,
                    sigma_scale: float = 1.0) -> dict:
    """G_pred of conditioned (post) vs unconditioned (prior) predictive samples.

    post, prior: (n, d) or (n,) arrays of predicted observables. x_obs, sigma: (d,) measured
    values and errors. periodic: per-column period (e.g. 2 pi for a phase) or None/0.
    sigma_scale: multiply sigma (exp 021 also reports 3, against tail-sample noise).
    Returns G_pred, logp_post, logp_prior (bits) and min chi2 of each sample set.
    """
    s = np.asarray(sigma, float) * sigma_scale
    cp, cq = _chi2(post, x_obs, s, periodic), _chi2(prior, x_obs, s, periodic)
    lp, lq = logmeanexp(-cp / 2), logmeanexp(-cq / 2)
    return dict(G_pred=float((lp - lq) / LN2), logp_post=float(lp / LN2), logp_prior=float(lq / LN2),
                min_chi2_post=float(cp.min()), min_chi2_prior=float(cq.min()))


def kl_shift(post, prior, n_bins: int = 40) -> float:
    """KL(posterior predictive || prior predictive) of one observable in bits (histogram estimate)."""
    a, b = np.asarray(post, float).ravel(), np.asarray(prior, float).ravel()
    edges = np.quantile(b, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = min(edges[0], a.min()) - 1e-12, max(edges[-1], a.max()) + 1e-12
    P = np.histogram(a, edges)[0] + 0.5
    Q = np.histogram(b, edges)[0] + 0.5
    P, Q = P / P.sum(), Q / Q.sum()
    return float(np.sum(P * np.log2(P / Q)))


def i_gauss(r2: float) -> float:
    """Bits of a Gaussian channel with explained variance r2: -0.5 log2(1 - max(r2, 0))."""
    return float(-0.5 * math.log2(1 - max(float(r2), 0.0)))


def sharing_net_bits(joint_chi2: float, own_a_chi2: float, own_b_chi2: float, occam_bits: float) -> dict:
    """Net bits of sharing a parameter between two sectors (exp 021 section 5).

    occam_bits: the maximum gain from not fitting the shared parameter twice, e.g. log2(prior
    volume / one sector's allowed volume). Returns delta_chi2_sharing and net_bits.
    """
    d = joint_chi2 - own_a_chi2 - own_b_chi2
    return dict(joint=joint_chi2, own_a=own_a_chi2, own_b=own_b_chi2, delta_chi2_sharing=d,
                occam_gain_max_bits=occam_bits, net_bits=occam_bits - d / (2 * LN2))

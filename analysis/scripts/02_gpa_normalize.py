"""Normalize the missing 3.0-3.3 GPA bin on the pooled (n=57) data.

Survey design error: the GPA single-choice question jumps
  "2.7-3.0" --> "3.3-3.7"
so students with true GPA in [3.0, 3.3) had no option and rounded into one of
the neighbours (empirically almost all into [3.3, 3.7) -- the spike in the
reported 3.3-3.7 bin makes that visible).

Three reconstructions are produced, so the reader can gauge sensitivity:

  1. Truncated-normal MLE  -- parametric, smooth, preserves N.
  2. Beta-distribution MLE -- alternative parametric with different tail shape.
  3. Log-linear interpolation -- non-parametric sanity check.

The truncated normal is adopted as the PRIMARY reconstruction because every
interview consistently describes a unimodal, upper-heavy distribution, which
a truncated normal handles cleanly.

The key trick: the "3.3-3.7" bin as reported is actually the merged interval
[3.0, 3.7) -- so we fit the distribution on the bin set
    {[0,2.3), [2.3,2.7), [2.7,3.0), [3.0,3.7), [3.7,4.3]}
and then SPLIT the merged bin using the fitted CDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Bin edges in GPA units. The survey's "Below 2.3" and "Above 3.7" are open
# intervals; we bound them at [0, 4.3] so the trunc-normal has finite support.
GPA_MIN, GPA_MAX = 0.0, 4.3
FULL_BINS = [
    ("Below 2.3", 0.0, 2.3),
    ("2.3-2.7", 2.3, 2.7),
    ("2.7-3.0", 2.7, 3.0),
    ("3.0-3.3", 3.0, 3.3),
    ("3.3-3.7", 3.3, 3.7),
    ("Above 3.7", 3.7, 4.3),
]
# Bins as they appeared on the ACTUAL survey (3.0-3.3 absent -- merged into
# 3.3-3.7):
SURVEY_BINS = [
    ("Below 2.3", 0.0, 2.3),
    ("2.3-2.7", 2.3, 2.7),
    ("2.7-3.0", 2.7, 3.0),
    ("3.0-3.7", 3.0, 3.7),  # the option labelled "3.3-3.7" on the form
    ("Above 3.7", 3.7, 4.3),
]


def neg_log_lik_trunc_normal(params: np.ndarray, counts: np.ndarray, bins) -> float:
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    a, b = (GPA_MIN - mu) / sigma, (GPA_MAX - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    probs = np.array([dist.cdf(hi) - dist.cdf(lo) for _, lo, hi in bins])
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(counts * np.log(probs))


def neg_log_lik_beta(params: np.ndarray, counts: np.ndarray, bins) -> float:
    log_a, log_b = params
    a, b = np.exp(log_a), np.exp(log_b)
    dist = stats.beta(a, b, loc=GPA_MIN, scale=(GPA_MAX - GPA_MIN))
    probs = np.array([dist.cdf(hi) - dist.cdf(lo) for _, lo, hi in bins])
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(counts * np.log(probs))


def fit_trunc_normal(counts: np.ndarray, bins):
    res = optimize.minimize(
        neg_log_lik_trunc_normal,
        x0=np.array([3.3, np.log(0.4)]),
        args=(counts, bins),
        method="Nelder-Mead",
    )
    mu, log_sigma = res.x
    return {"mu": float(mu), "sigma": float(np.exp(log_sigma)), "nll": float(res.fun)}


def fit_beta(counts: np.ndarray, bins):
    res = optimize.minimize(
        neg_log_lik_beta,
        x0=np.array([np.log(6.0), np.log(2.0)]),
        args=(counts, bins),
        method="Nelder-Mead",
    )
    log_a, log_b = res.x
    return {"alpha": float(np.exp(log_a)), "beta": float(np.exp(log_b)), "nll": float(res.fun)}


def trunc_normal_probs(mu: float, sigma: float, bins):
    a, b = (GPA_MIN - mu) / sigma, (GPA_MAX - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    return np.array([dist.cdf(hi) - dist.cdf(lo) for _, lo, hi in bins])


def beta_probs(alpha: float, beta: float, bins):
    dist = stats.beta(alpha, beta, loc=GPA_MIN, scale=(GPA_MAX - GPA_MIN))
    return np.array([dist.cdf(hi) - dist.cdf(lo) for _, lo, hi in bins])


def log_linear_impute(survey_counts: dict) -> dict:
    """Impute 3.0-3.3 by log-linear interpolation on bin midpoints."""
    # Using midpoints 2.85 (for 2.7-3.0) and 3.5 (for 3.3-3.7).
    c_left = max(survey_counts["2.7-3.0"], 0.5)
    c_right = max(survey_counts["3.3-3.7"], 0.5)
    x_left, x_right = 2.85, 3.5
    x_target = 3.15
    log_c = np.log(c_left) + (np.log(c_right) - np.log(c_left)) * (x_target - x_left) / (x_right - x_left)
    raw_est = float(np.exp(log_c))
    # Rescale so that counts in [3.0,3.7) sum to the originally reported 3.3-3.7 count
    # (i.e. we ONLY split the merged bin -- we never invent new respondents).
    merged = survey_counts["3.3-3.7"]
    # Estimate the *share* of the merged bin that belongs to 3.0-3.3 using
    # the ratio raw_est / (raw_est + c_right).
    share = raw_est / (raw_est + c_right)
    new_30_33 = merged * share
    new_33_37 = merged * (1 - share)
    out = {
        **{k: float(v) for k, v in survey_counts.items()},
        "3.0-3.3": float(new_30_33),
        "3.3-3.7": float(new_33_37),
    }
    return out


def ak_criterion(nll: float, k: int) -> float:
    return 2 * k + 2 * nll


def normalize_survey(aggregated: pd.DataFrame) -> dict:
    gpa_rows = aggregated[aggregated.question_id == "gpa"]
    survey_counts = {row.option: int(row["count"]) for _, row in gpa_rows.iterrows()}

    # Survey-as-observed counts for fitting (note: 3.3-3.7 really means 3.0-3.7):
    obs_counts = np.array(
        [
            survey_counts.get("Below 2.3", 0),
            survey_counts.get("2.3-2.7", 0),
            survey_counts.get("2.7-3.0", 0),
            survey_counts.get("3.3-3.7", 0),  # this absorbs the missing 3.0-3.3 bin
            survey_counts.get("Above 3.7", 0),
        ],
        dtype=float,
    )

    # Fit trunc normal & beta.
    tn = fit_trunc_normal(obs_counts, SURVEY_BINS)
    bt = fit_beta(obs_counts, SURVEY_BINS)
    tn["aic"] = ak_criterion(tn["nll"], 2)
    bt["aic"] = ak_criterion(bt["nll"], 2)
    primary = "trunc_normal" if tn["aic"] <= bt["aic"] else "beta"

    # Full 6-bin probabilities from each fit.
    full_p_tn = trunc_normal_probs(tn["mu"], tn["sigma"], FULL_BINS)
    full_p_bt = beta_probs(bt["alpha"], bt["beta"], FULL_BINS)

    n_total = int(obs_counts.sum())

    # Method 1: trunc-normal -- split the merged bin using the fitted share.
    p_30_33 = full_p_tn[3]
    p_33_37 = full_p_tn[4]
    share_tn = p_30_33 / (p_30_33 + p_33_37)
    merged_n = survey_counts["3.3-3.7"]
    tn_counts = {
        "Below 2.3": survey_counts["Below 2.3"],
        "2.3-2.7": survey_counts["2.3-2.7"],
        "2.7-3.0": survey_counts["2.7-3.0"],
        "3.0-3.3": round(merged_n * share_tn, 2),
        "3.3-3.7": round(merged_n * (1 - share_tn), 2),
        "Above 3.7": survey_counts["Above 3.7"],
    }

    # Method 2: beta -- same split procedure.
    p_30_33_b = full_p_bt[3]
    p_33_37_b = full_p_bt[4]
    share_bt = p_30_33_b / (p_30_33_b + p_33_37_b)
    bt_counts = {
        "Below 2.3": survey_counts["Below 2.3"],
        "2.3-2.7": survey_counts["2.3-2.7"],
        "2.7-3.0": survey_counts["2.7-3.0"],
        "3.0-3.3": round(merged_n * share_bt, 2),
        "3.3-3.7": round(merged_n * (1 - share_bt), 2),
        "Above 3.7": survey_counts["Above 3.7"],
    }

    # Method 3: log-linear interpolation.
    ll_counts = log_linear_impute(survey_counts)
    ll_counts = {k: round(v, 2) for k, v in ll_counts.items() if k in [
        "Below 2.3", "2.3-2.7", "2.7-3.0", "3.0-3.3", "3.3-3.7", "Above 3.7"]}

    # Sanity: all three reconstructions preserve N.
    for label_m, c in [("trunc_normal", tn_counts), ("beta", bt_counts), ("loglin", ll_counts)]:
        total = sum(c.values())
        assert abs(total - n_total) < 0.05, f"{label_m} total {total} != {n_total}"

    return {
        "n_total": n_total,
        "original": survey_counts,
        "trunc_normal_fit": tn,
        "beta_fit": bt,
        "primary": primary,
        "trunc_normal_counts": tn_counts,
        "beta_counts": bt_counts,
        "loglin_counts": ll_counts,
    }


def write_reports(result: dict) -> None:
    rows = []
    for bin_name, _, _ in FULL_BINS:
        rows.append({
            "bin": bin_name,
            "original": result["original"].get(bin_name, np.nan),
            "trunc_normal": result["trunc_normal_counts"].get(bin_name, np.nan),
            "beta": result["beta_counts"].get(bin_name, np.nan),
            "log_linear": result["loglin_counts"].get(bin_name, np.nan),
        })
    pd.DataFrame(rows).to_csv(DATA / "gpa_imputation_comparison.csv", index=False)

    # Primary (adopted) imputed table.
    primary_counts = (result["trunc_normal_counts"] if result["primary"] == "trunc_normal"
                      else result["beta_counts"])
    primary_rows = [{"bin": k, "count": v} for k, v in primary_counts.items()]
    pd.DataFrame(primary_rows).to_csv(DATA / "gpa_imputed_primary.csv", index=False)

    with (DATA / "gpa_normalization.json").open("w") as f:
        json.dump(result, f, indent=2, default=float)


def main() -> None:
    pooled = pd.read_csv(DATA / "survey_aggregated.csv")
    result = normalize_survey(pooled)
    write_reports(result)

    print("=" * 70)
    print("02_gpa_normalize.py -- verification")
    print("=" * 70)
    print(f"\nPooled sample  N={result['n_total']}")
    print(f"  Original (5 bins, 3.0-3.3 missing):")
    for b_name, _, _ in FULL_BINS:
        if b_name != "3.0-3.3":
            print(f"    {b_name:>10s}: {result['original'].get(b_name, 0):>4}")
    print(f"  Trunc-normal fit: mu={result['trunc_normal_fit']['mu']:.3f}, "
          f"sigma={result['trunc_normal_fit']['sigma']:.3f}, "
          f"AIC={result['trunc_normal_fit']['aic']:.2f}")
    print(f"  Beta fit:         alpha={result['beta_fit']['alpha']:.3f}, "
          f"beta={result['beta_fit']['beta']:.3f}, "
          f"AIC={result['beta_fit']['aic']:.2f}")
    print(f"  --> primary model: {result['primary']}")
    print(f"  Imputed 3.0-3.3 bin count by method:")
    print(f"    truncated normal : {result['trunc_normal_counts']['3.0-3.3']}")
    print(f"    beta             : {result['beta_counts']['3.0-3.3']}")
    print(f"    log-linear       : {result['loglin_counts']['3.0-3.3']}")
    print(f"  Post-imputation (primary):")
    primary = (result["trunc_normal_counts"] if result["primary"] == "trunc_normal"
               else result["beta_counts"])
    for k, v in primary.items():
        print(f"    {k:>10s}: {v}")

    print(f"\nWrote: {DATA / 'gpa_imputation_comparison.csv'}")
    print(f"Wrote: {DATA / 'gpa_imputed_primary.csv'}")
    print(f"Wrote: {DATA / 'gpa_normalization.json'}")


if __name__ == "__main__":
    main()

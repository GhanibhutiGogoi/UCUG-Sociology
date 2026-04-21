"""Statistical tests for the HKUST(GZ) pooled survey + interview data.

With row-level data now available from the xlsx exports we can run real
joint tests instead of the null-model bounds used previously.

Outputs in data/results.json:
  - gpa_descriptive        (pooled, post-imputation)
  - gpa_goodness_of_fit    (trunc-normal vs imputed observed)
  - gpa_descriptive_by_group       (mainland vs non-mainland, pre-imputation)
  - gpa_mann_whitney               (non-mainland vs mainland, GPA codes)
  - gpa_fisher_low_vs_high         (2x2: below/above 3.0 by group)
  - gpa_cliffs_delta               (non-mainland vs mainland)
  - likert_by_group                (per-item Mann-Whitney + agree %)
  - interview_theme_tests          (Fisher + BH on 14 themes)
  - interview_theme_burden         (permutation test)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RNG = np.random.default_rng(20260421)

GPA_BINS_6 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.0-3.3", "3.3-3.7", "Above 3.7"]
# Pre-imputation 5-bin scale (as observed in the xlsx). Codes 1..5 here map to
# 1,2,3,5,6 in the 6-bin scale because the 3.0-3.3 bin (code 4) was missing.
GPA_BINS_5 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.3-3.7", "Above 3.7"]
GPA_CODE_5 = {b: i + 1 for i, b in enumerate(GPA_BINS_5)}
GPA_MIDPOINT_5 = {
    "Below 2.3": 2.15,
    "2.3-2.7": 2.5,
    "2.7-3.0": 2.85,
    "3.3-3.7": 3.5,
    "Above 3.7": 3.85,
}
GPA_BIN_MIDPOINTS = {
    "Below 2.3": 2.15,
    "2.3-2.7": 2.5,
    "2.7-3.0": 2.85,
    "3.0-3.3": 3.15,
    "3.3-3.7": 3.5,
    "Above 3.7": 3.85,
}
LIKERT_ORDER = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
LIKERT_CODE = {lab: i + 1 for i, lab in enumerate(LIKERT_ORDER)}
MAINLAND = "Domestic Chinese student"
HKMT = "Hong Kong, Macau, or Taiwan student"
INTL = "International student"


def non_mainland(status: pd.Series) -> pd.Series:
    return status != MAINLAND


# ---------------------------------------------------------------------------
# GPA descriptive (pooled + by group)
# ---------------------------------------------------------------------------
def gpa_descriptive(primary_imputed: pd.DataFrame) -> dict:
    sub = primary_imputed.set_index("bin")
    counts = sub["count"].reindex(GPA_BINS_6).fillna(0).values
    midpoints = np.array([GPA_BIN_MIDPOINTS[b] for b in GPA_BINS_6])
    n = counts.sum()
    weights = counts / n
    mean = float(np.sum(weights * midpoints))
    var = float(np.sum(weights * (midpoints - mean) ** 2))
    std = math.sqrt(var)
    cdf = np.cumsum(weights)

    def q(p):
        idx = np.searchsorted(cdf, p)
        return float(midpoints[min(idx, len(midpoints) - 1)])

    return {
        "n": float(n),
        "mean_gpa_midpoint": round(mean, 3),
        "sd_gpa_midpoint": round(std, 3),
        "median": round(q(0.5), 3),
        "q1": round(q(0.25), 3),
        "q3": round(q(0.75), 3),
        "below_3_share": round(float(sum(counts[:3]) / n), 3),
        "three_to_three_three_share": round(float(counts[3] / n), 3),
        "above_3_3_share": round(float(sum(counts[4:]) / n), 3),
    }


def gpa_descriptive_by_group(rowlevel: pd.DataFrame) -> dict:
    """Pre-imputation, group-level descriptive using the 5-bin midpoints."""
    out = {}
    for label, mask in [
        ("mainland", rowlevel.status == MAINLAND),
        ("non_mainland", rowlevel.status != MAINLAND),
        ("international_only", rowlevel.status == INTL),
        ("hkmt_only", rowlevel.status == HKMT),
    ]:
        sub = rowlevel.loc[mask, "gpa"].map(GPA_MIDPOINT_5).dropna()
        if len(sub) == 0:
            continue
        out[label] = {
            "n": int(len(sub)),
            "mean_gpa_midpoint": round(float(sub.mean()), 3),
            "sd": round(float(sub.std(ddof=1)) if len(sub) > 1 else 0.0, 3),
            "median": round(float(sub.median()), 3),
            "share_below_3": round(float((sub < 3.0).mean()), 3),
        }
    return out


def gpa_goodness_of_fit(primary_imputed: pd.DataFrame, fit: dict) -> dict:
    sub = primary_imputed.set_index("bin")
    obs = sub["count"].reindex(GPA_BINS_6).fillna(0).values.astype(float)
    mu, sigma = fit["mu"], fit["sigma"]
    a = (0.0 - mu) / sigma
    b = (4.3 - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    edges = [(0.0, 2.3), (2.3, 2.7), (2.7, 3.0), (3.0, 3.3), (3.3, 3.7), (3.7, 4.3)]
    probs = np.array([dist.cdf(hi) - dist.cdf(lo) for lo, hi in edges])
    probs = probs / probs.sum()
    exp = probs * obs.sum()
    merged_obs, merged_exp = [], []
    buf_o = buf_e = 0.0
    for o, e in zip(obs, exp):
        buf_o += o
        buf_e += e
        if buf_e >= 5:
            merged_obs.append(buf_o)
            merged_exp.append(buf_e)
            buf_o = buf_e = 0.0
    if buf_e > 0:
        if merged_exp:
            merged_obs[-1] += buf_o
            merged_exp[-1] += buf_e
        else:
            merged_obs.append(buf_o)
            merged_exp.append(buf_e)
    chi2 = float(sum((o - e) ** 2 / e for o, e in zip(merged_obs, merged_exp)))
    dof = max(len(merged_obs) - 1 - 2, 1)
    pval = 1 - stats.chi2.cdf(chi2, dof)
    return {
        "chi2": round(chi2, 3),
        "dof": dof,
        "p_chi2": round(pval, 3),
        "merged_bins_used": len(merged_obs),
    }


# ---------------------------------------------------------------------------
# Real joint tests on GPA x status
# ---------------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    gt = sum(1 for a in x for b in y if a > b)
    lt = sum(1 for a in x for b in y if a < b)
    return (gt - lt) / (nx * ny)


def gpa_joint_tests(rowlevel: pd.DataFrame) -> dict:
    """Mann-Whitney U + Fisher exact + Cliff's delta on GPA codes
    (non-mainland vs mainland)."""
    codes = rowlevel["gpa"].map(GPA_CODE_5)
    nm = codes[non_mainland(rowlevel.status)].dropna().astype(int).values
    mn = codes[~non_mainland(rowlevel.status)].dropna().astype(int).values

    u_stat, p_u_two = stats.mannwhitneyu(nm, mn, alternative="two-sided")
    _, p_u_less = stats.mannwhitneyu(nm, mn, alternative="less")
    delta = cliffs_delta(nm, mn)

    # 2x2 Fisher: "below 3.0" (codes 1-3) vs "at or above 3.3" (codes 4-5)
    nm_low = int((nm <= 3).sum())
    nm_high = int((nm >= 4).sum())
    mn_low = int((mn <= 3).sum())
    mn_high = int((mn >= 4).sum())
    fisher_or, fisher_p = stats.fisher_exact([[nm_low, nm_high], [mn_low, mn_high]])

    # Permutation test on mean-code difference
    observed = nm.mean() - mn.mean()
    combined = np.concatenate([nm, mn])
    n1 = len(nm)
    n_iter = 10_000
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        RNG.shuffle(combined)
        diffs[i] = combined[:n1].mean() - combined[n1:].mean()
    perm_p_two = float((np.abs(diffs) >= abs(observed)).mean())
    perm_p_less = float((diffs <= observed).mean())

    return {
        "n_non_mainland": int(len(nm)),
        "n_mainland": int(len(mn)),
        "mean_code_non_mainland": round(float(nm.mean()), 3),
        "mean_code_mainland": round(float(mn.mean()), 3),
        "mean_diff_nm_minus_mn": round(float(observed), 3),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p_two_sided": round(float(p_u_two), 4),
        "mannwhitney_p_less": round(float(p_u_less), 4),
        "cliffs_delta": round(float(delta), 3),
        "fisher_table": {
            "non_mainland": {"below_3": nm_low, "at_or_above_3.3": nm_high},
            "mainland": {"below_3": mn_low, "at_or_above_3.3": mn_high},
        },
        "fisher_odds_ratio": round(float(fisher_or), 3),
        "fisher_p": round(float(fisher_p), 4),
        "permutation_p_two_sided": round(perm_p_two, 4),
        "permutation_p_less": round(perm_p_less, 4),
        "permutation_n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# Likert by group
# ---------------------------------------------------------------------------
def likert_by_group(rowlevel: pd.DataFrame) -> dict:
    """For each Likert item, per-group mean/agree %, Mann-Whitney vs mainland,
    with BH correction across items."""
    likert_cols = [c for c in rowlevel.columns
                   if rowlevel[c].dropna().astype(str).isin(LIKERT_ORDER).all()
                   and rowlevel[c].notna().any()]
    rows = []
    pvals = []
    for qid in likert_cols:
        codes = rowlevel[qid].map(LIKERT_CODE)
        nm = codes[non_mainland(rowlevel.status)].dropna().astype(int).values
        mn = codes[~non_mainland(rowlevel.status)].dropna().astype(int).values
        if len(nm) == 0 or len(mn) == 0:
            continue
        try:
            u, p = stats.mannwhitneyu(nm, mn, alternative="two-sided")
        except ValueError:
            u, p = float("nan"), 1.0
        delta = cliffs_delta(nm, mn)
        agree_nm = float((nm >= 4).mean())
        agree_mn = float((mn >= 4).mean())
        rows.append({
            "question_id": qid,
            "n_non_mainland": int(len(nm)),
            "n_mainland": int(len(mn)),
            "mean_non_mainland": round(float(nm.mean()), 3),
            "mean_mainland": round(float(mn.mean()), 3),
            "mean_diff": round(float(nm.mean() - mn.mean()), 3),
            "agree_pct_non_mainland": round(agree_nm, 3),
            "agree_pct_mainland": round(agree_mn, 3),
            "agree_diff": round(agree_nm - agree_mn, 3),
            "mannwhitney_u": float(u),
            "mannwhitney_p": round(float(p), 4),
            "cliffs_delta": round(float(delta), 3),
        })
        pvals.append(p)
    if pvals:
        _, p_bh, _, _ = multipletests(pvals, method="fdr_bh")
        for row, q in zip(rows, p_bh):
            row["p_bh"] = round(float(q), 4)
    return {"per_item": rows}


# ---------------------------------------------------------------------------
# Categorical (single-choice / ordinal) variables crossed with status
# ---------------------------------------------------------------------------
CATEGORICAL_BY_GROUP = [
    ("social_circle", ["0", "1-2", "3-5", "6-10", "More than 10"], "ordinal"),
    ("aware_services", ["No", "Somewhat", "Yes"], "ordinal"),
    ("year", ["Year 1", "Year 2", "Year 3"], "ordinal"),
    ("gender", ["Male", "Female", "Prefer not to say"], "nominal"),
]


def categorical_by_group(rowlevel: pd.DataFrame) -> dict:
    """For each categorical variable, return a crosstab (non-mainland vs mainland)
    with Fisher / chi-square test and ordinal Mann-Whitney where applicable."""
    out = {}
    for qid, ordering, kind in CATEGORICAL_BY_GROUP:
        if qid not in rowlevel.columns:
            continue
        sub = rowlevel[[qid, "status"]].dropna()
        if sub.empty:
            continue
        sub = sub.assign(group=np.where(sub.status == MAINLAND, "mainland", "non_mainland"))
        ct = pd.crosstab(sub[qid], sub.group).reindex(ordering).fillna(0).astype(int)
        # Ensure both group columns exist
        for g in ["non_mainland", "mainland"]:
            if g not in ct.columns:
                ct[g] = 0
        ct = ct[["non_mainland", "mainland"]]

        result = {
            "kind": kind,
            "n_non_mainland": int(ct["non_mainland"].sum()),
            "n_mainland": int(ct["mainland"].sum()),
            "crosstab": ct.astype(int).to_dict(orient="index"),
        }

        # Overall Fisher (for 2xK) / chi-square
        table = ct.values
        try:
            _, p_fisher = stats.fisher_exact(table) if table.shape == (2, 2) else (None, None)
        except Exception:
            p_fisher = None
        try:
            chi2, p_chi2, _, _ = stats.chi2_contingency(table)
            result["chi2"] = round(float(chi2), 3)
            result["chi2_p"] = round(float(p_chi2), 4)
        except (ValueError, stats._stats_py.LinAlgError):
            result["chi2"] = None
            result["chi2_p"] = None
        if p_fisher is not None:
            result["fisher_p"] = round(float(p_fisher), 4)

        # Ordinal test + effect size if ordinal
        if kind == "ordinal":
            code_map = {v: i + 1 for i, v in enumerate(ordering)}
            codes = sub[qid].map(code_map).astype(float)
            nm = codes[sub.group == "non_mainland"].values
            mn = codes[sub.group == "mainland"].values
            try:
                u, p_mw = stats.mannwhitneyu(nm, mn, alternative="two-sided")
                result["mannwhitney_p"] = round(float(p_mw), 4)
                result["mean_code_non_mainland"] = round(float(nm.mean()), 3)
                result["mean_code_mainland"] = round(float(mn.mean()), 3)
                result["mean_code_diff"] = round(float(nm.mean() - mn.mean()), 3)
                result["cliffs_delta"] = round(cliffs_delta(nm, mn), 3)
            except ValueError:
                pass

        out[qid] = result
    return out


# ---------------------------------------------------------------------------
# Pooled Likert summary (for back-compat with plots/report)
# ---------------------------------------------------------------------------
def likert_summary(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid in agg.question_id.unique():
        sub = agg[agg.question_id == qid]
        if not sub.option.isin(LIKERT_ORDER).all():
            continue
        codes = sub.option.map(LIKERT_CODE)
        weights = sub["count"].values
        total = int(weights.sum())
        mean = float(np.average(codes.values, weights=weights))
        var = float(np.average((codes.values - mean) ** 2, weights=weights))
        sd = math.sqrt(var)
        agree = int(sub[sub.option.isin(["Agree", "Strongly agree"])]["count"].sum())
        agree_pct = agree / total
        lo, hi = stats.binomtest(agree, total).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        rows.append({
            "question_id": qid,
            "question_text": sub.question_text.iloc[0],
            "mean_1to5": round(mean, 3),
            "sd": round(sd, 3),
            "agree_plus_pct": round(agree_pct, 3),
            "agree_ci_lo": round(float(lo), 3),
            "agree_ci_hi": round(float(hi), 3),
        })
    return pd.DataFrame(rows).sort_values("question_id")


# ---------------------------------------------------------------------------
# Interview theme tests (unchanged from prior pass)
# ---------------------------------------------------------------------------
def interview_theme_tests(themes: pd.DataFrame) -> dict:
    intl = themes[themes.group == "international"]
    mnl = themes[themes.group == "mainland"]
    theme_cols = [c for c in themes.columns if c not in ("interview_id", "group")]
    rows = []
    pvals = []
    for t in theme_cols:
        a = int(intl[t].sum())
        b = len(intl) - a
        c = int(mnl[t].sum())
        d = len(mnl) - c
        _, p = stats.fisher_exact([[a, b], [c, d]])
        p_intl = a / len(intl) if len(intl) else 0
        p_mnl = c / len(mnl) if len(mnl) else 0
        rows.append({
            "theme": t,
            "intl_count": a, "intl_n": len(intl), "intl_pct": round(p_intl, 3),
            "mnl_count": c, "mnl_n": len(mnl), "mnl_pct": round(p_mnl, 3),
            "diff_pct": round(p_intl - p_mnl, 3),
            "fisher_p": round(float(p), 4),
        })
        pvals.append(p)
    _, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
    for row, q in zip(rows, p_adj):
        row["fisher_p_bh"] = round(float(q), 4)
    return {"per_theme": rows}


def theme_permutation_test(themes: pd.DataFrame, n_iter: int = 10_000) -> dict:
    theme_cols = [c for c in themes.columns if c not in ("interview_id", "group")]
    t = themes.copy()
    t["burden"] = t[theme_cols].sum(axis=1)
    intl = t[t.group == "international"]["burden"].values.astype(float)
    mnl = t[t.group == "mainland"]["burden"].values.astype(float)
    observed = intl.mean() - mnl.mean()
    combined = np.concatenate([intl, mnl])
    n1 = len(intl)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        RNG.shuffle(combined)
        diffs[i] = combined[:n1].mean() - combined[n1:].mean()
    p = float((np.abs(diffs) >= abs(observed)).mean())
    return {
        "intl_mean_burden": round(float(intl.mean()), 3),
        "mnl_mean_burden": round(float(mnl.mean()), 3),
        "observed_diff": round(float(observed), 3),
        "perm_p": round(p, 4),
        "n_iter": n_iter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    rowlevel = pd.read_csv(DATA / "survey_rowlevel.csv")
    agg = pd.read_csv(DATA / "survey_aggregated.csv")
    imputed = pd.read_csv(DATA / "gpa_imputed_primary.csv")
    with (DATA / "gpa_normalization.json").open() as f:
        normalization = json.load(f)
    themes = pd.read_csv(DATA / "interview_themes.csv")

    n_total = len(rowlevel)

    desc = gpa_descriptive(imputed)
    gof = gpa_goodness_of_fit(imputed, normalization["trunc_normal_fit"])
    desc_by_group = gpa_descriptive_by_group(rowlevel)
    gpa_joint = gpa_joint_tests(rowlevel)
    likert_gr = likert_by_group(rowlevel)
    categoricals = categorical_by_group(rowlevel)

    likert_df = likert_summary(agg)
    likert_df.to_csv(DATA / "likert_summary.csv", index=False)
    pd.DataFrame(likert_gr["per_item"]).to_csv(DATA / "likert_by_group.csv", index=False)

    theme_tests = interview_theme_tests(themes)
    burden = theme_permutation_test(themes)

    results = {
        "n_total": n_total,
        "gpa_descriptive": desc,
        "gpa_descriptive_by_group": desc_by_group,
        "gpa_goodness_of_fit": gof,
        "gpa_joint_tests": gpa_joint,
        "likert_by_group": likert_gr,
        "categorical_by_group": categoricals,
        "interview_theme_tests": theme_tests,
        "interview_theme_burden": burden,
    }
    with (DATA / "results.json").open("w") as f:
        json.dump(results, f, indent=2, default=float)

    # Verification prints ---------------------------------------------------
    print("=" * 70)
    print("03_statistics.py -- verification")
    print("=" * 70)
    print(f"\nGPA descriptive (pooled N={int(desc['n'])}, post-imputation): "
          f"mean={desc['mean_gpa_midpoint']}, median={desc['median']}")
    print(f"GPA goodness-of-fit: chi2={gof['chi2']}, dof={gof['dof']}, p={gof['p_chi2']}")

    print("\nGPA by group (pre-imputation, bin midpoints):")
    for k, v in desc_by_group.items():
        print(f"  {k:<20s} n={v['n']:>3d}  mean={v['mean_gpa_midpoint']:.3f}  "
              f"sd={v['sd']:.3f}  share<3.0={v['share_below_3']:.2f}")

    gj = gpa_joint
    print(f"\nGPA: non-mainland vs mainland (real joint test):")
    print(f"  mean codes: nm={gj['mean_code_non_mainland']}, mn={gj['mean_code_mainland']}, "
          f"diff={gj['mean_diff_nm_minus_mn']}")
    print(f"  Mann-Whitney U: p(two-sided)={gj['mannwhitney_p_two_sided']}, "
          f"p(less)={gj['mannwhitney_p_less']}")
    print(f"  Cliff's delta: {gj['cliffs_delta']}")
    print(f"  Fisher (below 3.0 vs >=3.3): OR={gj['fisher_odds_ratio']}, "
          f"p={gj['fisher_p']}")
    print(f"  Permutation: p(two-sided)={gj['permutation_p_two_sided']}, "
          f"p(less)={gj['permutation_p_less']}")

    print(f"\nLikert by group ({len(likert_gr['per_item'])} items, BH-corrected):")
    sig = [r for r in likert_gr["per_item"] if r.get("p_bh", 1) < 0.1]
    for r in sorted(likert_gr["per_item"], key=lambda x: x["p_bh"]):
        mark = "*" if r["p_bh"] < 0.05 else " "
        print(f"  {mark} {r['question_id']:<25s} "
              f"nm={r['mean_non_mainland']:.2f}  mn={r['mean_mainland']:.2f}  "
              f"diff={r['mean_diff']:+.2f}  delta={r['cliffs_delta']:+.2f}  "
              f"p={r['mannwhitney_p']:.3f}  p_BH={r['p_bh']:.3f}")

    print(f"\nInterview theme burden: intl={burden['intl_mean_burden']}, "
          f"mnl={burden['mnl_mean_burden']}, perm p={burden['perm_p']}")

    print(f"\nCategorical variables by group:")
    for qid, res in categoricals.items():
        ptxt = f"chi2 p={res.get('chi2_p')}" if res.get("chi2_p") is not None else ""
        if "mannwhitney_p" in res:
            ptxt += f", MW p={res['mannwhitney_p']}, delta={res.get('cliffs_delta')}"
        print(f"  {qid:<20s} nm_n={res['n_non_mainland']}, mn_n={res['n_mainland']}  {ptxt}")

    print(f"\nWrote: {DATA / 'likert_summary.csv'}")
    print(f"Wrote: {DATA / 'likert_by_group.csv'}")
    print(f"Wrote: {DATA / 'results.json'}")


if __name__ == "__main__":
    main()

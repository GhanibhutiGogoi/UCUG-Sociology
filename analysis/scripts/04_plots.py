"""Build the figure set for the HKUST(GZ) sociology report (pooled n=57).

All figures are saved to analysis/figures/ as 300 DPI PNGs.
Colour palette is colour-blind-friendly.

With row-level data available, plots 09 and 11 now use real joint
status x GPA and status x Likert tables rather than the scenario / null-model
placeholders used earlier.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
})

GPA_BINS_6 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.0-3.3", "3.3-3.7", "Above 3.7"]
GPA_BINS_5 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.3-3.7", "Above 3.7"]
LIKERT_ORDER = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
LIKERT_CODE = {lab: i + 1 for i, lab in enumerate(LIKERT_ORDER)}
GPA_MIN, GPA_MAX = 0.0, 4.3
MAINLAND = "Domestic Chinese student"

NEW_THEMES = [
    "clubs_english_only",
    "chinese_peers_avoid_english",
    "non_gaokao_prep_gap",
    "intl_comms_chinese_default",
    "wants_structured_programs",
]


def _save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: GPA distribution -- raw vs three imputation methods (pooled).
# ---------------------------------------------------------------------------
def plot_gpa_imputation_comparison(imp_cmp: pd.DataFrame, n_total: int) -> None:
    sub = imp_cmp.set_index("bin").reindex(GPA_BINS_6)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(GPA_BINS_6))
    width = 0.2
    ax.bar(x - 1.5 * width, sub["original"].fillna(0), width,
           label="original (survey)", color="#bbbbbb", edgecolor="black")
    ax.bar(x - 0.5 * width, sub["trunc_normal"], width,
           label="trunc-normal (primary)", color="#4E79A7", edgecolor="black")
    ax.bar(x + 0.5 * width, sub["beta"], width,
           label="beta", color="#F28E2B", edgecolor="black")
    ax.bar(x + 1.5 * width, sub["log_linear"], width,
           label="log-linear", color="#59A14F", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(GPA_BINS_6, rotation=30, ha="right")
    ax.set_ylabel("respondents")
    ax.set_xlabel("GPA bin")
    ax.set_title(f"GPA distribution: original (5 bins) vs three imputed 6-bin reconstructions\n"
                 f"(N = {n_total})", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    _save(fig, "01_gpa_imputation_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: fitted truncated-normal density overlaid on histogram.
# ---------------------------------------------------------------------------
def plot_gpa_density(normalization: dict, imputed: pd.DataFrame) -> None:
    mu = normalization["trunc_normal_fit"]["mu"]
    sigma = normalization["trunc_normal_fit"]["sigma"]
    a, b = (GPA_MIN - mu) / sigma, (GPA_MAX - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    xs = np.linspace(GPA_MIN, GPA_MAX, 400)
    density = dist.pdf(xs)
    sub = imputed.set_index("bin").reindex(GPA_BINS_6)
    bin_midpts = [2.15, 2.5, 2.85, 3.15, 3.5, 3.85]
    widths = [2.3, 0.4, 0.3, 0.3, 0.4, 0.6]
    densities = sub["count"].values / (sub["count"].sum() * np.array(widths))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(bin_midpts, densities, width=widths, align="center",
           alpha=0.45, edgecolor="black", color="#4E79A7",
           label="observed (imputed)")
    ax.plot(xs, density, color="#E15759", lw=2.5,
            label=f"truncated normal\n($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})")
    ax.axvspan(3.0, 3.3, color="gold", alpha=0.25, label="imputed 3.0-3.3 bin")
    ax.set_xlim(2.0, 4.1)
    ax.set_xlabel("GPA")
    ax.set_ylabel("density")
    n = int(round(sub["count"].sum()))
    ax.set_title(f"Truncated-normal MLE fit to observed GPA histogram (N = {n})",
                 fontsize=13)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "02_gpa_density_fit.png")


# ---------------------------------------------------------------------------
# Figure 3: GPA distribution (post-imputation) with 3.0-3.3 highlighted.
# ---------------------------------------------------------------------------
def plot_gpa_distribution(imputed: pd.DataFrame) -> None:
    counts = imputed.set_index("bin")["count"].reindex(GPA_BINS_6).fillna(0)
    n = int(round(counts.sum()))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(counts.index, counts.values, color="#4E79A7", edgecolor="black")
    for b, c in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.3, f"{c:.1f}",
                ha="center", va="bottom", fontsize=11)
    idx = GPA_BINS_6.index("3.0-3.3")
    bars[idx].set_facecolor("gold")
    bars[idx].set_edgecolor("black")
    ax.set_title(f"GPA distribution (N = {n}, post-imputation).  "
                 f"Gold bar = imputed 3.0-3.3 bin.", fontsize=13)
    ax.set_ylabel("respondents (fractional allowed)")
    ax.set_xticklabels(GPA_BINS_6, rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, "03_gpa_distribution.png")


# ---------------------------------------------------------------------------
# Figure 4: Agree-% with 95% Wilson CI, sorted.
# ---------------------------------------------------------------------------
def plot_agree_ci(likert_df: pd.DataFrame, n_total: int) -> None:
    d = likert_df.sort_values("agree_plus_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y, d["agree_plus_pct"] * 100, color="#4E79A7", alpha=0.8)
    ax.errorbar(
        d["agree_plus_pct"] * 100, y,
        xerr=[
            (d["agree_plus_pct"] - d["agree_ci_lo"]) * 100,
            (d["agree_ci_hi"] - d["agree_plus_pct"]) * 100,
        ],
        fmt="none", color="black", capsize=4,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(d["question_text"], fontsize=10)
    ax.set_xlabel("% Agree + Strongly agree   (95% Wilson CI)")
    ax.set_xlim(0, 100)
    ax.set_title(f"Likert agreement (N = {n_total})", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_agree_ci.png")


# ---------------------------------------------------------------------------
# Figure 5: Social-circle size distribution (B-only, n=43).
# ---------------------------------------------------------------------------
def plot_social_circle(agg: pd.DataFrame) -> None:
    sub = agg[agg.question_id == "social_circle"]
    n = int(sub["count"].sum())
    order = ["0", "1-2", "3-5", "6-10", "More than 10"]
    counts = sub.set_index("option")["count"].reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(order, counts, color="#4E79A7", edgecolor="black")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.3, int(c),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel(f"respondents (n = {n}, later-batch only)")
    ax.set_xlabel("Size of main social circle")
    ax.set_title("Social circle size (asked only in the second export batch, n = 43)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, "05_social_circle.png")


# ---------------------------------------------------------------------------
# Figure 6: Stacked Likert for integration-related items (pooled).
# ---------------------------------------------------------------------------
def plot_integration_stack(agg: pd.DataFrame, n_total: int) -> None:
    focus = {
        "belong": "I feel that I belong at HKUST(GZ)",
        "meaningful_friendships": "Meaningful friendships across backgrounds",
        "interact_chinese": "Regularly interact with local Chinese",
        "comfort_init": "Comfortable initiating cross-culture conversations",
        "language_hard": "Language makes friendships hard",
        "stay_in_groups": "Students stay in own cultural/national groups",
        "excluded_study_group": "Excluded from study group (language/culture)",
        "social_motivation": "Social difficulties hurt my motivation",
        "social_grades": "Social difficulties hurt my actual grades",
    }
    rows = []
    for q_id, title in focus.items():
        sub = agg[agg.question_id == q_id].set_index("option")["count"]
        row = {"question": title}
        for opt in LIKERT_ORDER:
            row[opt] = sub.get(opt, 0)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("question")
    df_pct = df.div(df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(df) + 1.5))
    colors = ["#B30000", "#E17A6A", "#D8D8D8", "#7FA6C9", "#2E5894"]
    bottom = np.zeros(len(df))
    y = np.arange(len(df))
    for lab, color in zip(LIKERT_ORDER, colors):
        vals = df_pct[lab].values
        ax.barh(y, vals, left=bottom, color=color, label=lab, edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 6:
                ax.text(b + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9,
                        color="white" if lab in ("Strongly disagree", "Strongly agree") else "black")
        bottom += vals
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of respondents")
    ax.set_title(f"Social integration & academic-link items (N = {n_total})", fontsize=13)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=10)
    fig.tight_layout()
    _save(fig, "06_integration_likert_stack.png")


# ---------------------------------------------------------------------------
# Figure 7: Interview theme prevalence by group, with fisher-test stars.
# ---------------------------------------------------------------------------
def plot_theme_prevalence(theme_tests: dict) -> None:
    per = pd.DataFrame(theme_tests["per_theme"]).set_index("theme")
    per = per.sort_values("diff_pct", ascending=True)
    intl_n = int(per["intl_n"].iloc[0])
    mnl_n = int(per["mnl_n"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(per) + 1.5))
    y = np.arange(len(per))
    ax.barh(y - 0.2, per["intl_pct"] * 100, 0.4,
            label=f"International (n={intl_n})", color="#E15759")
    ax.barh(y + 0.2, per["mnl_pct"] * 100, 0.4,
            label=f"Mainland Chinese (n={mnl_n})", color="#4E79A7")
    for i, (_, row) in enumerate(per.iterrows()):
        mark = ""
        if row["fisher_p_bh"] < 0.001:
            mark = "***"
        elif row["fisher_p_bh"] < 0.01:
            mark = "**"
        elif row["fisher_p_bh"] < 0.05:
            mark = "*"
        if mark:
            ax.text(max(row["intl_pct"], row["mnl_pct"]) * 100 + 3, i, mark,
                    va="center", fontsize=16, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(per.index.str.replace("_", " "), fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of respondents reporting the theme")
    ax.set_title("Interview themes by group (Fisher exact, BH-adjusted:\n"
                 "* p<.05, ** p<.01, *** p<.001)", fontsize=13)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "07_interview_theme_prevalence.png")


# ---------------------------------------------------------------------------
# Figure 8: Interview theme burden (distribution).
# ---------------------------------------------------------------------------
def plot_theme_burden(themes: pd.DataFrame, burden_result: dict) -> None:
    theme_cols = [c for c in themes.columns if c not in ("interview_id", "group")]
    n_themes = len(theme_cols)
    t = themes.copy()
    t["burden"] = t[theme_cols].sum(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.boxplot(data=t, x="group", y="burden", ax=ax, hue="group",
                palette="colorblind", width=0.45, legend=False)
    sns.stripplot(data=t, x="group", y="burden", ax=ax, color="black",
                  size=8, jitter=0.12)
    ax.set_title(f"Interview 'theme burden' (0-{n_themes})\n"
                 f"permutation p = {burden_result['perm_p']:.4f}",
                 fontsize=13)
    ax.set_ylabel(f"# of themes reported (out of {n_themes})")
    ax.set_xlabel("group")
    fig.tight_layout()
    _save(fig, "08_theme_burden.png")


# ---------------------------------------------------------------------------
# Figure 9: REAL GPA distribution by group (intl+HKMT vs mainland).
# ---------------------------------------------------------------------------
def plot_gpa_by_group_real(rowlevel: pd.DataFrame, joint: dict) -> None:
    # Stacked/grouped bars: % of each group in each of the 5 observed bins.
    bins_order = GPA_BINS_5
    mainland = rowlevel[rowlevel.status == MAINLAND]["gpa"].value_counts(normalize=True).reindex(bins_order).fillna(0)
    non_mainland = rowlevel[rowlevel.status != MAINLAND]["gpa"].value_counts(normalize=True).reindex(bins_order).fillna(0)
    n_mnl = int((rowlevel.status == MAINLAND).sum())
    n_nm = int((rowlevel.status != MAINLAND).sum())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(bins_order))
    w = 0.4
    b1 = ax.bar(x - w / 2, non_mainland.values * 100, w,
                label=f"Non-mainland (intl + HK/Macau/TW, n={n_nm})",
                color="#E15759", edgecolor="black")
    b2 = ax.bar(x + w / 2, mainland.values * 100, w,
                label=f"Mainland Chinese (n={n_mnl})",
                color="#4E79A7", edgecolor="black")
    for b, v in list(zip(b1, non_mainland.values)) + list(zip(b2, mainland.values)):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bins_order, rotation=20, ha="right")
    ax.set_ylabel("% of group")
    ax.set_title(
        "GPA distribution by group (row-level data, pre-imputation)\n"
        f"mean GPA-code: non-mainland {joint['mean_code_non_mainland']}, "
        f"mainland {joint['mean_code_mainland']}  |  "
        f"Mann-Whitney p = {joint['mannwhitney_p_two_sided']:.3f} (two-sided), "
        f"{joint['mannwhitney_p_less']:.3f} (one-sided)  |  "
        f"Cliff's delta {joint['cliffs_delta']}",
        fontsize=11,
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "09_gpa_by_group.png")


# ---------------------------------------------------------------------------
# Figure 10: Second-pass interview themes.
# ---------------------------------------------------------------------------
def plot_new_themes(theme_tests: dict) -> None:
    per = pd.DataFrame(theme_tests["per_theme"]).set_index("theme")
    per = per.loc[[t for t in NEW_THEMES if t in per.index]]
    per = per.sort_values("diff_pct", ascending=True)
    intl_n = int(per["intl_n"].iloc[0])
    mnl_n = int(per["mnl_n"].iloc[0])
    fig, ax = plt.subplots(figsize=(11, 0.7 * len(per) + 2))
    y = np.arange(len(per))
    ax.barh(y - 0.2, per["intl_pct"] * 100, 0.4,
            label=f"International (n={intl_n})", color="#E15759")
    ax.barh(y + 0.2, per["mnl_pct"] * 100, 0.4,
            label=f"Mainland Chinese (n={mnl_n})", color="#4E79A7")
    for i, (_, row) in enumerate(per.iterrows()):
        mark = ""
        if row["fisher_p_bh"] < 0.001:
            mark = "***"
        elif row["fisher_p_bh"] < 0.01:
            mark = "**"
        elif row["fisher_p_bh"] < 0.05:
            mark = "*"
        if mark:
            ax.text(max(row["intl_pct"], row["mnl_pct"]) * 100 + 3, i, mark,
                    va="center", fontsize=16, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(per.index.str.replace("_", " "), fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_xlabel("% of respondents reporting the theme")
    ax.set_title("Second-pass themes (from long-transcript re-read)\n"
                 "Fisher exact, BH-adjusted across all 14 themes", fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "10_new_interview_themes.png")


# ---------------------------------------------------------------------------
# Figure 11: REAL per-Likert-item difference by group (survey data).
# ---------------------------------------------------------------------------
def plot_likert_by_group(likert_gr: dict, question_texts: dict) -> None:
    df = pd.DataFrame(likert_gr["per_item"])
    df["question_text"] = df["question_id"].map(question_texts)
    df = df.sort_values("mean_diff", ascending=True)
    n_nm = int(df["n_non_mainland"].iloc[0])
    n_mn = int(df["n_mainland"].iloc[0])

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(df) + 2))
    y = np.arange(len(df))
    diffs = df["mean_diff"].values
    colors = ["#E15759" if d > 0 else "#4E79A7" for d in diffs]
    ax.barh(y, diffs, color=colors, edgecolor="black")
    for i, (_, row) in enumerate(df.iterrows()):
        d = row["mean_diff"]
        x_label = d + (0.04 if d >= 0 else -0.04)
        ha = "left" if d >= 0 else "right"
        mark = ""
        if row["p_bh"] < 0.05:
            mark = " *"
        if row["p_bh"] < 0.01:
            mark = " **"
        txt = f"nm {row['mean_non_mainland']:.2f} vs mn {row['mean_mainland']:.2f}  (δ={row['cliffs_delta']:+.2f}{mark})"
        ax.text(x_label, i, txt, va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["question_text"], fontsize=10)
    ax.set_xlabel("non-mainland mean Likert  -  mainland mean Likert  (1-5 scale)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_title(
        f"Likert items: difference by group  (non-mainland n={n_nm}, mainland n={n_mn})\n"
        f"negative = non-mainland agrees less  |  "
        f"Mann-Whitney with BH correction (* p<.05, ** p<.01)",
        fontsize=12,
    )
    fig.tight_layout()
    _save(fig, "11_likert_by_group.png")


# ---------------------------------------------------------------------------
# Figure 12: Interview opinion differences (kept as qualitative supplement).
# ---------------------------------------------------------------------------
def plot_opinion_differences(theme_tests: dict) -> None:
    per = pd.DataFrame(theme_tests["per_theme"]).set_index("theme")
    per = per.sort_values("diff_pct", ascending=True)
    intl_n = int(per["intl_n"].iloc[0])
    mnl_n = int(per["mnl_n"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 0.55 * len(per) + 2))
    y = np.arange(len(per))
    diffs = per["diff_pct"].values * 100
    colors = ["#4E79A7" if d < 0 else "#E15759" for d in diffs]
    ax.barh(y, diffs, color=colors, edgecolor="black")
    for i, (_, row) in enumerate(per.iterrows()):
        d = row["diff_pct"] * 100
        label_x = d + (2 if d >= 0 else -2)
        ha = "left" if d >= 0 else "right"
        mark = ""
        if row["fisher_p_bh"] < 0.001:
            mark = " ***"
        elif row["fisher_p_bh"] < 0.01:
            mark = " **"
        elif row["fisher_p_bh"] < 0.05:
            mark = " *"
        ax.text(label_x, i, f"intl {row['intl_pct']*100:.0f}% vs mnl {row['mnl_pct']*100:.0f}%{mark}",
                va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(per.index.str.replace("_", " "), fontsize=11)
    ax.set_xlabel("percentage-point difference  (intl % - mainland %)")
    ax.set_xlim(-40, 120)
    ax.set_title(f"Interview-based opinion differences: "
                 f"international (n={intl_n}) vs mainland (n={mnl_n})\n"
                 f"across 14 themes  (Fisher BH: * p<.05, ** p<.01, *** p<.001)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "12_interview_opinion_differences.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    imp_cmp = pd.read_csv(DATA / "gpa_imputation_comparison.csv")
    imputed = pd.read_csv(DATA / "gpa_imputed_primary.csv")
    with (DATA / "gpa_normalization.json").open() as f:
        normalization = json.load(f)
    with (DATA / "results.json").open() as f:
        results = json.load(f)
    likert_df = pd.read_csv(DATA / "likert_summary.csv")
    themes = pd.read_csv(DATA / "interview_themes.csv")
    agg = pd.read_csv(DATA / "survey_aggregated.csv")
    rowlevel = pd.read_csv(DATA / "survey_rowlevel.csv")

    n_total = int(results["n_total"])
    question_texts = (
        agg.drop_duplicates("question_id")
        .set_index("question_id")["question_text"]
        .to_dict()
    )

    plot_gpa_imputation_comparison(imp_cmp, n_total)
    plot_gpa_density(normalization, imputed)
    plot_gpa_distribution(imputed)
    plot_agree_ci(likert_df, n_total)
    plot_social_circle(agg)
    plot_integration_stack(agg, n_total)
    plot_theme_prevalence(results["interview_theme_tests"])
    plot_theme_burden(themes, results["interview_theme_burden"])
    plot_gpa_by_group_real(rowlevel, results["gpa_joint_tests"])
    plot_new_themes(results["interview_theme_tests"])
    plot_likert_by_group(results["likert_by_group"], question_texts)
    plot_opinion_differences(results["interview_theme_tests"])

    pngs = sorted(FIG.glob("*.png"))
    print("=" * 70)
    print("04_plots.py -- verification")
    print("=" * 70)
    print(f"Wrote {len(pngs)} PNGs to {FIG}:")
    for p in pngs:
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name:50s} {size_kb:7.1f} KB")
        assert size_kb > 5, f"{p.name} is suspiciously small"


if __name__ == "__main__":
    main()

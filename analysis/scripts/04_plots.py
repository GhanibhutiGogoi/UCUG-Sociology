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
# Figure 5: Social-circle size by group (B-only, n=43).
# ---------------------------------------------------------------------------
def _grouped_percent_bars(ax, crosstab: pd.DataFrame, order: list[str],
                          title: str, xlabel: str,
                          nm_label: str, mn_label: str,
                          annotation: str = "") -> None:
    """Side-by-side percent bars for non-mainland vs mainland on a categorical."""
    ct = crosstab.reindex(order).fillna(0)
    nm_pct = ct["non_mainland"] / max(1, ct["non_mainland"].sum()) * 100
    mn_pct = ct["mainland"] / max(1, ct["mainland"].sum()) * 100
    x = np.arange(len(order))
    w = 0.4
    b1 = ax.bar(x - w / 2, nm_pct.values, w, label=nm_label,
                color="#E15759", edgecolor="black")
    b2 = ax.bar(x + w / 2, mn_pct.values, w, label=mn_label,
                color="#4E79A7", edgecolor="black")
    for b, v in list(zip(b1, nm_pct.values)) + list(zip(b2, mn_pct.values)):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("% of group")
    ax.set_xlabel(xlabel)
    ax.set_title(title + (f"\n{annotation}" if annotation else ""), fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, max(nm_pct.max(), mn_pct.max()) * 1.25 + 5)


def plot_social_circle_by_group(rowlevel: pd.DataFrame, categoricals: dict) -> None:
    d = rowlevel.dropna(subset=["social_circle"]).copy()
    d["group"] = np.where(d.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(d.social_circle, d.group)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    res = categoricals.get("social_circle", {})
    n_nm = res.get("n_non_mainland", 0)
    n_mn = res.get("n_mainland", 0)
    annot = ""
    if "mannwhitney_p" in res:
        annot = (f"Mann-Whitney p = {res['mannwhitney_p']:.3f}, "
                 f"Cliff's delta = {res['cliffs_delta']:+.2f}  "
                 f"(later-batch only, n = {n_nm+n_mn})")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _grouped_percent_bars(ax, ct,
                          ["0", "1-2", "3-5", "6-10", "More than 10"],
                          "Social circle size by group",
                          "Size of main social circle",
                          f"Non-mainland (n={n_nm})",
                          f"Mainland Chinese (n={n_mn})",
                          annotation=annot)
    fig.tight_layout()
    _save(fig, "05_social_circle_by_group.png")


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
# Figure 14: Likert PCA biplot (points coloured by group).
# ---------------------------------------------------------------------------
def plot_likert_pca(pca_out: dict) -> None:
    scores = pd.DataFrame(pca_out["scores"])
    loadings = pd.DataFrame(pca_out["loadings"]).T
    var = pca_out["explained_variance_ratio"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: scatter of respondents on PC1 vs PC2
    palette = {
        "mainland": "#4E79A7",
        "non_mainland": "#E15759",
    }
    for grp, col in palette.items():
        sub = scores[scores.group == grp]
        ax1.scatter(sub.PC1, sub.PC2, s=60, alpha=0.75, color=col,
                    edgecolor="black", linewidth=0.5,
                    label=f"{grp.replace('_', '-')} (n={len(sub)})")
    ax1.axhline(0, color="gray", lw=0.8, ls=":")
    ax1.axvline(0, color="gray", lw=0.8, ls=":")
    ax1.set_xlabel(f"PC1  ({var[0]*100:.1f}% variance)")
    ax1.set_ylabel(f"PC2  ({var[1]*100:.1f}% variance)")
    ax1.set_title("Respondents on first two principal components\n"
                  "(Likert items reverse-coded so high = positive experience)",
                  fontsize=12)
    ax1.legend(loc="best")

    # Panel 2: loadings as arrows / bar chart
    order = loadings["PC1"].abs().sort_values(ascending=True).index
    y = np.arange(len(order))
    w = 0.4
    ax2.barh(y - w / 2, loadings.loc[order, "PC1"], w, color="#4E79A7",
             edgecolor="black", label="PC1 loading")
    ax2.barh(y + w / 2, loadings.loc[order, "PC2"], w, color="#F28E2B",
             edgecolor="black", label="PC2 loading")
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(order, fontsize=9)
    ax2.set_xlabel("loading")
    ax2.set_title(f"Component loadings\n"
                  f"cumulative variance: 2 PCs = {sum(var[:2])*100:.1f}%,  "
                  f"3 PCs = {sum(var)*100:.1f}%",
                  fontsize=12)
    ax2.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, "14_likert_pca.png")


# ---------------------------------------------------------------------------
# Figure 15: Q23 sentiment distribution by group + keyword comparison.
# ---------------------------------------------------------------------------
def plot_q23_sentiment_and_keywords(q23: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    # Panel 1: sentiment % by group
    # The saved JSON uses `orient="index"`, so rows sit in the outer dict.
    # pd.DataFrame(dict) treats outer keys as columns, so we transpose.
    ct = pd.DataFrame(q23["sentiment_counts"]).T
    order = ["positive", "neutral", "negative", "unscorable"]
    colors = {"positive": "#2E8B57", "neutral": "#BBBBBB",
              "negative": "#C0392B", "unscorable": "#EFEFEF"}
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    ct = ct.reindex(order).fillna(0)
    nm_pct = ct["non_mainland"] / max(1, ct["non_mainland"].sum()) * 100
    mn_pct = ct["mainland"] / max(1, ct["mainland"].sum()) * 100

    x = np.arange(len(order))
    w = 0.4
    for idx, lab in enumerate(order):
        ax1.bar(x[idx] - w / 2, nm_pct[lab], w,
                color=colors[lab], edgecolor="black",
                label="Non-mainland" if idx == 0 else None,
                hatch="//")
        ax1.bar(x[idx] + w / 2, mn_pct[lab], w,
                color=colors[lab], edgecolor="black",
                label="Mainland" if idx == 0 else None)
    for i, lab in enumerate(order):
        ax1.text(x[i] - w / 2, nm_pct[lab] + 1, f"{int(ct.loc[lab,'non_mainland'])}",
                 ha="center", va="bottom", fontsize=9)
        ax1.text(x[i] + w / 2, mn_pct[lab] + 1, f"{int(ct.loc[lab,'mainland'])}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(order, rotation=15, ha="right")
    ax1.set_ylabel("% of respondents in group")
    ax1.set_title("Q23 response sentiment by group\n"
                  "(hatched = non-mainland, solid = mainland; "
                  "count labels above bars)",
                  fontsize=11)
    fs = q23.get("fisher_neg_vs_not_neg", {})
    if fs.get("p") is not None:
        ax1.text(0.5, -0.22,
                 f"Fisher (negative vs not-negative, unscorable dropped): "
                 f"OR={fs['odds_ratio']}, p={fs['p']:.3f}",
                 transform=ax1.transAxes, ha="center", fontsize=9, style="italic")

    # Panel 2: top keywords comparison (union of top 12 from each group)
    nm_kw = pd.Series(q23["top_keywords_non_mainland"])
    mn_kw = pd.Series(q23["top_keywords_mainland"])
    union_kw = list(dict.fromkeys(
        list(nm_kw.head(12).index) + list(mn_kw.head(12).index)
    ))[:18]
    d = pd.DataFrame({
        "non_mainland": nm_kw.reindex(union_kw).fillna(0),
        "mainland": mn_kw.reindex(union_kw).fillna(0),
    })
    d = d.sort_values("non_mainland", ascending=True)
    y = np.arange(len(d))
    ax2.barh(y - 0.2, d["non_mainland"], 0.4, color="#E15759",
             edgecolor="black", label=f"Non-mainland")
    ax2.barh(y + 0.2, d["mainland"], 0.4, color="#4E79A7",
             edgecolor="black", label=f"Mainland")
    ax2.set_yticks(y)
    ax2.set_yticklabels(d.index, fontsize=10)
    ax2.set_xlabel("keyword frequency")
    ax2.set_title("Top Q23 keywords by group (union of top 12 from each)\n"
                  "Language-oriented words dominate non-mainland; "
                  "academic/social positives dominate mainland",
                  fontsize=11)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, "15_q23_sentiment_and_keywords.png")


# ---------------------------------------------------------------------------
# Figure 13: aware_services + year + gender by group.
# ---------------------------------------------------------------------------
def plot_categoricals_by_group(rowlevel: pd.DataFrame, categoricals: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: aware_services (pooled n=57)
    d = rowlevel.dropna(subset=["aware_services"]).copy()
    d["group"] = np.where(d.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(d.aware_services, d.group)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    res = categoricals.get("aware_services", {})
    annot = ""
    if "mannwhitney_p" in res:
        annot = (f"MW p = {res['mannwhitney_p']:.3f}, "
                 f"delta = {res['cliffs_delta']:+.2f}  (n = {res['n_non_mainland']+res['n_mainland']})")
    _grouped_percent_bars(
        axes[0], ct, ["No", "Somewhat", "Yes"],
        "Awareness of international-student support services",
        "Aware of support services?",
        f"Non-mainland (n={res.get('n_non_mainland', 0)})",
        f"Mainland (n={res.get('n_mainland', 0)})",
        annotation=annot,
    )

    # Panel 2: year of study (B-only)
    d = rowlevel.dropna(subset=["year"]).copy()
    d["group"] = np.where(d.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(d.year, d.group)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    res = categoricals.get("year", {})
    annot = ""
    if "mannwhitney_p" in res:
        annot = (f"MW p = {res['mannwhitney_p']:.3f}, "
                 f"delta = {res['cliffs_delta']:+.2f}  (later-batch, n = {res['n_non_mainland']+res['n_mainland']})")
    _grouped_percent_bars(
        axes[1], ct, ["Year 1", "Year 2", "Year 3"],
        "Year of study",
        "Year",
        f"Non-mainland (n={res.get('n_non_mainland', 0)})",
        f"Mainland (n={res.get('n_mainland', 0)})",
        annotation=annot,
    )

    # Panel 3: gender (B-only)
    d = rowlevel.dropna(subset=["gender"]).copy()
    d["group"] = np.where(d.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(d.gender, d.group)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    res = categoricals.get("gender", {})
    annot = ""
    if res.get("chi2_p") is not None:
        annot = f"chi2 p = {res['chi2_p']:.3f}  (later-batch, n = {res['n_non_mainland']+res['n_mainland']})"
    _grouped_percent_bars(
        axes[2], ct, ["Male", "Female", "Prefer not to say"],
        "Gender",
        "Gender",
        f"Non-mainland (n={res.get('n_non_mainland', 0)})",
        f"Mainland (n={res.get('n_mainland', 0)})",
        annotation=annot,
    )

    fig.suptitle("Other categorical variables by group", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "13_categoricals_by_group.png")


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
    plot_social_circle_by_group(rowlevel, results["categorical_by_group"])
    plot_integration_stack(agg, n_total)
    plot_theme_prevalence(results["interview_theme_tests"])
    plot_theme_burden(themes, results["interview_theme_burden"])
    plot_gpa_by_group_real(rowlevel, results["gpa_joint_tests"])
    plot_new_themes(results["interview_theme_tests"])
    plot_likert_by_group(results["likert_by_group"], question_texts)
    plot_opinion_differences(results["interview_theme_tests"])
    plot_categoricals_by_group(rowlevel, results["categorical_by_group"])
    plot_likert_pca(results["likert_pca"])
    plot_q23_sentiment_and_keywords(results["q23_text_analysis"])

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

"""Build a 13-slide matplotlib deck summarising the study.

Each slide is a 16:9 PNG at 1920x1080 saved to analysis/slides/. Text is
laid out with fig.text() for fine control; charts are embedded via add_axes.
Colours match the main analysis figures: mainland = #4E79A7, non-mainland =
#E15759.

Run from analysis/ (or via run_all.sh) after the data pipeline has produced
data/results.json and related CSVs.
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
SLIDES = ROOT / "slides"
SLIDES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Study metadata (edit these to re-brand the deck)
# ---------------------------------------------------------------------------
STUDY_TITLE_LINE1 = "Social Integration and the GPA Gap"
STUDY_TITLE_LINE2 = "at HKUST(GZ)"
STUDY_SUBTITLE = (
    "A Mixed-Methods Study of International and Mainland\n"
    "Undergraduate Experience"
)
AUTHOR_LINE = "Division of Social Science  ·  HKUST(GZ)  ·  April 2026"
FOOTER_LEFT = "HKUST(GZ) Social Integration Study  ·  April 2026"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
TITLE_COLOR = "#1F3A5F"
SUBTITLE_COLOR = "#555555"
ACCENT_MN = "#4E79A7"     # mainland
ACCENT_NM = "#E15759"     # non-mainland
ACCENT_HIGHLIGHT = "#F28E2B"
GRAY_TEXT = "#333333"
MUTED = "#888888"
BG = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 120,
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

SLIDE_SIZE = (16, 9)
GPA_BINS_6 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.0-3.3", "3.3-3.7", "Above 3.7"]
GPA_BINS_5 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.3-3.7", "Above 3.7"]
LIKERT_ORDER = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]
LIKERT_CODE = {lab: i + 1 for i, lab in enumerate(LIKERT_ORDER)}
MAINLAND = "Domestic Chinese student"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def new_slide(num: int, total: int) -> plt.Figure:
    fig = plt.figure(figsize=SLIDE_SIZE)
    fig.patch.set_facecolor(BG)
    fig.text(0.02, 0.02, FOOTER_LEFT, ha="left", va="bottom",
             fontsize=10, color=MUTED)
    fig.text(0.98, 0.02, f"{num} / {total}", ha="right", va="bottom",
             fontsize=10, color=MUTED)
    return fig


def add_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.05, 0.91, title, fontsize=30, fontweight="bold", color=TITLE_COLOR)
    fig.text(0.05, 0.875, "", fontsize=1)  # spacer
    # Horizontal accent line under the title
    ax_line = fig.add_axes([0.05, 0.865, 0.90, 0.002])
    ax_line.axhline(0.5, color=ACCENT_HIGHLIGHT, lw=2.5)
    ax_line.set_axis_off()
    if subtitle:
        fig.text(0.05, 0.83, subtitle, fontsize=16, color=SUBTITLE_COLOR,
                 style="italic")


def add_bullets(fig: plt.Figure, bullets: list[str], x: float = 0.05,
                y_top: float = 0.78, dy: float = 0.075,
                fontsize: int = 16) -> None:
    for i, line in enumerate(bullets):
        fig.text(x, y_top - i * dy, line, fontsize=fontsize,
                 color=GRAY_TEXT, verticalalignment="top")


def save_slide(fig: plt.Figure, num: int, tag: str) -> None:
    fig.savefig(SLIDES / f"{num:02d}_{tag}.png", bbox_inches="tight",
                pad_inches=0.3, facecolor=BG)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load computed data once
# ---------------------------------------------------------------------------
def load_all() -> dict:
    with (DATA / "results.json").open() as f:
        r = json.load(f)
    with (DATA / "gpa_normalization.json").open() as f:
        norm = json.load(f)
    rowlevel = pd.read_csv(DATA / "survey_rowlevel.csv")
    imp_cmp = pd.read_csv(DATA / "gpa_imputation_comparison.csv")
    imputed = pd.read_csv(DATA / "gpa_imputed_primary.csv")
    themes = pd.read_csv(DATA / "interview_themes.csv")
    return {
        "results": r,
        "norm": norm,
        "rowlevel": rowlevel,
        "imp_cmp": imp_cmp,
        "imputed": imputed,
        "themes": themes,
    }


# ---------------------------------------------------------------------------
# Individual slides
# ---------------------------------------------------------------------------
def slide_01_title(num: int, total: int) -> None:
    fig = new_slide(num, total)
    # Big centered title
    fig.text(0.5, 0.66, STUDY_TITLE_LINE1, fontsize=44, fontweight="bold",
             ha="center", color=TITLE_COLOR)
    fig.text(0.5, 0.58, STUDY_TITLE_LINE2, fontsize=44, fontweight="bold",
             ha="center", color=TITLE_COLOR)
    # Accent bar
    ax_bar = fig.add_axes([0.35, 0.52, 0.30, 0.004])
    ax_bar.axhline(0.5, color=ACCENT_HIGHLIGHT, lw=3)
    ax_bar.set_axis_off()
    # Subtitle
    fig.text(0.5, 0.44, STUDY_SUBTITLE, fontsize=20, ha="center",
             color=SUBTITLE_COLOR, style="italic")
    # Author + date + sample sizes
    fig.text(0.5, 0.28, AUTHOR_LINE, fontsize=18, ha="center", color=GRAY_TEXT)
    fig.text(0.5, 0.22, "57 survey respondents  ·  21 interview transcripts",
             fontsize=14, ha="center", color=MUTED)
    save_slide(fig, num, "title")


def slide_02_question(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Research question & data")
    bullets = [
        "•  Why do international students at HKUST(GZ) tend to record",
        "    lower GPAs than their mainland Chinese peers?",
        "",
        "•  Mixed-methods design:",
        "      –  Survey (n = 57): 20 questions incl. GPA, 14 Likert items",
        "      –  Interviews (n = 21): international + mainland + Taiwanese students",
        "",
        "•  Status composition of the survey sample:",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.055, fontsize=15)

    # Small status bar chart in the bottom right
    rl = d["rowlevel"]
    counts = rl["status"].value_counts()
    order = ["Domestic Chinese student", "International student",
             "Hong Kong, Macau, or Taiwan student"]
    counts = counts.reindex(order)
    ax = fig.add_axes([0.55, 0.10, 0.40, 0.45])
    colors = [ACCENT_MN, ACCENT_NM, ACCENT_HIGHLIGHT]
    bars = ax.barh(["Mainland Chinese", "International", "HK / Macau / TW"],
                   counts.values, color=colors, edgecolor="black")
    for b, v in zip(bars, counts.values):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"n = {int(v)}",
                va="center", fontsize=12)
    ax.set_xlim(0, counts.max() * 1.2)
    ax.set_xlabel("respondents")
    ax.set_title("Sample composition (survey, N = 57)", fontsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "question")


def slide_03_survey_error(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "A critical survey design flaw",
              subtitle="The GPA question skipped the 3.0 – 3.3 bin")
    bullets = [
        "•  The options read:  Below 2.3  ·  2.3 – 2.7  ·  2.7 – 3.0  ·  3.3 – 3.7  ·  Above 3.7",
        "",
        "•  Students whose true GPA sits in 3.0 – 3.3 were forced into a neighbour,",
        "    almost all into 3.3 – 3.7.",
        "",
        "•  The tell-tale sign: in the raw data, the 2.7 – 3.0 bin holds only 21 %,",
        "    but the next available bin (3.3 – 3.7) jumps to 60 %.",
        "",
        "•  No smooth unimodal GPA distribution produces a jump that large —",
        "    the 3.3 – 3.7 bar is really the merged interval [3.0, 3.7).",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.06, fontsize=15)

    # Raw (pre-imputation) bar chart highlighting the jump
    ax = fig.add_axes([0.52, 0.10, 0.43, 0.40])
    rl = d["rowlevel"]
    counts = rl["gpa"].value_counts().reindex(GPA_BINS_5).fillna(0)
    bars = ax.bar(GPA_BINS_5, counts.values,
                  color=[ACCENT_MN] * 5, edgecolor="black")
    # Highlight the suspicious 3.3-3.7 bar
    bars[3].set_facecolor(ACCENT_NM)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{int(v)}",
                ha="center", fontsize=11)
    ax.set_title("Raw GPA distribution as collected (n = 57)", fontsize=12)
    ax.set_ylabel("respondents")
    ax.set_xticklabels(GPA_BINS_5, rotation=20, ha="right")
    # Annotate the missing bin with an arrow
    ax.annotate("3.3 – 3.7 bar is inflated:\n" + r"really $[3.0, 3.7)$",
                xy=(3, counts.values[3]),
                xytext=(1.1, counts.values[3] * 0.85),
                fontsize=11, color=ACCENT_NM,
                arrowprops=dict(arrowstyle="->", color=ACCENT_NM, lw=1.5))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "survey_error")


def slide_04_math_fix(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Fixing the gap with truncated-normal MLE",
              subtitle="A principled way to recover the missing bin")
    norm = d["norm"]
    tn = norm["trunc_normal_fit"]
    bt = norm["beta_fit"]
    gof = d["results"]["gpa_goodness_of_fit"]

    bullets = [
        "Method",
        "  •  Treat reported 3.3 – 3.7 as the merged interval [3.0, 3.7).",
        "  •  Fit a truncated normal on GPA ∈ [0, 4.3] via multinomial MLE over",
        "      the 5 observed-as-surveyed bins.",
        r"      $\max_{\mu, \sigma} \; \sum_i n_i \log P(\text{bin}_i \mid \mu, \sigma)$",
        "  •  Split [3.0, 3.7) into 3.0 – 3.3 and 3.3 – 3.7 using the fitted CDF.",
        "",
        "Validation",
        f"  •  Fit: μ = {tn['mu']:.3f},  σ = {tn['sigma']:.3f}  (AIC = {tn['aic']:.2f}).",
        f"  •  Goodness-of-fit χ² = {gof['chi2']}, dof = {gof['dof']}, p = {gof['p_chi2']}",
        "      —  cannot reject the truncated-normal shape.",
        f"  •  Sensitivity: beta fit (α = {bt['alpha']:.2f}, β = {bt['beta']:.2f}) and",
        "      log-linear interpolation produce consistent splits.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.052, fontsize=14)

    # Density-fit overlay in bottom right
    mu, sigma = tn["mu"], tn["sigma"]
    a, b = (0 - mu) / sigma, (4.3 - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    xs = np.linspace(0, 4.3, 400)
    density = dist.pdf(xs)
    sub = d["imputed"].set_index("bin").reindex(GPA_BINS_6)
    bin_midpts = [2.15, 2.5, 2.85, 3.15, 3.5, 3.85]
    widths = [2.3, 0.4, 0.3, 0.3, 0.4, 0.6]
    densities = sub["count"].values / (sub["count"].sum() * np.array(widths))
    ax = fig.add_axes([0.58, 0.08, 0.37, 0.36])
    ax.bar(bin_midpts, densities, width=widths, align="center",
           alpha=0.45, edgecolor="black", color=ACCENT_MN,
           label="observed (imputed)")
    ax.plot(xs, density, color=ACCENT_NM, lw=2.5,
            label=f"trunc-normal  μ={mu:.2f}, σ={sigma:.2f}")
    ax.axvspan(3.0, 3.3, color="gold", alpha=0.25, label="imputed 3.0 – 3.3 bin")
    ax.set_xlim(2.0, 4.1)
    ax.set_xlabel("GPA")
    ax.set_ylabel("density")
    ax.set_title("Fitted density on observed histogram", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "math_fix")


def slide_05_imputation_result(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "The corrected GPA distribution",
              subtitle="~ 44 % of the inflated 3.3 – 3.7 bin reassigned to 3.0 – 3.3")
    imputed = d["imputed"].set_index("bin")["count"].reindex(GPA_BINS_6).fillna(0)

    # Two side-by-side bars: before vs after
    ax1 = fig.add_axes([0.08, 0.18, 0.38, 0.55])
    rl = d["rowlevel"]
    pre = rl["gpa"].value_counts().reindex(GPA_BINS_5).fillna(0)
    pre_labels = GPA_BINS_5
    bars1 = ax1.bar(pre_labels, pre.values, color=ACCENT_MN, edgecolor="black")
    bars1[3].set_facecolor(ACCENT_NM)
    for b, v in zip(bars1, pre.values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{int(v)}",
                 ha="center", fontsize=11)
    ax1.set_title("Before: raw 5-bin (red = inflated)", fontsize=12)
    ax1.set_xticklabels(pre_labels, rotation=25, ha="right")
    ax1.set_ylabel("respondents")
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    ax2 = fig.add_axes([0.52, 0.18, 0.44, 0.55])
    bars2 = ax2.bar(GPA_BINS_6, imputed.values, color=ACCENT_MN, edgecolor="black")
    bars2[GPA_BINS_6.index("3.0-3.3")].set_facecolor("gold")
    bars2[GPA_BINS_6.index("3.0-3.3")].set_edgecolor("black")
    for b, v in zip(bars2, imputed.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                 ha="center", fontsize=11)
    ax2.set_title("After: imputed 6-bin (gold = new 3.0 – 3.3 bin)",
                  fontsize=12)
    ax2.set_xticklabels(GPA_BINS_6, rotation=25, ha="right")
    ax2.set_ylabel("respondents (fractional allowed)")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    # Summary stats underneath
    desc = d["results"]["gpa_descriptive"]
    stats_line = (
        f"Post-imputation:  N = {int(desc['n'])}  ·  "
        f"mean GPA (midpoint) = {desc['mean_gpa_midpoint']}  ·  "
        f"SD = {desc['sd_gpa_midpoint']}  ·  "
        f"median = {desc['median']}"
    )
    fig.text(0.5, 0.10, stats_line, ha="center", fontsize=14, color=GRAY_TEXT)

    save_slide(fig, num, "imputation_result")


def slide_06_gpa_by_group(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "The GPA gap: real but modest",
              subtitle="Direction matches hypothesis; n = 57 is under-powered")
    joint = d["results"]["gpa_joint_tests"]
    desc_gr = d["results"]["gpa_descriptive_by_group"]

    bullets = [
        f"•  Non-mainland (intl + HK/Macau/TW, n = {joint['n_non_mainland']}) mean = "
        f"{desc_gr['non_mainland']['mean_gpa_midpoint']:.2f}",
        f"•  Mainland Chinese (n = {joint['n_mainland']}) mean = "
        f"{desc_gr['mainland']['mean_gpa_midpoint']:.2f}",
        f"•  Difference: {desc_gr['non_mainland']['mean_gpa_midpoint'] - desc_gr['mainland']['mean_gpa_midpoint']:+.2f}",
        "",
        f"•  Mann-Whitney U:  two-sided p = {joint['mannwhitney_p_two_sided']:.3f},",
        f"                              one-sided p = {joint['mannwhitney_p_less']:.3f}",
        f"•  Cliff's δ = {joint['cliffs_delta']}  (small – medium effect)",
        f"•  Permutation (10k iter):  one-sided p = {joint['permutation_p_less']:.3f}",
        "",
        f"•  HK / Macau / Taiwan subgroup (n = {desc_gr['hkmt_only']['n']}) is the",
        f"    lowest performer: mean = {desc_gr['hkmt_only']['mean_gpa_midpoint']:.2f},",
        f"    {desc_gr['hkmt_only']['share_below_3']*100:.0f} % below 3.0.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.053, fontsize=14)

    # Grouped bars for distribution
    rl = d["rowlevel"]
    mainland = rl[rl.status == MAINLAND]["gpa"].value_counts(normalize=True).reindex(GPA_BINS_5).fillna(0)
    non_mainland = rl[rl.status != MAINLAND]["gpa"].value_counts(normalize=True).reindex(GPA_BINS_5).fillna(0)
    ax = fig.add_axes([0.52, 0.12, 0.44, 0.55])
    x = np.arange(len(GPA_BINS_5))
    w = 0.4
    b1 = ax.bar(x - w / 2, non_mainland.values * 100, w, label=f"Non-mainland (n={joint['n_non_mainland']})",
                color=ACCENT_NM, edgecolor="black")
    b2 = ax.bar(x + w / 2, mainland.values * 100, w, label=f"Mainland (n={joint['n_mainland']})",
                color=ACCENT_MN, edgecolor="black")
    for b, v in list(zip(b1, non_mainland.values)) + list(zip(b2, mainland.values)):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1, f"{v*100:.0f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(GPA_BINS_5, rotation=20, ha="right")
    ax.set_ylabel("% of group")
    ax.set_title("GPA distribution by group (raw, pre-imputation)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "gpa_by_group")


def slide_07_likert_diff(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Where the groups actually differ",
              subtitle="Two Likert items separate non-mainland from mainland")
    likert = pd.DataFrame(d["results"]["likert_by_group"]["per_item"])
    likert = likert.sort_values("mean_diff")
    top = pd.concat([likert.head(3), likert.tail(3)]).drop_duplicates("question_id")

    bullets = [
        "•  interact_chinese:  non-mainland 3.10  vs  mainland 4.03",
        "      (Cliff's δ = −0.41,  raw p = 0.007,  BH p = 0.10)",
        "",
        "•  excluded_study_group:  non-mainland 2.85  vs  mainland 2.11",
        "      (Cliff's δ = +0.37,  raw p = 0.015,  BH p = 0.10)",
        "",
        "•  Both directions match the interview evidence exactly:",
        "    non-mainland students interact less with Chinese peers",
        "    AND report more study-group exclusion.",
        "",
        "•  Neither item passes BH at α = 0.05 with n = 57,  but the",
        "    effect sizes are non-trivial and robust.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.058, fontsize=14)

    # Diverging bar chart of mean differences
    ax = fig.add_axes([0.52, 0.10, 0.44, 0.55])
    likert_sorted = likert.sort_values("mean_diff")
    y = np.arange(len(likert_sorted))
    colors = [ACCENT_NM if d > 0 else ACCENT_MN for d in likert_sorted["mean_diff"]]
    ax.barh(y, likert_sorted["mean_diff"], color=colors, edgecolor="black")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(likert_sorted["question_id"], fontsize=9)
    ax.set_xlabel("non-mainland mean  −  mainland mean  (1–5 scale)")
    ax.set_title("Mean Likert difference by group", fontsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(-1.2, 1.2)

    save_slide(fig, num, "likert_diff")


def slide_08_interviews(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Interviews: an unambiguous signal",
              subtitle="Joint status × opinion data is strongest here")
    themes = pd.DataFrame(d["results"]["interview_theme_tests"]["per_theme"])
    burden = d["results"]["interview_theme_burden"]
    sig = themes[themes.fisher_p_bh < 0.05].sort_values("fisher_p_bh")

    bullets = [
        f"•  {len(themes)} binary themes coded across 21 interview respondents.",
        "",
        "•  Four themes separate international from mainland respondents",
        "    with BH-adjusted p < 0.05:",
    ]
    for _, row in sig.head(4).iterrows():
        bullets.append(
            f"      –  {row['theme'].replace('_', ' ')}: "
            f"{row['intl_pct']*100:.0f} %  vs  {row['mnl_pct']*100:.0f} %  "
            f"(BH p = {row['fisher_p_bh']:.3f})"
        )
    bullets += [
        "",
        f"•  Theme burden permutation test: international = "
        f"{burden['intl_mean_burden']}  vs  mainland = {burden['mnl_mean_burden']}",
        f"    out of {len(themes)}  ·  p = {burden['perm_p']:.4f}",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.049, fontsize=14)

    # Boxplot of theme burden on the right
    t = d["themes"].copy()
    theme_cols = [c for c in t.columns if c not in ("interview_id", "group")]
    t["burden"] = t[theme_cols].sum(axis=1)
    ax = fig.add_axes([0.60, 0.10, 0.35, 0.50])
    sns.boxplot(
        data=t, x="group", y="burden", ax=ax, hue="group",
        palette={"international": ACCENT_NM, "mainland": ACCENT_MN,
                 "hkmt": ACCENT_HIGHLIGHT},
        width=0.45, legend=False,
    )
    sns.stripplot(data=t, x="group", y="burden", ax=ax,
                  color="black", size=7, jitter=0.12)
    ax.set_title(f"Theme burden (0 – {len(theme_cols)})", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("# of themes reported")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "interviews")


def slide_09_keywords(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "What students write, unprompted",
              subtitle="Q23 open-ended keyword contrast tells the clearest story")
    q23 = d["results"]["q23_text_analysis"]
    nm_kw = pd.Series(q23["top_keywords_non_mainland"]).head(10)
    mn_kw = pd.Series(q23["top_keywords_mainland"]).head(10)

    # Bullets
    sc = pd.DataFrame(q23["sentiment_counts"]).T
    nm_neg = sc.loc["negative", "non_mainland"] if "non_mainland" in sc.columns else 0
    mn_neg = sc.loc["negative", "mainland"] if "mainland" in sc.columns else 0
    nm_scored = int(sc[["non_mainland"]].drop("unscorable", errors="ignore").sum().iloc[0])
    mn_scored = int(sc[["mainland"]].drop("unscorable", errors="ignore").sum().iloc[0])
    bullets = [
        "•  56 / 57 respondents wrote a free-text answer.",
        "",
        "•  Non-mainland vocabulary  (language + research contexts):",
        "      " + "  ·  ".join(nm_kw.head(6).index),
        "",
        "•  Mainland vocabulary  (positive social + academic):",
        "      " + "  ·  ".join(mn_kw.head(6).index),
        "",
        f"•  Among scorable responses, non-mainland are ~{int(round(100*nm_neg/max(1,nm_scored)))}%",
        f"    negative  vs  ~{int(round(100*mn_neg/max(1,mn_scored)))}% for mainland.",
        "",
        "•  Different vocabularies, not different tones only:  one group",
        "    talks about language, the other about friends.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.050, fontsize=14)

    # Side-by-side keyword bars
    ax = fig.add_axes([0.55, 0.10, 0.42, 0.55])
    union = list(dict.fromkeys(list(nm_kw.index) + list(mn_kw.index)))[:14]
    nm_vals = nm_kw.reindex(union).fillna(0)
    mn_vals = mn_kw.reindex(union).fillna(0)
    y = np.arange(len(union))
    ax.barh(y - 0.2, nm_vals.values, 0.4, color=ACCENT_NM,
            edgecolor="black", label="Non-mainland")
    ax.barh(y + 0.2, mn_vals.values, 0.4, color=ACCENT_MN,
            edgecolor="black", label="Mainland")
    ax.set_yticks(y)
    ax.set_yticklabels(union, fontsize=10)
    ax.set_xlabel("keyword frequency")
    ax.set_title("Top Q23 keywords (union)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "keywords")


def slide_10_null_finding(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "A striking null: services aren't reaching the target",
              subtitle="Awareness of international-student services is identical")
    cats = d["results"]["categorical_by_group"]
    aw = cats["aware_services"]

    bullets = [
        "•  The question:  ‘I am aware of support services available to",
        "    international students at HKUST(GZ).’",
        "",
        "•  Expectation:  the audience these services are FOR",
        "    (non-mainland) should be more aware than mainland students.",
        "",
        f"•  Actual result (n = {aw['n_non_mainland'] + aw['n_mainland']}):",
        f"      –  Non-mainland mean code = {aw.get('mean_code_non_mainland')}",
        f"      –  Mainland      mean code = {aw.get('mean_code_mainland')}",
        f"      –  Cliff's δ = {aw.get('cliffs_delta', 0):+.2f}, Mann-Whitney p = {aw.get('mannwhitney_p')}",
        "",
        "•  Groups are statistically indistinguishable.",
        "    The outreach is not visibly reaching the target audience.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.049, fontsize=14)

    # Bar chart of aware_services by group
    rl = d["rowlevel"].dropna(subset=["aware_services"]).copy()
    rl["group"] = np.where(rl.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(rl.aware_services, rl.group).reindex(["No", "Somewhat", "Yes"]).fillna(0)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    nm_pct = ct["non_mainland"] / max(1, ct["non_mainland"].sum()) * 100
    mn_pct = ct["mainland"] / max(1, ct["mainland"].sum()) * 100
    ax = fig.add_axes([0.58, 0.12, 0.37, 0.50])
    x = np.arange(3)
    w = 0.4
    ax.bar(x - w / 2, nm_pct.values, w, color=ACCENT_NM, edgecolor="black",
           label=f"Non-mainland (n={aw['n_non_mainland']})")
    ax.bar(x + w / 2, mn_pct.values, w, color=ACCENT_MN, edgecolor="black",
           label=f"Mainland (n={aw['n_mainland']})")
    for i, (nm, mn) in enumerate(zip(nm_pct.values, mn_pct.values)):
        ax.text(x[i] - w / 2, nm + 1, f"{nm:.0f}%", ha="center", fontsize=10)
        ax.text(x[i] + w / 2, mn + 1, f"{mn:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["No", "Somewhat", "Yes"])
    ax.set_ylabel("% of group")
    ax.set_title("Aware of support services?", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "null_finding")


def slide_11_pca(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "One underlying integration factor",
              subtitle="PCA on the 14 Likert items (reverse-coded where needed)")
    pca = d["results"]["likert_pca"]
    var = pca["explained_variance_ratio"]

    bullets = [
        f"•  PC1 = {var[0]*100:.1f} %   PC2 = {var[1]*100:.1f} %   PC3 = {var[2]*100:.1f} %  of variance.",
        f"    Two PCs capture {sum(var[:2])*100:.1f} %  of the Likert structure.",
        "",
        "•  PC1 is a clean ‘general integration’ factor:",
        "      –  belonging  →  high PC1",
        "      –  motivation  →  high PC1",
        "      –  better grades  →  high PC1",
        "      –  less exclusion  →  high PC1",
        "      –  less language difficulty  →  high PC1",
        "",
        "•  Practical implication: you can compress the 14 Likert items",
        "    into a single ‘integration score’ without much loss.",
        "    Respondents who feel they belong ALSO feel the other",
        "    four things — these are not independent struggles.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.050, fontsize=14)

    # PCA scatter on the right
    scores = pd.DataFrame(pca["scores"])
    ax = fig.add_axes([0.55, 0.10, 0.40, 0.55])
    palette = {"mainland": ACCENT_MN, "non_mainland": ACCENT_NM}
    for grp, col in palette.items():
        sub = scores[scores.group == grp]
        ax.scatter(sub.PC1, sub.PC2, s=60, alpha=0.75, color=col,
                   edgecolor="black", linewidth=0.5,
                   label=f"{grp.replace('_', '-')} (n={len(sub)})")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(f"PC1  ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2  ({var[1]*100:.1f}%)")
    ax.set_title("Respondents on PC1 × PC2", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    save_slide(fig, num, "pca")


def slide_12_conclusions(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Conclusions")
    joint = d["results"]["gpa_joint_tests"]
    desc_gr = d["results"]["gpa_descriptive_by_group"]
    burden = d["results"]["interview_theme_burden"]

    n_themes = len(pd.DataFrame(d["results"]["interview_theme_tests"]["per_theme"]))
    bullets = [
        "1.  The GPA gap is real but modest.",
        f"     Non-mainland mean {desc_gr['non_mainland']['mean_gpa_midpoint']:.2f}  vs  "
        f"mainland {desc_gr['mainland']['mean_gpa_midpoint']:.2f}  "
        f"(Cliff's δ = {joint['cliffs_delta']}).  Under-powered at n = 57.",
        "",
        "2.  HK / Macau / Taiwan is the lowest-performing subgroup",
        f"     (mean {desc_gr['hkmt_only']['mean_gpa_midpoint']:.2f}), not internationals proper — "
        "‘international’ is not a monolith.",
        "",
        "3.  The mechanisms are language + exclusion, not ability.",
        "     Survey, interviews, and open-ended text all converge on:  less",
        "     interaction with Chinese peers, more study-group exclusion,",
        "     Mandarin-dominant clubs, no Gaokao STEM preparation.",
        "",
        "4.  Institutional support is not reaching its audience.",
        "     Awareness of international-student services is statistically",
        "     identical across groups.",
        "",
        "5.  The friction is STRUCTURAL, not individual.",
        f"     Theme-burden permutation test:  international reports "
        f"{burden['intl_mean_burden']:.1f} of {n_themes}",
        f"     themes  vs  mainland {burden['mnl_mean_burden']:.1f}  "
        f"(p = {burden['perm_p']:.3f}).",
    ]
    add_bullets(fig, bullets, y_top=0.79, dy=0.040, fontsize=13)

    save_slide(fig, num, "conclusions")


def slide_13_recommendations(num: int, total: int, d: dict) -> None:
    fig = new_slide(num, total)
    add_title(fig, "Recommendations for follow-up")
    bullets = [
        "Survey design",
        "  •   Fix the GPA option set — include 3.0 – 3.3.",
        "  •   Export row-level data from the start (not PDF summaries).",
        "  •   Add a UCUG-vs-UFUG self-reported GPA question to test",
        "       the course-type asymmetry directly.",
        "",
        "Sample size",
        "  •   Target n ≈ 25 – 30 per group for Cliff's δ ≈ 0.2 power.",
        "  •   Oversample HK / Macau / Taiwan — currently n = 7 is too",
        "       small to estimate the sub-group effect with any precision.",
        "",
        "Institutional actions",
        "  •   Language-paired study groups or buddy scheme.",
        "  •   Clubs / RC events with a bilingual (or English-default) policy.",
        "  •   Redesign international-student communications:  non-Chinese",
        "       subject lines, separate announcement channels.",
        "  •   Active referral to support services, not passive availability.",
    ]
    add_bullets(fig, bullets, y_top=0.78, dy=0.044, fontsize=14)
    save_slide(fig, num, "recommendations")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    d = load_all()
    slides = [
        slide_01_title,
        slide_02_question,
        slide_03_survey_error,
        slide_04_math_fix,
        slide_05_imputation_result,
        slide_06_gpa_by_group,
        slide_07_likert_diff,
        slide_08_interviews,
        slide_09_keywords,
        slide_10_null_finding,
        slide_11_pca,
        slide_12_conclusions,
        slide_13_recommendations,
    ]
    total = len(slides)
    for i, fn in enumerate(slides, 1):
        # slide_01_title doesn't take data; dispatch accordingly
        if fn is slide_01_title:
            fn(i, total)
        else:
            fn(i, total, d)

    # Verification
    pngs = sorted(SLIDES.glob("*.png"))
    print("=" * 70)
    print("06_slides.py  --  verification")
    print("=" * 70)
    print(f"Wrote {len(pngs)} slides to {SLIDES}:")
    for p in pngs:
        print(f"  {p.name:40s} {p.stat().st_size/1024:7.1f} KB")


if __name__ == "__main__":
    main()

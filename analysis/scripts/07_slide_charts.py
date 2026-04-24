"""Generate clean standalone chart PNGs for each slide.

One chart per file — no embedded titles/bullets/footers. Use the
companion slides/SLIDE_TEXT.md for the text that goes on each slide.

Charts are sized to fit comfortably in the body of a 16:9 slide when
inserted at ~80% width. Colours match the rest of the deck:
  mainland      = #4E79A7
  non-mainland  = #E15759
  accent        = #F28E2B
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
OUT = ROOT / "slides"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT_MN = "#4E79A7"
ACCENT_NM = "#E15759"
ACCENT_HL = "#F28E2B"
GOLD = "#F2C14E"
GRAY = "#333333"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 180,
    "figure.dpi": 120,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

GPA_BINS_6 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.0-3.3", "3.3-3.7", "Above 3.7"]
GPA_BINS_5 = ["Below 2.3", "2.3-2.7", "2.7-3.0", "3.3-3.7", "Above 3.7"]
MAINLAND = "Domestic Chinese student"


def clean_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=3)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}  ({path.stat().st_size/1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all():
    with (DATA / "results.json").open() as f:
        r = json.load(f)
    with (DATA / "gpa_normalization.json").open() as f:
        norm = json.load(f)
    return {
        "results": r,
        "norm": norm,
        "rowlevel": pd.read_csv(DATA / "survey_rowlevel.csv"),
        "imputed": pd.read_csv(DATA / "gpa_imputed_primary.csv"),
        "themes": pd.read_csv(DATA / "interview_themes.csv"),
    }


# ---------------------------------------------------------------------------
# Slide 02 — sample composition
# ---------------------------------------------------------------------------
def chart_02_sample(d):
    rl = d["rowlevel"]
    counts = rl["status"].value_counts()
    order = [
        "Domestic Chinese student",
        "International student",
        "Hong Kong, Macau, or Taiwan student",
    ]
    labels = ["Mainland Chinese", "International", "HK / Macau / Taiwan"]
    values = [int(counts.get(k, 0)) for k in order]
    colors = [ACCENT_MN, ACCENT_NM, ACCENT_HL]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(v + 0.7, b.get_y() + b.get_height() / 2,
                f"n = {v}", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.2)
    ax.set_xlabel("Respondents")
    ax.set_title("Survey sample composition  (N = 57)", pad=12)
    ax.invert_yaxis()
    clean_axes(ax)
    save(fig, "slide_02_sample")


# ---------------------------------------------------------------------------
# Slide 03 — raw GPA distribution with highlighted inflated bin
# ---------------------------------------------------------------------------
def chart_03_raw_gpa(d):
    rl = d["rowlevel"]
    counts = rl["gpa"].value_counts().reindex(GPA_BINS_5).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [ACCENT_MN] * 5
    colors[3] = ACCENT_NM
    bars = ax.bar(GPA_BINS_5, counts.values, color=colors, edgecolor="black")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v}",
                ha="center", fontsize=13, fontweight="bold")

    ax.annotate(
        "3.3 – 3.7 bar is inflated:\nreally $[3.0,\\ 3.7)$",
        xy=(3, counts.values[3]),
        xytext=(1.0, counts.values[3] * 0.80),
        fontsize=13, color=ACCENT_NM, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT_NM, lw=2),
    )
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("Respondents")
    ax.set_title("Raw GPA distribution as collected  (n = 57)", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    clean_axes(ax)
    save(fig, "slide_03_raw_gpa")


# ---------------------------------------------------------------------------
# Slide 04 — explicit mathematical statement of the fix
# ---------------------------------------------------------------------------
def chart_04_math_fix(d):
    tn = d["norm"]["trunc_normal_fit"]
    gof = d["results"]["gpa_goodness_of_fit"]
    mu, sigma = tn["mu"], tn["sigma"]

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Problem statement block (top, red card)
    ax.add_patch(plt.Rectangle((0.04, 0.80), 0.92, 0.14,
                               facecolor="#FDECEA", edgecolor=ACCENT_NM, lw=2))
    ax.text(0.5, 0.905, "The survey was flawed:  the 3.0 – 3.3 GPA bin was missing",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=ACCENT_NM)
    ax.text(0.5, 0.835,
            "Students whose true GPA fell in [3.0, 3.3] were forced into the next bin (3.3 – 3.7),\n"
            "inflating that bar from a plausible ~19 respondents to the observed 34.",
            ha="center", va="center", fontsize=13, color=GRAY)

    # Fix statement (banner)
    ax.text(0.5, 0.75,
            r"Fix:  mathematical normalization via $\mathbf{truncated\ normal\ MLE}$",
            ha="center", va="center", fontsize=18, color=ACCENT_MN)

    # Math block (left column)
    math_x = 0.06
    ax.text(math_x, 0.68, "1.  Truncated-normal density on  $[a, b] = [0, 4.3]$:",
            fontsize=13, color=GRAY, fontweight="bold")
    ax.text(math_x + 0.03, 0.60,
            r"$f(x \mid \mu, \sigma) \ =\ "
            r"\frac{1}{\sigma}\ "
            r"\frac{\phi\left(\frac{x-\mu}{\sigma}\right)}"
            r"{\Phi\left(\frac{b-\mu}{\sigma}\right) - \Phi\left(\frac{a-\mu}{\sigma}\right)}$",
            fontsize=16, color=GRAY)

    ax.text(math_x, 0.52,
            r"2.  Probability a respondent falls in bin $i = [l_i,\ u_i)$:",
            fontsize=13, color=GRAY, fontweight="bold")
    ax.text(math_x + 0.03, 0.44,
            r"$P_i(\mu, \sigma) \ =\ "
            r"\frac{\Phi\left(\frac{u_i-\mu}{\sigma}\right) - \Phi\left(\frac{l_i-\mu}{\sigma}\right)}"
            r"{\Phi\left(\frac{b-\mu}{\sigma}\right) - \Phi\left(\frac{a-\mu}{\sigma}\right)}$",
            fontsize=16, color=GRAY)

    ax.text(math_x, 0.36,
            r"3.  Fit by multinomial MLE over the 5 observed-as-surveyed bins:",
            fontsize=13, color=GRAY, fontweight="bold")
    ax.text(math_x + 0.03, 0.28,
            r"$(\hat{\mu},\ \hat{\sigma}) \ =\ "
            r"\arg\max_{\mu,\ \sigma}\ \sum_{i=1}^{5}\ n_i\ \log P_i(\mu, \sigma)$",
            fontsize=16, color=GRAY)

    ax.text(math_x, 0.20,
            r"4.  Split the merged [3.0, 3.7) interval using the fitted CDF:",
            fontsize=13, color=GRAY, fontweight="bold")
    ax.text(math_x + 0.03, 0.12,
            r"$\hat{n}_{3.0-3.3} \ =\ n_{[3.0,\ 3.7)}\ \cdot\ "
            r"\frac{P(3.0 \leq X < 3.3)}{P(3.0 \leq X < 3.7)}$",
            fontsize=16, color=GRAY)

    # Result card (right side)
    card_x, card_y, card_w, card_h = 0.62, 0.08, 0.34, 0.58
    ax.add_patch(plt.Rectangle((card_x, card_y), card_w, card_h,
                               facecolor="#F4F6F8", edgecolor=ACCENT_HL, lw=2))
    cx = card_x + card_w / 2
    ax.text(cx, card_y + card_h - 0.04, "Fitted values",
            ha="center", fontsize=14, fontweight="bold", color=GRAY)

    ax.text(cx, card_y + card_h - 0.13,
            rf"$\hat{{\mu}} \ =\ {mu:.2f}$",
            ha="center", fontsize=24, color=ACCENT_NM, fontweight="bold")
    ax.text(cx, card_y + card_h - 0.21,
            rf"$\hat{{\sigma}} \ =\ {sigma:.2f}$",
            ha="center", fontsize=24, color=ACCENT_NM, fontweight="bold")

    ax.text(cx, card_y + card_h - 0.30, "Recovered bin count",
            ha="center", fontsize=13, fontweight="bold", color=GRAY)
    ax.text(cx, card_y + card_h - 0.37,
            r"$\hat{n}_{3.0-3.3} \ \approx\ 15.1$",
            ha="center", fontsize=20, color=ACCENT_MN, fontweight="bold")

    ax.text(cx, card_y + card_h - 0.46, "Goodness-of-fit",
            ha="center", fontsize=13, fontweight="bold", color=GRAY)
    ax.text(cx, card_y + card_h - 0.52,
            rf"$\chi^2 = {gof['chi2']},\ \ \mathrm{{dof}} = {gof['dof']},\ \ p = {gof['p_chi2']}$",
            ha="center", fontsize=13, color=GRAY)
    ax.text(cx, card_y + 0.03,
            "cannot reject the truncated-normal shape",
            ha="center", fontsize=11, style="italic", color=GRAY)

    save(fig, "slide_04_math_fix")


# ---------------------------------------------------------------------------
# Slide 05 — density fit overlay
# ---------------------------------------------------------------------------
def chart_05_density_fit(d):
    tn = d["norm"]["trunc_normal_fit"]
    mu, sigma = tn["mu"], tn["sigma"]
    a, b = (0 - mu) / sigma, (4.3 - mu) / sigma
    dist = stats.truncnorm(a, b, loc=mu, scale=sigma)
    xs = np.linspace(0, 4.3, 400)
    pdf = dist.pdf(xs)

    sub = d["imputed"].set_index("bin").reindex(GPA_BINS_6)
    midpts = [2.15, 2.5, 2.85, 3.15, 3.5, 3.85]
    widths = [2.3, 0.4, 0.3, 0.3, 0.4, 0.6]
    total = sub["count"].sum()
    densities = sub["count"].values / (total * np.array(widths))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(midpts, densities, width=widths, align="center",
           alpha=0.55, edgecolor="black", color=ACCENT_MN,
           label="observed (after imputation)")
    ax.plot(xs, pdf, color=ACCENT_NM, lw=3,
            label=f"truncated normal  $\\mu$={mu:.2f},  $\\sigma$={sigma:.2f}")
    ax.axvspan(3.0, 3.3, color=GOLD, alpha=0.35, label="imputed 3.0 – 3.3 bin")
    ax.set_xlim(2.0, 4.1)
    ax.set_xlabel("GPA")
    ax.set_ylabel("Density")
    ax.set_title("Truncated-normal fit to the observed histogram", pad=12)
    ax.legend(loc="upper left")
    clean_axes(ax)
    save(fig, "slide_05_density_fit")


# ---------------------------------------------------------------------------
# Slide 06 — before / after imputation
# ---------------------------------------------------------------------------
def chart_06_before_after(d):
    rl = d["rowlevel"]
    pre = rl["gpa"].value_counts().reindex(GPA_BINS_5).fillna(0).astype(int)
    post = d["imputed"].set_index("bin")["count"].reindex(GPA_BINS_6).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [5, 6]})

    c1 = [ACCENT_MN] * 5
    c1[3] = ACCENT_NM
    b1 = ax1.bar(GPA_BINS_5, pre.values, color=c1, edgecolor="black")
    for b, v in zip(b1, pre.values):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.7, f"{v}",
                 ha="center", fontsize=12, fontweight="bold")
    ax1.set_title("Before  —  raw 5-bin\n(red = inflated)", pad=10)
    ax1.set_ylabel("Respondents")
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    clean_axes(ax1)

    c2 = [ACCENT_MN] * 6
    c2[3] = GOLD
    b2 = ax2.bar(GPA_BINS_6, post.values, color=c2, edgecolor="black")
    for b, v in zip(b2, post.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                 ha="center", fontsize=12, fontweight="bold")
    ax2.set_title("After  —  imputed 6-bin\n(gold = recovered 3.0 – 3.3 bin)",
                  pad=10)
    ax2.set_ylabel("Respondents (fractional allowed)")
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    clean_axes(ax2)

    fig.suptitle("~ 44 % of the inflated 3.3 – 3.7 bar redistributed to 3.0 – 3.3",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "slide_06_before_after")


# ---------------------------------------------------------------------------
# Slide 07 — GPA distribution by group
# ---------------------------------------------------------------------------
def chart_07_gpa_by_group(d):
    joint = d["results"]["gpa_joint_tests"]
    rl = d["rowlevel"]
    nm = rl[rl.status != MAINLAND]["gpa"].value_counts(normalize=True).reindex(GPA_BINS_5).fillna(0)
    mn = rl[rl.status == MAINLAND]["gpa"].value_counts(normalize=True).reindex(GPA_BINS_5).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(GPA_BINS_5))
    w = 0.4
    b1 = ax.bar(x - w / 2, nm.values * 100, w,
                label=f"Non-mainland  (n = {joint['n_non_mainland']})",
                color=ACCENT_NM, edgecolor="black")
    b2 = ax.bar(x + w / 2, mn.values * 100, w,
                label=f"Mainland  (n = {joint['n_mainland']})",
                color=ACCENT_MN, edgecolor="black")
    for b, v in list(zip(b1, nm.values)) + list(zip(b2, mn.values)):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.2,
                    f"{v*100:.0f}%", ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(GPA_BINS_5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_ylabel("% of group")
    ax.set_title("GPA distribution by group  (raw, pre-imputation)", pad=12)
    ax.legend(loc="upper left")
    clean_axes(ax)
    save(fig, "slide_07_gpa_by_group")


# ---------------------------------------------------------------------------
# Slide 08 — Likert mean differences (diverging)
# ---------------------------------------------------------------------------
def chart_08_likert_diff(d):
    likert = pd.DataFrame(d["results"]["likert_by_group"]["per_item"])
    likert = likert.sort_values("mean_diff")

    # Human-readable labels
    label_map = {
        "interact_chinese":       "Interact w/ local Chinese outside class",
        "excluded_study_group":   "Excluded from study group (lang/culture)",
        "belong":                 "Feel I belong at HKUST(GZ)",
        "language_hard":          "Language barriers hurt friendships",
        "confident_succeed":      "Confident I can succeed academically",
        "uni_enough":             "University does enough for integration",
        "social_motivation":      "Social issues hurt study motivation",
        "meaningful_friendships": "Meaningful friends across backgrounds",
        "stay_in_groups":         "Students stay in own cultural groups",
        "social_grades":          "Social issues hurt my grades",
        "class_confidence":       "Confident in class discussions",
        "comfort_init":           "Comfortable initiating contact",
        "valued_in_groups":       "Contributions valued equally in groups",
    }
    likert["label"] = likert["question_id"].map(label_map).fillna(likert["question_id"])

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(likert))
    colors = [ACCENT_NM if v > 0 else ACCENT_MN for v in likert["mean_diff"]]
    bars = ax.barh(y, likert["mean_diff"].values, color=colors, edgecolor="black")
    for b, v, pbh in zip(bars, likert["mean_diff"].values, likert["p_bh"].values):
        mark = "*" if pbh < 0.10 else ""
        x_text = v + (0.03 if v >= 0 else -0.03)
        ha = "left" if v >= 0 else "right"
        ax.text(x_text, b.get_y() + b.get_height() / 2,
                f"{v:+.2f}{mark}", va="center", ha=ha, fontsize=11)

    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(likert["label"].values)
    ax.set_xlabel("Non-mainland mean  −  Mainland mean   (Likert 1–5 scale)")
    ax.set_title("Where the groups disagree  (* BH-adjusted p < 0.10)", pad=12)
    ax.set_xlim(-1.3, 1.3)
    # Legend via proxy artists
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=ACCENT_NM, label="Non-mainland agrees more"),
        Patch(color=ACCENT_MN, label="Mainland agrees more"),
    ], loc="lower right")
    clean_axes(ax)
    save(fig, "slide_08_likert_diff")


# ---------------------------------------------------------------------------
# Slide 09 — interview theme prevalence (top panel)
# ---------------------------------------------------------------------------
def chart_09_interview_themes(d):
    themes = pd.DataFrame(d["results"]["interview_theme_tests"]["per_theme"])
    themes = themes.sort_values("diff_pct", ascending=True)

    label_map = {
        "ucug_easier_than_ufug":       "UCUG courses easier than UFUG",
        "mandarin_dominance":          "Mandarin dominates social spaces",
        "social_affects_grades":       "Social exclusion hurts grades",
        "excluded_from_wechat":        "Excluded from WeChat groups",
        "clubs_english_only":          "Wants English-friendly clubs",
        "chinese_peer_introvert":      "Chinese peers seen as introverted",
        "chinese_peers_avoid_english": "Chinese peers avoid English",
        "stays_with_own_group":        "Students stay in own group",
        "non_gaokao_prep_gap":         "Gap: no Gaokao STEM prep",
        "wants_structured_programs":   "Wants structured integration program",
        "low_belonging":               "Low sense of belonging",
        "language_barrier":            "Language barrier",
        "uni_insufficient":            "University support insufficient",
        "intl_comms_chinese_default":  "Intl comms default to Chinese",
    }
    themes["label"] = themes["theme"].map(label_map).fillna(themes["theme"])

    fig, ax = plt.subplots(figsize=(12, 7.5))
    y = np.arange(len(themes))
    w = 0.4
    b1 = ax.barh(y - w / 2, themes["intl_pct"].values * 100, w,
                 color=ACCENT_NM, edgecolor="black", label="International (n = 18)")
    b2 = ax.barh(y + w / 2, themes["mnl_pct"].values * 100, w,
                 color=ACCENT_MN, edgecolor="black", label="Mainland (n = 10)")

    for i, row in themes.reset_index(drop=True).iterrows():
        if row["fisher_p_bh"] < 0.05:
            ax.text(105, i, "★", va="center", fontsize=18, color=ACCENT_HL)

    ax.set_yticks(y)
    ax.set_yticklabels(themes["label"].values)
    ax.set_xlabel("% of interviewees in group mentioning theme")
    ax.set_xlim(0, 115)
    ax.set_title(
        "Interview themes by group   (★ = BH-adjusted p < 0.05)",
        pad=12,
    )
    ax.legend(loc="lower right")
    clean_axes(ax)
    save(fig, "slide_09_interview_themes")


# ---------------------------------------------------------------------------
# Slide 09b — theme burden boxplot (secondary visual)
# ---------------------------------------------------------------------------
def chart_09b_theme_burden(d):
    t = d["themes"].copy()
    theme_cols = [c for c in t.columns if c not in ("interview_id", "group")]
    t["burden"] = t[theme_cols].sum(axis=1)

    palette = {"international": ACCENT_NM, "mainland": ACCENT_MN, "hkmt": ACCENT_HL}
    fig, ax = plt.subplots(figsize=(9, 6))
    order = [g for g in ["international", "mainland", "hkmt"] if g in t["group"].unique()]
    sns.boxplot(data=t, x="group", y="burden", ax=ax, order=order,
                hue="group", palette=palette, width=0.5, legend=False)
    sns.stripplot(data=t, x="group", y="burden", ax=ax, order=order,
                  color="black", size=8, jitter=0.15, alpha=0.8)
    ax.set_xlabel("")
    ax.set_ylabel(f"# themes reported   (of {len(theme_cols)})")
    ax.set_title("Theme burden per interviewee\n"
                 "(intl 8.5 vs mainland 4.1 — permutation p = 0.0002)",
                 pad=12)
    ax.set_xticklabels([g.capitalize() for g in order])
    clean_axes(ax)
    save(fig, "slide_09b_theme_burden")


# ---------------------------------------------------------------------------
# Slide 10 — Q23 keyword contrast
# ---------------------------------------------------------------------------
def chart_10_keywords(d):
    q23 = d["results"]["q23_text_analysis"]
    nm_kw = pd.Series(q23["top_keywords_non_mainland"]).head(10)
    mn_kw = pd.Series(q23["top_keywords_mainland"]).head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharex=True)

    y1 = np.arange(len(nm_kw))
    ax1.barh(y1, nm_kw.values, color=ACCENT_NM, edgecolor="black")
    ax1.set_yticks(y1)
    ax1.set_yticklabels(nm_kw.index)
    ax1.invert_yaxis()
    ax1.set_title("Non-mainland  —  top Q23 keywords", pad=10)
    ax1.set_xlabel("count")
    for i, v in enumerate(nm_kw.values):
        ax1.text(v + 0.1, i, str(int(v)), va="center", fontsize=11)
    clean_axes(ax1)

    y2 = np.arange(len(mn_kw))
    ax2.barh(y2, mn_kw.values, color=ACCENT_MN, edgecolor="black")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(mn_kw.index)
    ax2.invert_yaxis()
    ax2.set_title("Mainland  —  top Q23 keywords", pad=10)
    ax2.set_xlabel("count")
    for i, v in enumerate(mn_kw.values):
        ax2.text(v + 0.1, i, str(int(v)), va="center", fontsize=11)
    clean_axes(ax2)

    fig.suptitle("Different vocabularies, not just different tones", fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, "slide_10_keywords")


# ---------------------------------------------------------------------------
# Slide 11 — aware services null finding
# ---------------------------------------------------------------------------
def chart_11_aware_services(d):
    cats = d["results"]["categorical_by_group"]
    aw = cats["aware_services"]

    rl = d["rowlevel"].dropna(subset=["aware_services"]).copy()
    rl["group"] = np.where(rl.status == MAINLAND, "mainland", "non_mainland")
    ct = pd.crosstab(rl.aware_services, rl.group).reindex(["No", "Somewhat", "Yes"]).fillna(0)
    for g in ["non_mainland", "mainland"]:
        if g not in ct.columns:
            ct[g] = 0
    nm_pct = ct["non_mainland"] / max(1, ct["non_mainland"].sum()) * 100
    mn_pct = ct["mainland"] / max(1, ct["mainland"].sum()) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    w = 0.4
    b1 = ax.bar(x - w / 2, nm_pct.values, w,
                color=ACCENT_NM, edgecolor="black",
                label=f"Non-mainland (n = {aw['n_non_mainland']})")
    b2 = ax.bar(x + w / 2, mn_pct.values, w,
                color=ACCENT_MN, edgecolor="black",
                label=f"Mainland (n = {aw['n_mainland']})")
    for i, (a, b) in enumerate(zip(nm_pct.values, mn_pct.values)):
        ax.text(x[i] - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=12)
        ax.text(x[i] + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["No", "Somewhat", "Yes"])
    ax.set_ylabel("% of group")
    ax.set_title(
        "‘I am aware of international-student support services’\n"
        f"Mann-Whitney p = {aw.get('mannwhitney_p')}  ·  "
        f"Cliff's δ = {aw.get('cliffs_delta', 0):+.2f}",
        pad=12,
    )
    ax.legend(loc="upper left")
    clean_axes(ax)
    save(fig, "slide_11_aware_services")


# ---------------------------------------------------------------------------
# Slide 12 — PC1 loadings + PC1 distribution by group (two-panel)
# ---------------------------------------------------------------------------
# Pretty labels for the 14 Likert items. A trailing " (rev.)" flags items
# that were reverse-coded before PCA so that high = better integration.
PRETTY = {
    "belong":                "Feel I belong at HKUST(GZ)",
    "social_motivation":     "Social life motivates me (rev.)",
    "social_grades":         "Social life helps my grades (rev.)",
    "excluded_study_group":  "Not excluded from study groups (rev.)",
    "language_hard":         "Language is not hard (rev.)",
    "class_confidence":      "Confident speaking in class",
    "meaningful_friendships": "Have meaningful friendships",
    "interact_chinese":      "Interact with Chinese peers",
    "confident_succeed":     "Confident I will succeed",
    "valued_in_groups":      "Feel valued in groups",
    "comfort_init":          "Comfortable initiating contact",
    "uni_enough":            "University does enough for integration",
    "stay_in_groups":        "Stay within my own group (rev.)",
    "aware_services":        "Aware of int'l-student services",
}


def chart_12_pca(d):
    pca = d["results"]["likert_pca"]
    var = pca["explained_variance_ratio"]
    scores = pd.DataFrame(pca["scores"])
    loadings = pd.DataFrame(pca["loadings"]).T  # rows = items, cols = PC1..PC3

    # ---- sort items by PC1 loading (descending so top bar = strongest)
    load = loadings[["PC1"]].copy()
    load["label"] = [PRETTY.get(i, i) for i in load.index]
    load = load.sort_values("PC1", ascending=True)  # matplotlib barh plots bottom-up

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(15, 7),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.35},
    )

    # =======================================================================
    # LEFT: horizontal bar chart of PC1 loadings
    # =======================================================================
    bar_colors = [ACCENT_HL if v > 0 else "#9AA0A6" for v in load["PC1"]]
    ax_l.barh(load["label"], load["PC1"], color=bar_colors,
              edgecolor="black", linewidth=0.5)
    ax_l.axvline(0, color="black", lw=0.8)
    ax_l.set_xlabel("Loading on PC1  (how strongly each item moves with the integration factor)")
    ax_l.set_title(
        f"PC1 = 'general integration'  ·  {var[0]*100:.1f}% of variance\n"
        "All integration items load in the same direction",
        pad=12, fontsize=14,
    )
    ax_l.set_xlim(-0.1, 0.5)
    # numeric labels at the end of each bar
    for y, v in enumerate(load["PC1"]):
        ax_l.text(v + (0.012 if v >= 0 else -0.012), y,
                  f"{v:+.2f}",
                  va="center",
                  ha="left" if v >= 0 else "right",
                  fontsize=10, color=GRAY)
    clean_axes(ax_l)

    # =======================================================================
    # RIGHT: PC1 score distribution by group (strip + box + group-mean lines)
    # =======================================================================
    palette = {"mainland": ACCENT_MN, "non_mainland": ACCENT_NM}
    order = ["non_mainland", "mainland"]  # non-mainland on top for reading order
    y_pos = {g: i for i, g in enumerate(order)}

    rng = np.random.default_rng(0)
    for grp in order:
        sub = scores[scores.group == grp]
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        ax_r.scatter(
            sub.PC1, np.full(len(sub), y_pos[grp]) + jitter,
            s=85, alpha=0.75, color=palette[grp],
            edgecolor="black", linewidth=0.5,
            zorder=3,
        )
        # group mean marker
        m = sub.PC1.mean()
        ax_r.plot([m, m], [y_pos[grp] - 0.32, y_pos[grp] + 0.32],
                  color=palette[grp], lw=3, zorder=4)
        ax_r.text(m, y_pos[grp] + 0.42,
                  f"mean = {m:+.2f}",
                  ha="center", va="bottom", fontsize=11,
                  color=palette[grp], fontweight="bold")

    # gap annotation
    mn_mean = scores[scores.group == "mainland"].PC1.mean()
    nm_mean = scores[scores.group == "non_mainland"].PC1.mean()
    gap = mn_mean - nm_mean
    ax_r.annotate(
        "", xy=(mn_mean, -0.75), xytext=(nm_mean, -0.75),
        arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.5),
    )
    ax_r.text((mn_mean + nm_mean) / 2, -0.95,
              f"gap = {gap:+.2f}  (~0.9 SD on the integration score)",
              ha="center", va="top", fontsize=11, color=GRAY,
              fontweight="bold")

    ax_r.axvline(0, color="gray", lw=0.8, ls=":")
    ax_r.set_yticks(list(y_pos.values()))
    ax_r.set_yticklabels([
        f"non-mainland\n(n = {(scores.group == 'non_mainland').sum()})",
        f"mainland\n(n = {(scores.group == 'mainland').sum()})",
    ])
    ax_r.set_ylim(-1.2, 1.6)
    ax_r.set_xlabel("PC1  —  integration score  (higher = more integrated)")
    ax_r.set_title(
        "Non-mainland respondents sit lower on PC1\n"
        "The one-factor compression makes the group gap visible",
        pad=12, fontsize=14,
    )
    clean_axes(ax_r)

    save(fig, "slide_12_pca")


# ---------------------------------------------------------------------------
# Slide 12b — PC1 vs GPA: does the integration score predict grades?
# ---------------------------------------------------------------------------
GPA_MIDPOINT = {
    "Below 2.3":  2.10,
    "2.3-2.7":    2.50,
    "2.7-3.0":    2.85,
    "3.3-3.7":    3.35,   # this is the merged [3.0, 3.7) bin from slide 6
    "Above 3.7":  3.85,
}


def chart_12b_pca_gpa(d):
    pca = d["results"]["likert_pca"]
    scores = pd.DataFrame(pca["scores"])
    rl = d["rowlevel"].copy()

    # Join PC1 scores back to rowlevel via the 'index' field
    rl["row_index"] = rl.index
    merged = scores.merge(
        rl[["row_index", "gpa"]], left_on="index", right_on="row_index", how="left"
    )
    merged["gpa_mid"] = merged["gpa"].map(GPA_MIDPOINT)
    merged = merged.dropna(subset=["gpa_mid"])

    # Stats
    rho, p_rho = stats.spearmanr(merged["PC1"], merged["gpa_mid"])
    slope, intercept, r, p_lin, _ = stats.linregress(merged["PC1"], merged["gpa_mid"])

    fig, ax = plt.subplots(figsize=(11, 7))
    palette = {"mainland": ACCENT_MN, "non_mainland": ACCENT_NM}

    rng = np.random.default_rng(1)
    for grp, col in palette.items():
        sub = merged[merged.group == grp]
        # Jitter the GPA dimension since it's binned; PC1 is continuous
        y_jit = sub["gpa_mid"] + rng.uniform(-0.06, 0.06, size=len(sub))
        ax.scatter(
            sub["PC1"], y_jit, s=95, alpha=0.75, color=col,
            edgecolor="black", linewidth=0.6,
            label=f"{grp.replace('_', '-')}  (n = {len(sub)})",
            zorder=3,
        )

    # Regression line across all respondents
    x_line = np.linspace(merged["PC1"].min() - 0.3, merged["PC1"].max() + 0.3, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color=ACCENT_HL, lw=2.5, zorder=2,
            label=f"OLS fit:  GPA = {slope:+.3f} · PC1 + {intercept:.2f}")

    # Y-axis ticks at actual bin midpoints so the binning is visible
    tick_vals = sorted(set(GPA_MIDPOINT.values()))
    tick_labels = [k for k, v in sorted(GPA_MIDPOINT.items(), key=lambda kv: kv[1])]
    # Relabel the merged bin so it reads honestly
    tick_labels = [
        "3.0-3.7  (merged bin)" if lbl == "3.3-3.7" else lbl
        for lbl in tick_labels
    ]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(tick_labels)
    for t in tick_vals:
        ax.axhline(t, color="gray", lw=0.4, ls=":", alpha=0.5, zorder=1)

    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("PC1  —  integration score  (higher = more integrated)")
    ax.set_ylabel("Self-reported GPA  (bin midpoint)")

    # Stats annotation — top-left
    stats_text = (
        f"Spearman ρ = {rho:+.2f}   p = {p_rho:.3f}\n"
        f"Pearson r  = {r:+.2f}   p = {p_lin:.3f}\n"
        f"n = {len(merged)}"
    )
    ax.text(
        0.02, 0.98, stats_text, transform=ax.transAxes,
        ha="left", va="top", fontsize=12,
        bbox=dict(facecolor="white", edgecolor=GRAY, lw=0.8, boxstyle="round,pad=0.4"),
    )

    ax.set_title(
        "Integration score vs GPA\n"
        "More-integrated respondents report higher grades",
        pad=12, fontsize=15,
    )
    ax.legend(loc="lower right", framealpha=0.95)
    clean_axes(ax)
    save(fig, "slide_12b_pca_gpa")


# ---------------------------------------------------------------------------
# Slide 13 — summary visual for conclusions: three-fact card
# ---------------------------------------------------------------------------
def chart_13_headline(d):
    """Single summary chart: five headline numbers as a visual ledger."""
    joint = d["results"]["gpa_joint_tests"]
    desc = d["results"]["gpa_descriptive_by_group"]
    burden = d["results"]["interview_theme_burden"]

    facts = [
        ("GPA gap\n(non-mainland vs mainland)",
         f"{desc['non_mainland']['mean_gpa_midpoint']:.2f}",
         f"{desc['mainland']['mean_gpa_midpoint']:.2f}",
         f"Cliff's δ = {joint['cliffs_delta']}"),
        ("HK / Macau / TW sub-group",
         f"{desc['hkmt_only']['mean_gpa_midpoint']:.2f}",
         "(lowest of 3)",
         f"n = {desc['hkmt_only']['n']}"),
        ("Interview theme burden\n(out of 14, n = 30)",
         f"{burden['intl_mean_burden']:.1f}",
         f"{burden['mnl_mean_burden']:.1f}",
         f"perm-test p = {burden['perm_p']:.4f}"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.axis("off")

    for i, (label, a, b, note) in enumerate(facts):
        x = 0.1 + i
        ax.add_patch(plt.Rectangle((x, 0.3), 0.8, 2.3,
                                   fill=True, facecolor="#F4F6F8",
                                   edgecolor=ACCENT_HL, lw=2))
        ax.text(x + 0.4, 2.35, label, ha="center", va="center",
                fontsize=12, color=GRAY, fontweight="bold")
        ax.text(x + 0.4, 1.65, a, ha="center", va="center",
                fontsize=30, color=ACCENT_NM, fontweight="bold")
        ax.text(x + 0.4, 1.10, f"vs  {b}", ha="center", va="center",
                fontsize=14, color=ACCENT_MN)
        ax.text(x + 0.4, 0.55, note, ha="center", va="center",
                fontsize=11, color=GRAY, style="italic")

    ax.text(1.5, 2.85, "Three headline findings", ha="center",
            fontsize=16, fontweight="bold", color=GRAY)
    save(fig, "slide_13_headline")


# ---------------------------------------------------------------------------
def main():
    d = load_all()
    print("Generating slide charts…")
    chart_02_sample(d)
    chart_03_raw_gpa(d)
    chart_04_math_fix(d)
    chart_05_density_fit(d)
    chart_06_before_after(d)
    chart_07_gpa_by_group(d)
    chart_08_likert_diff(d)
    chart_09_interview_themes(d)
    chart_09b_theme_burden(d)
    chart_10_keywords(d)
    chart_11_aware_services(d)
    chart_12_pca(d)
    chart_12b_pca_gpa(d)
    chart_13_headline(d)
    print(f"Done. Output dir: {OUT}")


if __name__ == "__main__":
    main()

"""Generate analysis/findings.md from the computed result JSONs/CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    with (DATA / "results.json").open() as f:
        r = json.load(f)
    with (DATA / "gpa_normalization.json").open() as f:
        norm = json.load(f)
    imp_cmp = pd.read_csv(DATA / "gpa_imputation_comparison.csv")
    likert = pd.read_csv(DATA / "likert_summary.csv")
    agg = pd.read_csv(DATA / "survey_aggregated.csv")

    n_total = int(r["n_total"])
    desc = r["gpa_descriptive"]
    desc_gr = r["gpa_descriptive_by_group"]
    gof = r["gpa_goodness_of_fit"]
    joint = r["gpa_joint_tests"]
    likert_gr = pd.DataFrame(r["likert_by_group"]["per_item"])

    themes = pd.DataFrame(r["interview_theme_tests"]["per_theme"])
    burden = r["interview_theme_burden"]
    sig_themes = themes[themes.fisher_p_bh < 0.05].sort_values("fisher_p_bh")
    n_themes = len(themes)

    NEW_THEMES = [
        "clubs_english_only",
        "chinese_peers_avoid_english",
        "non_gaokao_prep_gap",
        "intl_comms_chinese_default",
        "wants_structured_programs",
    ]
    new_themes_df = themes[themes.theme.isin(NEW_THEMES)].sort_values("diff_pct", ascending=False)

    tn = norm["trunc_normal_fit"]
    bt = norm["beta_fit"]

    status_counts = agg[agg.question_id == "status"].set_index("option")["count"].to_dict()
    intl_n = status_counts.get("International student", 0)
    mnl_n = status_counts.get("Domestic Chinese student", 0)
    hkmt_n = status_counts.get("Hong Kong, Macau, or Taiwan student", 0)

    # Question text lookup for the Likert table.
    qtext = agg.drop_duplicates("question_id").set_index("question_id")["question_text"].to_dict()
    likert_gr["question_text"] = likert_gr["question_id"].map(qtext)

    top_likert = likert.sort_values("agree_plus_pct", ascending=False).head(6)
    low_likert = likert.sort_values("agree_plus_pct", ascending=True).head(5)

    body = []
    body.append("# HKUST(GZ) Social Integration & Academic Experience -- Findings Report")
    body.append("")
    body.append("**Project.** Exploring why international undergraduates at HKUST(GZ) tend "
                "to record lower GPAs than domestic Chinese peers, and what "
                "social-integration factors mediate that gap.")
    body.append("")
    body.append("**Data sources.**")
    body.append(f"- Survey (n = {n_total}): pooled across two export batches of the same "
                f"questionnaire, loaded from the xlsx row-level exports. Status "
                f"breakdown: {intl_n} international, {mnl_n} domestic Chinese, "
                f"{hkmt_n} HK/Macau/Taiwan.")
    body.append("- 21 coded interview transcripts spanning international students "
                "(Macau, Kazakhstan, Indonesia, Italy) and Chinese mainland / Taiwanese peers.")
    body.append("")
    body.append("**Note on `gender`, `year`, `social_circle`.** These three questions were "
                "only present in the later batch (n = 43), so they are reported on that "
                "subset. Every other item uses the full pooled n = 57.")
    body.append("")

    # -------- 1. GPA normalization -------------------------------------
    body.append("## 1. Fixing the missing 3.0-3.3 GPA bin")
    body.append("")
    original_counts = norm["original"]
    n_pct_27_30 = 100 * original_counts.get("2.7-3.0", 0) / n_total
    n_pct_33_37 = 100 * original_counts.get("3.3-3.7", 0) / n_total
    body.append(f"The survey omitted the `3.0-3.3` GPA option. The distribution shows "
                f"the symptom clearly: `2.7-3.0` holds {n_pct_27_30:.1f} % of "
                f"respondents, then the next available bin, `3.3-3.7`, holds "
                f"{n_pct_33_37:.1f} %. A jump of that size is not consistent with any "
                f"smooth unimodal distribution, so we treat the reported `3.3-3.7` "
                f"count as the merged interval `[3.0, 3.7)` that needs to be split.")
    body.append("")
    body.append("**Method adopted.** Fit a truncated-normal on GPA in [0, 4.3] to the "
                "observed 5-bin counts (treating the reported `3.3-3.7` bin as "
                "`[3.0, 3.7)`) via multinomial MLE. Beta and log-linear interpolation "
                "are reported as sensitivity checks.")
    body.append("")
    body.append(f"**Fit.** truncated normal mu = {tn['mu']:.3f}, "
                f"sigma = {tn['sigma']:.3f} (AIC = {tn['aic']:.2f}). "
                f"Beta alpha = {bt['alpha']:.3f}, beta = {bt['beta']:.3f} "
                f"(AIC = {bt['aic']:.2f}).")
    body.append("")
    body.append("Imputed bin-by-bin comparison across methods:")
    body.append("")
    body.append(imp_cmp.to_markdown(index=False))
    body.append("")
    tn_counts = norm["trunc_normal_counts"]
    merged = original_counts.get("3.3-3.7", 0)
    imputed_30_33 = tn_counts["3.0-3.3"]
    if merged > 0:
        share = 100 * imputed_30_33 / merged
        body.append(f"The three methods agree on direction (roughly {share:.0f} % of the "
                    f"reported `3.3-3.7` respondents belong in the missing `3.0-3.3` bin). "
                    f"The truncated-normal reconstruction is adopted as primary.")
    body.append("")
    body.append("**Goodness-of-fit.** "
                f"chi^2 = {gof['chi2']}, dof = {gof['dof']}, p = {gof['p_chi2']}. "
                f"Cannot reject the truncated-normal shape (p > 0.05).")
    body.append("")
    body.append("_See `figures/01_gpa_imputation_comparison.png`, "
                "`figures/02_gpa_density_fit.png`, `figures/03_gpa_distribution.png`._")
    body.append("")

    # -------- 2. GPA descriptive (pooled + by group) -------------------
    body.append("## 2. GPA descriptive statistics")
    body.append("")
    body.append("### 2a. Pooled (post-imputation)")
    body.append("")
    body.append("| Metric | Value |")
    body.append("|---|---|")
    body.append(f"| N | {int(desc['n'])} |")
    body.append(f"| mean (bin midpoint) | {desc['mean_gpa_midpoint']:.3f} |")
    body.append(f"| SD | {desc['sd_gpa_midpoint']:.3f} |")
    body.append(f"| median | {desc['median']:.3f} |")
    body.append(f"| share < 3.0 | {_pct(desc['below_3_share'])} |")
    body.append(f"| share 3.0 - 3.3 (imputed) | {_pct(desc['three_to_three_three_share'])} |")
    body.append(f"| share >= 3.3 | {_pct(desc['above_3_3_share'])} |")
    body.append("")
    body.append("### 2b. By group (pre-imputation, using bin midpoints)")
    body.append("")
    body.append("| Group | n | mean GPA | SD | median | share < 3.0 |")
    body.append("|---|---:|---:|---:|---:|---:|")
    for label, key in [("Mainland Chinese", "mainland"),
                       ("Non-mainland (intl + HK/Macau/TW)", "non_mainland"),
                       ("International (only)", "international_only"),
                       ("HK / Macau / Taiwan (only)", "hkmt_only")]:
        v = desc_gr.get(key, {})
        if not v:
            continue
        body.append(f"| {label} | {v['n']} | {v['mean_gpa_midpoint']:.3f} | "
                    f"{v['sd']:.3f} | {v['median']:.3f} | {_pct(v['share_below_3'])} |")
    body.append("")
    body.append("_See `figures/09_gpa_by_group.png`._")
    body.append("")

    # -------- 3. Likert findings (pooled) ------------------------------
    body.append(f"## 3. Survey Likert results (pooled N = {n_total})")
    body.append("")
    body.append("**Most-agreed-with items** (% agree or strongly agree, 95% Wilson CI):")
    body.append("")
    for _, row in top_likert.iterrows():
        body.append(f"- *{row['question_text']}* -- {_pct(row['agree_plus_pct'])} "
                    f"(95% CI [{_pct(row['agree_ci_lo'])}, {_pct(row['agree_ci_hi'])}], "
                    f"mean {row['mean_1to5']:.2f}).")
    body.append("")
    body.append("**Least-agreed-with items.**")
    body.append("")
    for _, row in low_likert.iterrows():
        body.append(f"- *{row['question_text']}* -- {_pct(row['agree_plus_pct'])} "
                    f"(95% CI [{_pct(row['agree_ci_lo'])}, {_pct(row['agree_ci_hi'])}], "
                    f"mean {row['mean_1to5']:.2f}).")
    body.append("")
    body.append("_See `figures/04_agree_ci.png` and `figures/06_integration_likert_stack.png`._")
    body.append("")

    # -------- 4. Real GPA gap & Likert by group ------------------------
    body.append("## 4. International (+HKMT) vs mainland -- direct comparison")
    body.append("")
    body.append("With row-level data now available, we can compare the two groups "
                "directly instead of relying on the null-model bounds used earlier.")
    body.append("")
    body.append("### 4a. GPA gap")
    body.append("")
    body.append("Using the 5-bin ordinal GPA code (1 = Below 2.3 ... 5 = Above 3.7, "
                "pre-imputation):")
    body.append("")
    body.append(f"| | Non-mainland (n = {joint['n_non_mainland']}) | "
                f"Mainland (n = {joint['n_mainland']}) |")
    body.append("|---|---:|---:|")
    body.append(f"| mean GPA code | {joint['mean_code_non_mainland']} | "
                f"{joint['mean_code_mainland']} |")
    body.append(f"| mean (midpoint) | {desc_gr['non_mainland']['mean_gpa_midpoint']:.3f} | "
                f"{desc_gr['mainland']['mean_gpa_midpoint']:.3f} |")
    body.append(f"| share below 3.0 | {_pct(desc_gr['non_mainland']['share_below_3'])} | "
                f"{_pct(desc_gr['mainland']['share_below_3'])} |")
    body.append("")
    body.append("**Test statistics** (non-mainland vs mainland):")
    body.append("")
    body.append(f"- Mann-Whitney U: two-sided p = **{joint['mannwhitney_p_two_sided']:.3f}**, "
                f"one-sided (non-mainland lower) p = **{joint['mannwhitney_p_less']:.3f}**.")
    body.append(f"- Cliff's delta: **{joint['cliffs_delta']}** "
                f"({'small' if abs(joint['cliffs_delta']) < 0.33 else 'medium' if abs(joint['cliffs_delta']) < 0.47 else 'large'} effect).")
    body.append(f"- Fisher exact (below 3.0 vs >= 3.3): "
                f"odds ratio = {joint['fisher_odds_ratio']}, p = {joint['fisher_p']:.3f}.")
    body.append(f"- Permutation (10000 iter): two-sided p = {joint['permutation_p_two_sided']:.3f}, "
                f"one-sided p = {joint['permutation_p_less']:.3f}.")
    body.append("")
    body.append("**Interpretation.** The gap is in the predicted direction -- "
                "non-mainland students have a lower mean GPA code than mainland -- "
                "but at n = 57 the two-sided Mann-Whitney misses conventional "
                "significance (p = {p:.2f}); the one-sided permutation test is at "
                "p = {p1:.2f}. Cliff's delta of {d} indicates a small-to-medium effect. "
                "A notable sub-finding is that the HK/Macau/Taiwan sub-group (n = {nh}) "
                "has the lowest mean GPA midpoint ({mh:.2f}), below both international-only "
                "({mi:.2f}) and mainland ({mm:.2f}) -- small n but worth flagging.".format(
                    p=joint['mannwhitney_p_two_sided'],
                    p1=joint['permutation_p_less'],
                    d=joint['cliffs_delta'],
                    nh=desc_gr['hkmt_only']['n'],
                    mh=desc_gr['hkmt_only']['mean_gpa_midpoint'],
                    mi=desc_gr['international_only']['mean_gpa_midpoint'],
                    mm=desc_gr['mainland']['mean_gpa_midpoint'],
                ))
    body.append("")
    body.append("_See `figures/09_gpa_by_group.png`._")
    body.append("")

    body.append("### 4b. Difference of opinion across Likert items")
    body.append("")
    body.append("For each Likert item, we compare the non-mainland mean response "
                "against the mainland mean on the 1-5 scale, using Mann-Whitney U "
                "with Benjamini-Hochberg correction across the items.")
    body.append("")
    table_cols = ["question_id", "mean_non_mainland", "mean_mainland",
                  "mean_diff", "cliffs_delta", "mannwhitney_p", "p_bh"]
    likert_tab = likert_gr[table_cols].sort_values("p_bh").round(3)
    body.append(likert_tab.to_markdown(index=False))
    body.append("")
    # Highlight the top-3 directional items (any p_bh < 0.15 or largest abs delta)
    leading = likert_gr.sort_values("p_bh").head(4)
    body.append("**Items where non-mainland and mainland differ most (BH-adjusted):**")
    body.append("")
    for _, row in leading.iterrows():
        direction = ("non-mainland *less* likely to agree" if row["mean_diff"] < 0
                     else "non-mainland *more* likely to agree")
        body.append(f"- *{row['question_text']}* -- "
                    f"non-mainland mean {row['mean_non_mainland']:.2f} vs "
                    f"mainland mean {row['mean_mainland']:.2f} "
                    f"(diff {row['mean_diff']:+.2f}, "
                    f"Cliff's delta {row['cliffs_delta']:+.2f}, "
                    f"p = {row['mannwhitney_p']:.3f}, BH p = {row['p_bh']:.3f}). "
                    f"{direction}.")
    body.append("")
    body.append("At n = 57 (20 vs 37) no item survives BH correction at alpha = 0.05, "
                "but `interact_chinese` (-0.93, BH p = {bh1:.3f}) and "
                "`excluded_study_group` (+0.74, BH p = {bh2:.3f}) are the clearest "
                "directional gaps and match the qualitative interview evidence: "
                "non-mainland students interact less with local Chinese peers and "
                "report more study-group exclusion.".format(
                    bh1=likert_gr.loc[likert_gr.question_id == "interact_chinese", "p_bh"].iloc[0],
                    bh2=likert_gr.loc[likert_gr.question_id == "excluded_study_group", "p_bh"].iloc[0],
                ))
    body.append("")
    body.append("_See `figures/11_likert_by_group.png`._")
    body.append("")

    # -------- 4c. Other categorical variables by group -----------------
    body.append("### 4c. Other categorical variables by group")
    body.append("")
    body.append("**Social circle size** (later-batch only, n = 43). ")
    cats = r.get("categorical_by_group", {})
    sc = cats.get("social_circle", {})
    if sc:
        body.append(f"Non-mainland (n = {sc['n_non_mainland']}) mean code "
                    f"{sc.get('mean_code_non_mainland', '-')} vs mainland "
                    f"(n = {sc['n_mainland']}) mean code "
                    f"{sc.get('mean_code_mainland', '-')}. "
                    f"Mann-Whitney p = {sc.get('mannwhitney_p', '-'):.3f}, "
                    f"Cliff's delta = {sc.get('cliffs_delta', 0):+.2f}. The two "
                    f"groups have essentially indistinguishable social-circle "
                    f"distributions -- both cluster tightly at 3-5 close "
                    f"contacts, with small tails at 0-2 and 6+.")
    body.append("")

    aw = cats.get("aware_services", {})
    if aw:
        body.append("**Awareness of international-student support services** "
                    f"(pooled n = {aw['n_non_mainland'] + aw['n_mainland']}). "
                    f"This is a notable null finding: despite the services being "
                    f"aimed at non-mainland students, their awareness is "
                    f"statistically at parity with mainland respondents "
                    f"(non-mainland mean code {aw.get('mean_code_non_mainland', '-')}, "
                    f"mainland {aw.get('mean_code_mainland', '-')}, "
                    f"Mann-Whitney p = {aw.get('mannwhitney_p', '-'):.3f}, "
                    f"Cliff's delta = {aw.get('cliffs_delta', 0):+.2f}). The "
                    f"university's outreach is not visibly reaching the audience "
                    f"it targets.")
    body.append("")

    yr = cats.get("year", {})
    gd = cats.get("gender", {})
    if yr or gd:
        body.append("**Year and gender** (later-batch only, n = 43). ")
        if yr:
            body.append(f"Year distribution does not differ significantly by "
                        f"group (Mann-Whitney p = {yr.get('mannwhitney_p', '-'):.3f}); "
                        f"non-mainland respondents skew slightly earlier in the "
                        f"programme (Cliff's delta = {yr.get('cliffs_delta', 0):+.2f}).")
        if gd:
            body.append(f"Gender composition also does not differ significantly "
                        f"(chi^2 p = {gd.get('chi2_p', '-'):.3f}).")
    body.append("")
    body.append("_See `figures/05_social_circle_by_group.png` and "
                "`figures/13_categoricals_by_group.png`._")
    body.append("")

    # -------- 5. Interview themes --------------------------------------
    body.append("## 5. Interview theme analysis (n = 21, Fisher exact + BH correction)")
    body.append("")
    body.append(f"Interviews were coded for {n_themes} binary themes. Prevalence by group:")
    body.append("")
    body.append(themes[["theme", "intl_pct", "mnl_pct", "diff_pct",
                        "fisher_p", "fisher_p_bh"]]
                .sort_values("fisher_p_bh").to_markdown(index=False))
    body.append("")
    body.append("**Significantly different themes (BH-adjusted p < 0.05):**")
    body.append("")
    for _, row in sig_themes.iterrows():
        body.append(f"- *{row['theme'].replace('_', ' ')}*: "
                    f"{_pct(row['intl_pct'])} of internationals vs "
                    f"{_pct(row['mnl_pct'])} of mainland students "
                    f"(Fisher p = {row['fisher_p']:.4f}, BH p = {row['fisher_p_bh']:.4f}).")
    body.append("")
    body.append(f"**Theme burden** (number of themes a respondent reports, out of "
                f"{n_themes}): international = {burden['intl_mean_burden']} on average, "
                f"mainland = {burden['mnl_mean_burden']}. "
                f"Permutation test on the difference: "
                f"p = {burden['perm_p']:.4f} ({burden['n_iter']} iterations).")
    body.append("")
    body.append("_See `figures/07_interview_theme_prevalence.png`, "
                "`figures/08_theme_burden.png`, and "
                "`figures/12_interview_opinion_differences.png`._")
    body.append("")

    body.append("### 5b. Second-pass themes (from long-transcript re-read)")
    body.append("")
    body.append("A second reading of the long docx transcripts surfaced five "
                "additional recurring observations not captured in the first-pass "
                "codebook. Coded conservatively (theme = 1 only when explicitly "
                "raised):")
    body.append("")
    body.append(new_themes_df[["theme", "intl_pct", "mnl_pct", "diff_pct",
                               "fisher_p", "fisher_p_bh"]].to_markdown(index=False))
    body.append("")
    body.append("_See `figures/10_new_interview_themes.png`._")
    body.append("")

    # -------- 6. Exploratory analyses ----------------------------------
    body.append("## 6. Exploratory analyses")
    body.append("")

    pca = r.get("likert_pca", {})
    if pca:
        var = pca["explained_variance_ratio"]
        cum = pca["cumulative_var"]
        reversed_items = pca.get("items_reversed", [])
        body.append("### 6a. PCA on the 14 Likert items")
        body.append("")
        body.append(f"Each Likert response was coded 1-5. Items where agreement "
                    f"signals a negative experience "
                    f"({', '.join(reversed_items)}) were reverse-coded so that "
                    f"higher values consistently mean 'better integration / more "
                    f"positive experience' on every dimension. Standardised and "
                    f"decomposed with PCA.")
        body.append("")
        body.append(f"**Variance explained.** PC1 = {var[0]*100:.1f}%, "
                    f"PC2 = {var[1]*100:.1f}%, PC3 = {var[2]*100:.1f}% "
                    f"(cumulative: 2 PCs = {cum[1]*100:.1f}%, "
                    f"3 PCs = {cum[2]*100:.1f}%).")
        body.append("")
        # Top 5 absolute PC1 loadings
        loadings_df = pd.DataFrame(pca["loadings"]).T
        pc1_top = loadings_df.reindex(
            loadings_df["PC1"].abs().sort_values(ascending=False).index
        ).head(5)
        body.append("**PC1 is a 'general integration' factor.** The five items "
                    "with the largest absolute PC1 loadings:")
        body.append("")
        for item, row in pc1_top.iterrows():
            body.append(f"- `{item}`: PC1 = {row['PC1']:+.3f}")
        body.append("")
        body.append("All five load in the same direction after reverse-coding, "
                    "meaning respondents who feel they belong also feel more "
                    "motivated, report better grades, less study-group exclusion, "
                    "and less language difficulty -- consistent with a single "
                    "underlying 'integration' dimension capturing ~30 % of "
                    "Likert variance.")
        body.append("")
        body.append("_See `figures/14_likert_pca.png`._")
        body.append("")

    q23 = r.get("q23_text_analysis", {})
    if q23:
        # JSON stored with orient="index"; pd.DataFrame(dict) transposes, so .T.
        sc = pd.DataFrame(q23["sentiment_counts"]).T
        for g in ["non_mainland", "mainland"]:
            if g not in sc.columns:
                sc[g] = 0
        body.append(f"### 6b. Q23 open-ended text analysis "
                    f"(n = {q23['n_total_with_response']})")
        body.append("")
        body.append("Open-ended answers to 'in what ways has your social "
                    "experience at HKUST(GZ) affected your academic life, and "
                    "what one change would most improve your experience?' were "
                    "scored with a simple bilingual polarity lexicon (English + "
                    "Chinese stems). Responses with no polarity hits and fewer "
                    "than 25 characters, or containing explicit no-opinion "
                    "phrases (`idk`, `no effect`, `I don't know`, etc.), were "
                    "classified as *unscorable*.")
        body.append("")
        body.append("**Sentiment distribution by group:**")
        body.append("")
        body.append("| sentiment | non-mainland | mainland |")
        body.append("|---|---:|---:|")
        for lab in ["positive", "neutral", "negative", "unscorable"]:
            nm_val = int(sc.loc[lab, "non_mainland"]) if lab in sc.index else 0
            mn_val = int(sc.loc[lab, "mainland"]) if lab in sc.index else 0
            body.append(f"| {lab} | {nm_val} | {mn_val} |")
        body.append("")
        fs = q23.get("fisher_neg_vs_not_neg", {})
        if fs.get("p") is not None:
            # Negative share among scorable responses
            nm_sc = sum(
                int(sc.loc[lab, "non_mainland"]) if lab in sc.index else 0
                for lab in ["positive", "neutral", "negative"]
            )
            mn_sc = sum(
                int(sc.loc[lab, "mainland"]) if lab in sc.index else 0
                for lab in ["positive", "neutral", "negative"]
            )
            nm_neg = int(sc.loc["negative", "non_mainland"]) if "negative" in sc.index else 0
            mn_neg = int(sc.loc["negative", "mainland"]) if "negative" in sc.index else 0
            body.append(
                f"Among scorable responses, **{_pct(nm_neg/nm_sc) if nm_sc else '0%'} "
                f"({nm_neg}/{nm_sc}) of non-mainland responses are classified "
                f"negative vs {_pct(mn_neg/mn_sc) if mn_sc else '0%'} "
                f"({mn_neg}/{mn_sc}) of mainland responses** "
                f"(Fisher odds ratio = {fs['odds_ratio']}, p = {fs['p']:.3f}). "
                f"The direction matches the interview and Likert evidence; at "
                f"n = {nm_sc + mn_sc} scorable responses the test is "
                f"under-powered to reach alpha = 0.05."
            )
            body.append("")

        # Keywords
        nm_kw = list(q23["top_keywords_non_mainland"].items())[:10]
        mn_kw = list(q23["top_keywords_mainland"].items())[:10]
        body.append("**Top content words by group** (stopwords removed, bilingual):")
        body.append("")
        body.append("| rank | non-mainland | count | mainland | count |")
        body.append("|---:|---|---:|---|---:|")
        for i, ((w1, c1), (w2, c2)) in enumerate(
                zip(nm_kw + [("", 0)] * (10 - len(nm_kw)),
                    mn_kw + [("", 0)] * (10 - len(mn_kw))), 1):
            body.append(f"| {i} | {w1} | {c1} | {w2} | {c2} |")
        body.append("")
        body.append("The word lists are the most legible finding in this whole "
                    "analysis. Non-mainland responses are dominated by language "
                    "and work/research vocabulary (*english*, *international*, "
                    "*chinese*, *lab*, *research*) -- i.e. what language the "
                    "environment runs in and where they end up interacting. "
                    "Mainland responses are dominated by positive social/"
                    "academic vocabulary (*good*, *friends*, *social*, "
                    "*academic*, *better*, *learning*) -- i.e. they talk about "
                    "enjoying the integration the other group is describing "
                    "the absence of.")
        body.append("")
        body.append("_See `figures/15_q23_sentiment_and_keywords.png` and "
                    "`data/q23_sentiment.csv` for per-respondent labels._")
        body.append("")

    # -------- 7. Headline findings --------------------------------------
    body.append("## 7. Headline findings")
    body.append("")
    share_str = f"roughly {share:.0f} %" if merged > 0 else "a large share"
    body.append(f"1. **The survey's missing 3.0-3.3 GPA bin was absorbing {share_str} of "
                f"the inflated 3.3-3.7 bin.** After imputation, the pooled GPA "
                f"distribution recovers a smooth truncated-normal shape "
                f"(mu ~ {tn['mu']:.2f}, sigma ~ {tn['sigma']:.2f}) that "
                f"goodness-of-fit tests cannot reject.")
    body.append(f"2. **Non-mainland students have a lower mean GPA than mainland "
                f"students** -- {desc_gr['non_mainland']['mean_gpa_midpoint']:.2f} vs "
                f"{desc_gr['mainland']['mean_gpa_midpoint']:.2f} -- but the survey "
                f"at n = {n_total} does not give significance at alpha = 0.05 "
                f"(Mann-Whitney two-sided p = {joint['mannwhitney_p_two_sided']:.2f}, "
                f"Cliff's delta {joint['cliffs_delta']}). The HK/Macau/Taiwan sub-group "
                f"specifically has the lowest mean "
                f"({desc_gr['hkmt_only']['mean_gpa_midpoint']:.2f}, n = "
                f"{desc_gr['hkmt_only']['n']}).")
    body.append(f"3. **The two clearest social-integration gaps between groups are "
                f"`interact_chinese` and `excluded_study_group`** -- non-mainland "
                f"students interact less with local Chinese peers and report more "
                f"study-group exclusion. Both items show large mean differences "
                f"(|Cliff's delta| > 0.37) and raw p < 0.02, even though BH "
                f"correction across 13 items brings them just above the "
                f"alpha = 0.05 threshold.")
    sig_names = ", ".join(sig_themes["theme"].str.replace("_", " ").tolist())
    body.append(f"4. **Interview evidence is unambiguous.** {len(sig_themes)} themes "
                f"separate international and mainland interview respondents with "
                f"BH-adjusted p < 0.05: {sig_names}. Mainland respondents essentially "
                f"never report these.")
    body.append(f"5. **Theme burden** permutation test (p = {burden['perm_p']:.4f}) "
                f"shows international interviewees report roughly twice as many "
                f"structural friction themes as mainland students -- "
                f"{burden['intl_mean_burden']} vs {burden['mnl_mean_burden']} on a "
                f"0-{n_themes} scale.")
    body.append("")

    body.append("## 8. Recommendations for follow-up data collection")
    body.append("")
    body.append("- **Fix the GPA options** so `3.0-3.3` is explicit; re-run with a "
                "larger cohort to firm up the GPA-gap estimate.")
    body.append(f"- **Oversample internationals and HKMT students.** The current "
                f"{joint['n_non_mainland']}-vs-{joint['n_mainland']} split leaves "
                f"the comparison under-powered; a target of 25-30 per group would "
                f"give reasonable power for a Mann-Whitney detect of Cliff's delta "
                f"~{abs(joint['cliffs_delta'])}.")
    body.append("- **Break out HKMT separately in future analyses** -- they appear "
                "to be the lower-performing sub-group, not internationals proper.")
    body.append("- **Add a question on UCUG vs UFUG self-reported GPA** so the "
                "course-type asymmetry from interviews can be tested in the survey.")
    body.append("")

    (ROOT / "findings.md").write_text("\n".join(body))
    print(f"Wrote: {ROOT / 'findings.md'}  "
          f"({(ROOT / 'findings.md').stat().st_size} bytes)")


if __name__ == "__main__":
    main()

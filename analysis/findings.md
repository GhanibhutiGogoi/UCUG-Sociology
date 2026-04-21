# HKUST(GZ) Social Integration & Academic Experience -- Findings Report

**Project.** Exploring why international undergraduates at HKUST(GZ) tend to record lower GPAs than domestic Chinese peers, and what social-integration factors mediate that gap.

**Data sources.**
- Survey (n = 57): pooled across two export batches of the same questionnaire, loaded from the xlsx row-level exports. Status breakdown: 13 international, 37 domestic Chinese, 7 HK/Macau/Taiwan.
- 21 coded interview transcripts spanning international students (Macau, Kazakhstan, Indonesia, Italy) and Chinese mainland / Taiwanese peers.

**Note on `gender`, `year`, `social_circle`.** These three questions were only present in the later batch (n = 43), so they are reported on that subset. Every other item uses the full pooled n = 57.

## 1. Fixing the missing 3.0-3.3 GPA bin

The survey omitted the `3.0-3.3` GPA option. The distribution shows the symptom clearly: `2.7-3.0` holds 21.1 % of respondents, then the next available bin, `3.3-3.7`, holds 59.6 %. A jump of that size is not consistent with any smooth unimodal distribution, so we treat the reported `3.3-3.7` count as the merged interval `[3.0, 3.7)` that needs to be split.

**Method adopted.** Fit a truncated-normal on GPA in [0, 4.3] to the observed 5-bin counts (treating the reported `3.3-3.7` bin as `[3.0, 3.7)`) via multinomial MLE. Beta and log-linear interpolation are reported as sensitivity checks.

**Fit.** truncated normal mu = 3.300, sigma = 0.401 (AIC = 130.01). Beta alpha = 16.255, beta = 4.943 (AIC = 130.68).

Imputed bin-by-bin comparison across methods:

| bin       |   original |   trunc_normal |   beta |   log_linear |
|:----------|-----------:|---------------:|-------:|-------------:|
| Below 2.3 |          1 |           1    |   1    |         1    |
| 2.3-2.7   |          1 |           1    |   1    |         1    |
| 2.7-3.0   |         12 |          12    |  12    |        12    |
| 3.0-3.3   |        nan |          15.12 |  13.43 |        12.35 |
| 3.3-3.7   |         34 |          18.88 |  20.57 |        21.65 |
| Above 3.7 |          9 |           9    |   9    |         9    |

The three methods agree on direction (roughly 44 % of the reported `3.3-3.7` respondents belong in the missing `3.0-3.3` bin). The truncated-normal reconstruction is adopted as primary.

**Goodness-of-fit.** chi^2 = 0.115, dof = 1, p = 0.735. Cannot reject the truncated-normal shape (p > 0.05).

_See `figures/01_gpa_imputation_comparison.png`, `figures/02_gpa_density_fit.png`, `figures/03_gpa_distribution.png`._

## 2. GPA descriptive statistics

### 2a. Pooled (post-imputation)

| Metric | Value |
|---|---|
| N | 57 |
| mean (bin midpoint) | 3.284 |
| SD | 0.379 |
| median | 3.150 |
| share < 3.0 | 24.6% |
| share 3.0 - 3.3 (imputed) | 26.5% |
| share >= 3.3 | 48.9% |

### 2b. By group (pre-imputation, using bin midpoints)

| Group | n | mean GPA | SD | median | share < 3.0 |
|---|---:|---:|---:|---:|---:|
| Mainland Chinese | 37 | 3.443 | 0.320 | 3.500 | 18.9% |
| Non-mainland (intl + HK/Macau/TW) | 20 | 3.255 | 0.458 | 3.500 | 35.0% |
| International (only) | 13 | 3.354 | 0.372 | 3.500 | 30.8% |
| HK / Macau / Taiwan (only) | 7 | 3.071 | 0.571 | 3.500 | 42.9% |

_See `figures/09_gpa_by_group.png`._

## 3. Survey Likert results (pooled N = 57)

**Most-agreed-with items** (% agree or strongly agree, 95% Wilson CI):

- *Students at HKUST(GZ) tend to stay within their own cultural or national groups.* -- 70.2% (95% CI [57.3%, 80.5%], mean 3.77).
- *I regularly interact with local Chinese students outside of class.* -- 63.2% (95% CI [50.2%, 74.5%], mean 3.70).
- *I feel confident participating in class discussions.* -- 57.9% (95% CI [45.0%, 69.8%], mean 3.58).
- *When working on group projects, I feel my contributions are valued equally.* -- 57.9% (95% CI [45.0%, 69.8%], mean 3.60).
- *I feel confident in my ability to succeed academically at HKUST(GZ).* -- 56.1% (95% CI [43.3%, 68.2%], mean 3.42).
- *I have meaningful friendships with students from different backgrounds.* -- 56.1% (95% CI [43.3%, 68.2%], mean 3.47).

**Least-agreed-with items.**

- *Social difficulties on campus have negatively affected my actual grades.* -- 15.8% (95% CI [8.5%, 27.4%], mean 2.40).
- *Social difficulties on campus have negatively affected my motivation to study.* -- 15.8% (95% CI [8.5%, 27.4%], mean 2.44).
- *I have been excluded from, or had difficulty joining, a study group due to language or cultural differences.* -- 19.3% (95% CI [11.1%, 31.3%], mean 2.37).
- *Language differences make it difficult for me to form meaningful friendships here.* -- 29.8% (95% CI [19.5%, 42.7%], mean 2.81).
- *The university does enough to help international students integrate and succeed.* -- 29.8% (95% CI [19.5%, 42.7%], mean 3.09).

_See `figures/04_agree_ci.png` and `figures/06_integration_likert_stack.png`._

## 4. International (+HKMT) vs mainland -- direct comparison

With row-level data now available, we can compare the two groups directly instead of relying on the null-model bounds used earlier.

### 4a. GPA gap

Using the 5-bin ordinal GPA code (1 = Below 2.3 ... 5 = Above 3.7, pre-imputation):

| | Non-mainland (n = 20) | Mainland (n = 37) |
|---|---:|---:|
| mean GPA code | 3.6 | 4.0 |
| mean (midpoint) | 3.255 | 3.443 |
| share below 3.0 | 35.0% | 18.9% |

**Test statistics** (non-mainland vs mainland):

- Mann-Whitney U: two-sided p = **0.122**, one-sided (non-mainland lower) p = **0.061**.
- Cliff's delta: **-0.222** (small effect).
- Fisher exact (below 3.0 vs >= 3.3): odds ratio = 2.308, p = 0.209.
- Permutation (10000 iter): two-sided p = 0.067, one-sided p = 0.045.

**Interpretation.** The gap is in the predicted direction -- non-mainland students have a lower mean GPA code than mainland -- but at n = 57 the two-sided Mann-Whitney misses conventional significance (p = 0.12); the one-sided permutation test is at p = 0.04. Cliff's delta of -0.222 indicates a small-to-medium effect. A notable sub-finding is that the HK/Macau/Taiwan sub-group (n = 7) has the lowest mean GPA midpoint (3.07), below both international-only (3.35) and mainland (3.44) -- small n but worth flagging.

_See `figures/09_gpa_by_group.png`._

### 4b. Difference of opinion across Likert items

For each Likert item, we compare the non-mainland mean response against the mainland mean on the 1-5 scale, using Mann-Whitney U with Benjamini-Hochberg correction across the items.

| question_id            |   mean_non_mainland |   mean_mainland |   mean_diff |   cliffs_delta |   mannwhitney_p |   p_bh |
|:-----------------------|--------------------:|----------------:|------------:|---------------:|----------------:|-------:|
| interact_chinese       |                3.1  |           4.027 |      -0.927 |         -0.412 |           0.007 |  0.096 |
| excluded_study_group   |                2.85 |           2.108 |       0.742 |          0.372 |           0.015 |  0.098 |
| belong                 |                3.25 |           3.622 |      -0.372 |         -0.265 |           0.085 |  0.193 |
| language_hard          |                3.25 |           2.568 |       0.682 |          0.297 |           0.057 |  0.193 |
| confident_succeed      |                3.2  |           3.541 |      -0.341 |         -0.257 |           0.085 |  0.193 |
| uni_enough             |                2.8  |           3.243 |      -0.443 |         -0.258 |           0.089 |  0.193 |
| social_motivation      |                2.6  |           2.351 |       0.249 |          0.138 |           0.361 |  0.671 |
| meaningful_friendships |                3.65 |           3.378 |       0.272 |          0.091 |           0.556 |  0.722 |
| stay_in_groups         |                3.85 |           3.73  |       0.12  |          0.088 |           0.536 |  0.722 |
| social_grades          |                2.5  |           2.351 |       0.149 |          0.095 |           0.53  |  0.722 |
| class_confidence       |                3.65 |           3.541 |       0.109 |          0.064 |           0.681 |  0.805 |
| comfort_init           |                3.6  |           3.541 |       0.059 |          0.039 |           0.803 |  0.87  |
| valued_in_groups       |                3.65 |           3.568 |       0.082 |          0.004 |           0.986 |  0.986 |

**Items where non-mainland and mainland differ most (BH-adjusted):**

- *I regularly interact with local Chinese students outside of class.* -- non-mainland mean 3.10 vs mainland mean 4.03 (diff -0.93, Cliff's delta -0.41, p = 0.007, BH p = 0.096). non-mainland *less* likely to agree.
- *I have been excluded from, or had difficulty joining, a study group due to language or cultural differences.* -- non-mainland mean 2.85 vs mainland mean 2.11 (diff +0.74, Cliff's delta +0.37, p = 0.015, BH p = 0.098). non-mainland *more* likely to agree.
- *I feel that I belong at HKUST(GZ).* -- non-mainland mean 3.25 vs mainland mean 3.62 (diff -0.37, Cliff's delta -0.27, p = 0.085, BH p = 0.193). non-mainland *less* likely to agree.
- *Language differences make it difficult for me to form meaningful friendships here.* -- non-mainland mean 3.25 vs mainland mean 2.57 (diff +0.68, Cliff's delta +0.30, p = 0.057, BH p = 0.193). non-mainland *more* likely to agree.

At n = 57 (20 vs 37) no item survives BH correction at alpha = 0.05, but `interact_chinese` (-0.93, BH p = 0.096) and `excluded_study_group` (+0.74, BH p = 0.098) are the clearest directional gaps and match the qualitative interview evidence: non-mainland students interact less with local Chinese peers and report more study-group exclusion.

_See `figures/11_likert_by_group.png`._

## 5. Interview theme analysis (n = 21, Fisher exact + BH correction)

Interviews were coded for 14 binary themes. Prevalence by group:

| theme                       |   intl_pct |   mnl_pct |   diff_pct |   fisher_p |   fisher_p_bh |
|:----------------------------|-----------:|----------:|-----------:|-----------:|--------------:|
| ucug_easier_than_ufug       |      0.909 |     0     |      0.909 |     0.0001 |        0.0017 |
| mandarin_dominance          |      0.818 |     0     |      0.818 |     0.0003 |        0.0023 |
| social_affects_grades       |      0.727 |     0     |      0.727 |     0.0014 |        0.0064 |
| excluded_from_wechat        |      0.636 |     0     |      0.636 |     0.0047 |        0.0165 |
| clubs_english_only          |      0.455 |     0.111 |      0.343 |     0.1571 |        0.4399 |
| chinese_peer_introvert      |      0.364 |     0.111 |      0.253 |     0.3189 |        0.6378 |
| chinese_peers_avoid_english |      0.364 |     0.111 |      0.253 |     0.3189 |        0.6378 |
| stays_with_own_group        |      1     |     0.889 |      0.111 |     0.45   |        0.7875 |
| non_gaokao_prep_gap         |      0.273 |     0.111 |      0.162 |     0.5913 |        0.8279 |
| wants_structured_programs   |      0.273 |     0.111 |      0.162 |     0.5913 |        0.8279 |
| low_belonging               |      0.636 |     0.444 |      0.192 |     0.6534 |        0.8316 |
| language_barrier            |      0.909 |     1     |     -0.091 |     1      |        1      |
| uni_insufficient            |      0.818 |     0.889 |     -0.071 |     1      |        1      |
| intl_comms_chinese_default  |      0.091 |     0.111 |     -0.02  |     1      |        1      |

**Significantly different themes (BH-adjusted p < 0.05):**

- *ucug easier than ufug*: 90.9% of internationals vs 0.0% of mainland students (Fisher p = 0.0001, BH p = 0.0017).
- *mandarin dominance*: 81.8% of internationals vs 0.0% of mainland students (Fisher p = 0.0003, BH p = 0.0023).
- *social affects grades*: 72.7% of internationals vs 0.0% of mainland students (Fisher p = 0.0014, BH p = 0.0064).
- *excluded from wechat*: 63.6% of internationals vs 0.0% of mainland students (Fisher p = 0.0047, BH p = 0.0165).

**Theme burden** (number of themes a respondent reports, out of 14): international = 8.273 on average, mainland = 3.889. Permutation test on the difference: p = 0.0030 (10000 iterations).

_See `figures/07_interview_theme_prevalence.png`, `figures/08_theme_burden.png`, and `figures/12_interview_opinion_differences.png`._

### 5b. Second-pass themes (from long-transcript re-read)

A second reading of the long docx transcripts surfaced five additional recurring observations not captured in the first-pass codebook. Coded conservatively (theme = 1 only when explicitly raised):

| theme                       |   intl_pct |   mnl_pct |   diff_pct |   fisher_p |   fisher_p_bh |
|:----------------------------|-----------:|----------:|-----------:|-----------:|--------------:|
| clubs_english_only          |      0.455 |     0.111 |      0.343 |     0.1571 |        0.4399 |
| chinese_peers_avoid_english |      0.364 |     0.111 |      0.253 |     0.3189 |        0.6378 |
| non_gaokao_prep_gap         |      0.273 |     0.111 |      0.162 |     0.5913 |        0.8279 |
| wants_structured_programs   |      0.273 |     0.111 |      0.162 |     0.5913 |        0.8279 |
| intl_comms_chinese_default  |      0.091 |     0.111 |     -0.02  |     1      |        1      |

_See `figures/10_new_interview_themes.png`._

## 6. Headline findings

1. **The survey's missing 3.0-3.3 GPA bin was absorbing roughly 44 % of the inflated 3.3-3.7 bin.** After imputation, the pooled GPA distribution recovers a smooth truncated-normal shape (mu ~ 3.30, sigma ~ 0.40) that goodness-of-fit tests cannot reject.
2. **Non-mainland students have a lower mean GPA than mainland students** -- 3.25 vs 3.44 -- but the survey at n = 57 does not give significance at alpha = 0.05 (Mann-Whitney two-sided p = 0.12, Cliff's delta -0.222). The HK/Macau/Taiwan sub-group specifically has the lowest mean (3.07, n = 7).
3. **The two clearest social-integration gaps between groups are `interact_chinese` and `excluded_study_group`** -- non-mainland students interact less with local Chinese peers and report more study-group exclusion. Both items show large mean differences (|Cliff's delta| > 0.37) and raw p < 0.02, even though BH correction across 13 items brings them just above the alpha = 0.05 threshold.
4. **Interview evidence is unambiguous.** 4 themes separate international and mainland interview respondents with BH-adjusted p < 0.05: ucug easier than ufug, mandarin dominance, social affects grades, excluded from wechat. Mainland respondents essentially never report these.
5. **Theme burden** permutation test (p = 0.0030) shows international interviewees report roughly twice as many structural friction themes as mainland students -- 8.273 vs 3.889 on a 0-14 scale.

## 7. Recommendations for follow-up data collection

- **Fix the GPA options** so `3.0-3.3` is explicit; re-run with a larger cohort to firm up the GPA-gap estimate.
- **Oversample internationals and HKMT students.** The current 20-vs-37 split leaves the comparison under-powered; a target of 25-30 per group would give reasonable power for a Mann-Whitney detect of Cliff's delta ~0.222.
- **Break out HKMT separately in future analyses** -- they appear to be the lower-performing sub-group, not internationals proper.
- **Add a question on UCUG vs UFUG self-reported GPA** so the course-type asymmetry from interviews can be tested in the survey.

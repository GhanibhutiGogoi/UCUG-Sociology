# HKUST(GZ) Social Integration & Academic Experience -- Findings Report

**Project.** Exploring why international undergraduates at HKUST(GZ) tend to record lower GPAs than domestic Chinese peers, and what social-integration factors mediate that gap.

**Data sources.**
- Survey (n = 57): pooled across two export batches of the same questionnaire, loaded from the xlsx row-level exports. Status breakdown: 13 international, 37 domestic Chinese, 7 HK/Macau/Taiwan.
- 30 coded interview transcripts spanning international students (Macau, Kazakhstan, Kyrgyzstan, Indonesia, Italy, India, Malaysia), Chinese mainland, Hong Kong, and Taiwanese peers.

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

### 4c. Other categorical variables by group

**Social circle size** (later-batch only, n = 43). 
Non-mainland (n = 16) mean code 3.562 vs mainland (n = 27) mean code 3.333. Mann-Whitney p = 0.554, Cliff's delta = +0.10. The two groups have essentially indistinguishable social-circle distributions -- both cluster tightly at 3-5 close contacts, with small tails at 0-2 and 6+.

**Awareness of international-student support services** (pooled n = 57). This is a notable null finding: despite the services being aimed at non-mainland students, their awareness is statistically at parity with mainland respondents (non-mainland mean code 2.1, mainland 2.054, Mann-Whitney p = 0.774, Cliff's delta = +0.04). The university's outreach is not visibly reaching the audience it targets.

**Year and gender** (later-batch only, n = 43). 
Year distribution does not differ significantly by group (Mann-Whitney p = 0.343); non-mainland respondents skew slightly earlier in the programme (Cliff's delta = -0.17).
Gender composition also does not differ significantly (chi^2 p = 0.302).

_See `figures/05_social_circle_by_group.png` and `figures/13_categoricals_by_group.png`._

## 5. Interview theme analysis (n = 30, Fisher exact + BH correction)

Interviews were coded for 14 binary themes. Prevalence by group:

| theme                       |   intl_pct |   mnl_pct |   diff_pct |   fisher_p |   fisher_p_bh |
|:----------------------------|-----------:|----------:|-----------:|-----------:|--------------:|
| ucug_easier_than_ufug       |      0.889 |       0   |      0.889 |     0      |        0.0001 |
| mandarin_dominance          |      0.778 |       0   |      0.778 |     0.0002 |        0.0007 |
| social_affects_grades       |      0.778 |       0   |      0.778 |     0.0002 |        0.0007 |
| excluded_from_wechat        |      0.611 |       0   |      0.611 |     0.0016 |        0.0054 |
| low_belonging               |      0.722 |       0.4 |      0.322 |     0.1245 |        0.3487 |
| clubs_english_only          |      0.444 |       0.2 |      0.244 |     0.2474 |        0.4329 |
| chinese_peers_avoid_english |      0.389 |       0.1 |      0.289 |     0.1937 |        0.4329 |
| wants_structured_programs   |      0.5   |       0.2 |      0.3   |     0.2264 |        0.4329 |
| stays_with_own_group        |      1     |       0.9 |      0.1   |     0.3571 |        0.5556 |
| uni_insufficient            |      0.778 |       0.9 |     -0.122 |     0.6264 |        0.7308 |
| non_gaokao_prep_gap         |      0.222 |       0.1 |      0.122 |     0.6264 |        0.7308 |
| intl_comms_chinese_default  |      0.222 |       0.1 |      0.122 |     0.6264 |        0.7308 |
| language_barrier            |      0.944 |       1   |     -0.056 |     1      |        1      |
| chinese_peer_introvert      |      0.222 |       0.2 |      0.022 |     1      |        1      |

**Significantly different themes (BH-adjusted p < 0.05):**

- *ucug easier than ufug*: 88.9% of internationals vs 0.0% of mainland students (Fisher p = 0.0000, BH p = 0.0001).
- *mandarin dominance*: 77.8% of internationals vs 0.0% of mainland students (Fisher p = 0.0002, BH p = 0.0007).
- *social affects grades*: 77.8% of internationals vs 0.0% of mainland students (Fisher p = 0.0002, BH p = 0.0007).
- *excluded from wechat*: 61.1% of internationals vs 0.0% of mainland students (Fisher p = 0.0016, BH p = 0.0054).

**Theme burden** (number of themes a respondent reports, out of 14): international = 8.5 on average, mainland = 4.1. Permutation test on the difference: p = 0.0002 (10000 iterations).

_See `figures/07_interview_theme_prevalence.png`, `figures/08_theme_burden.png`, and `figures/12_interview_opinion_differences.png`._

### 5b. Second-pass themes (from long-transcript re-read)

A second reading of the long docx transcripts surfaced five additional recurring observations not captured in the first-pass codebook. Coded conservatively (theme = 1 only when explicitly raised):

| theme                       |   intl_pct |   mnl_pct |   diff_pct |   fisher_p |   fisher_p_bh |
|:----------------------------|-----------:|----------:|-----------:|-----------:|--------------:|
| wants_structured_programs   |      0.5   |       0.2 |      0.3   |     0.2264 |        0.4329 |
| chinese_peers_avoid_english |      0.389 |       0.1 |      0.289 |     0.1937 |        0.4329 |
| clubs_english_only          |      0.444 |       0.2 |      0.244 |     0.2474 |        0.4329 |
| non_gaokao_prep_gap         |      0.222 |       0.1 |      0.122 |     0.6264 |        0.7308 |
| intl_comms_chinese_default  |      0.222 |       0.1 |      0.122 |     0.6264 |        0.7308 |

_See `figures/10_new_interview_themes.png`._

## 6. Exploratory analyses

### 6a. PCA on the 14 Likert items

Each Likert response was coded 1-5. Items where agreement signals a negative experience (excluded_study_group, language_hard, social_grades, social_motivation, stay_in_groups) were reverse-coded so that higher values consistently mean 'better integration / more positive experience' on every dimension. Standardised and decomposed with PCA.

**Variance explained.** PC1 = 30.4%, PC2 = 15.4%, PC3 = 10.8% (cumulative: 2 PCs = 45.8%, 3 PCs = 56.6%).

**PC1 is a 'general integration' factor.** The five items with the largest absolute PC1 loadings:

- `belong`: PC1 = +0.385
- `social_motivation`: PC1 = +0.381
- `social_grades`: PC1 = +0.376
- `excluded_study_group`: PC1 = +0.327
- `language_hard`: PC1 = +0.308

All five load in the same direction after reverse-coding, meaning respondents who feel they belong also feel more motivated, report better grades, less study-group exclusion, and less language difficulty -- consistent with a single underlying 'integration' dimension capturing ~30 % of Likert variance.

_See `figures/14_likert_pca.png`._

### 6b. Q23 open-ended text analysis (n = 56)

Open-ended answers to 'in what ways has your social experience at HKUST(GZ) affected your academic life, and what one change would most improve your experience?' were scored with a simple bilingual polarity lexicon (English + Chinese stems). Responses with no polarity hits and fewer than 25 characters, or containing explicit no-opinion phrases (`idk`, `no effect`, `I don't know`, etc.), were classified as *unscorable*.

**Sentiment distribution by group:**

| sentiment | non-mainland | mainland |
|---|---:|---:|
| positive | 5 | 13 |
| neutral | 7 | 8 |
| negative | 5 | 3 |
| unscorable | 3 | 13 |

Among scorable responses, **29.4% (5/17) of non-mainland responses are classified negative vs 12.5% (3/24) of mainland responses** (Fisher odds ratio = 2.917, p = 0.241). The direction matches the interview and Likert evidence; at n = 41 scorable responses the test is under-powered to reach alpha = 0.05.

**Top content words by group** (stopwords removed, bilingual):

| rank | non-mainland | count | mainland | count |
|---:|---|---:|---|---:|
| 1 | english | 7 | academic | 7 |
| 2 | international | 6 | good | 6 |
| 3 | chinese | 5 | friends | 6 |
| 4 | lab | 4 | social | 5 |
| 5 | work | 4 | course | 4 |
| 6 | research | 3 | learning | 3 |
| 7 | study | 3 | life | 3 |
| 8 | maybe | 3 | better | 3 |
| 9 | due | 3 | courses | 3 |
| 10 | because | 3 | love | 3 |

The word lists are the most legible finding in this whole analysis. Non-mainland responses are dominated by language and work/research vocabulary (*english*, *international*, *chinese*, *lab*, *research*) -- i.e. what language the environment runs in and where they end up interacting. Mainland responses are dominated by positive social/academic vocabulary (*good*, *friends*, *social*, *academic*, *better*, *learning*) -- i.e. they talk about enjoying the integration the other group is describing the absence of.

_See `figures/15_q23_sentiment_and_keywords.png` and `data/q23_sentiment.csv` for per-respondent labels._

## 7. Headline findings

1. **The survey's missing 3.0-3.3 GPA bin was absorbing roughly 44 % of the inflated 3.3-3.7 bin.** After imputation, the pooled GPA distribution recovers a smooth truncated-normal shape (mu ~ 3.30, sigma ~ 0.40) that goodness-of-fit tests cannot reject.
2. **Non-mainland students have a lower mean GPA than mainland students** -- 3.25 vs 3.44 -- but the survey at n = 57 does not give significance at alpha = 0.05 (Mann-Whitney two-sided p = 0.12, Cliff's delta -0.222). The HK/Macau/Taiwan sub-group specifically has the lowest mean (3.07, n = 7).
3. **The two clearest social-integration gaps between groups are `interact_chinese` and `excluded_study_group`** -- non-mainland students interact less with local Chinese peers and report more study-group exclusion. Both items show large mean differences (|Cliff's delta| > 0.37) and raw p < 0.02, even though BH correction across 13 items brings them just above the alpha = 0.05 threshold.
4. **Interview evidence is unambiguous.** 4 themes separate international and mainland interview respondents with BH-adjusted p < 0.05: ucug easier than ufug, mandarin dominance, social affects grades, excluded from wechat. Mainland respondents essentially never report these.
5. **Theme burden** permutation test (p = 0.0002) shows international interviewees report roughly twice as many structural friction themes as mainland students -- 8.5 vs 4.1 on a 0-14 scale.

## 8. Recommendations for follow-up data collection

- **Fix the GPA options** so `3.0-3.3` is explicit; re-run with a larger cohort to firm up the GPA-gap estimate.
- **Oversample internationals and HKMT students.** The current 20-vs-37 split leaves the comparison under-powered; a target of 25-30 per group would give reasonable power for a Mann-Whitney detect of Cliff's delta ~0.222.
- **Break out HKMT separately in future analyses** -- they appear to be the lower-performing sub-group, not internationals proper.
- **Add a question on UCUG vs UFUG self-reported GPA** so the course-type asymmetry from interviews can be tested in the survey.

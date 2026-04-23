# Slide text & speaker notes

Companion file to the chart PNGs in this directory. Each section below is one slide.
Copy the **Title**, **Bullets**, and (optionally) **Speaker notes** into your slide tool,
and insert the named chart image where indicated.

Colour key used throughout the deck: **Mainland = blue**, **Non-mainland = red**,
**accent / highlight = orange**.

**Deck structure (14 slides):**

1. Title
2. Research question & data
3. A critical survey design flaw
4. Mathematical normalization of the flawed bin   ← *new — explicit formulas*
5. Truncated-normal density fit (visual)
6. The corrected GPA distribution (before / after)
7. The GPA gap: real but modest
8. Where the groups actually differ (Likert)
9. Interviews: an unambiguous signal
10. What students write, unprompted (Q23 keywords)
11. A striking null: services aren't reaching the target
12. One underlying 'integration' factor (PCA)
13. Conclusions
14. Recommendations for follow-up

---

## Slide 1 — Title slide  (no chart)

**Title (large)**
Social Integration and the GPA Gap at HKUST(GZ)

**Subtitle**
A mixed-methods study of international and mainland undergraduate experience

**Footer line**
Division of Social Science · HKUST(GZ) · April 2026 · N = 57 survey + 30 interviews

**Speaker notes**
> Today I'll walk through a mixed-methods study we did on why international students at HKUST(GZ) tend to record lower GPAs than their mainland peers, and what social-integration factors seem to mediate that gap. We'll go from a survey-design problem we had to fix, through the GPA gap itself, into the Likert and interview evidence for *why* that gap exists.

---

## Slide 2 — Research question & data
**Chart:** `slide_02_sample.png`

**Title**
Research question & data

**Bullets**
- **Why do international students at HKUST(GZ) record lower GPAs than their mainland Chinese peers, and what social-integration factors mediate that gap?**
- Mixed methods:
  - **Survey (n = 57)** — 20 questions including GPA and 14 Likert items
  - **Interviews (n = 30, coded across 14 binary themes)** — 18 international, 10 mainland, 2 HK / Macau / Taiwan
- Sample skews mainland in the survey (37) — non-mainland survey sample (20) is modest but coherent

**Speaker notes**
> The survey was pooled from two export batches of the same questionnaire, which is why we end up at n = 57 rather than two separate samples. Three questions — gender, year, and social-circle size — were only present in the later batch, so those are reported on n = 43. Everything else uses the full 57.

---

## Slide 3 — A critical survey design flaw
**Chart:** `slide_03_raw_gpa.png`

**Title**
A critical survey design flaw

**Subtitle**
The GPA question skipped the 3.0 – 3.3 bin

**Bullets**
- The options given were: **Below 2.3 · 2.3–2.7 · 2.7–3.0 · 3.3–3.7 · Above 3.7**
- Students whose true GPA was in **3.0–3.3** had to pick the nearest bin — almost all went to **3.3–3.7**
- The tell: **2.7–3.0 holds 21 %**, but the next available bin **jumps to 60 %**
- No smooth unimodal GPA distribution produces a jump that large — the **3.3–3.7 bar is really the merged interval [3.0, 3.7)**

**Speaker notes**
> This is the first thing we noticed when we looked at the raw data, and we had to fix it before any group-level analysis would make sense. A 2.7–3.0 to 3.3–3.7 gap of 39 percentage points is not how any realistic GPA distribution behaves — it's a survey-design artifact, not a finding.

---

## Slide 4 — Mathematical normalization of the flawed bin
**Chart:** `slide_04_math_fix.png`

**Title**
Mathematical normalization of the flawed bin

**Subtitle**
We fixed the missing 3.0 – 3.3 bar with a truncated-normal maximum-likelihood fit

**Bullets**
- **The survey was flawed**: the 3.0 – 3.3 GPA bin was simply missing from the response options, so those respondents got absorbed into 3.3 – 3.7.
- **The fix**: we mathematically normalized the distribution — i.e., we fit a continuous probability distribution to the *observed-as-surveyed* five bins and used that distribution to recover the missing bin.
- **Model — truncated normal on [0, 4.3]:**
  - Density:  *f(x | μ, σ) = (1/σ) · φ((x−μ)/σ) / [Φ((b−μ)/σ) − Φ((a−μ)/σ)]*
  - Bin probability:  *P_i(μ, σ) = [Φ((u_i−μ)/σ) − Φ((l_i−μ)/σ)] / [Φ((b−μ)/σ) − Φ((a−μ)/σ)]*
  - Multinomial MLE:  *(μ̂, σ̂) = argmax_{μ,σ} Σ_i n_i · log P_i(μ, σ)*
  - Imputed split:  *n̂_{3.0-3.3} = n_{[3.0, 3.7)} · P(3.0 ≤ X < 3.3) / P(3.0 ≤ X < 3.7)*
- **Result:**  **μ̂ = 3.30**, **σ̂ = 0.40**, **n̂_{3.0-3.3} ≈ 15.1**
- **Validation:**  χ² = 0.12, dof = 1, **p = 0.74** — cannot reject the truncated-normal shape.

**Speaker notes**
> I want to be completely explicit about the fix here because it's central to everything that follows. The survey gave five GPA options but skipped 3.0–3.3, so every respondent whose true GPA was in that range had to pick the nearest available bin, and we can see in the raw data that almost all of them rolled up into 3.3–3.7 — inflating it to 34 people when a smooth distribution predicts about 19.
>
> The method is called *multinomial maximum likelihood estimation of a truncated normal distribution*. Concretely: we assume GPA is continuous on the interval [0, 4.3], and the density on that interval is a truncated normal (a normal renormalized so its integral over [0, 4.3] equals 1) parameterised by mean μ and standard deviation σ. Each of the five observed bins has a probability P_i under this model, which is just the CDF difference at the bin edges. We write down the multinomial log-likelihood of the observed counts and maximise it numerically over μ and σ. That gives us μ̂ = 3.30 and σ̂ = 0.40. Then we use the fitted CDF to split the inflated [3.0, 3.7) interval proportionally — about 15 respondents belong in the missing 3.0–3.3 bin, and about 19 in the real 3.3–3.7.
>
> The χ² goodness-of-fit test on the five observed bins comes back at p = 0.74, so the truncated-normal shape is not rejected. We also ran beta-distribution and log-linear sensitivity checks and they agree on the split within about one respondent — the result doesn't hinge on the choice of functional form.

---

## Slide 5 — Truncated-normal density fit
**Chart:** `slide_05_density_fit.png`

**Title**
Truncated-normal density fit to the observed histogram

**Subtitle**
Visual check that the fitted curve matches the observed bins

**Bullets**
- Red curve: fitted truncated normal with **μ̂ = 3.30, σ̂ = 0.40**
- Blue bars: observed histogram **after** the [3.0, 3.7) split
- Gold shaded region: the **recovered 3.0 – 3.3 bin** (absent from the survey itself)
- Goodness-of-fit **χ² = 0.12, p = 0.74** — visually and statistically a good fit

**Speaker notes**
> This slide is purely a visual sanity check of the model on the previous slide. The fitted curve sits on top of the observed histogram like you'd want — no bar is dramatically above or below the curve. The gold band is the 3.0–3.3 bin we recovered from the fit; you can see it's the peak of the underlying distribution, which explains why its omission created such a large pile-up in the next bin over.

---

## Slide 6 — The corrected GPA distribution
**Chart:** `slide_06_before_after.png`

**Title**
The corrected GPA distribution

**Subtitle**
~ 44 % of the inflated 3.3 – 3.7 bar moves to 3.0 – 3.3

**Bullets**
- Before: 34 respondents piled into 3.3–3.7 (red bar)
- After: that splits into **≈ 15 in 3.0–3.3** (gold bar) and **≈ 19 in 3.3–3.7**
- Post-imputation pooled summary: **N = 57 · mean 3.28 · SD 0.38 · median 3.15**
- Shares: **< 3.0 = 25 % · 3.0–3.3 = 27 % · ≥ 3.3 = 49 %**

**Speaker notes**
> This is what the corrected distribution looks like. The 3.3–3.7 bar comes down to a plausible ~19 respondents and the new 3.0–3.3 bar picks up ~15 — matching what you'd expect from a smooth unimodal distribution centered near 3.3.

---

## Slide 7 — The GPA gap: real but modest
**Chart:** `slide_07_gpa_by_group.png`

**Title**
The GPA gap: real but modest

**Subtitle**
Direction matches hypothesis; n = 57 is under-powered

**Bullets**
- **Non-mainland (n = 20) mean = 3.26** vs **Mainland (n = 37) mean = 3.44**  (Δ = −0.19)
- Share below 3.0: **non-mainland 35 %** vs **mainland 19 %**
- Mann-Whitney: two-sided p = 0.12, one-sided (non-mainland lower) p = 0.06
- Cliff's δ = **−0.22** (small-to-medium effect)
- Permutation test (10 k iter): one-sided p = **0.045**
- **HK / Macau / Taiwan sub-group (n = 7) is the lowest performer**: mean 3.07, 43 % below 3.0

**Speaker notes**
> The gap is in the predicted direction, but at n = 57 we sit right on the edge of significance — two-sided Mann-Whitney p = 0.12, one-sided permutation p = 0.045. What's worth flagging is that 'international' isn't a monolith. The HK / Macau / Taiwan sub-group, despite speaking Cantonese/Mandarin, is actually the lowest-performing subset, below both internationals proper and mainland students. That's a finding the university probably wasn't expecting.

---

## Slide 8 — Where the groups actually differ
**Chart:** `slide_08_likert_diff.png`

**Title**
Where the groups actually differ

**Subtitle**
Two Likert items clearly separate non-mainland from mainland

**Bullets**
- `interact_chinese` — **non-mainland 3.10 vs mainland 4.03** (Δ = −0.93, Cliff's δ = −0.41, p = 0.007, BH p = 0.10)
- `excluded_study_group` — **non-mainland 2.85 vs mainland 2.11** (Δ = +0.74, Cliff's δ = +0.37, p = 0.015, BH p = 0.10)
- The direction **matches the interview evidence exactly**: non-mainland students interact less with Chinese peers **and** report more study-group exclusion
- Neither passes BH at α = 0.05 at n = 57, but effect sizes are non-trivial and robust
- 11 of 13 remaining items show no meaningful group difference — the gap is specific, not diffuse

**Speaker notes**
> We ran Mann-Whitney on each of the 13 Likert items with Benjamini-Hochberg correction across the battery. Two items pop: how often they interact with Chinese students outside class, and whether they feel excluded from study groups. That's tightly consistent with what interview respondents describe in the next slide. The point I'd make from this panel is that the disagreement between groups is *specific* — it's about contact and inclusion, not about confidence or belonging in general.

---

## Slide 9 — Interviews: an unambiguous signal
**Chart:** `slide_09_interview_themes.png`
**Optional second chart:** `slide_09b_theme_burden.png`

**Title**
Interviews: an unambiguous signal

**Subtitle**
Joint status × opinion data is strongest here

**Bullets**
- 14 binary themes coded across **30 interview transcripts** (18 international, 10 mainland, 2 HK/Macau/Taiwan)
- **4 themes separate international from mainland at BH p < 0.05:**
  - *UCUG easier than UFUG*: **89 % vs 0 %**  (BH p = 0.0001)
  - *Mandarin dominates social spaces*: **78 % vs 0 %**  (BH p = 0.0007)
  - *Social exclusion hurts grades*: **78 % vs 0 %**  (BH p = 0.0007)
  - *Excluded from WeChat / information networks*: **61 % vs 0 %**  (BH p = 0.005)
- **Theme burden**: international mean = **8.5 of 14**, mainland = **4.1 of 14** (permutation p = **0.0002**, 10 000 iterations)

**Speaker notes**
> This is the strongest evidence in the study. 30 interviews were coded end-to-end — 18 international, 10 mainland, 2 HK/Macau/Taiwan. Four themes appear in a clear majority of international interviews and in *zero* of the mainland interviews — those Fisher tests survive BH correction by a wide margin (smallest BH p is 0.0001 for UCUG-easier, largest among the four is 0.005 for WeChat exclusion). The theme-burden result is arguably more important than any individual theme: on average, international respondents raise more than twice as many structural friction themes as mainland respondents (8.5 vs 4.1 out of 14), with permutation p = 0.0002. The gap is structural, not about a single grievance.

---

## Slide 10 — What students write, unprompted
**Chart:** `slide_10_keywords.png`

**Title**
What students write, unprompted

**Subtitle**
Q23 open-ended keyword contrast tells the clearest story

**Bullets**
- 56 / 57 respondents wrote a free-text answer to Q23 ("how has your social experience affected your academic life, and what one change would improve it?")
- **Non-mainland vocabulary** (top 6): *english · international · chinese · lab · work · research*
- **Mainland vocabulary** (top 6): *academic · good · friends · social · course · learning*
- Among scorable responses: **29 % of non-mainland are negative** vs **13 % of mainland** (Fisher p = 0.24, under-powered)
- **Different vocabularies, not just different tones** — one group talks about language and labs, the other about friends and learning

**Speaker notes**
> This is the slide I'd point to as the most legible finding in the whole study. We didn't do any sentiment-model magic here — it's just the top unigrams after stopword removal. And you can almost see the two experiences in the word lists: non-mainland respondents write about the language the environment runs in and where they end up spending their time, which is the lab. Mainland respondents talk about enjoying exactly the thing the other group is describing the absence of — friends, social life, learning.

---

## Slide 11 — A striking null: services aren't reaching the target
**Chart:** `slide_11_aware_services.png`

**Title**
A striking null: services aren't reaching the target

**Subtitle**
Awareness of international-student services is statistically identical across groups

**Bullets**
- The question: *"I am aware of support services available to international students at HKUST(GZ)."*
- **Expectation**: non-mainland students should be more aware — these services are *for* them
- **Actual result (pooled n = 57):**
  - Non-mainland mean code = **2.10**
  - Mainland mean code = **2.05**
  - Cliff's δ ≈ **+0.04** · Mann-Whitney p = **0.77**
- Groups are **statistically indistinguishable**. The outreach is not visibly reaching the audience it targets.

**Speaker notes**
> This is a null finding but I think it's important. If you set up services specifically for international students and at the end of two years the non-mainland respondents are *not* more aware of them than mainland respondents, then whatever communication channel you are using is not working — probably because announcements default to Chinese-language WeChat groups that internationals don't reliably read. This ties directly to the *excluded from WeChat* theme from the interviews.

---

## Slide 12 — One underlying 'integration' factor
**Chart:** `slide_12_pca.png`

**Title**
One underlying 'integration' factor

**Subtitle**
PCA on the 14 Likert items, reverse-coded where needed

**Bullets**
- PC1 = **30.4 %** · PC2 = **15.4 %** · PC3 = **10.8 %**  (2 PCs = 46 %, 3 PCs = 57 %)
- **PC1 is a clean 'general integration' factor** — top loadings:
  - belonging → high PC1
  - motivation → high PC1
  - better grades → high PC1
  - less study-group exclusion → high PC1
  - less language difficulty → high PC1
- Respondents who feel they belong *also* feel more motivated, report better grades, etc. — **these are not independent struggles**
- Practically: the 14 Likert items can compress into a single **'integration score'** with little loss

**Speaker notes**
> After reverse-coding the items where agreement means a *negative* experience, the first principal component cleanly loads in one direction on five different items — belonging, motivation, grades, study-group inclusion, and language ease. That's PC1 behaving like a single latent 'integration' variable, which justifies treating these Likert items as a battery rather than 14 independent questions. In the scatter, you can see non-mainland respondents pulled leftward on PC1 on average — exactly what we'd predict.

---

## Slide 13 — Conclusions
**Chart:** `slide_13_headline.png`  (optional summary visual)

**Title**
Conclusions

**Bullets**
1. **The GPA gap is real but modest.** Non-mainland 3.26 vs mainland 3.44 (Cliff's δ = −0.22). Under-powered at n = 57.
2. **HK / Macau / Taiwan is the lowest-performing sub-group** (mean 3.07), not internationals proper — 'international' is not a monolith.
3. **The mechanism is language + exclusion, not ability.** Survey, interviews, and open-ended text all converge on: less interaction with Chinese peers, more study-group exclusion, Mandarin-dominant clubs, no Gaokao STEM preparation.
4. **Institutional support is not reaching its audience.** Awareness of international-student services is statistically identical across groups.
5. **The friction is structural, not individual.** Theme-burden permutation test on n = 30 interviews: international 8.5 themes vs mainland 4.1 (p = 0.0002).

**Speaker notes**
> Five takeaways. The GPA gap is the surface symptom. The real story is that non-mainland students are systematically less plugged in to the social and informational fabric of the university — WeChat groups, study groups, clubs, announcements — and this cashes out in both subjective experience and, modestly, in grades. It's a structural problem the institution can intervene on.

---

## Slide 14 — Recommendations for follow-up  (no chart)

**Title**
Recommendations

**Bullets**
- **Survey design**
  - Fix the GPA option set — include 3.0–3.3 explicitly
  - Export row-level data from the start (not PDF summaries)
  - Add a UCUG-vs-UFUG self-reported GPA question to test the course-type asymmetry directly
- **Sample size**
  - Target n ≈ 25–30 per group for Cliff's δ ≈ 0.2 power
  - **Oversample HK / Macau / Taiwan** — currently n = 7 is too small to estimate the sub-group effect with any precision
- **Institutional actions**
  - Language-paired study groups or buddy scheme
  - Clubs / RC events with bilingual (or English-default) policy
  - Redesign international-student comms: non-Chinese subject lines, separate announcement channels
  - **Active referral** to support services, not passive availability

**Speaker notes**
> Two kinds of recommendations: what we'd do differently in a follow-up study, and what the institution could try based on what we've already found. The survey-design fix is easy. The oversampling recommendation is more important — 20-vs-37 leaves us under-powered, and 7 HK/Macau/Taiwan respondents isn't enough to say anything definitive about the subgroup we suspect is actually driving the gap. On the institutional side, the biggest lever is probably the communication-channel redesign — if non-mainland students aren't on the WeChat groups where everything happens, no amount of 'support services' will find them.

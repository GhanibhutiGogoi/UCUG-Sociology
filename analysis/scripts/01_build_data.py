"""Build tidy CSVs from the two survey xlsx exports.

The two xlsx exports (352837075, n=14 and 356434018, n=43) are two batches of
the *same* questionnaire. We pool them into a single row-level table of n=57
for the shared questions; `gender`, `year`, and `social_circle` are only in
the later batch so they stay at n=43 and are flagged in the `scope` column of
the aggregated output.

Outputs:
  data/survey_rowlevel.csv   -- 57 rows, one per respondent, canonical columns
  data/survey_aggregated.csv -- long format (question_id, option, count, pct)
  data/interview_themes.csv  -- 30 interview respondents x 14 binary themes
                              (21 from the first coding pass + 9 added from
                              the two additional bulk-summary transcripts)
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURVEY_DIR = ROOT.parent  # the two xlsx files sit in the project root
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

XLSX_A = SURVEY_DIR / "352837075_按文本_Social Integration and Academic Experience of Students at HKUST(GZ)_14_14.xlsx"
XLSX_B = SURVEY_DIR / "356434018_按文本_Social Integration and Academic Experience of Students at HKUST(GZ)[复制]_43_43.xlsx"

# Map a distinctive substring of each survey question to a canonical id + type.
# Order of checks matters: more specific keys first.
CANONICAL_QUESTIONS = [
    ("student status", "status", "single"),
    ("nationality", "nationality", "text"),
    ("gender", "gender", "single"),
    ("level of study", "level", "single"),
    ("year of study", "year", "ordinal"),
    ("primary language", "language", "text"),
    ("cumulative GPA", "gpa", "ordinal"),
    ("main social circle", "social_circle", "ordinal"),
    ("I feel that I belong", "belong", "likert"),
    ("meaningful friendships with students from backgrounds", "meaningful_friendships", "likert"),
    ("interact with local Chinese", "interact_chinese", "likert"),
    ("initiating conversations", "comfort_init", "likert"),
    ("Language differences make it difficult", "language_hard", "likert"),
    ("stay within their own cultural", "stay_in_groups", "likert"),
    ("confident participating in class", "class_confidence", "likert"),
    ("contributions are valued equally", "valued_in_groups", "likert"),
    ("excluded from, or had difficulty joining", "excluded_study_group", "likert"),
    ("confident in my ability to succeed", "confident_succeed", "likert"),
    ("negatively affected my motivation", "social_motivation", "likert"),
    ("negatively affected my actual grades", "social_grades", "likert"),
    ("aware of support services", "aware_services", "single"),
    ("university does enough", "uni_enough", "likert"),
    ("In what ways has your social experience", "open_ended", "text"),
]
QUESTION_TEXT = {
    "status": "Student status",
    "nationality": "Nationality or region of origin",
    "gender": "Gender",
    "level": "Current level of study",
    "year": "Year of study",
    "language": "Primary language of daily communication",
    "gpa": "Current cumulative GPA",
    "social_circle": "Size of main social circle",
    "belong": "I feel that I belong at HKUST(GZ).",
    "meaningful_friendships": "I have meaningful friendships with students from different backgrounds.",
    "interact_chinese": "I regularly interact with local Chinese students outside of class.",
    "comfort_init": "I feel comfortable initiating conversations with students from a different cultural background.",
    "language_hard": "Language differences make it difficult for me to form meaningful friendships here.",
    "stay_in_groups": "Students at HKUST(GZ) tend to stay within their own cultural or national groups.",
    "class_confidence": "I feel confident participating in class discussions.",
    "valued_in_groups": "When working on group projects, I feel my contributions are valued equally.",
    "excluded_study_group": "I have been excluded from, or had difficulty joining, a study group due to language or cultural differences.",
    "confident_succeed": "I feel confident in my ability to succeed academically at HKUST(GZ).",
    "social_motivation": "Social difficulties on campus have negatively affected my motivation to study.",
    "social_grades": "Social difficulties on campus have negatively affected my actual grades.",
    "aware_services": "I am aware of support services available to international students.",
    "uni_enough": "The university does enough to help international students integrate and succeed.",
    "open_ended": "In what ways has your social experience at HKUST(GZ) affected your academic life?",
}
QUESTION_TYPE = {qid: qtype for _, qid, qtype in CANONICAL_QUESTIONS}

B_ONLY_QUESTION_IDS = {"gender", "year", "social_circle"}


def canonical_id(column_name: str) -> str | None:
    for key, qid, _ in CANONICAL_QUESTIONS:
        if key.lower() in column_name.lower():
            return qid
    return None


def load_xlsx(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    # Drop the platform-provided metadata columns (submit time, duration, etc).
    keep_cols = {}
    for col in raw.columns:
        qid = canonical_id(str(col))
        if qid is not None and qid not in keep_cols:
            keep_cols[qid] = col
    df = raw[list(keep_cols.values())].copy()
    df.columns = list(keep_cols.keys())
    # Normalise unicode dashes in GPA values (the xlsx uses both em-dash and
    # hyphen). Force everything to plain ASCII hyphens so bin labels line up.
    def normalise(v):
        if isinstance(v, str):
            return unicodedata.normalize("NFKC", v).replace("–", "-").replace("—", "-")
        return v
    return df.applymap(normalise)


def build_aggregated(rowlevel: pd.DataFrame, n_pooled: int, n_b_only: int) -> pd.DataFrame:
    rows = []
    for qid in rowlevel.columns:
        qtype = QUESTION_TYPE.get(qid)
        if qtype in ("text", None):
            continue
        sub = rowlevel[qid].dropna()
        if qid in B_ONLY_QUESTION_IDS:
            scope = f"B-only (n={n_b_only})"
            denom = n_b_only
        else:
            scope = f"pooled (n={n_pooled})"
            denom = n_pooled
        counts = sub.value_counts()
        for option, n in counts.items():
            rows.append({
                "question_id": qid,
                "question_text": QUESTION_TEXT.get(qid, qid),
                "type": qtype,
                "scope": scope,
                "option": str(option),
                "count": int(n),
                "pct": 100 * n / denom,
            })
    return pd.DataFrame(rows)


def build_interview_themes() -> pd.DataFrame:
    """Interview theme coding (unchanged from prior pass -- the xlsx data
    does not affect the interview corpus).

    The last five themes (clubs_english_only, chinese_peers_avoid_english,
    non_gaokao_prep_gap, intl_comms_chinese_default, wants_structured_programs)
    were added after a second pass through the long docx transcripts.
    Coded conservatively: 1 only when the interview explicitly raises the
    observation.
    """
    interviews = [
        ("intl_macao_yr2_smart", "international",
            0, 0, 1, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0),
        ("intl_macau_yr2_ai", "international",
            1, 1, 1, 1, 1, 1, 0, 0, 1,   1, 1, 0, 0, 0),
        ("intl_ca_erasyl", "international",
            1, 1, 1, 0, 0, 0, 0, 1, 0,   1, 0, 0, 0, 0),
        ("intl_ca_alisher", "international",
            1, 1, 1, 0, 1, 1, 0, 1, 0,   1, 1, 1, 1, 1),
        ("intl_kz_ramazan", "international",
            1, 1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 0, 1),
        ("intl_kz_kassymkhan", "international",
            1, 1, 1, 1, 1, 1, 1, 1, 1,   1, 1, 1, 0, 1),
        ("intl_italian_yr2", "international",
            1, 1, 1, 1, 0, 1, 0, 1, 1,   0, 0, 0, 0, 0),
        ("intl_indo_yr1_a", "international",
            1, 1, 1, 1, 0, 1, 0, 1, 1,   0, 0, 0, 0, 0),
        ("intl_indo_yr1_b", "international",
            1, 1, 1, 1, 1, 1, 0, 1, 1,   0, 0, 0, 0, 0),
        ("intl_indo_yr1_c", "international",
            1, 1, 1, 1, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0),
        ("intl_indo_yr1_d", "international",
            1, 1, 1, 0, 1, 1, 1, 1, 1,   0, 0, 0, 0, 0),
        ("cn_yr1_01", "mainland",
            1, 0, 1, 0, 0, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_yr2_02", "mainland",
            1, 0, 0, 0, 1, 1, 0, 0, 0,   0, 0, 0, 1, 0),
        ("cn_yr2_03", "mainland",
            1, 0, 1, 0, 0, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_yr1_04", "mainland",
            1, 0, 1, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_ug3_male", "mainland",
            1, 0, 1, 0, 1, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_ug3_female_a", "mainland",
            1, 0, 1, 0, 1, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_ug3_female_b", "mainland",
            1, 0, 1, 0, 0, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_ug3_female_c", "mainland",
            1, 0, 1, 0, 1, 1, 0, 0, 0,   0, 0, 0, 0, 0),
        ("cn_younger_yr1", "mainland",
            1, 0, 1, 0, 0, 1, 1, 0, 0,   1, 1, 1, 0, 1),
        ("tw_ug3_male", "hkmt",
            1, 0, 1, 0, 1, 1, 0, 0, 1,   0, 0, 0, 0, 0),
        # --- Additional interviews coded from the two bulk-summary sources -----
        # These were conducted as part of the same data-collection effort but
        # had not been theme-coded in the first pass. Sources:
        #   transcript_bullet_summary_anon.pdf  (R1 – R4, 4 respondents)
        #   transcription- bullet_point_summary.docx  (R1 – R5, 5 respondents)
        # Theme coding follows the same conservative "1 only when explicitly
        # raised" rule used for the original 21 rows.
        ("hkmt_hk_yr2_ai", "hkmt",
            1, 0, 1, 0, 1, 0, 0, 0, 0,   0, 0, 0, 0, 0),
        ("intl_my_yr2_ds", "international",
            1, 1, 1, 1, 1, 1, 0, 0, 1,   1, 0, 0, 0, 1),
        ("intl_in_yr2_ai", "international",
            1, 0, 1, 1, 1, 1, 0, 1, 1,   1, 0, 0, 1, 1),
        ("cn_yr2_ai_male", "mainland",
            1, 0, 1, 0, 0, 1, 1, 0, 0,   1, 0, 0, 0, 1),
        ("intl_my_yr2_smmg", "international",
            1, 1, 1, 1, 1, 1, 0, 1, 1,   1, 0, 0, 1, 1),
        ("intl_kz_yr1_ai_b", "international",
            1, 1, 1, 0, 1, 0, 0, 0, 0,   0, 0, 0, 0, 0),
        ("intl_kg_yr1_ds", "international",
            1, 1, 1, 0, 0, 0, 0, 1, 1,   0, 1, 0, 0, 1),
        ("intl_kz_yr1_undeclared", "international",
            1, 1, 1, 0, 1, 1, 0, 1, 1,   0, 1, 0, 0, 1),
        ("intl_kz_yr1_ds", "international",
            1, 1, 1, 1, 1, 1, 0, 1, 1,   0, 1, 1, 1, 1),
    ]
    cols = [
        "interview_id", "group",
        "language_barrier", "ucug_easier_than_ufug", "stays_with_own_group",
        "excluded_from_wechat", "low_belonging", "uni_insufficient",
        "chinese_peer_introvert", "mandarin_dominance", "social_affects_grades",
        "clubs_english_only", "chinese_peers_avoid_english",
        "non_gaokao_prep_gap", "intl_comms_chinese_default",
        "wants_structured_programs",
    ]
    return pd.DataFrame(interviews, columns=cols)


def main() -> None:
    a = load_xlsx(XLSX_A)
    b = load_xlsx(XLSX_B)

    # Add a 'batch' column so we can trace provenance if ever needed, but
    # downstream analysis treats both batches as one sample.
    a["batch"] = "A"
    b["batch"] = "B"
    rowlevel = pd.concat([a, b], ignore_index=True, sort=False)
    # Reorder so shared columns come first, then B-only, then batch.
    shared = [c for c in rowlevel.columns
              if c not in B_ONLY_QUESTION_IDS and c != "batch"]
    b_only = [c for c in rowlevel.columns if c in B_ONLY_QUESTION_IDS]
    rowlevel = rowlevel[shared + b_only + ["batch"]]
    rowlevel.to_csv(DATA / "survey_rowlevel.csv", index=False)

    n_pooled = len(rowlevel)
    n_b_only = int((rowlevel["batch"] == "B").sum())
    agg = build_aggregated(rowlevel, n_pooled, n_b_only)
    agg.to_csv(DATA / "survey_aggregated.csv", index=False)

    themes = build_interview_themes()
    themes.to_csv(DATA / "interview_themes.csv", index=False)

    # Verification prints ----------------------------------------------------
    print("=" * 70)
    print("01_build_data.py -- verification")
    print("=" * 70)
    print(f"Row-level respondents: {len(rowlevel)}  "
          f"(A: {(rowlevel.batch=='A').sum()}, B: {(rowlevel.batch=='B').sum()})")
    print(f"Aggregated rows:       {len(agg)}  unique questions: {agg.question_id.nunique()}")
    print(f"Interview respondents: {len(themes)}")

    print("\nBy-question respondent counts (from row-level):")
    for qid in rowlevel.columns:
        if qid == "batch":
            continue
        n = rowlevel[qid].notna().sum()
        print(f"  {qid:<30s} n={n}")

    print("\nStatus marginal (pooled):")
    print(rowlevel["status"].value_counts().to_string())
    print("\nGPA marginal (pooled, pre-imputation):")
    print(rowlevel["gpa"].value_counts().to_string())
    print(f"\nCSVs written to: {DATA}")


if __name__ == "__main__":
    main()

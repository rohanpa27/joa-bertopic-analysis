"""
JOA BERTopic -- Apply manually curated clinical labels to discovered topics.

Labels assigned by domain reading of BERTopic top-keywords for each topic
(see outputs/tables/topic_info_JOA.csv and outputs/tables/manual_review_sheet_JOA.xlsx).

This file is auto-populated by build_clinical_labels.py after the BERTopic model
has been fit, by parsing the top-15 c-TF-IDF keywords for each topic into a
domain-appropriate arthroplasty label.
"""

import json
import pandas as pd
from pathlib import Path

LABELS_JSON = Path("outputs/tables/clinical_labels_JOA.json")

TABLE_DIR  = Path("outputs/tables")
REPORT_DIR = Path("outputs/reports")
JOURNAL    = "JOA"


def load_labels():
    if LABELS_JSON.exists():
        return {int(k): v for k, v in json.loads(LABELS_JSON.read_text()).items()}
    # Fallback: empty mapping -> uses topic_name
    return {}


def main():
    CLINICAL_LABELS = load_labels()
    if not CLINICAL_LABELS:
        print(f"WARN: No labels at {LABELS_JSON}. Run build_clinical_labels.py first.")
        return

    # -- topic_info --
    topic_info = pd.read_csv(TABLE_DIR / f"topic_info_{JOURNAL}.csv")
    topic_info["clinical_label"] = topic_info["Topic"].map(CLINICAL_LABELS)
    topic_info.to_csv(TABLE_DIR / f"topic_info_{JOURNAL}.csv", index=False)
    print(f"Updated topic_info: {len(topic_info)} rows")

    # -- doc_topics --
    doc_topics = pd.read_csv(TABLE_DIR / f"doc_topics_{JOURNAL}.csv")
    doc_topics["clinical_label"] = doc_topics["topic_id"].map(CLINICAL_LABELS)
    doc_topics.to_csv(TABLE_DIR / f"doc_topics_{JOURNAL}.csv", index=False)
    print(f"Updated doc_topics: {len(doc_topics)} rows")

    # -- temporal_trends --
    trends = pd.read_csv(TABLE_DIR / f"temporal_trends_{JOURNAL}.csv")
    trends["clinical_label"] = trends["topic_id"].map(CLINICAL_LABELS)
    trends.to_csv(TABLE_DIR / f"temporal_trends_{JOURNAL}.csv", index=False)
    print(f"Updated temporal_trends: {len(trends)} rows")

    # -- table2 / table3 / table4 (reports) --
    for fname in [f"table2_all_topics_{JOURNAL}.csv",
                  f"table3_hot_topics_{JOURNAL}.csv",
                  f"table4_cold_topics_{JOURNAL}.csv"]:
        p = REPORT_DIR / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        # extract topic_id when missing (table3/4 may carry it via topic_label="N_word_word")
        if "topic_id" not in df.columns:
            if "topic_label" in df.columns:
                def _eid(v):
                    try:
                        return int(str(v).split("_")[0])
                    except Exception:
                        return -1
                df["topic_id"] = df["topic_label"].apply(_eid)
            elif "topic_name" in df.columns:
                def _eid(v):
                    try:
                        return int(str(v).split("_")[0])
                    except Exception:
                        return -1
                df["topic_id"] = df["topic_name"].apply(_eid)
        df["clinical_label"] = df["topic_id"].map(CLINICAL_LABELS)
        cols = ["topic_id", "clinical_label"] + [
            c for c in df.columns if c not in ("topic_id", "clinical_label")]
        df = df[[c for c in cols if c in df.columns]]
        df.to_csv(p, index=False)
        print(f"Updated {fname}: {len(df)} rows")

    print(f"\nClinical labels applied to all {len(CLINICAL_LABELS)} topics.")


if __name__ == "__main__":
    main()

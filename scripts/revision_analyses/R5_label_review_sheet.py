"""
Revision analysis 5: build the topic-label validation workbook requested by
Reviewer 2 comment 4.

For each of the 83 topics this emits: the top-15 c-TF-IDF keywords, the
rule-assigned clinical label, and the 5 most representative abstracts
(documents closest to the topic centroid in embedding space), plus blank
columns for two independent arthroplasty reviewers and an agreement tab.

NOTE: this script PREPARES the validation instrument. It does not fabricate
reviewer judgments. The reviewer columns are intentionally empty and must be
completed by the two clinician co-authors before resubmission.

Output -> Revisions/05_supplementary/topic_label_validation_worksheet.xlsx
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("Revisions/05_supplementary")
OUT.mkdir(parents=True, exist_ok=True)

N_REP = 5


def main():
    emb = np.load("data/embeddings/embeddings_pubmedbert_JOA.npy")
    doc = pd.read_csv("outputs/tables/doc_topics_JOA.csv",
                      usecols=["pmid", "title", "year", "topic_id"])
    tr = pd.read_csv("outputs/tables/temporal_trends_JOA.csv")
    review = pd.read_excel("outputs/tables/manual_review_sheet_JOA.xlsx")

    doc = doc.reset_index(drop=True)
    assert len(doc) == len(emb), f"row mismatch {len(doc)} vs {len(emb)}"

    rows = []
    for tid in sorted(t for t in doc.topic_id.unique() if t != -1):
        idx = np.where(doc.topic_id.values == tid)[0]
        centroid = emb[idx].mean(axis=0)
        v = emb[idx]
        sims = (v @ centroid) / (np.linalg.norm(v, axis=1) * np.linalg.norm(centroid))
        best = idx[np.argsort(sims)[::-1][:N_REP]]

        meta = tr[tr.topic_id == tid]
        kw = review.loc[review.topic_id == tid, "top_15_words"]
        rows.append({
            "topic_id": tid,
            "n_documents": int(len(idx)),
            "rule_assigned_label": (meta.clinical_label.iloc[0]
                                    if len(meta) else ""),
            "trend_class": meta.trend_class.iloc[0] if len(meta) else "",
            "top_15_keywords": kw.iloc[0] if len(kw) else "",
            **{f"representative_title_{i+1}": doc.title.iloc[b]
               for i, b in enumerate(best)},
            "reviewer_1_label_agrees_Y_N": "",
            "reviewer_1_proposed_label": "",
            "reviewer_2_label_agrees_Y_N": "",
            "reviewer_2_proposed_label": "",
            "adjudicated_final_label": "",
            "notes": "",
        })

    sheet = pd.DataFrame(rows)

    instructions = pd.DataFrame({
        "Instructions": [
            "Topic-label validation worksheet (Reviewer 2, comment 4).",
            "",
            "Two arthroplasty reviewers should INDEPENDENTLY complete their own "
            "columns without consulting each other.",
            "",
            "For each topic: read the top-15 keywords and the 5 representative "
            "article titles, then record whether the rule-assigned clinical label "
            "is an accurate description of the topic (Y/N).",
            "If N, propose a corrected label in the adjacent column.",
            "",
            "After both reviewers finish, compute Cohen's kappa on the Y/N "
            "agreement columns and adjudicate disagreements by consensus.",
            "Report the kappa value and the number of relabeled topics in the "
            "Methods and Results.",
            "",
            "This worksheet was generated automatically. The reviewer columns are "
            "deliberately blank; they must not be completed programmatically.",
        ]
    })

    path = OUT / "topic_label_validation_worksheet.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        instructions.to_excel(xl, sheet_name="Instructions", index=False)
        sheet.to_excel(xl, sheet_name="Topic labels", index=False)
    print(f"saved: {path}  ({len(sheet)} topics)")
    print(f"blank reviewer columns: awaiting clinician completion")


if __name__ == "__main__":
    main()

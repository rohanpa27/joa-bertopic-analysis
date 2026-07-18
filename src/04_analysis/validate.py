"""
Phase 8: Topic Model Validation
=================================
Computes all validation metrics and generates the validation report.

Metrics computed:
1. Topic coherence (NPMI) — internal semantic quality
2. Topic diversity — uniqueness of words across topics
3. Silhouette score — cluster separation quality
4. Noise ratio — proportion of uncategorized documents
5. Inter-rater reliability setup — exports topics for manual review

Usage:
    python src/04_analysis/validate.py --journal ARTHROSCOPY

Outputs:
    outputs/tables/validation_report_JOA.csv
    outputs/tables/manual_review_sheet_JOA.xlsx
"""

import argparse
import glob
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score
from loguru import logger
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase + keep only alphanum tokens — matches CountVectorizer defaults."""
    return _TOKEN_RE.findall(text.lower())


# ── Inline coherence helpers ───────────────────────────────────────────────────

def compute_coherence(topic_model: BERTopic, docs: list[str], top_n: int = 10) -> dict:
    """Compute NPMI topic coherence using gensim."""
    try:
        topics = topic_model.get_topics()
        topic_words = [
            [word for word, _ in topics[t_id][:top_n]]
            for t_id in sorted(topics.keys())
            if t_id != -1
        ]
        tokenized = [_tokenize(doc) for doc in docs]
        dictionary = Dictionary(tokenized)
        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        score = coherence_model.get_coherence()
        return {"coherence_npmi": round(score, 4), "n_topics": len(topic_words)}
    except Exception as e:
        logger.warning(f"Coherence computation failed: {e}")
        return {"coherence_npmi": None, "n_topics": len(topic_model.get_topics()) - 1}


def compute_diversity(topic_model: BERTopic, top_n: int = 10) -> float:
    """Compute topic diversity (proportion of unique words across all topics)."""
    try:
        topics = topic_model.get_topics()
        all_words = [
            word
            for t_id, words in topics.items()
            if t_id != -1
            for word, _ in words[:top_n]
        ]
        diversity = len(set(all_words)) / len(all_words) if all_words else 0
        return round(diversity, 4)
    except Exception:
        return 0.0


# ── Validation Functions ───────────────────────────────────────────────────────

def compute_silhouette(embeddings: np.ndarray, topics: list[int], sample_size: int = 5000) -> float:
    """Compute silhouette score on a subsample (full computation is O(n^2))."""
    try:
        topic_arr = np.array(topics)
        valid_idx = np.where(topic_arr != -1)[0]
        if len(valid_idx) < 100:
            return None

        if len(valid_idx) > sample_size:
            sample_idx = np.random.choice(valid_idx, size=sample_size, replace=False)
        else:
            sample_idx = valid_idx

        # We already subsampled above — do not pass sample_size again, which
        # would cause sklearn to subsample a second time on the already-small array.
        score = silhouette_score(
            embeddings[sample_idx],
            topic_arr[sample_idx],
            metric="cosine",
        )
        return round(float(score), 4)
    except Exception as e:
        logger.warning(f"Silhouette computation failed: {e}")
        return None


def generate_review_sheet(topic_model: BERTopic, topic_info: pd.DataFrame, journal: str):
    """Generate an Excel sheet for expert manual review of topic labels."""
    topics = topic_info[topic_info["Topic"] != -1].copy()

    review_rows = []
    for _, row in topics.iterrows():
        topic_id = row["Topic"]
        top_words = topic_model.get_topic(topic_id)
        words_str = ", ".join([w for w, _ in top_words[:15]])
        review_rows.append({
            "topic_id": topic_id,
            "n_documents": row["Count"],
            "top_15_words": words_str,
            "bertopic_name": row["Name"],
            "reviewer_1_label": "",
            "reviewer_2_label": "",
            "notes": "",
        })

    df_review = pd.DataFrame(review_rows)
    out_path = OUTPUT_TABLES / f"manual_review_sheet_{journal}.xlsx"
    df_review.to_excel(out_path, index=False, engine="openpyxl")
    logger.info(f"Expert review sheet: {out_path}")
    return df_review


def run_validation(journal: str = "JOA"):
    model_path = f"outputs/models/bertopic_model_{journal}"
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")

    # Load embeddings
    emb_files = sorted(glob.glob(f"data/embeddings/embeddings_*_{journal}.npy"))
    embeddings = np.load(emb_files[0]) if emb_files else None

    # Load model
    model = BERTopic.load(model_path)
    topics = doc_topics["topic_id"].tolist()

    # Merge document text back in only if not already present in doc_topics CSV.
    # If document is already there (saved by run_bertopic.py), a merge would create
    # document_x / document_y columns, causing the lookup below to silently fall back
    # to title-only text for coherence computation.
    corpus_path = Path("data/processed") / f"corpus_clean_{journal}.csv"
    if "document" not in doc_topics.columns and corpus_path.exists():
        corpus = pd.read_csv(corpus_path, usecols=["doc_id", "document"])
        doc_topics = doc_topics.merge(corpus, on="doc_id", how="left")

    docs = doc_topics["document"].tolist() if "document" in doc_topics.columns else doc_topics["title"].tolist()

    # 1. Coherence
    coherence = compute_coherence(model, docs)
    diversity = compute_diversity(model)

    # 2. Noise ratio
    noise_ratio = round(sum(t == -1 for t in topics) / len(topics) * 100, 2)

    # 3. Topic count
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)

    # 4. Silhouette (if embeddings available)
    silhouette = compute_silhouette(embeddings, topics) if embeddings is not None else None

    # 5. Summary stats
    topic_sizes = topic_info[topic_info["Topic"] != -1]["Count"]

    report = {
        "journal": journal,
        "n_documents": len(docs),
        "n_topics": n_topics,
        "noise_pct": noise_ratio,
        "coherence_npmi": coherence["coherence_npmi"],
        "diversity": diversity,
        "silhouette_score": silhouette,
        "median_topic_size": int(topic_sizes.median()),
        "min_topic_size": int(topic_sizes.min()),
        "max_topic_size": int(topic_sizes.max()),
    }

    logger.info("\nVALIDATION REPORT")
    logger.info("="*50)
    for k, v in report.items():
        logger.info(f"  {k:30s}: {v}")

    # Save
    pd.DataFrame([report]).to_csv(OUTPUT_TABLES / f"validation_report_{journal}.csv", index=False)

    # Generate review sheet
    generate_review_sheet(model, topic_info, journal)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="JOA")
    args = parser.parse_args()
    run_validation(args.journal)

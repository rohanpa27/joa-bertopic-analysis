"""
Phase 7: Temporal Trend Analysis
==================================
For each topic, fits a linear regression of document frequency vs. year
and classifies the trend. Also generates topics-over-time data for visualization.

Usage:
    python src/04_analysis/temporal_analysis.py --journal ARTHROSCOPY

Outputs:
    outputs/tables/temporal_trends_JOA.csv
    outputs/tables/topics_over_time_JOA.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from loguru import logger
from bertopic import BERTopic

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_FIGURES = Path("outputs/figures")
for d in [OUTPUT_TABLES, OUTPUT_FIGURES]:
    d.mkdir(parents=True, exist_ok=True)

# ── Trend Classification ───────────────────────────────────────────────────────

def classify_trend(slope: float, p_adj: float, years: np.ndarray, counts: np.ndarray) -> str:
    """
    Classify a topic's temporal trend using FDR-adjusted p-values.

    Rules (all use BH-adjusted p-value at α=0.05):
    - Emerging: significant positive slope AND ≥60% of documents in last 10 years
    - Hot:      significant positive slope
    - Cold:     significant negative slope
    - Stable:   no significant trend
    """
    alpha = 0.05
    recent_counts = counts[years >= years.max() - 9]
    recent_pct = recent_counts.sum() / counts.sum() if counts.sum() > 0 else 0

    if p_adj < alpha and slope > 0 and recent_pct >= 0.6:
        return "Emerging"
    elif p_adj < alpha and slope > 0:
        return "Hot"
    elif p_adj < alpha and slope < 0:
        return "Cold"
    else:
        return "Stable"


def compute_topic_trends(
    df: pd.DataFrame,
    min_year: int = 2005,
    max_year: int = 2026,
) -> pd.DataFrame:
    """Compute linear trend statistics for each topic."""
    year_range = np.arange(min_year, max_year + 1)
    results = []

    topic_ids = sorted(df["topic_id"].unique())
    topic_ids = [t for t in topic_ids if t != -1]

    for topic_id in topic_ids:
        topic_docs = df[df["topic_id"] == topic_id]

        year_counts = topic_docs.groupby("year").size().reindex(year_range, fill_value=0)
        counts = year_counts.values.astype(float)

        # Normalize to proportion of total publications per year
        total_per_year = df.groupby("year").size().reindex(year_range, fill_value=1)
        proportions = counts / total_per_year.values * 100

        slope, intercept, r_value, p_value, std_err = stats.linregress(year_range, proportions)

        results.append({
            "topic_id": topic_id,
            "n_documents": len(topic_docs),
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_value**2, 4),
            "p_value": round(p_value, 6),
            "mean_pct": round(proportions.mean(), 4),
            "peak_year": int(year_range[np.argmax(counts)]),
            "peak_count": int(counts.max()),
            "recent_5yr_pct": round(proportions[-5:].mean(), 4),
            "early_5yr_pct": round(proportions[:5].mean(), 4),
            # stash arrays for classify_trend after FDR correction
            "_year_range": year_range,
            "_counts": counts,
        })

    df_trends = pd.DataFrame(results)

    # Apply Benjamini-Hochberg FDR correction across all topics
    reject, p_adj, _, _ = multipletests(df_trends["p_value"], method="fdr_bh")
    df_trends["p_adj"] = p_adj.round(6)

    # Classify using adjusted p-values
    df_trends["trend_class"] = df_trends.apply(
        lambda row: classify_trend(
            row["slope"], row["p_adj"],
            row["_year_range"], row["_counts"]
        ),
        axis=1,
    )
    df_trends = df_trends.drop(columns=["_year_range", "_counts"])

    return df_trends


def run_temporal_analysis(journal: str = "JOA", min_year: int = 2005, max_year: int = 2026):
    # Load data
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")

    # Merge document text back in only if not already present in doc_topics CSV.
    # If document is already there (saved by run_bertopic.py), a merge would create
    # document_x / document_y columns, breaking the column lookup below.
    corpus_path = Path("data/processed") / f"corpus_clean_{journal}.csv"
    if "document" not in doc_topics.columns and corpus_path.exists():
        corpus = pd.read_csv(corpus_path, usecols=["doc_id", "document"])
        doc_topics = doc_topics.merge(corpus, on="doc_id", how="left")

    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")

    logger.info(f"Running temporal analysis for {journal}: {len(doc_topics)} documents")

    # Compute trends
    df_trends = compute_topic_trends(doc_topics, min_year=min_year, max_year=max_year)

    # Merge with topic labels
    topic_labels = topic_info[["Topic", "Name", "Count"]].rename(
        columns={"Topic": "topic_id", "Name": "topic_name", "Count": "topic_count"}
    )
    df_trends = df_trends.merge(topic_labels, on="topic_id", how="left")

    # Add LLM labels if available
    import json
    llm_path = OUTPUT_TABLES / f"topic_labels_llm_{journal}.json"
    if llm_path.exists():
        llm_labels = json.loads(llm_path.read_text())
        df_trends["llm_label"] = df_trends["topic_id"].map(
            {int(k): v for k, v in llm_labels.items()}
        )

    # Sort by trend class then slope
    trend_order = {"Emerging": 0, "Hot": 1, "Stable": 2, "Cold": 3}
    df_trends["trend_order"] = df_trends["trend_class"].map(trend_order)
    df_trends = df_trends.sort_values(["trend_order", "slope"], ascending=[True, False])
    df_trends = df_trends.drop(columns=["trend_order"])

    # Save
    out_path = OUTPUT_TABLES / f"temporal_trends_{journal}.csv"
    df_trends.to_csv(out_path, index=False)
    logger.success(f"Temporal trends saved: {out_path}")

    # Topics over time (using BERTopic built-in)
    if "document" in doc_topics.columns:
        text_col = doc_topics["document"].tolist()
    else:
        text_col = doc_topics["title"].tolist()

    timestamps = doc_topics["year"].tolist()
    # BERTopic.load() does NOT persist per-document topic assignments (model.topics_).
    # Pass them explicitly from the saved doc_topics CSV so the call succeeds.
    saved_topics = doc_topics["topic_id"].tolist()

    try:
        topics_over_time = model.topics_over_time(
            text_col,
            timestamps,
            topics=saved_topics,
            nr_bins=10,
            evolution_tuning=True,
            global_tuning=True,
        )
        tot_path = OUTPUT_TABLES / f"topics_over_time_{journal}.csv"
        topics_over_time.to_csv(tot_path, index=False)
        logger.success(f"Topics over time saved: {tot_path}")
    except Exception as e:
        logger.warning(f"topics_over_time failed (non-fatal): {e}")
        topics_over_time = None

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL TREND SUMMARY")
    logger.info("="*60)
    for trend_class in ["Emerging", "Hot", "Stable", "Cold"]:
        subset = df_trends[df_trends["trend_class"] == trend_class]
        logger.info(f"\n{trend_class} Topics ({len(subset)}):")
        for _, row in subset.iterrows():
            label = row.get("llm_label") or row.get("topic_name", f"Topic {row['topic_id']}")
            logger.info(f"  [{row['topic_id']}] {label} | n={row['n_documents']} | slope={row['slope']:.4f} | p={row['p_value']:.4f}")

    return df_trends, topics_over_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="JOA")
    parser.add_argument("--min_year", type=int, default=2005)
    parser.add_argument("--max_year", type=int, default=2026)
    args = parser.parse_args()
    run_temporal_analysis(args.journal, min_year=args.min_year, max_year=args.max_year)

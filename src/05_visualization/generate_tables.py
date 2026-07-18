"""
Phase 10: Generate publication-ready tables for manuscript.

Tables produced:
  Table 1: Corpus characteristics (N per year, total)
  Table 2: All topics with labels, size, trend class, top words
  Table 3: Hot topics (top 15 by slope, p<0.05)
  Table 4: Cold topics (top 15 by slope magnitude, p<0.05)
  Table 5: Validation metrics

Usage:
    python src/05_visualization/generate_tables.py --journal ARTHROSCOPY
"""

import argparse
import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from loguru import logger

OUTPUT_TABLES = Path("outputs/tables")
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS = Path("outputs/reports")
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)


def generate_all_tables(journal: str = "JOA"):
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    temporal_trends = pd.read_csv(OUTPUT_TABLES / f"temporal_trends_{journal}.csv")
    validation = pd.read_csv(OUTPUT_TABLES / f"validation_report_{journal}.csv")
    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")

    label_col = "llm_label" if "llm_label" in temporal_trends.columns else "topic_name"

    # ── Table 1: Corpus Summary ────────────────────────────────────────────────
    t1 = doc_topics.groupby("year").size().reset_index(name="n_publications")
    t1["cumulative"] = t1["n_publications"].cumsum()
    t1.to_csv(OUTPUT_REPORTS / f"table1_corpus_{journal}.csv", index=False)

    # ── Table 2: All Topics ────────────────────────────────────────────────────
    top_words_map = {
        t_id: ", ".join([w for w, _ in model.get_topic(t_id)[:10]])
        for t_id in temporal_trends["topic_id"]
        if model.get_topic(t_id)
    }
    topics_all = temporal_trends.copy()
    topics_all["top_10_words"] = topics_all["topic_id"].map(top_words_map)
    topics_all = topics_all[[
        "topic_id", label_col, "n_documents", "trend_class",
        "slope", "p_value", "r_squared", "peak_year", "top_10_words"
    ]].rename(columns={label_col: "topic_label"})
    topics_all.to_csv(OUTPUT_REPORTS / f"table2_all_topics_{journal}.csv", index=False)

    # ── Table 3: Hot Topics ────────────────────────────────────────────────────
    hot = temporal_trends[temporal_trends["trend_class"].isin(["Hot", "Emerging"])].copy()
    hot = hot.sort_values("slope", ascending=False).head(15)
    hot["top_words"] = hot["topic_id"].map(top_words_map)
    hot_cols = [c for c in [label_col, "n_documents", "slope", "p_value", "peak_year", "recent_5yr_pct", "top_words"] if c in hot.columns]
    hot[hot_cols].to_csv(OUTPUT_REPORTS / f"table3_hot_topics_{journal}.csv", index=False)

    # ── Table 4: Cold Topics ───────────────────────────────────────────────────
    cold = temporal_trends[temporal_trends["trend_class"] == "Cold"].copy()
    cold = cold.sort_values("slope", ascending=True).head(15)
    cold["top_words"] = cold["topic_id"].map(top_words_map)
    cold_cols = [c for c in [label_col, "n_documents", "slope", "p_value", "peak_year", "early_5yr_pct", "top_words"] if c in cold.columns]
    cold[cold_cols].to_csv(OUTPUT_REPORTS / f"table4_cold_topics_{journal}.csv", index=False)

    # ── Table 5: Validation ────────────────────────────────────────────────────
    validation.to_csv(OUTPUT_REPORTS / f"table5_validation_{journal}.csv", index=False)

    logger.success(f"All tables saved to outputs/reports/")
    logger.info(f"\nSummary for {journal}:")
    logger.info(f"  Total documents: {len(doc_topics)}")
    logger.info(f"  Topics discovered: {temporal_trends['topic_id'].nunique()}")
    logger.info(f"  Hot topics: {len(temporal_trends[temporal_trends['trend_class']=='Hot'])}")
    logger.info(f"  Emerging topics: {len(temporal_trends[temporal_trends['trend_class']=='Emerging'])}")
    logger.info(f"  Cold topics: {len(temporal_trends[temporal_trends['trend_class']=='Cold'])}")
    logger.info(f"  Stable topics: {len(temporal_trends[temporal_trends['trend_class']=='Stable'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="JOA")
    args = parser.parse_args()
    generate_all_tables(args.journal)

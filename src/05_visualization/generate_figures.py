"""
Phase 9: Visualization
=======================
Generates all publication-ready figures.

Figures produced:
  Fig 1: Intertopic distance map (2D UMAP)
  Fig 2: Top words per topic (bar chart)
  Fig 3: Topics over time (line chart)
  Fig 4: Temporal trend heatmap (topics x years)
  Fig 5: Topic size distribution
  Fig 6: Hot/Cold/Emerging/Stable classification summary
  Fig 7: Hierarchical topic tree (dendrogram)

Usage:
    python src/05_visualization/generate_figures.py --journal ARTHROSCOPY

Outputs:
    outputs/figures/fig1_topic_map_JOA.html
    outputs/figures/fig2_top_words_JOA.html
    outputs/figures/fig3_topics_over_time_JOA.html
    outputs/figures/fig4_temporal_heatmap_JOA.html
    outputs/figures/fig5_topic_sizes_JOA.html
    outputs/figures/fig6_trend_summary_JOA.html
    outputs/figures/fig7_hierarchy_JOA.html
"""

import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from bertopic import BERTopic
from loguru import logger

OUTPUT_FIGURES = Path("outputs/figures")
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLES = Path("outputs/tables")

TREND_COLORS = {
    "Emerging": "#2ECC71",
    "Hot": "#E74C3C",
    "Stable": "#95A5A6",
    "Cold": "#3498DB",
}


def load_data(journal: str):
    model = BERTopic.load(f"outputs/models/bertopic_model_{journal}")
    topic_info = pd.read_csv(OUTPUT_TABLES / f"topic_info_{journal}.csv")
    doc_topics = pd.read_csv(OUTPUT_TABLES / f"doc_topics_{journal}.csv")
    temporal_trends = pd.read_csv(OUTPUT_TABLES / f"temporal_trends_{journal}.csv")

    tot_path = OUTPUT_TABLES / f"topics_over_time_{journal}.csv"
    topics_over_time = pd.read_csv(tot_path) if tot_path.exists() else None

    return model, topic_info, doc_topics, temporal_trends, topics_over_time


def fig1_topic_map(model, journal):
    """Intertopic distance map using BERTopic built-in."""
    fig = model.visualize_topics(title=f"<b>Topic Map — {journal}</b>")
    fig.write_html(OUTPUT_FIGURES / f"fig1_topic_map_{journal}.html")
    logger.info("Fig 1: Topic map saved")


def fig2_top_words(model, topic_info, journal, top_n=20):
    """Bar chart of top words for top N topics by document count."""
    top_topics = topic_info[topic_info["Topic"] != -1].nlargest(top_n, "Count")["Topic"].tolist()
    fig = model.visualize_barchart(
        topics=top_topics,
        top_n_topics=top_n,
        n_words=10,
        title=f"<b>Top Words per Topic — {journal}</b>",
    )
    fig.write_html(OUTPUT_FIGURES / f"fig2_top_words_{journal}.html")
    logger.info("Fig 2: Top words saved")


def fig3_topics_over_time(model, topics_over_time, temporal_trends, journal, top_n=15):
    """Topics over time line chart for top N topics."""
    if topics_over_time is None:
        logger.warning("Topics over time data not available — skipping Fig 3")
        return

    top_topics = temporal_trends.nlargest(top_n, "n_documents")["topic_id"].tolist()
    fig = model.visualize_topics_over_time(
        topics_over_time,
        topics=top_topics,
        title=f"<b>Topics Over Time — {journal}</b>",
    )
    fig.write_html(OUTPUT_FIGURES / f"fig3_topics_over_time_{journal}.html")
    logger.info("Fig 3: Topics over time saved")


def fig4_temporal_heatmap(doc_topics, temporal_trends, journal):
    """Heatmap: topics (rows) x year bins (cols), color = normalized frequency."""
    doc_topics = doc_topics.copy()
    doc_topics["year_bin"] = (doc_topics["year"] // 5 * 5).astype(str) + "–" + \
                              ((doc_topics["year"] // 5 * 5) + 4).astype(str)

    pivot = doc_topics[doc_topics["topic_id"] != -1].groupby(
        ["topic_id", "year_bin"]
    ).size().unstack(fill_value=0)

    # Normalize by row (within-topic)
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Get topic labels
    label_map = {}
    if "llm_label" in temporal_trends.columns:
        label_map = temporal_trends.set_index("topic_id")["llm_label"].dropna().to_dict()
    elif "topic_name" in temporal_trends.columns:
        label_map = temporal_trends.set_index("topic_id")["topic_name"].to_dict()

    pivot_norm.index = [label_map.get(i, f"Topic {i}") for i in pivot_norm.index]

    # Sort rows by trend class (Emerging → Hot → Stable → Cold).
    # Build name→trend directly to avoid a fragile reverse-lookup through label_map.
    trend_map = temporal_trends.set_index("topic_id")["trend_class"].to_dict()
    trend_order = {"Emerging": 0, "Hot": 1, "Stable": 2, "Cold": 3}
    name_to_trend = {
        label_map.get(tid, f"Topic {tid}"): trend_map.get(tid, "Stable")
        for tid in trend_map
    }
    sorted_idx = sorted(
        pivot_norm.index,
        key=lambda x: trend_order.get(name_to_trend.get(x, "Stable"), 2),
    )
    pivot_norm = pivot_norm.loc[sorted_idx]

    fig = px.imshow(
        pivot_norm,
        labels=dict(x="Year Range", y="Topic", color="% of Topic Docs"),
        title=f"<b>Research Activity Heatmap — {journal}</b>",
        color_continuous_scale="RdYlGn",
        aspect="auto",
    )
    fig.update_layout(height=max(400, len(pivot_norm) * 20 + 100))
    fig.write_html(OUTPUT_FIGURES / f"fig4_temporal_heatmap_{journal}.html")
    logger.info("Fig 4: Temporal heatmap saved")


def fig5_topic_sizes(temporal_trends, journal):
    """Bar chart of topic sizes, colored by trend class."""
    df = temporal_trends.sort_values("n_documents", ascending=False).head(30).copy()
    label_col = "llm_label" if "llm_label" in df.columns else "topic_name"
    df["label"] = df[label_col].fillna(df["topic_id"].astype(str))

    fig = px.bar(
        df,
        x="label",
        y="n_documents",
        color="trend_class",
        color_discrete_map=TREND_COLORS,
        title=f"<b>Topic Size by Trend Class — {journal}</b>",
        labels={"label": "Topic", "n_documents": "Number of Documents"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=600)
    fig.write_html(OUTPUT_FIGURES / f"fig5_topic_sizes_{journal}.html")
    logger.info("Fig 5: Topic sizes saved")


def fig6_trend_summary(temporal_trends, journal):
    """Summary figure: scatter of slope vs. document count, colored by trend."""
    df = temporal_trends.copy()
    label_col = "llm_label" if "llm_label" in df.columns else "topic_name"
    df["label"] = df[label_col].fillna(df["topic_id"].astype(str))

    fig = px.scatter(
        df,
        x="slope",
        y="n_documents",
        color="trend_class",
        color_discrete_map=TREND_COLORS,
        text="label",
        size="n_documents",
        title=f"<b>Topic Trends: Growth Rate vs. Volume — {journal}</b>",
        labels={"slope": "Annual Growth Rate (% publications)", "n_documents": "Total Documents"},
        hover_data=["p_value", "r_squared", "peak_year"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=700)
    fig.write_html(OUTPUT_FIGURES / f"fig6_trend_summary_{journal}.html")
    logger.info("Fig 6: Trend summary saved")


def fig7_hierarchy(model, journal):
    """Hierarchical topic clustering dendrogram."""
    fig = model.visualize_hierarchy(title=f"<b>Topic Hierarchy — {journal}</b>")
    fig.write_html(OUTPUT_FIGURES / f"fig7_hierarchy_{journal}.html")
    logger.info("Fig 7: Hierarchy saved")


def generate_all_figures(journal: str = "JOA"):
    logger.info(f"Generating all figures for {journal}...")
    model, topic_info, doc_topics, temporal_trends, topics_over_time = load_data(journal)

    fig1_topic_map(model, journal)
    fig2_top_words(model, topic_info, journal)
    fig3_topics_over_time(model, topics_over_time, temporal_trends, journal)
    fig4_temporal_heatmap(doc_topics, temporal_trends, journal)
    fig5_topic_sizes(temporal_trends, journal)
    fig6_trend_summary(temporal_trends, journal)
    fig7_hierarchy(model, journal)

    logger.success(f"All figures saved to outputs/figures/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", default="JOA")
    args = parser.parse_args()
    generate_all_figures(args.journal)

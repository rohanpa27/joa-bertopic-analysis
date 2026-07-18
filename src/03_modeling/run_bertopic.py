"""
Phase 6: Full BERTopic Pipeline
================================
Runs the complete BERTopic pipeline with hyperparameter tuning.

Usage:
    python src/03_modeling/run_bertopic.py \
        --corpus data/processed/corpus_clean_JOA.csv \
        --embeddings data/embeddings/embeddings_pubmedbert_JOA.npy \
        --journal ARTHROSCOPY

    python src/03_modeling/run_bertopic.py \
        --corpus data/processed/corpus_clean_JOA.csv \
        --embeddings data/embeddings/embeddings_pubmedbert_JOA.npy \
        --journal ARTHROSCOPY \
        --grid_search

Outputs:
    outputs/models/bertopic_model_JOA/
    outputs/tables/topic_info_JOA.csv
    outputs/tables/doc_topics_JOA.csv
    outputs/tables/grid_search_JOA.csv  (if --grid_search)
"""

import argparse
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from loguru import logger
from tqdm import tqdm

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [MODEL_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Coherence Evaluation ───────────────────────────────────────────────────────

import re as _re
_TOKEN_RE = _re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase + keep only alphanum tokens — matches CountVectorizer defaults."""
    return _TOKEN_RE.findall(text.lower())


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


# ── Model Building ─────────────────────────────────────────────────────────────

def build_topic_model(
    n_neighbors: int = 15,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    cluster_selection_method: str = "eom",
    top_n_words: int = 10,
    calculate_probabilities: bool = True,
    embedding_model=None,
    for_grid_search: bool = False,
) -> BERTopic:
    """Construct a BERTopic model with specified hyperparameters.

    for_grid_search=True skips KeyBERTInspired (which requires a loaded embedding
    model) and relaxes min_df so the vectorizer doesn't fail when few topics form.
    """

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,
        core_dist_n_jobs=-1,
    )

    if for_grid_search:
        # KeyBERTInspired requires embedding_model != None; skip it during search.
        # min_df=2 prevents max_df < min_df when few topics form.
        representation_model = None
        min_df = 2
    else:
        representation_model = {
            "KeyBERT": KeyBERTInspired(),
            "MMR": MaximalMarginalRelevance(diversity=0.3),
        }
        min_df = 5

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=min_df,
        max_df=0.85,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,
        vectorizer_model=vectorizer_model,
        top_n_words=top_n_words,
        verbose=False,
        calculate_probabilities=calculate_probabilities,
    )

    return topic_model


# ── Grid Search ────────────────────────────────────────────────────────────────

def run_grid_search(
    docs: list[str],
    embeddings: np.ndarray,
    journal: str,
) -> pd.DataFrame:
    """Hyperparameter search over UMAP n_neighbors and HDBSCAN min_cluster_size."""
    grid = {
        "n_neighbors": [5, 10, 15, 20],
        "min_cluster_size": [5, 10, 15, 20, 30],
        "min_samples": [5],
        "cluster_selection_method": ["eom"],
    }

    results = []
    combos = list(product(*grid.values()))
    logger.info(f"Running grid search: {len(combos)} combinations")

    for combo in tqdm(combos, desc="Grid search"):
        params = dict(zip(grid.keys(), combo))
        try:
            # Use calculate_probabilities=False during grid search for speed;
            # soft probabilities are only needed for the final selected model.
            model = build_topic_model(**params, calculate_probabilities=False, for_grid_search=True)
            topics, _ = model.fit_transform(docs, embeddings)

            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            noise_pct = (sum(t == -1 for t in topics) / len(topics)) * 100
            coherence = compute_coherence(model, docs)
            diversity = compute_diversity(model)

            result = {
                **params,
                "n_topics": n_topics,
                "noise_pct": round(noise_pct, 1),
                "coherence_npmi": coherence["coherence_npmi"],
                "diversity": diversity,
            }
            results.append(result)
            logger.info(f"  {params} → {n_topics} topics, NPMI={coherence['coherence_npmi']}, noise={noise_pct:.1f}%")
        except Exception as e:
            logger.warning(f"  {params} failed: {e}")
            continue

    df_results = pd.DataFrame(results)
    out_path = TABLE_DIR / f"grid_search_{journal}.csv"
    df_results.to_csv(out_path, index=False)
    logger.success(f"Grid search saved: {out_path}")
    return df_results


def select_best_params(df_grid: pd.DataFrame) -> dict:
    """Select best hyperparameters from grid search by coherence with noise constraint."""
    valid = df_grid[df_grid["noise_pct"] < 45.0].copy()
    if valid.empty:
        valid = df_grid.copy()
        logger.warning("All configs exceed 45% noise — relaxing constraint")

    best_row = valid.loc[valid["coherence_npmi"].idxmax()]
    best_params = {
        "n_neighbors": int(best_row["n_neighbors"]),
        "min_cluster_size": int(best_row["min_cluster_size"]),
        "min_samples": int(best_row["min_samples"]),
        "cluster_selection_method": best_row["cluster_selection_method"],
    }
    logger.info(f"Best params: {best_params}")
    logger.info(f"  n_topics={best_row['n_topics']}, NPMI={best_row['coherence_npmi']}, noise={best_row['noise_pct']}%")
    return best_params


# ── Main Run ───────────────────────────────────────────────────────────────────

def run_bertopic(args):
    # Load data
    df = pd.read_csv(args.corpus)
    embeddings = np.load(args.embeddings)
    docs = df["document"].tolist()
    journal = args.journal

    logger.info(f"Corpus: {len(docs)} documents | Embeddings: {embeddings.shape}")

    # Grid search or use defaults
    if args.grid_search:
        df_grid = run_grid_search(docs, embeddings, journal)
        best_params = select_best_params(df_grid)
    else:
        best_params = {
            "n_neighbors": 15,
            "min_cluster_size": 50,
            "min_samples": 10,
            "cluster_selection_method": "eom",
        }

    # Build and fit final model — load embedding model so KeyBERTInspired works
    logger.info("Loading embedding model for final model representation...")
    emb_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    logger.info(f"Fitting final BERTopic model with params: {best_params}")
    topic_model = build_topic_model(**best_params, embedding_model=emb_model)
    topics, probs = topic_model.fit_transform(docs, embeddings)

    # Assign to dataframe
    df["topic_id"] = topics
    if probs is None:
        df["topic_probability"] = 1.0
    elif hasattr(probs, "ndim") and probs.ndim > 1:
        df["topic_probability"] = probs.max(axis=1)
    else:
        df["topic_probability"] = probs

    # Topic info
    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info["Topic"] != -1])
    noise_pct = (sum(t == -1 for t in topics) / len(topics)) * 100
    logger.info(f"Topics discovered: {n_topics}")
    logger.info(f"Noise documents: {noise_pct:.1f}%")

    # Evaluate
    coherence = compute_coherence(topic_model, docs)
    diversity = compute_diversity(topic_model)
    logger.info(f"Coherence (NPMI): {coherence['coherence_npmi']}")
    logger.info(f"Diversity: {diversity}")

    # Save model
    model_path = MODEL_DIR / f"bertopic_model_{journal}"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    logger.success(f"Model saved: {model_path}")

    # Save topic info table
    topic_info["coherence_npmi"] = coherence["coherence_npmi"]
    topic_info["diversity"] = diversity
    topic_info_path = TABLE_DIR / f"topic_info_{journal}.csv"
    topic_info.to_csv(topic_info_path, index=False)

    # Save per-document assignments — include document text so downstream scripts
    # (temporal analysis, validation) use the same input the model was trained on,
    # not a silent fallback to titles only.
    keep_cols = ["doc_id", "pmid", "year", "journal", "title", "abstract",
                 "document", "topic_id", "topic_probability"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    doc_topics = df[keep_cols]
    doc_topics_path = TABLE_DIR / f"doc_topics_{journal}.csv"
    doc_topics.to_csv(doc_topics_path, index=False)

    # Save model config
    config = {
        **best_params,
        "journal": journal,
        "n_documents": len(docs),
        "n_topics": n_topics,
        "noise_pct": round(noise_pct, 2),
        "coherence_npmi": coherence["coherence_npmi"],
        "diversity": diversity,
    }
    config_path = MODEL_DIR / f"model_config_{journal}.json"
    config_path.write_text(json.dumps(config, indent=2))

    logger.success("Phase 6 complete.")
    return topic_model, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BERTopic pipeline")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--journal", default="JOA")
    parser.add_argument("--grid_search", action="store_true",
                        help="Run hyperparameter grid search before final fit")
    args = parser.parse_args()
    run_bertopic(args)

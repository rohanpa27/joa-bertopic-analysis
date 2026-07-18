"""
Phase 3: Document Embedding
============================
Converts clean documents to dense semantic embeddings using sentence transformers.

Usage:
    python src/03_modeling/embed.py --input data/processed/corpus_clean_JOA.csv
    python src/03_modeling/embed.py --input data/processed/corpus_clean_JOA.csv \
        --model mpnet --batch_size 64

Outputs:
    data/embeddings/embeddings_pubmedbert_JOA.npy
    data/embeddings/embedding_metadata_pubmedbert_JOA.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from loguru import logger

OUTPUT_DIR = Path("data/embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_MAP = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "biomedbert-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
}


def report_truncation(docs: list[str], model, model_key: str) -> dict:
    """
    Report what fraction of documents exceed the model's max token limit.
    BERT-family models truncate at 512 tokens; longer docs lose their conclusions.
    """
    tokenizer = model.tokenizer
    max_len = getattr(model, "max_seq_length", 512)
    n_truncated = sum(
        len(tokenizer.encode(doc, add_special_tokens=True)) > max_len
        for doc in docs
    )
    pct = round(n_truncated / len(docs) * 100, 1)
    logger.info(f"Truncation report ({model_key}, max_seq_length={max_len}):")
    logger.info(f"  Docs exceeding limit: {n_truncated}/{len(docs)} ({pct}%)")
    if pct > 20:
        logger.warning(
            "  >20% of documents are truncated. Consider increasing max_seq_length "
            "or disclosing this limitation in the manuscript."
        )
    return {"n_truncated": n_truncated, "truncation_pct": pct, "max_seq_length": max_len}


def embed_documents(
    docs: list,
    model_key: str = "pubmedbert",
    batch_size: int = 32,
    normalize: bool = True,
    device: str = None,
):
    """Load model, report truncation, and embed documents. Returns (embeddings, trunc_stats)."""
    import torch
    model_id = MODEL_MAP.get(model_key, model_key)
    if not device:
        device = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model: {model_id} on device={device}")
    model = SentenceTransformer(model_id, device=device)

    trunc_stats = report_truncation(docs, model, model_key)

    logger.info(f"Embedding {len(docs)} documents (batch_size={batch_size})...")
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
    )
    logger.info(f"Embedding shape: {embeddings.shape}")
    return embeddings, trunc_stats


def main(args):
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} documents from {args.input}")

    docs = df["document"].tolist()
    journal = args.journal or Path(args.input).stem.replace("corpus_clean_", "")

    embeddings, trunc_stats = embed_documents(docs, args.model, args.batch_size, device=args.device)
    logger.info(f"Embedding shape: {embeddings.shape}")

    # Save embeddings
    model_short = args.model.replace("/", "_")
    emb_path = OUTPUT_DIR / f"embeddings_{model_short}_{journal}.npy"
    np.save(emb_path, embeddings)
    logger.success(f"Saved embeddings: {emb_path}")

    # Save metadata
    meta = {
        "model": args.model,
        "model_id": MODEL_MAP.get(args.model, args.model),
        "journal": journal,
        "n_documents": len(docs),
        "embedding_dim": embeddings.shape[1],
        "normalized": True,
        "created_at": datetime.now().isoformat(),
        "input_file": str(args.input),
        **trunc_stats,
    }
    meta_path = OUTPUT_DIR / f"embedding_metadata_{model_short}_{journal}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed documents for BERTopic")
    parser.add_argument("--input", required=True, help="Path to clean corpus CSV")
    parser.add_argument("--model", default="pubmedbert",
                        choices=list(MODEL_MAP.keys()), help="Embedding model")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Embedding batch size (reduce if OOM)")
    parser.add_argument("--journal", default=None, help="Label for output files")
    parser.add_argument("--device", default=None,
                        help="Force device: cpu, cuda, mps. Default: auto-detect.")
    args = parser.parse_args()
    main(args)

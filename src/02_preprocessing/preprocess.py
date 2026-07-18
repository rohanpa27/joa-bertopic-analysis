"""
Phase 2: Text Preprocessing
============================
Cleans raw PubMed abstracts for BERTopic embedding.

Key decisions:
- DO NOT remove stopwords (BERT handles this via attention)
- DO remove structured abstract labels ("BACKGROUND:", "METHODS:", etc.)
- DO concatenate title + abstract (increases signal)
- DO filter by abstract length (min 50 words)
- DO NOT stem or lemmatize (BERT uses full subword tokenization)

Usage:
    python src/02_preprocessing/preprocess.py --input "data/raw/pubmed_raw_JOA_*.csv"
    python src/02_preprocessing/preprocess.py --input "data/raw/pubmed_raw_JOA_*.csv" --journal ARTHROSCOPY

Outputs:
    data/processed/corpus_clean_JOA.csv
    data/processed/preprocessing_report_JOA.txt
"""

import re
import argparse
import glob
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_ABSTRACT_WORDS = 50
MIN_ABSTRACT_CHARS = 200

# Structured abstract section labels to strip
STRUCTURED_LABELS = re.compile(
    r"\b(BACKGROUND|OBJECTIVE[S]?|PURPOSE|AIM[S]?|HYPOTHESIS|STUDY DESIGN|"
    r"METHODS?|MATERIALS? AND METHODS?|PATIENTS? AND METHODS?|DESIGN|"
    r"SETTING|PARTICIPANTS?|INTERVENTIONS?|RESULTS?|FINDINGS?|"
    r"CONCLUSIONS?|SIGNIFICANCE|CLINICAL RELEVANCE|LEVEL OF EVIDENCE|"
    r"EVIDENCE|WHAT IS KNOWN|WHAT THIS STUDY ADDS|TAKE[\-\s]HOME MESSAGE|"
    r"KEY POINTS?|TRIAL REGISTRATION)\s*:?\s*",
    re.IGNORECASE,
)

# Common boilerplate phrases to remove
BOILERPLATE = [
    r"Copyright ©.*?\.",
    r"All rights reserved\.",
    r"Published by.*?Elsevier.*?\.",
    r"This is an open[- ]access article.*?\.",
    r"ClinicalTrials\.gov.*?NCT\d+",
    r"Level of evidence:.*?\.",
    r"Study design:.*?\.",
]

# Publication types to EXCLUDE
EXCLUDE_PUB_TYPES = {
    "Editorial",
    "Letter",
    "Comment",
    "Published Erratum",
    "Retracted Publication",
    "News",
    "Biography",
    "Personal Narrative",
}

# ── Cleaning Functions ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Apply all text cleaning steps to an abstract or title."""
    if pd.isna(text) or not str(text).strip():
        return ""

    text = str(text)

    # Remove structured abstract labels
    text = STRUCTURED_LABELS.sub(" ", text)

    # Remove boilerplate
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove HTML entities and tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove leading/trailing punctuation artifacts
    text = text.strip(".,;:-")

    return text.strip()


def build_document(row: pd.Series) -> str:
    """Concatenate title + abstract into a single document string."""
    title = clean_text(str(row.get("title", "")))
    abstract = clean_text(str(row.get("abstract", "")))

    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract


def is_valid_document(doc: str, min_words: int = MIN_ABSTRACT_WORDS) -> bool:
    """Check if a document meets minimum length requirements."""
    if not doc:
        return False
    words = doc.split()
    return len(words) >= min_words and len(doc) >= MIN_ABSTRACT_CHARS


def filter_pub_types(pub_types_str: str) -> bool:
    """Return True if publication type should be INCLUDED."""
    if pd.isna(pub_types_str):
        return True
    for exclude_type in EXCLUDE_PUB_TYPES:
        if exclude_type.lower() in str(pub_types_str).lower():
            return False
    return True


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def preprocess(input_glob: str, journal_label: str = None) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files found matching: {input_glob}")

    logger.info(f"Loading {len(files)} file(s)...")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    n_raw = len(df)
    logger.info(f"Loaded {n_raw} raw records")

    # ── Step 1: Remove duplicates ──────────────────────────────────────────────
    df = df.drop_duplicates(subset="pmid")
    n_after_dedup = len(df)
    logger.info(f"After dedup: {n_after_dedup} records ({n_raw - n_after_dedup} removed)")

    # ── Step 2: Filter publication types ──────────────────────────────────────
    df["include"] = df["pub_types"].apply(filter_pub_types)
    df = df[df["include"]].drop(columns=["include"])
    n_after_pubtype = len(df)
    logger.info(f"After pub type filter: {n_after_pubtype} records")

    # ── Step 3: Year validation ────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1990) & (df["year"] <= 2026)]
    n_after_year = len(df)
    logger.info(f"After year filter: {n_after_year} records")

    # ── Step 4: Build document strings ────────────────────────────────────────
    logger.info("Building documents (title + abstract)...")
    tqdm.pandas(desc="Building docs")
    df["document"] = df.progress_apply(build_document, axis=1)

    # ── Step 5: Filter short documents ────────────────────────────────────────
    df["is_valid"] = df["document"].apply(is_valid_document)
    df_invalid = df[~df["is_valid"]]
    df = df[df["is_valid"]].drop(columns=["is_valid"])
    n_final = len(df)
    logger.info(f"Removed {len(df_invalid)} short/empty documents")
    logger.info(f"Final corpus: {n_final} documents")

    # ── Step 6: Add derived columns ───────────────────────────────────────────
    df["doc_length_words"] = df["document"].str.split().str.len()
    df["doc_length_chars"] = df["document"].str.len()
    df["decade"] = (df["year"] // 10 * 10).astype(int)

    # ── Step 7: Sort ──────────────────────────────────────────────────────────
    df = df.sort_values("year").reset_index(drop=True)
    df["doc_id"] = df.index  # Stable index for all downstream joins

    # ── Save ───────────────────────────────────────────────────────────────────
    label = journal_label or "corpus"
    out_path = OUTPUT_DIR / f"corpus_clean_{label}.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved clean corpus: {out_path}")

    # ── Report ─────────────────────────────────────────────────────────────────
    report = f"""
PREPROCESSING REPORT — {label}
{'='*50}
Raw records loaded:        {n_raw}
After deduplication:       {n_after_dedup}
After pub type filter:     {n_after_pubtype}
After year filter:         {n_after_year}
After length filter:       {n_final}
Retention rate:            {n_final/n_raw*100:.1f}%

Year range:                {df['year'].min()} – {df['year'].max()}
Median doc length (words): {df['doc_length_words'].median():.0f}
Mean doc length (words):   {df['doc_length_words'].mean():.0f}

Records per year (last 5):
{df[df['year'] >= df['year'].max()-4].groupby('year').size().to_string()}
"""
    print(report)
    report_path = OUTPUT_DIR / f"preprocessing_report_{label}.txt"
    report_path.write_text(report)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PubMed abstracts")
    parser.add_argument("--input", required=True, help="Glob to raw CSV(s)")
    parser.add_argument("--journal", default="JOA", help="Label for output files")
    args = parser.parse_args()
    preprocess(args.input, args.journal)

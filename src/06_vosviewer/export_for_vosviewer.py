"""
VOSviewer Comparison Export
============================
Exports the corpus in the formats required for VOSviewer keyword co-occurrence
analysis, which serves as the traditional bibliometric comparison method against
BERTopic (Protocol Objective 3).

Two export formats are produced:

1. RIS file  — import directly into VOSviewer via "Create a map based on
               bibliographic data" → RIS format. VOSviewer extracts keywords
               automatically (author keywords + title words + abstract words).

2. Co-occurrence matrix CSV — a pre-computed keyword co-occurrence matrix that
               can be loaded into VOSviewer via "Create a map based on network
               data". This gives finer control over which keyword types to use
               (author keywords vs MeSH vs title-abstract derived).

Comparison methodology
-----------------------
BERTopic discovers latent topics from full abstract text using transformer
embeddings.  VOSviewer identifies explicit research fronts from keyword
co-occurrence networks.  Comparison is qualitative (do the topic labels match
co-occurring keyword clusters?) and quantitative (Jaccard similarity between
the top-N keywords per BERTopic topic and the top-N co-occurring keywords per
VOSviewer cluster, and proportion of BERTopic topics that have a corresponding
VOSviewer cluster by visual inspection).

Usage:
    python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY
    python src/06_vosviewer/export_for_vosviewer.py --journal ARTHROSCOPY \
        --keyword_type author_keywords --min_keyword_freq 5

Outputs:
    outputs/vosviewer/vosviewer_ris_JOA.ris
    outputs/vosviewer/keyword_cooccurrence_JOA.csv
    outputs/vosviewer/vosviewer_readme_JOA.txt
"""

import argparse
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd
from loguru import logger

OUTPUT_DIR = Path("outputs/vosviewer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── RIS export ─────────────────────────────────────────────────────────────────

def row_to_ris(row: pd.Series) -> str:
    """Convert one corpus row to a RIS record string."""
    lines = ["TY  - JOUR"]
    if pd.notna(row.get("pmid")):
        lines.append(f"AN  - {row['pmid']}")
    if pd.notna(row.get("title")):
        lines.append(f"TI  - {row['title']}")
    if pd.notna(row.get("abstract")):
        lines.append(f"AB  - {row['abstract']}")
    if pd.notna(row.get("year")):
        lines.append(f"PY  - {int(row['year'])}")
    if pd.notna(row.get("journal")):
        lines.append(f"JO  - {row['journal']}")
    if pd.notna(row.get("authors")):
        for author in str(row["authors"]).split(";"):
            author = author.strip()
            if author:
                lines.append(f"AU  - {author}")
    if pd.notna(row.get("keywords")):
        for kw in str(row["keywords"]).split(";"):
            kw = kw.strip()
            if kw:
                lines.append(f"KW  - {kw}")
    if pd.notna(row.get("doi")):
        lines.append(f"DO  - {row['doi']}")
    lines.append("ER  - ")
    return "\n".join(lines)


def export_ris(df: pd.DataFrame, journal: str):
    """Write a full RIS file for the corpus."""
    ris_path = OUTPUT_DIR / f"vosviewer_ris_{journal}.ris"
    with open(ris_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(row_to_ris(row))
            f.write("\n\n")
    logger.success(f"RIS export: {ris_path}  ({len(df)} records)")
    return ris_path


# ── Keyword co-occurrence matrix ───────────────────────────────────────────────

_STOP = {
    "a", "an", "the", "in", "of", "for", "and", "or", "to", "with",
    "on", "at", "by", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "not", "this", "that", "from", "as", "it", "its",
    "we", "our", "their", "after", "before", "between", "among",
}


def extract_author_keywords(kw_str: str) -> list[str]:
    """Parse semicolon-separated author keyword strings into clean tokens."""
    if pd.isna(kw_str) or not str(kw_str).strip():
        return []
    return [k.strip().lower() for k in str(kw_str).split(";") if k.strip()]


def extract_mesh_terms(mesh_str: str) -> list[str]:
    if pd.isna(mesh_str) or not str(mesh_str).strip():
        return []
    return [m.strip().lower() for m in str(mesh_str).split(";") if m.strip()]


def extract_title_keywords(title: str, min_len: int = 4) -> list[str]:
    """Very basic noun-phrase extraction from title: filtered unigrams."""
    if pd.isna(title):
        return []
    tokens = re.findall(r"[a-zA-Z]+", str(title).lower())
    return [t for t in tokens if len(t) >= min_len and t not in _STOP]


def build_cooccurrence_matrix(
    keyword_lists: list[list[str]],
    min_freq: int = 5,
    max_keywords: int = 500,
) -> pd.DataFrame:
    """
    Build a keyword co-occurrence matrix from per-document keyword lists.

    Only keywords appearing in at least `min_freq` documents are included.
    The matrix value at [kw_i, kw_j] is the number of documents where both
    keywords appear together.
    """
    # Count keyword frequencies
    freq = Counter(kw for doc_kws in keyword_lists for kw in set(doc_kws))
    vocab = [kw for kw, cnt in freq.most_common(max_keywords) if cnt >= min_freq]
    vocab_set = set(vocab)

    logger.info(f"Vocabulary: {len(vocab)} keywords (min_freq={min_freq})")

    # Count co-occurrences
    cooc = defaultdict(int)
    for doc_kws in keyword_lists:
        filtered = list(set(doc_kws) & vocab_set)
        for a, b in combinations(sorted(filtered), 2):
            cooc[(a, b)] += 1

    # Build matrix
    matrix = pd.DataFrame(0, index=vocab, columns=vocab)
    for (a, b), cnt in cooc.items():
        matrix.loc[a, b] = cnt
        matrix.loc[b, a] = cnt

    return matrix


def export_cooccurrence(df: pd.DataFrame, journal: str, keyword_type: str, min_freq: int):
    """Build and save co-occurrence matrix for the chosen keyword type."""
    if keyword_type == "author_keywords":
        keyword_lists = df["keywords"].apply(extract_author_keywords).tolist()
        label = "author keywords"
    elif keyword_type == "mesh":
        keyword_lists = df["mesh_terms"].apply(extract_mesh_terms).tolist()
        label = "MeSH terms"
    elif keyword_type == "title":
        keyword_lists = df["title"].apply(extract_title_keywords).tolist()
        label = "title-derived unigrams"
    else:
        raise ValueError(f"Unknown keyword_type: {keyword_type}")

    matrix = build_cooccurrence_matrix(keyword_lists, min_freq=min_freq)
    out_path = OUTPUT_DIR / f"keyword_cooccurrence_{journal}.csv"
    matrix.to_csv(out_path)
    logger.success(
        f"Co-occurrence matrix ({label}): {matrix.shape[0]}×{matrix.shape[1]} → {out_path}"
    )
    return matrix


# ── README for VOSviewer operator ─────────────────────────────────────────────

def write_readme(journal: str, keyword_type: str, min_freq: int):
    readme = f"""VOSviewer Import Instructions — {journal}
{'='*60}

OPTION 1: RIS import (recommended for beginners)
-------------------------------------------------
1. Open VOSviewer
2. File → Create a map based on bibliographic data
3. Select "RIS" format
4. Load: outputs/vosviewer/vosviewer_ris_{journal}.ris
5. Choose "Co-occurrence" → "Author keywords" (or "All keywords")
6. Set minimum keyword occurrences to {min_freq}
7. Run → Visualize

OPTION 2: Pre-computed co-occurrence matrix
-------------------------------------------
1. Open VOSviewer
2. File → Create a map based on network data
3. Select "co-occurrence matrix (CSV)"
4. Load: outputs/vosviewer/keyword_cooccurrence_{journal}.csv
5. Keyword type used: {keyword_type}
6. Minimum frequency threshold: {min_freq}

COMPARISON WITH BERTOPIC
-------------------------
For each VOSviewer cluster:
  a. List its top-10 keywords by link strength
  b. For each BERTopic topic: compute Jaccard similarity between
     its top-10 words and the VOSviewer cluster keywords
  c. A BERTopic topic is "matched" if Jaccard >= 0.2 with any cluster
  d. Report: % of BERTopic topics matched, mean Jaccard, visual overlay

Manuscript language:
  "We compared BERTopic-derived topics with VOSviewer keyword co-occurrence
   clusters using Jaccard similarity on the top-10 representative keywords
   (threshold J≥0.20) and qualitative visual inspection of cluster labels."
"""
    out_path = OUTPUT_DIR / f"vosviewer_readme_{journal}.txt"
    out_path.write_text(readme)
    logger.info(f"README written: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def export_for_vosviewer(journal: str, keyword_type: str = "author_keywords", min_freq: int = 5):
    corpus_path = Path("data/processed") / f"corpus_clean_{journal}.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path} — run preprocessing first")

    df = pd.read_csv(corpus_path)
    logger.info(f"Loaded {len(df)} documents from {corpus_path}")

    export_ris(df, journal)
    export_cooccurrence(df, journal, keyword_type, min_freq)
    write_readme(journal, keyword_type, min_freq)

    logger.success("VOSviewer export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export corpus for VOSviewer comparison")
    parser.add_argument("--journal", default="JOA")
    parser.add_argument(
        "--keyword_type",
        default="author_keywords",
        choices=["author_keywords", "mesh", "title"],
        help="Which keyword type to use for the co-occurrence matrix",
    )
    parser.add_argument(
        "--min_keyword_freq", type=int, default=5,
        help="Minimum number of documents a keyword must appear in",
    )
    args = parser.parse_args()
    export_for_vosviewer(args.journal, args.keyword_type, args.min_keyword_freq)

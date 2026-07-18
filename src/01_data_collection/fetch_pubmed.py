"""
Phase 1: PubMed Abstract Retrieval (Skill Template)
=============================================================
Fetches qualifying abstracts from any biomedical journal via NCBI Entrez.

This is the bertopic-study skill's bundled template. It is parameterized
via CLI args so you do not need to edit the script per-journal.

Usage:
    python src/01_data_collection/fetch_pubmed.py \\
        --journal JOA \\
        --query   '("J Arthroplasty"[Journal] OR "1532-8406"[ISSN] OR "0883-5403"[ISSN])' \\
        --journal_filter Arthroplasty \\
        --start_year 1995 \\
        --end_year   2026

Why date chunking?
    NCBI's efetch hard-caps retstart at 9999 even with WebEnv history.
    A journal with >10k matching records will silently truncate.
    Chunking by 5-year buckets keeps each batch under the cap.

Why a journal_filter?
    PubMed's `0883-5403[ISSN]`-style queries can bleed in sibling titles
    (European editions, oncology spinoffs, name-collision journals).
    We post-filter on the journal name field to drop contamination.

Outputs:
    data/raw/pubmed_raw_<KEY>_YYYYMMDD.csv
    data/raw/fetch_log_YYYYMMDD.txt
"""

import os
import sys
import time
import argparse
from typing import Optional
import pandas as pd
from datetime import datetime
from pathlib import Path
from Bio import Entrez
from dotenv import load_dotenv
from tqdm import tqdm
from loguru import logger

load_dotenv()

Entrez.email = os.getenv("PUBMED_EMAIL", "researcher@example.com")
_api_key = os.getenv("PUBMED_API_KEY", "").strip().lstrip("#").strip()
if _api_key and not _api_key.startswith("#"):
    Entrez.api_key = _api_key


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_log_path = OUTPUT_DIR / f"fetch_log_{datetime.now().strftime('%Y%m%d')}.txt"
logger.add(str(_log_path), format="{time} | {level} | {message}", level="INFO")


def build_query(journal_query: str, start_year: int, end_year: int,
                final_end_year: int = None, final_end_month: str = "05/01") -> str:
    """Compose the full PubMed search string from a journal query and date range.

    For intermediate chunks (end_year < final_end_year) we use Dec 31 so we don't
    silently drop Jun-Dec records. The May-01 cutoff is applied only to the last chunk.
    """
    if final_end_year is not None and end_year < final_end_year:
        end_date = f"{end_year}/12/31"
    else:
        end_date = f"{end_year}/{final_end_month}"
    date_term = f'("{start_year}/01/01"[PDAT] : "{end_date}"[PDAT])'
    return f'{journal_query} AND {date_term} AND hasabstract[text] AND English[lang]'


def search_pubmed(query: str, retmax: int = 100000, retries: int = 5) -> tuple:
    """Return (total_count, webenv, query_key) using NCBI history server."""
    logger.info(f"Searching PubMed: {query}")
    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=0, usehistory="y")
            record = Entrez.read(handle)
            handle.close()
            total = int(record["Count"])
            webenv = record["WebEnv"]
            query_key = record["QueryKey"]
            logger.info(f"Total matching records: {total}")
            return total, webenv, query_key
        except Exception as e:
            wait = 15 * attempt
            logger.warning(f"Attempt {attempt}/{retries} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"PubMed search failed after {retries} attempts")


def fetch_records_batch(total: int, webenv: str, query_key: str, batch_size: int = 200) -> list:
    """Fetch records using WebEnv. Caller is responsible for staying under retstart=9999."""
    records = []
    n_batches = (total + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Fetching batches"):
        retstart = i * batch_size
        try:
            handle = Entrez.efetch(
                db="pubmed", rettype="xml", retmode="xml",
                retstart=retstart, retmax=batch_size,
                webenv=webenv, query_key=query_key,
            )
            batch_records = Entrez.read(handle)
            handle.close()
            for article in batch_records["PubmedArticle"]:
                parsed = parse_article(article)
                if parsed:
                    records.append(parsed)
        except Exception as e:
            logger.error(f"Batch {i} (retstart={retstart}) failed: {e}")
            time.sleep(5)
            continue
        sleep_time = 0.11 if Entrez.api_key else 0.34
        time.sleep(sleep_time)
    return records


def parse_article(article: dict) -> Optional[dict]:
    try:
        medline = article["MedlineCitation"]
        art = medline["Article"]

        title = str(art.get("ArticleTitle", "")).strip()

        abstract_text = ""
        if "Abstract" in art:
            abstract_obj = art["Abstract"].get("AbstractText", "")
            if isinstance(abstract_obj, list):
                abstract_text = " ".join([
                    f"{getattr(sec, 'attributes', {}).get('Label', '')}: {str(sec)}"
                    for sec in abstract_obj
                ]).strip()
            else:
                abstract_text = str(abstract_obj).strip()

        if not abstract_text:
            return None

        pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = str(pub_date.get("Year", "")).strip()
        month = str(pub_date.get("Month", "")).strip()
        if not year:
            year = str(medline.get("DateCompleted", {}).get("Year", "")).strip()

        journal_info = art.get("Journal", {})
        journal_name = str(journal_info.get("Title", "")).strip()
        journal_abbrev = str(journal_info.get("ISOAbbreviation", "")).strip()
        volume = str(journal_info.get("JournalIssue", {}).get("Volume", "")).strip()
        issue = str(journal_info.get("JournalIssue", {}).get("Issue", "")).strip()
        pages = str(art.get("Pagination", {}).get("MedlinePgn", "")).strip()

        doi = ""
        for id_obj in article.get("PubmedData", {}).get("ArticleIdList", []):
            if getattr(id_obj, "attributes", {}).get("IdType") == "doi":
                doi = str(id_obj).strip()
                break

        author_list = art.get("AuthorList", [])
        authors = []
        for author in author_list:
            last = str(author.get("LastName", "")).strip()
            fore = str(author.get("ForeName", "")).strip()
            if last:
                authors.append(f"{last} {fore}".strip())
        authors_str = "; ".join(authors)

        affiliation = ""
        if author_list:
            aff_list = author_list[0].get("AffiliationInfo", [])
            if aff_list:
                affiliation = str(aff_list[0].get("Affiliation", "")).strip()

        keyword_list = medline.get("KeywordList", [[]])
        keywords = "; ".join([str(kw) for sublist in keyword_list for kw in sublist])

        mesh_list = medline.get("MeshHeadingList", [])
        mesh_terms = "; ".join([
            str(mh.get("DescriptorName", "")) for mh in mesh_list
        ])

        pub_types = "; ".join([
            str(pt) for pt in art.get("PublicationTypeList", [])
        ])

        pmid = str(medline["PMID"])

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract_text,
            "year": year,
            "month": month,
            "journal": journal_name,
            "journal_abbrev": journal_abbrev,
            "volume": volume,
            "issue": issue,
            "pages": pages,
            "doi": doi,
            "authors": authors_str,
            "keywords": keywords,
            "mesh_terms": mesh_terms,
            "pub_types": pub_types,
            "affiliation": affiliation,
            "n_authors": len(authors),
        }
    except Exception as e:
        logger.warning(f"Failed to parse article: {e}")
        return None


def fetch_journal(
    journal_key: str,
    journal_query: str,
    journal_filter: str,
    start_year: int,
    end_year: int,
    chunk_years: int = 5,
) -> pd.DataFrame:
    """Chunk-fetch all records for journal_key in [start_year, end_year]."""
    all_records = []
    chunk_starts = list(range(start_year, end_year + 1, chunk_years))
    chunk_ranges = [(s, min(s + chunk_years - 1, end_year)) for s in chunk_starts]

    for cs, ce in chunk_ranges:
        logger.info(f"\n--- Chunk {cs}-{ce} ---")
        query = build_query(journal_query, cs, ce, final_end_year=end_year)
        total, webenv, query_key = search_pubmed(query)
        if total == 0:
            logger.warning(f"  No records in {cs}-{ce}")
            continue
        if total > 9999:
            logger.warning(
                f"  Chunk {cs}-{ce} has {total} records (> 9999 efetch cap). "
                f"Drop chunk_years to keep each bucket under 10k."
            )
        records = fetch_records_batch(total, webenv, query_key)
        all_records.extend(records)
        logger.success(f"  Chunk {cs}-{ce}: fetched {len(records)} records")

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning(f"No records parsed for {journal_key}")
        return df

    df = df.drop_duplicates(subset="pmid")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "abstract"])
    df = df[df["year"] >= start_year]

    # ── Contamination guard ───────────────────────────────────────────────────
    # ISSN queries can bleed sibling titles. Keep only records whose journal
    # field contains the substring the caller passed (case-insensitive).
    if journal_filter and "journal" in df.columns:
        mask = df["journal"].str.contains(journal_filter, case=False, na=False)
        n_dropped = (~mask).sum()
        if n_dropped > 0:
            logger.warning(
                f"Contamination guard dropped {n_dropped} records "
                f"not matching '{journal_filter}' in journal name."
            )
            dropped_journals = df.loc[~mask, "journal"].value_counts().head(10)
            logger.warning(f"Top contaminating journals:\n{dropped_journals.to_string()}")
        df = df[mask]

    df = df.sort_values("year")
    date_str = datetime.now().strftime("%Y%m%d")
    outpath = OUTPUT_DIR / f"pubmed_raw_{journal_key}_{date_str}.csv"
    df.to_csv(outpath, index=False)
    logger.success(f"Saved {len(df)} records to {outpath}")

    logger.info(f"\n{'='*50}")
    logger.info(f"Journal:       {journal_key}")
    logger.info(f"Filter:        '{journal_filter}'")
    logger.info(f"Final records: {len(df)}")
    logger.info(f"Year range:    {df['year'].min():.0f}-{df['year'].max():.0f}")
    logger.info(f"{'='*50}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts (skill template)")
    parser.add_argument("--journal",        required=True,
                        help="Short journal abbreviation key (e.g. JOA). Used in output filenames.")
    parser.add_argument("--query",          required=True,
                        help="PubMed journal selector, e.g. '(\"J Arthroplasty\"[Journal] OR \"1532-8406\"[ISSN])'")
    parser.add_argument("--journal_filter", default="",
                        help="Case-insensitive substring required in the journal-name field "
                             "of returned records (contamination guard). E.g. 'Arthroplasty'.")
    parser.add_argument("--start_year", type=int, required=True)
    parser.add_argument("--end_year",   type=int, required=True,
                        help="Use the current calendar year, not a hardcoded value.")
    parser.add_argument("--chunk_years", type=int, default=5,
                        help="Years per fetch bucket. Lower if any chunk exceeds 9999 records.")
    args = parser.parse_args()

    fetch_journal(args.journal, args.query, args.journal_filter,
                  args.start_year, args.end_year, args.chunk_years)

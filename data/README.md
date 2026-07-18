# Data

## What is here

| File | Contents |
|---|---|
| `included_records.csv` | PMID, DOI, year and title of all 10,597 analyzed records |
| `excluded_records.csv` | The 29 excluded records with exclusion reason |
| `annual_counts.csv` | Records per publication year |
| `pubtype_composition.csv` | PubMed publication-type mix of the corpus |

## What is deliberately not here

**Abstract full text is not redistributed.** Abstracts published in The
Journal of Arthroplasty are copyright Elsevier. This repository ships the
PubMed identifiers instead, which is sufficient to regenerate the exact
corpus:

```bash
python src/01_data_collection/fetch_pubmed.py --journal JOA
python src/02_preprocessing/preprocess.py --journal JOA
```

The search is fully specified in `docs/REPRODUCIBILITY.md`. Because the
query is date-bounded and PubMed indexing is append-only for closed years,
re-running it reproduces the same record set for 2005 through 2025.

Embeddings (32 MB) and the fitted BERTopic model are also omitted; both are
regenerated deterministically by the pipeline using the seeds recorded in
`docs/REPRODUCIBILITY.md`.

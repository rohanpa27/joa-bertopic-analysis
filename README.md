# BERTopic analysis of The Journal of Arthroplasty (2005-2026)

Analysis code and derived results for a transformer-based topic model of 10,597
abstracts published in *The Journal of Arthroplasty*, covering topic discovery,
temporal trend classification, and the clustering and model-specification
sensitivity analyses added during peer review.

This repository is released so the analysis can be independently rerun,
audited, and reused.

## Quick start

```bash
git clone https://github.com/rohanpa27/joa-bertopic-analysis.git
cd joa-bertopic-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# rebuild the corpus from PubMed (abstracts are not redistributed, see below)
python src/01_data_collection/fetch_pubmed.py --journal JOA
python src/02_preprocessing/preprocess.py --journal JOA

# full pipeline: embed -> cluster -> label -> trends -> figures
bash run_pipeline.sh
```

Exact search string, hyperparameters, random seeds, and package versions are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## A correction worth reading first

The originally submitted manuscript stated that partial-year 2026 was excluded
from the temporal regressions. It was not: the year vector ran to 2026 and every
published slope included a year indexed only through May. This was found during
peer review and corrected. Emerging/Stable/Cold counts moved from 34/33/16 to
**36/31/16**, four topics were reclassified, and the rank order of the top
gainers and decliners was unchanged.

Both the original and corrected trend tables are published here for comparison.
Details in [`docs/CORRECTION_NOTE.md`](docs/CORRECTION_NOTE.md).

## Layout

```
src/                        the pipeline, phase by phase
  01_data_collection/       PubMed retrieval via Entrez
  02_preprocessing/         cleaning and eligibility filtering
  03_modeling/              PubMedBERT embedding, UMAP + HDBSCAN, BERTopic
  04_analysis/              temporal trends, model validation
  05_visualization/         figures and tables
  06_vosviewer/             keyword co-occurrence export

scripts/
  build_clinical_labels.py            the 250-rule keyword-to-label mapping
  apply_clinical_labels.py            merges labels into result tables
  generate_publication_figures_*.py   data-driven figures
  revision_analyses/R1..R7            everything added during peer review

data/                       PMID/DOI identifiers, exclusions, annual counts
results/                    derived tables (trends, sensitivity, similarity)
docs/                       reproducibility notes and the correction note
```

## What the revision analyses do

| Script | Question it answers |
|---|---|
| `R1_provenance_and_trends.py` | Record provenance, annual noise, why no topic was classified "Hot", slope CIs and negative binomial refits |
| `R2_similarity_and_params.py` | Re-clusters under 8 parameter/seed configurations; topic similarity |
| `R3_reporting_language.py` | Did abstract writing style drift confound the trends? MeSH-based corroboration |
| `R4_conclusion_stability.py` | Do the *conclusions* survive re-clustering, even when the partition does not? |
| `R5_label_review_sheet.py` | Builds the topic-label validation worksheet (representative abstracts per topic) |
| `R6_renumber_and_supp_figs.py` | Supplementary figures: annual noise, OLS vs negative binomial concordance |
| `R7_recompute_trends_2005_2025.py` | The corrected complete-year temporal analysis |

Two results are worth highlighting. The exact document partition is
seed-sensitive (adjusted Rand index 0.51 to 0.68), but the **direction and
significance of every thematic trend was identical across all eight clustering
configurations** - periprosthetic joint infection, patient-reported outcomes and
machine learning rising; cup orientation, polyethylene wear, ceramic bearings and
case reports falling. Separately, abstract prose changed enormously over 21 years
(structured headings 0.1% to 97.5%; explicit P values 18.1% to 61.4%), so the
methods-flavored topics are partly a reporting-style artifact - but MeSH-indexed
signals, which are independent of prose, corroborate the substantive trends.

## Data availability and copyright

**Abstract full text is not redistributed.** Abstracts published in *The Journal
of Arthroplasty* are copyright Elsevier. This repository ships PubMed
identifiers instead, which is sufficient to regenerate the exact corpus with the
included fetch script. Topic assignments are published keyed on PMID with the
text columns removed.

Also omitted, because both are large and deterministically regenerable: the
768-dimensional embedding matrix and the fitted BERTopic model binary.

See [`data/README.md`](data/README.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the article once published.

## License

MIT for the code and derived result tables (see [`LICENSE`](LICENSE)). The
license does not extend to the underlying journal abstracts, which are not
included here.

# Reproducibility

## Search strategy

Executed against NCBI PubMed via the Entrez API on **26 May 2026**, in
date-partitioned blocks to stay within the NCBI retrieval ceiling:

```
("J Arthroplasty"[Journal] OR "1532-8406"[ISSN] OR "0883-5403"[ISSN])
AND ("YYYY/MM/DD"[PDAT] : "YYYY/MM/DD"[PDAT])
AND hasabstract[text] AND English[lang]
```

Blocks: 2005-2007, 2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022,
2023-2025, 2026 (through 1 May).

Retrieved 10,626 records; 0 duplicate PMIDs; 17 excluded as editorials,
letters, comments or errata; 12 excluded for abstracts under 50 words;
10,597 analyzed (99.7% retention).

## Model configuration

| Stage | Setting |
|---|---|
| Embedding | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` (abstract-only) |
| Max sequence length | 512 tokens (186 abstracts, 1.8%, truncated) |
| Device | CPU, for bitwise reproducibility |
| UMAP | `n_neighbors=10, n_components=5, min_dist=0.0, metric=cosine` |
| HDBSCAN | `min_cluster_size=20, min_samples=5, method=eom` |
| Representation | c-TF-IDF + KeyBERT-Inspired + MMR |
| Random seed | **42** (all UMAP/HDBSCAN runs unless stated) |
| Sensitivity seeds | 42, 7, 2024 |

The 20-combination hyperparameter grid and its coherence/noise results are in
`results/hyperparameter_grid_search.csv`. The selected configuration maximized
c-NPMI coherence subject to a 45% noise ceiling.

## Temporal model

OLS on annual proportional share over **complete calendar years 2005-2025**
(partial-year 2026 excluded from all fitting), Benjamini-Hochberg FDR at
alpha = 0.05, with 95% confidence intervals. Every trend re-estimated with
negative binomial regression on annual counts using log total annual output as
an offset, dispersion estimated by Cameron-Trivedi auxiliary regression.
Nonlinearity tested with a quadratic year term.

See `docs/CORRECTION_NOTE.md` for an important correction to the originally
submitted analysis.

## Order of execution

```bash
pip install -r requirements.txt
bash run_pipeline.sh                                   # phases 1-10
python scripts/build_clinical_labels.py
python scripts/apply_clinical_labels.py

python scripts/revision_analyses/R1_provenance_and_trends.py
python scripts/revision_analyses/R2_similarity_and_params.py   # slow, ~10 min
python scripts/revision_analyses/R3_reporting_language.py
python scripts/revision_analyses/R4_conclusion_stability.py    # needs R2 output
python scripts/revision_analyses/R5_label_review_sheet.py
python scripts/revision_analyses/R6_renumber_and_supp_figs.py
python scripts/revision_analyses/R7_recompute_trends_2005_2025.py
```

## A note on paths

The scripts expect the pipeline's *working* layout, which is what
`run_pipeline.sh` creates in the directory you run it from:

```
data/raw/  data/processed/  data/embeddings/
outputs/tables/  outputs/figures/  outputs/models/
Revisions/01_analyses/  Revisions/02_figures/     <- revision script outputs
```

The `results/` and `data/` directories in *this repository* are a published
snapshot of those outputs, provided so the tables can be inspected without
rerunning anything. They are not the paths the scripts read from or write to.

To rerun the revision analyses end to end, run `run_pipeline.sh` first to
populate `outputs/tables/`, create `Revisions/01_analyses/`, then run the R1-R7
scripts from the same working directory. Each script defines its input and
output paths in the first few lines if you prefer to point them elsewhere.

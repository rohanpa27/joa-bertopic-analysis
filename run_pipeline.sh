#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# BERTopic Journal Study -- Full Pipeline (Skill Template)
#
# Customize the 4 lines below for your study, then run from the project root
# with the JSES venv active:
#   source /Users/rohan/Documents/BERTopic\ Studies/JSES\ BERT/venv/bin/activate
#   cd <PROJECT_DIR>
#   bash run_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ====== CUSTOMIZE THESE PER-STUDY ======
JOURNAL="JOA"                    # short abbreviation key, e.g. JOA, JHS, JBJS
JOURNAL_QUERY='("J Arthroplasty"[Journal] OR "1532-8406"[ISSN] OR "0883-5403"[ISSN])'
JOURNAL_FILTER="Arthroplasty"    # required substring in journal name (contamination guard)
START_YEAR=2005
END_YEAR=2026                    # May 2026 cutoff per study specification
# ========================================

echo "=================================================="
echo " BERTopic Journal Study"
echo " Journal: $JOURNAL  ($START_YEAR -- $END_YEAR)"
echo " Started: $(date)"
echo "=================================================="

# -- Phase 1: Data collection -------------------------------------------------
echo ""
echo "-- Phase 1: Fetching abstracts from PubMed ---------------------------------"
python src/01_data_collection/fetch_pubmed.py \
    --journal        "$JOURNAL" \
    --query          "$JOURNAL_QUERY" \
    --journal_filter "$JOURNAL_FILTER" \
    --start_year     $START_YEAR \
    --end_year       $END_YEAR

# -- Phase 2: Preprocessing ---------------------------------------------------
echo ""
echo "-- Phase 2: Preprocessing corpus -------------------------------------------"
python src/02_preprocessing/preprocess.py \
    --input   "data/raw/pubmed_raw_${JOURNAL}_*.csv" \
    --journal "$JOURNAL"

# -- Phase 2b: VOSviewer export -----------------------------------------------
echo ""
echo "-- Phase 2b: Exporting for VOSviewer comparison ----------------------------"
python src/06_vosviewer/export_for_vosviewer.py --journal "$JOURNAL"

# -- VALIDATION GATE ----------------------------------------------------------
# Before the long-running embedding step, sanity-check the corpus. If this
# prints anything unexpected (wrong year range, wrong journal mix, anomalous N)
# STOP and investigate before proceeding. The embedding step takes 30-90 min.
echo ""
echo "-- Validation gate: corpus characteristics ---------------------------------"
python - <<PY
import pandas as pd
df = pd.read_csv("data/processed/corpus_clean_${JOURNAL}.csv")
print(f"N records:    {len(df)}")
print(f"Year range:   {df['year'].min()}-{df['year'].max()}")
print(f"Top journals: {df['journal'].value_counts().head(5).to_dict()}")
print("Sample titles:")
for t in df.sample(min(5, len(df)), random_state=42)['title']:
    print(f"  - {t[:100]}")
PY
echo ""
echo "If the above looks correct, embedding will start in 10 seconds (Ctrl-C to abort)..."
sleep 10

# -- Phase 3: Embedding -------------------------------------------------------
echo ""
echo "-- Phase 3: Generating PubMedBERT embeddings -------------------------------"
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -i -s python -u src/03_modeling/embed.py \
    --input      "data/processed/corpus_clean_${JOURNAL}.csv" \
    --model      pubmedbert \
    --batch_size 64 \
    --journal    "$JOURNAL"
# Note: caffeinate -i -s prevents the Mac from sleeping during the long embed,
# which would otherwise thermal-throttle the process to 50x slower (see known-gotchas.md).

# -- Phases 4-6: BERTopic modeling --------------------------------------------
echo ""
echo "-- Phases 4-6: BERTopic modeling with grid search --------------------------"
python src/03_modeling/run_bertopic.py \
    --corpus     "data/processed/corpus_clean_${JOURNAL}.csv" \
    --embeddings "data/embeddings/embeddings_pubmedbert_${JOURNAL}.npy" \
    --journal    "$JOURNAL" \
    --grid_search

# -- Phase 7: Temporal analysis -----------------------------------------------
echo ""
echo "-- Phase 7: Temporal trend analysis (BH-FDR corrected) ---------------------"
python src/04_analysis/temporal_analysis.py \
    --journal "$JOURNAL" \
    --min_year $START_YEAR \
    --max_year $END_YEAR

# -- Phase 8: Validation ------------------------------------------------------
echo ""
echo "-- Phase 8: Validation metrics ---------------------------------------------"
python src/04_analysis/validate.py --journal "$JOURNAL"

# -- Phase 9: Figures ---------------------------------------------------------
echo ""
echo "-- Phase 9: Generating interactive figures ---------------------------------"
python src/05_visualization/generate_figures.py --journal "$JOURNAL"

# -- Phase 10: Tables ---------------------------------------------------------
echo ""
echo "-- Phase 10: Generating publication tables ---------------------------------"
python src/05_visualization/generate_tables.py --journal "$JOURNAL"

# -- Summary ------------------------------------------------------------------
echo ""
echo "=================================================="
echo " Pipeline complete!  $(date)"
echo "=================================================="
echo ""
echo "Outputs:"
echo "  Tables   --> outputs/tables/"
echo "  Reports  --> outputs/reports/"
echo "  Model    --> outputs/models/bertopic_model_${JOURNAL}/"
echo ""
echo "Next steps:"
echo "  1. python build_clinical_labels.py    # auto-curate clinical labels"
echo "  2. python apply_clinical_labels.py    # merge labels into output tables"
echo "  3. python generate_publication_figures_v2.py    # 4 PNG figures"
echo "  4. python create_abstract_docx.py     # final DOCX manuscript"

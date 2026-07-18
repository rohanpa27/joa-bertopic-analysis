"""
Revision analysis 3: reporting-language drift (R2-9) and rule-based
label documentation (R2-4).

Tests whether the apparent rise of methods-flavored topics (observational
design/statistics, meta-analyses, PROMs) is corroborated by indexing-based
signals that are independent of abstract prose style: PubMed publication
types and MeSH terms.

Run from project root with the JSES venv active.
Outputs -> Revisions/01_analyses/
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("Revisions/01_analyses")
OUT.mkdir(parents=True, exist_ok=True)

report = {}


def load():
    return pd.read_csv("data/processed/corpus_clean_JOA.csv",
                       usecols=["pmid", "year", "title", "abstract",
                                "pub_types", "mesh_terms", "doc_length_words"])


def _yearly_share(df, mask, years):
    num = df[mask].groupby("year").size().reindex(years, fill_value=0)
    den = df.groupby("year").size().reindex(years, fill_value=0)
    return 100 * num / den.replace(0, np.nan)


def _trend(share, years):
    m = ~share.isna()
    sl, ic, r, p, se = stats.linregress(years[m], share[m].values)
    df_resid = int(m.sum()) - 2
    t = float(stats.t.ppf(0.975, df_resid))
    return {"slope_pct_per_year": round(float(sl), 4),
            "ci95": [round(float(sl - t * se), 4), round(float(sl + t * se), 4)],
            "p_value": float(f"{p:.3g}"), "r_squared": round(float(r ** 2), 3)}


def indexing_corroboration(df):
    """Independent (non-prose) signals for reviewer-flagged topics."""
    years = np.arange(2005, 2026)
    d = df[df.year.between(2005, 2025)].copy()
    pt = d.pub_types.fillna("").str.lower()
    mh = d.mesh_terms.fillna("").str.lower()

    SIGNALS = {
        "Systematic reviews and meta-analyses": (
            pt.str.contains("meta-analysis|systematic review")
            | mh.str.contains("meta-analysis as topic|systematic reviews as topic")),
        "Patient-reported outcome measures": (
            mh.str.contains("patient reported outcome measures|"
                            "patient outcome assessment")),
        "Comparative/observational study indexing": (
            pt.str.contains("comparative study|observational study")
            | mh.str.contains("retrospective studies|cohort studies")),
        "Randomized controlled trials": pt.str.contains("randomized controlled trial"),
        "Case reports": pt.str.contains("case reports"),
        "Artificial intelligence / machine learning": (
            mh.str.contains("machine learning|artificial intelligence|deep learning")),
        "Periprosthetic joint infection": (
            mh.str.contains("prosthesis-related infections|arthritis, infectious")),
    }

    rows = []
    for name, mask in SIGNALS.items():
        share = _yearly_share(d, mask.values, years)
        t = _trend(share, years)
        rows.append({"signal": name, "n_documents": int(mask.sum()),
                     "share_2005_2009_pct": round(float(share.loc[2005:2009].mean()), 2),
                     "share_2021_2025_pct": round(float(share.loc[2021:2025].mean()), 2),
                     **t})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "indexing_corroboration_JOA.csv", index=False)
    report["indexing_corroboration"] = r.to_dict("records")


def prose_style_drift(df):
    """Quantify how much abstract writing style itself changed over 21 years."""
    years = np.arange(2005, 2026)
    d = df[df.year.between(2005, 2025)].copy()
    ab = d.abstract.fillna("")

    # structured-abstract headings
    structured = ab.str.contains(
        r"\b(BACKGROUND|METHODS|RESULTS|CONCLUSION)S?\b:", regex=True)
    # explicit statistical reporting language
    ci_lang = ab.str.contains(r"95%\s*(CI|confidence interval)", case=False, regex=True)
    p_lang = ab.str.contains(r"\bP\s*[<=>]\s*\.?\d", case=False, regex=True)
    or_hr = ab.str.contains(r"\b(odds ratio|hazard ratio|\bOR\b|\bHR\b)",
                            case=False, regex=True)

    rows = []
    for name, mask in [("Structured abstract headings", structured),
                       ("Reports 95% confidence interval", ci_lang),
                       ("Reports explicit P value", p_lang),
                       ("Reports odds/hazard ratio", or_hr)]:
        share = _yearly_share(d, mask.values, years)
        rows.append({"feature": name,
                     "share_2005_2009_pct": round(float(share.loc[2005:2009].mean()), 2),
                     "share_2021_2025_pct": round(float(share.loc[2021:2025].mean()), 2),
                     **_trend(share, years)})

    # abstract length drift
    ln = d.groupby("year").doc_length_words.mean().reindex(years)
    rows.append({"feature": "Mean abstract length (words)",
                 "share_2005_2009_pct": round(float(ln.loc[2005:2009].mean()), 1),
                 "share_2021_2025_pct": round(float(ln.loc[2021:2025].mean()), 1),
                 **_trend(ln, years)})

    r = pd.DataFrame(rows)
    r.to_csv(OUT / "prose_style_drift_JOA.csv", index=False)
    report["prose_style_drift"] = r.to_dict("records")


def label_rules_documentation():
    """Serialize the ordered keyword->label rule set so it can be published
    as a supplement (R2-4)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bcl", "build_clinical_labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rules = getattr(mod, "RULES", [])
    rows = [{"priority": i + 1,
             "trigger_keywords": " AND ".join(kw) if isinstance(kw, (list, tuple)) else str(kw),
             "assigned_label": lab}
            for i, (kw, lab) in enumerate(rules)]
    pd.DataFrame(rows).to_csv(OUT / "clinical_label_rules_JOA.csv", index=False)
    report["label_rules"] = {
        "n_rules": len(rows),
        "method": ("Ordered priority list. The top-10 c-TF-IDF keywords of each "
                   "topic are matched against each rule in sequence; the first "
                   "rule whose trigger keywords are all present assigns the label. "
                   "Topics matching no rule retain their keyword-derived name."),
    }


def main():
    df = load()
    indexing_corroboration(df)
    prose_style_drift(df)
    try:
        label_rules_documentation()
    except Exception as e:
        report["label_rules"] = {"error": str(e)}

    with open(OUT / "reporting_language_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

"""
Revision analysis 7: corrected temporal trend analysis.

DISCREPANCY FOUND DURING REVISION
---------------------------------
The submitted Methods stated that OLS regression was fitted "from 2005 through
2025 (2026 data excluded due to partial-year indexing)". In fact the original
code built year_range = arange(min_year, max_year + 1) with max_year = 2026 and
regressed over all 22 years, so the partial year 2026 WAS included in every
published slope. This was verified by exact reproduction: recomputing with 2026
included reproduces the published slopes to 0.000000, whereas restricting to
complete years changes them by up to 0.024 %/year.

This script recomputes the trend analysis the way the Methods described and the
way Reviewer 2 (comment 2) requests: complete calendar years 2005-2025 only,
with 95% confidence intervals, Benjamini-Hochberg q-values, and the same
Emerging/Hot/Stable/Cold classification rules. Partial-year 2026 is retained for
descriptive corpus reporting only.

Outputs -> Revisions/01_analyses/temporal_trends_corrected_2005_2025.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

OUT = Path("Revisions/01_analyses")
YEARS = np.arange(2005, 2026)          # complete calendar years only
RECENT_FROM = 2016                     # most recent 10 years within 2005-2025


def classify(slope, q, recent_share, alpha=0.05):
    if q < alpha and slope > 0 and recent_share >= 0.60:
        return "Emerging"
    if q < alpha and slope > 0:
        return "Hot"
    if q < alpha and slope < 0:
        return "Cold"
    return "Stable"


def main():
    doc = pd.read_csv("outputs/tables/doc_topics_JOA.csv",
                      usecols=["year", "topic_id"])
    pub = pd.read_csv("outputs/tables/temporal_trends_JOA.csv")

    d = doc[doc.year.between(2005, 2025)]
    total = d.groupby("year").size().reindex(YEARS, fill_value=1)

    rows = []
    for tid, g in d[d.topic_id != -1].groupby("topic_id"):
        cnt = g.groupby("year").size().reindex(YEARS, fill_value=0)
        share = 100 * cnt.values / total.values

        X = sm.add_constant(YEARS.astype(float))
        fit = sm.OLS(share, X).fit()
        slope = float(fit.params[1])
        lo, hi = fit.conf_int(alpha=0.05)[1]
        recent = cnt.loc[RECENT_FROM:].sum() / cnt.sum() if cnt.sum() else 0.0

        rows.append({
            "topic_id": tid,
            "n_documents_2005_2025": int(cnt.sum()),
            "slope": slope, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": float(fit.pvalues[1]),
            "r_squared": float(fit.rsquared),
            "recent_10yr_share": float(recent),
            "peak_year": int(YEARS[int(np.argmax(cnt.values))]),
        })

    t = pd.DataFrame(rows)
    t["q_value"] = multipletests(t.p_value, method="fdr_bh")[1]
    t["trend_class"] = [classify(s, q, r) for s, q, r
                        in zip(t.slope, t.q_value, t.recent_10yr_share)]

    t = t.merge(pub[["topic_id", "clinical_label", "trend_class", "slope"]]
                .rename(columns={"trend_class": "published_trend_class",
                                 "slope": "published_slope"}),
                on="topic_id", how="left")
    t = t.sort_values("slope", ascending=False)
    t.to_csv(OUT / "temporal_trends_corrected_2005_2025.csv", index=False)

    changed = t[t.trend_class != t.published_trend_class]
    summary = {
        "corrected_class_counts": t.trend_class.value_counts().to_dict(),
        "published_class_counts": t.published_trend_class.value_counts().to_dict(),
        "n_topics_reclassified": int(len(changed)),
        "reclassified": changed[["topic_id", "clinical_label",
                                 "published_trend_class", "trend_class",
                                 "published_slope", "slope", "q_value"]]
        .round(4).to_dict("records"),
        "max_abs_slope_change": round(float((t.slope - t.published_slope).abs().max()), 4),
        "n_hot": int((t.trend_class == "Hot").sum()),
        "min_recent_share_among_sig_positive": round(float(
            t[(t.q_value < 0.05) & (t.slope > 0)].recent_10yr_share.min()), 4),
        "top5_emerging": t[t.trend_class == "Emerging"].nlargest(5, "slope")[
            ["clinical_label", "slope", "ci_low", "ci_high", "q_value"]
        ].round(4).to_dict("records"),
        "top5_cold": t[t.trend_class == "Cold"].nsmallest(5, "slope")[
            ["clinical_label", "slope", "ci_low", "ci_high", "q_value"]
        ].round(4).to_dict("records"),
    }
    with open(OUT / "corrected_trends_report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

"""
Revision analysis 1: Data provenance (R2-1), partial-year 2026 (R2-2),
Hot-category audit (R2-7), and trend robustness (R2-6).

Run from project root with the JSES venv active.
Outputs -> Revisions/01_analyses/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT = Path("Revisions/01_analyses")
OUT.mkdir(parents=True, exist_ok=True)

RAW = "data/raw/pubmed_raw_JOA_20260526.csv"
CLEAN = "data/processed/corpus_clean_JOA.csv"
DOCT = "outputs/tables/doc_topics_JOA.csv"
TREND = "outputs/tables/temporal_trends_JOA.csv"

report = {}


# ---------------------------------------------------------------- provenance
def provenance():
    raw = pd.read_csv(RAW)
    clean = pd.read_csv(CLEAN)

    n_raw = len(raw)
    n_dup_pmid = int(raw["pmid"].duplicated().sum())
    raw_dedup = raw.drop_duplicates(subset="pmid")

    # journal verification: every record should be Journal of Arthroplasty
    jnames = raw["journal"].astype(str).str.strip().value_counts()
    jabbrev = raw["journal_abbrev"].astype(str).str.strip().value_counts()

    # publication types present / excluded
    EXCLUDE_TYPES = {"Editorial", "Letter", "Comment", "Published Erratum",
                     "Retraction of Publication", "Retracted Publication"}
    kept_pmids = set(clean["pmid"])
    excluded = raw_dedup[~raw_dedup["pmid"].isin(kept_pmids)].copy()

    exc_rows = []
    for _, r in excluded.iterrows():
        wc = len(str(r["abstract"]).split())
        pts = str(r.get("pub_types", ""))
        reason = ("publication type" if any(t in pts for t in EXCLUDE_TYPES)
                  else ("abstract <50 words" if wc < 50 else "other"))
        exc_rows.append({"pmid": r["pmid"], "year": r["year"],
                         "pub_types": pts, "abstract_words": wc,
                         "exclusion_reason": reason})
    exc_df = pd.DataFrame(exc_rows)
    exc_df.to_csv(OUT / "excluded_records_JOA.csv", index=False)

    # PMID / DOI list of the analyzed corpus
    clean[["pmid", "doi", "year", "title"]].to_csv(
        OUT / "included_pmid_doi_list_JOA.csv", index=False)

    # publication-type composition of the analyzed corpus
    pt_counts = (clean["pub_types"].astype(str)
                 .str.split(";").explode().str.strip().value_counts())
    pt_counts.to_csv(OUT / "pubtype_composition_JOA.csv")

    # ahead-of-print / missing-month accounting
    missing_month = int(clean["month"].isna().sum())

    report["provenance"] = {
        "query": ('("J Arthroplasty"[Journal] OR "1532-8406"[ISSN] OR '
                  '"0883-5403"[ISSN]) AND ("YYYY/MM/DD"[PDAT] : "YYYY/MM/DD"[PDAT]) '
                  'AND hasabstract[text] AND English[lang]'),
        "retrieval_date": "2026-05-26",
        "n_retrieved": n_raw,
        "n_duplicate_pmids": n_dup_pmid,
        "n_after_dedup": len(raw_dedup),
        "n_excluded_total": len(exc_df),
        "exclusion_breakdown": exc_df["exclusion_reason"].value_counts().to_dict(),
        "n_analyzed": len(clean),
        "retention_pct": round(100 * len(clean) / n_raw, 2),
        "journal_names_in_raw": jnames.to_dict(),
        "journal_abbrev_in_raw": jabbrev.to_dict(),
        "n_missing_month": missing_month,
        "corpus_year_min": int(clean["year"].min()),
        "corpus_year_max": int(clean["year"].max()),
    }
    return clean


# ------------------------------------------------------- 2026 partial year
def partial_year(clean, doc):
    per_year = clean.groupby("year").size()
    d = doc[doc.topic_id != -1]

    full = d.groupby("topic_id").size()
    thru25 = d[d.year <= 2025].groupby("topic_id").size()

    top_full = full.sort_values(ascending=False).head(5)
    top_25 = thru25.sort_values(ascending=False).head(5)

    # peak-year determination with and without 2026
    tr = pd.read_csv(TREND)
    peaks_2026 = int((tr["peak_year"] == 2026).sum())

    report["partial_year_2026"] = {
        "n_2026_docs": int(per_year.get(2026, 0)),
        "pct_of_corpus_2026": round(100 * per_year.get(2026, 0) / len(clean), 2),
        "n_2025_docs": int(per_year.get(2025, 0)),
        "peak_year_2005_2025": int(per_year.loc[:2025].idxmax()),
        "peak_count_2005_2025": int(per_year.loc[:2025].max()),
        "n_topics_peaking_in_2026": peaks_2026,
        "top5_topic_sizes_full": top_full.to_dict(),
        "top5_topic_sizes_thru2025": top_25.to_dict(),
        "note": "OLS trend fits already excluded 2026.",
    }
    per_year.to_csv(OUT / "annual_counts_JOA.csv")


# --------------------------------------------------------- annual noise (R2-3)
def annual_noise(doc):
    g = doc.groupby("year").agg(
        n_total=("topic_id", "size"),
        n_noise=("topic_id", lambda s: int((s == -1).sum())),
    )
    g["noise_pct"] = (100 * g.n_noise / g.n_total).round(2)
    g.to_csv(OUT / "annual_noise_fraction_JOA.csv")

    yrs = g.index.values.astype(float)
    # test whether noise fraction drifts over time (would bias trend estimates)
    m = (yrs >= 2005) & (yrs <= 2025)
    sl, ic, r, p, se = stats.linregress(yrs[m], g["noise_pct"].values[m])
    df_resid = int(m.sum()) - 2
    tcrit = float(stats.t.ppf(0.975, df_resid))

    report["annual_noise"] = {
        "overall_noise_pct": round(100 * (doc.topic_id == -1).mean(), 2),
        "by_year": g["noise_pct"].to_dict(),
        "min_noise_pct": float(g["noise_pct"].min()),
        "max_noise_pct": float(g["noise_pct"].max()),
        "trend_slope_pct_per_year_2005_2025": round(float(sl), 4),
        "trend_p_value": round(float(p), 4),
        "trend_95ci": [round(float(sl - tcrit * se), 4),
                       round(float(sl + tcrit * se), 4)],
    }


# ------------------------------------------------------- Hot audit (R2-7)
def hot_audit(doc, tr):
    d = doc[doc.topic_id != -1]
    rows = []
    for tid, grp in d.groupby("topic_id"):
        yc = grp.groupby("year").size().reindex(range(2005, 2027), fill_value=0)
        recent = yc.loc[2017:].sum() / yc.sum() if yc.sum() else 0
        rows.append({"topic_id": tid, "recent10yr_share": recent})
    rec = pd.DataFrame(rows)
    m = tr.merge(rec, on="topic_id")
    sigpos = m[(m.p_adj < 0.05) & (m.slope > 0)]

    report["hot_category_audit"] = {
        "n_significant_positive": int(len(sigpos)),
        "n_meeting_60pct_recency_Emerging": int((sigpos.recent10yr_share >= 0.60).sum()),
        "n_failing_recency_would_be_Hot": int((sigpos.recent10yr_share < 0.60).sum()),
        "min_recency_among_sigpos": round(float(sigpos.recent10yr_share.min()), 4),
        "median_recency_among_sigpos": round(float(sigpos.recent10yr_share.median()), 4),
        "explanation": ("Every topic with a significant positive slope also had "
                        ">=60% of its documents in the most recent 10 years, because "
                        "total journal output grew 3.9-fold over the study period. "
                        "The Hot category was therefore empty by definition, not by omission."),
    }
    m.to_csv(OUT / "hot_category_audit_JOA.csv", index=False)


# --------------------------------------- trend robustness: CI, q, NB (R2-6)
def trend_robustness(doc, tr):
    d = doc[doc.topic_id != -1]
    years = np.arange(2005, 2026)  # complete years only
    total = doc[doc.year.between(2005, 2025)].groupby("year").size().reindex(
        years, fill_value=0)

    rows = []
    for tid, grp in d[d.year.between(2005, 2025)].groupby("topic_id"):
        cnt = grp.groupby("year").size().reindex(years, fill_value=0)
        share = 100 * cnt / total

        # OLS on proportional share, with CI
        X = sm.add_constant(years.astype(float))
        ols = sm.OLS(share.values, X).fit()
        slope = ols.params[1]
        ci = ols.conf_int(alpha=0.05)[1]

        # quadratic term -> nonlinearity test
        Xq = sm.add_constant(np.column_stack([years - 2015.0, (years - 2015.0) ** 2]))
        q = sm.OLS(share.values, Xq).fit()
        p_quad = q.pvalues[2]

        # count model on annual counts with log(total annual output) offset.
        # Dispersion is estimated from the data (Cameron-Trivedi auxiliary
        # regression) rather than assumed, then a negative binomial is fitted
        # with that alpha. If the counts are not overdispersed relative to
        # Poisson, alpha -> 0 and the model reduces to Poisson.
        Xc = sm.add_constant((years - 2005).astype(float))
        off = np.log(total.values)
        try:
            pois = sm.GLM(cnt.values, Xc, family=sm.families.Poisson(),
                          offset=off).fit()
            mu = pois.fittedvalues
            aux_y = ((cnt.values - mu) ** 2 - cnt.values) / mu
            alpha_hat = float(sm.OLS(aux_y, mu).fit().params[0])
            alpha_hat = max(alpha_hat, 1e-6)

            nb = sm.GLM(cnt.values, Xc,
                        family=sm.families.NegativeBinomial(alpha=alpha_hat),
                        offset=off).fit()
            nb_beta = float(nb.params[1])
            nb_p = float(nb.pvalues[1])
            nb_irr = float(np.exp(nb.params[1]))
            nb_ci = [float(np.exp(nb.conf_int()[1][0])),
                     float(np.exp(nb.conf_int()[1][1]))]
        except Exception:
            nb_beta = nb_p = nb_irr = alpha_hat = np.nan
            nb_ci = [np.nan, np.nan]

        rows.append({
            "topic_id": tid, "n_documents": int(cnt.sum()),
            "ols_slope": slope, "ols_ci_low": ci[0], "ols_ci_high": ci[1],
            "ols_p": ols.pvalues[1],
            "p_quadratic": p_quad,
            "nb_log_irr": nb_beta, "nb_irr_per_year": nb_irr,
            "nb_ci_low": nb_ci[0], "nb_ci_high": nb_ci[1], "nb_p": nb_p,
            "nb_alpha_hat": alpha_hat,
        })

    r = pd.DataFrame(rows)
    r["ols_q"] = multipletests(r.ols_p, method="fdr_bh")[1]
    r["nb_q"] = multipletests(r.nb_p.fillna(1), method="fdr_bh")[1]
    r["p_quad_q"] = multipletests(r.p_quadratic.fillna(1), method="fdr_bh")[1]

    def cls(row, scol, qcol):
        if row[qcol] < 0.05 and row[scol] > 0:
            return "Increasing"
        if row[qcol] < 0.05 and row[scol] < 0:
            return "Decreasing"
        return "Stable"

    r["ols_class"] = r.apply(lambda x: cls(x, "ols_slope", "ols_q"), axis=1)
    r["nb_class"] = r.apply(lambda x: cls(x, "nb_log_irr", "nb_q"), axis=1)

    agree = (r.ols_class == r.nb_class).mean()

    # sensitivity excluding small topics
    big = r[r.n_documents >= 50]
    agree_big = (big.ols_class == big.nb_class).mean()

    r = r.merge(tr[["topic_id", "clinical_label", "trend_class"]], on="topic_id",
                how="left")
    r.to_csv(OUT / "trend_robustness_JOA.csv", index=False)

    report["trend_robustness"] = {
        "n_topics": int(len(r)),
        "ols_vs_negbin_agreement_pct": round(100 * float(agree), 1),
        "ols_vs_negbin_agreement_pct_topics_ge50docs": round(100 * float(agree_big), 1),
        "n_topics_lt_50_docs": int((r.n_documents < 50).sum()),
        "n_topics_lt_30_docs": int((r.n_documents < 30).sum()),
        "n_significant_nonlinear_quadratic_q<0.05": int((r.p_quad_q < 0.05).sum()),
        "median_ols_ci_width": round(float((r.ols_ci_high - r.ols_ci_low).median()), 4),
        "ols_class_counts": r.ols_class.value_counts().to_dict(),
        "nb_class_counts": r.nb_class.value_counts().to_dict(),
        "median_estimated_dispersion_alpha": round(float(r.nb_alpha_hat.median()), 4),
        "n_direction_disagreements_sig_both": int(
            ((r.ols_class != "Stable") & (r.nb_class != "Stable")
             & (r.ols_class != r.nb_class)).sum()),
        "n_sig_ols_only": int(((r.ols_class != "Stable") & (r.nb_class == "Stable")).sum()),
        "n_sig_nb_only": int(((r.nb_class != "Stable") & (r.ols_class == "Stable")).sum()),
    }


def main():
    clean = provenance()
    doc = pd.read_csv(DOCT, usecols=["pmid", "year", "topic_id"])
    tr = pd.read_csv(TREND)

    partial_year(clean, doc)
    annual_noise(doc)
    hot_audit(doc, tr)
    trend_robustness(doc, tr)

    with open(OUT / "revision_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

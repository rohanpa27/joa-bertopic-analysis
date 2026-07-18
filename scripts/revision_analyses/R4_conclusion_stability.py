"""
Revision analysis 4: (a) discriminative topic-similarity metrics for the
over-segmentation question (R2-5), and (b) conclusion-level stability of the
temporal findings across clustering configurations and seeds (R2-3).

Raw cosine similarity between BERTopic topic embeddings is uninformative
because transformer embeddings occupy a narrow cone (anisotropy): every pair
exceeds 0.99. We therefore recompute similarity on mean-centered embeddings
and on c-TF-IDF keyword profiles, and add top-keyword Jaccard overlap.

The scientifically relevant robustness question is not whether an identical
partition is recovered, but whether the same themes rise and fall. Part (b)
tests exactly that.

PREREQUISITE: run R2_similarity_and_params.py first. Part (b) reads the cluster
assignments it writes to 01_analyses/sensitivity_assignments.npz (an
intermediate that is not kept in the delivered package).

Run from project root with the JSES venv active.
Outputs -> Revisions/01_analyses/
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

OUT = Path("Revisions/01_analyses")
report = {}

# signature terms that identify a theme from a cluster's top keywords
THEMES = {
    "Periprosthetic joint infection": ["infection", "pji", "periprosthetic joint",
                                       "septic", "synovial", "debridement", "spacer"],
    "Patient-reported outcome measures": ["koos", "prom", "womac", "promis",
                                          "patient-reported", "mcid", "hoos"],
    "Tranexamic acid / blood management": ["tranexamic", "txa", "transfusion",
                                           "blood loss", "hemoglobin"],
    "Machine learning / AI": ["machine learning", "deep learning", "neural",
                              "artificial intelligence", "algorithm", "prediction model"],
    "Case reports of complications": ["case report", "rare", "unusual",
                                      "we report", "case of"],
    "Acetabular cup orientation": ["acetabular", "cup", "inclination",
                                   "anteversion", "component position"],
    "Polyethylene wear / osteolysis": ["polyethylene", "wear", "osteolysis",
                                       "liner", "crosslinked", "hxlpe"],
    "Ceramic bearings": ["ceramic", "alumina", "zirconia", "squeaking"],
}


# ------------------------------------------------ (a) discriminative similarity
def similarity_metrics():
    from safetensors.numpy import load_file

    te = load_file("outputs/models/bertopic_model_JOA/topic_embeddings.safetensors")
    emb = te[list(te.keys())[0]]
    tr = pd.read_csv("outputs/tables/temporal_trends_JOA.csv").sort_values("topic_id")
    if emb.shape[0] == len(tr) + 1:
        emb = emb[1:]
    labels = tr.clinical_label.tolist()
    tids = tr.topic_id.tolist()

    # raw cosine (anisotropic, shown for contrast)
    raw = cosine_similarity(emb)
    # mean-centered cosine (removes the shared component)
    cent = cosine_similarity(emb - emb.mean(axis=0, keepdims=True))

    # c-TF-IDF keyword-profile similarity, built from the documents themselves
    doc = pd.read_csv("outputs/tables/doc_topics_JOA.csv",
                      usecols=["topic_id", "document"])
    doc = doc[doc.topic_id != -1]
    joined = (doc.groupby("topic_id").document
              .apply(lambda s: " ".join(s.astype(str))).reindex(tids))
    cv = CountVectorizer(stop_words="english", min_df=1, max_features=50000)
    Xc = cv.fit_transform(joined.values)
    Xt = TfidfTransformer().fit_transform(Xc)
    ctf = cosine_similarity(Xt)

    # top-20 keyword Jaccard
    feat = np.array(cv.get_feature_names_out())
    tops = [set(feat[np.asarray(Xt[i].todense()).ravel().argsort()[::-1][:20]])
            for i in range(len(tids))]
    jac = np.zeros((len(tids), len(tids)))
    for i in range(len(tids)):
        for j in range(len(tids)):
            a, b = tops[i], tops[j]
            jac[i, j] = len(a & b) / len(a | b) if (a | b) else 0

    for m in (raw, cent, ctf, jac):
        np.fill_diagonal(m, np.nan)

    rows = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            rows.append({"topic_i": tids[i], "topic_j": tids[j],
                         "label_i": labels[i], "label_j": labels[j],
                         "cosine_raw": float(raw[i, j]),
                         "cosine_centered": float(cent[i, j]),
                         "cosine_ctfidf": float(ctf[i, j]),
                         "keyword_jaccard": float(jac[i, j])})
    p = pd.DataFrame(rows)
    p.sort_values("cosine_ctfidf", ascending=False).to_csv(
        OUT / "topic_similarity_metrics_JOA.csv", index=False)

    THEME_KEYS = {"PJI": "Periprosthetic Joint Infection",
                  "PROMs": "KOOS and Knee-Specific PROMs",
                  "Medicare": "Medicare Reimbursement",
                  "Survivorship": "Survivorship"}
    theme_rows = []
    for name, key in THEME_KEYS.items():
        idx = [k for k, l in enumerate(labels) if key.lower() in l.lower()]
        if len(idx) < 2:
            continue
        sub_c = cent[np.ix_(idx, idx)]
        sub_t = ctf[np.ix_(idx, idx)]
        sub_j = jac[np.ix_(idx, idx)]
        theme_rows.append({
            "theme": name, "n_clusters": len(idx),
            "max_cosine_centered": round(float(np.nanmax(sub_c)), 4),
            "max_cosine_ctfidf": round(float(np.nanmax(sub_t)), 4),
            "max_keyword_jaccard": round(float(np.nanmax(sub_j)), 4),
            "clusters": "; ".join(labels[k] for k in idx)})
    pd.DataFrame(theme_rows).to_csv(OUT / "theme_similarity_JOA.csv", index=False)

    report["similarity"] = {
        "raw_cosine_median": round(float(np.nanmedian(raw)), 4),
        "raw_cosine_max": round(float(np.nanmax(raw)), 4),
        "centered_cosine_median": round(float(np.nanmedian(cent)), 4),
        "centered_cosine_max": round(float(np.nanmax(cent)), 4),
        "ctfidf_cosine_median": round(float(np.nanmedian(ctf)), 4),
        "ctfidf_cosine_max": round(float(np.nanmax(ctf)), 4),
        "keyword_jaccard_median": round(float(np.nanmedian(jac)), 4),
        "keyword_jaccard_max": round(float(np.nanmax(jac)), 4),
        "n_pairs_ctfidf_gt_0.80": int((p.cosine_ctfidf > 0.80).sum()),
        "n_pairs_jaccard_ge_0.50": int((p.keyword_jaccard >= 0.50).sum()),
        "note": ("Raw topic-embedding cosine is uninformative (all pairs >0.99) "
                 "because transformer embeddings are anisotropic; centered and "
                 "c-TF-IDF metrics are discriminative and are reported instead."),
        "flagged_themes": theme_rows,
        "top10_by_ctfidf": p.nlargest(10, "cosine_ctfidf")[
            ["label_i", "label_j", "cosine_ctfidf", "keyword_jaccard"]
        ].to_dict("records"),
    }


# --------------------------------------- (b) conclusion stability across configs
def _theme_of(terms):
    hits = []
    blob = " ".join(terms)
    for theme, sigs in THEMES.items():
        if sum(s in blob for s in sigs) >= 2:
            hits.append(theme)
    return hits


def conclusion_stability():
    corpus = pd.read_csv("data/processed/corpus_clean_JOA.csv",
                         usecols=["year", "document"])
    z = np.load(OUT / "sensitivity_assignments.npz")
    years = np.arange(2005, 2026)

    out = []
    for cfg in z.files:
        lab = z[cfg]
        if len(set(lab) - {-1}) < 5:
            continue  # degenerate partition
        df = corpus.copy()
        df["cl"] = lab
        d = df[(df.cl != -1) & (df.year.between(2005, 2025))]

        joined = d.groupby("cl").document.apply(lambda s: " ".join(s.astype(str)))
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2),
                             min_df=1, max_features=60000)
        X = cv.fit_transform(joined.values)
        Xt = TfidfTransformer().fit_transform(X)
        feat = np.array(cv.get_feature_names_out())

        cl_theme = {}
        for i, cl in enumerate(joined.index):
            top = feat[np.asarray(Xt[i].todense()).ravel().argsort()[::-1][:25]]
            for th in _theme_of(list(top)):
                cl_theme.setdefault(th, []).append(cl)

        total = df[df.year.between(2005, 2025)].groupby("year").size().reindex(
            years, fill_value=0)
        for th, cls in cl_theme.items():
            cnt = (d[d.cl.isin(cls)].groupby("year").size()
                   .reindex(years, fill_value=0))
            share = 100 * cnt / total
            sl, ic, r, p, se = stats.linregress(years, share.values)
            out.append({"config": cfg, "theme": th, "n_clusters": len(cls),
                        "n_documents": int(cnt.sum()),
                        "slope_pct_per_year": round(float(sl), 4),
                        "p_value": float(f"{p:.3g}"),
                        "direction": ("increase" if sl > 0 and p < 0.05
                                      else "decrease" if sl < 0 and p < 0.05
                                      else "no significant change")})
    s = pd.DataFrame(out)
    s.to_csv(OUT / "conclusion_stability_JOA.csv", index=False)

    summary = []
    for th, g in s.groupby("theme"):
        vc = g.direction.value_counts()
        top = vc.index[0]
        summary.append({"theme": th, "n_configs": int(len(g)),
                        "modal_direction": top,
                        "pct_configs_agreeing": round(100 * float(vc.iloc[0] / len(g)), 1),
                        "median_slope": round(float(g.slope_pct_per_year.median()), 4)})
    sm = pd.DataFrame(summary).sort_values("median_slope", ascending=False)
    sm.to_csv(OUT / "conclusion_stability_summary_JOA.csv", index=False)
    report["conclusion_stability"] = sm.to_dict("records")


def main():
    print("similarity metrics ...")
    similarity_metrics()
    print("conclusion stability ...")
    conclusion_stability()
    with open(OUT / "conclusion_stability_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

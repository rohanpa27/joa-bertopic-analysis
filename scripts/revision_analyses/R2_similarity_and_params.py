"""
Revision analysis 2: topic similarity / over-segmentation (R2-5),
clustering parameter + random-seed sensitivity (R2-3, R1-15),
and silhouette recomputation in the clustering space.

Run from project root with the JSES venv active.
Outputs -> Revisions/01_analyses/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

OUT = Path("Revisions/01_analyses")
OUT.mkdir(parents=True, exist_ok=True)

EMB = "data/embeddings/embeddings_pubmedbert_JOA.npy"
DOCT = "outputs/tables/doc_topics_JOA.csv"
TREND = "outputs/tables/temporal_trends_JOA.csv"

report = {}


# ------------------------------------------- topic similarity / merging (R2-5)
def topic_similarity():
    from safetensors.numpy import load_file

    te = load_file("outputs/models/bertopic_model_JOA/topic_embeddings.safetensors")
    emb = te[list(te.keys())[0]]
    tr = pd.read_csv(TREND).sort_values("topic_id")

    # BERTopic stores outlier topic -1 first when present
    n_topics = len(tr)
    if emb.shape[0] == n_topics + 1:
        emb = emb[1:]
    labels = tr["clinical_label"].tolist()
    tids = tr["topic_id"].tolist()

    sim = cosine_similarity(emb)
    np.fill_diagonal(sim, np.nan)

    pairs = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            pairs.append({"topic_i": tids[i], "topic_j": tids[j],
                          "label_i": labels[i], "label_j": labels[j],
                          "cosine_similarity": float(sim[i, j])})
    p = pd.DataFrame(pairs).sort_values("cosine_similarity", ascending=False)
    p.to_csv(OUT / "topic_pairwise_similarity_JOA.csv", index=False)

    # themes the reviewer flagged as potentially over-segmented
    THEMES = {
        "PJI": "Periprosthetic Joint Infection",
        "PROMs": "PROM",
        "Medicare": "Medicare",
        "Blood management": "Blood Management",
        "Survivorship": "Survivorship",
    }
    theme_rows = []
    for name, key in THEMES.items():
        idx = [k for k, lab in enumerate(labels) if key.lower() in lab.lower()]
        if len(idx) < 2:
            theme_rows.append({"theme": name, "n_clusters": len(idx),
                               "max_pairwise_cosine": None,
                               "mean_pairwise_cosine": None,
                               "clusters": [labels[k] for k in idx]})
            continue
        sub = sim[np.ix_(idx, idx)]
        theme_rows.append({
            "theme": name, "n_clusters": len(idx),
            "max_pairwise_cosine": round(float(np.nanmax(sub)), 4),
            "mean_pairwise_cosine": round(float(np.nanmean(sub)), 4),
            "clusters": [labels[k] for k in idx],
        })
    pd.DataFrame(theme_rows).to_csv(OUT / "theme_oversegmentation_JOA.csv", index=False)

    report["topic_similarity"] = {
        "n_topics": n_topics,
        "max_pairwise_cosine_overall": round(float(p.cosine_similarity.max()), 4),
        "median_pairwise_cosine_overall": round(float(p.cosine_similarity.median()), 4),
        "n_pairs_cosine_gt_0.90": int((p.cosine_similarity > 0.90).sum()),
        "n_pairs_cosine_gt_0.95": int((p.cosine_similarity > 0.95).sum()),
        "top10_most_similar_pairs": p.head(10)[
            ["label_i", "label_j", "cosine_similarity"]].to_dict("records"),
        "flagged_themes": theme_rows,
    }


# ----------------------------------- parameter + seed sensitivity (R2-3, R1-15)
def param_sensitivity():
    import hdbscan
    from umap import UMAP

    X = np.load(EMB)
    base = pd.read_csv(DOCT, usecols=["topic_id"])["topic_id"].values

    CONFIGS = [
        # (n_neighbors, min_cluster_size, seed, note)
        (10, 20, 42, "published configuration"),
        (10, 20, 7, "published configuration, alternate seed"),
        (10, 20, 2024, "published configuration, alternate seed"),
        (5, 20, 42, "lower n_neighbors (noise-minimizing)"),
        (5, 15, 42, "lower n_neighbors, smaller clusters"),
        (5, 30, 42, "lower n_neighbors, larger clusters"),
        (15, 20, 42, "higher n_neighbors"),
        (10, 15, 42, "smaller minimum cluster size"),
        (10, 30, 42, "larger minimum cluster size"),
    ]

    rows = []
    assignments = {}
    for nn, mcs, seed, note in CONFIGS:
        um = UMAP(n_neighbors=nn, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=seed)
        Z = um.fit_transform(X)
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5,
                             metric="euclidean",
                             cluster_selection_method="eom")
        lab = cl.fit_predict(Z)

        noise = float((lab == -1).mean() * 100)
        ntop = int(len(set(lab)) - (1 if -1 in lab else 0))
        mask = lab != -1
        sil = float(silhouette_score(Z[mask], lab[mask])) if mask.sum() > 10 else np.nan
        ari_vs_base = float(adjusted_rand_score(base, lab))

        key = f"nn{nn}_mcs{mcs}_seed{seed}"
        assignments[key] = lab
        rows.append({"config": key, "n_neighbors": nn, "min_cluster_size": mcs,
                     "seed": seed, "note": note, "n_topics": ntop,
                     "noise_pct": round(noise, 2),
                     "silhouette_in_umap_space": round(sil, 4),
                     "ARI_vs_published": round(ari_vs_base, 4)})
        print(f"  {key}: topics={ntop} noise={noise:.1f}% ARI={ari_vs_base:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "parameter_sensitivity_JOA.csv", index=False)
    np.savez_compressed(OUT / "sensitivity_assignments.npz", **assignments)

    seeds = df[df.config.str.startswith("nn10_mcs20")]
    report["parameter_sensitivity"] = {
        "configurations_tested": len(df),
        "noise_range_pct": [float(df.noise_pct.min()), float(df.noise_pct.max())],
        "n_topics_range": [int(df.n_topics.min()), int(df.n_topics.max())],
        "seed_stability_ARI_mean": round(float(seeds.ARI_vs_published.mean()), 4),
        "seed_stability_ARI_min": round(float(seeds.ARI_vs_published.min()), 4),
        "lowest_noise_config": df.loc[df.noise_pct.idxmin()].to_dict(),
        "table": df.to_dict("records"),
    }


# ---------------------- silhouette in the space where clustering happened
def silhouette_note():
    """The published silhouette was computed on the 768-d embeddings; HDBSCAN
    operates on the 5-d UMAP projection. Report both so the negative value is
    interpretable (R2-3)."""
    X = np.load(EMB)
    doc = pd.read_csv(DOCT, usecols=["topic_id"])["topic_id"].values
    mask = doc != -1
    sil_768 = float(silhouette_score(X[mask], doc[mask], metric="cosine"))
    report["silhouette"] = {
        "silhouette_768d_cosine_published": round(sil_768, 4),
        "interpretation": ("Silhouette assumes globular, equidistant clusters and is "
                           "computed here in the 768-dimensional embedding space, "
                           "whereas HDBSCAN defines clusters by density in the "
                           "5-dimensional UMAP projection. Near-zero or slightly "
                           "negative silhouette is therefore expected and does not "
                           "by itself indicate invalid clusters."),
    }


def main():
    print("topic similarity ...")
    topic_similarity()
    print("silhouette ...")
    silhouette_note()
    print("parameter sensitivity (slow) ...")
    param_sensitivity()

    with open(OUT / "similarity_and_params_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main()

"""
Revision analysis 6: renumber figures into citation order (Reviewer 1, comment 14)
and build the supplementary figures requested by Reviewer 2 (annual noise
fraction, comment 3; OLS vs negative binomial concordance, comment 6).

Figure renumbering, driven by order of first citation in the revised text:
    NEW Figure 1  methods / study flow          (new this revision)
    NEW Figure 2  publication trends, 10 largest topics   (was Figure 1)
    NEW Figure 3  document count per topic                (was Figure 3)
    NEW Figure 4  temporal trend slopes, all 83 topics    (was Figure 4)
    NEW Figure 5  temporal heatmap                        (was Figure 2)
    NEW Figure 6  clinical research domains               (was Figure 5)

Output -> Revisions/02_figures/
"""

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SRC = Path("Submission JOA")
OUT = Path("Revisions/02_figures")
OUT.mkdir(parents=True, exist_ok=True)

RENUMBER = [
    ("fig1_topics_over_time_JOA.png", "fig2_topics_over_time_JOA.png"),
    ("fig3_topic_sizes_JOA.png",      "fig3_topic_sizes_JOA.png"),
    ("fig4_trend_summary_JOA.png",    "fig4_trend_summary_JOA.png"),
    ("fig2_temporal_heatmap_JOA.png", "fig5_temporal_heatmap_JOA.png"),
    ("fig5_clinical_domains_JOA.png", "fig6_clinical_domains_JOA.png"),
]

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
INK, GRID, SPINE = "#111111", "#EBEBEB", "#B8B8B8"
DPI = 150


def renumber():
    for old, new in RENUMBER:
        s, d = SRC / old, OUT / new
        if s.exists():
            shutil.copy2(s, d)
            print(f"  {old}  ->  {new}")
        else:
            print(f"  MISSING: {s}")


def supp_noise():
    """Supplementary Figure S1: annual unclustered (noise) fraction."""
    g = pd.read_csv("Revisions/01_analyses/annual_noise_fraction_JOA.csv")
    g = g[g.year <= 2025]

    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=DPI)
    ax.plot(g.year, g.noise_pct, marker="o", markersize=5, linewidth=2.0,
            color="#4878D0", zorder=3)
    mean_noise = g.noise_pct.mean()
    ax.axhline(mean_noise, linestyle="--", linewidth=1.2,
               color="#8C8C8C", zorder=2)
    ax.annotate(f"Mean {mean_noise:.1f}%",
                xy=(2019, mean_noise), xytext=(2015.5, 54),
                fontsize=9, color="#555555", ha="center",
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", linewidth=0.9))

    ax.set_xlabel("Year", fontsize=10, color="#333333")
    ax.set_ylabel("Unclustered documents (%)", fontsize=10, color="#333333")
    ax.set_title("Annual proportion of documents assigned to the "
                 "unclustered category",
                 fontsize=10.5, fontweight="bold", color=INK, pad=8)
    ax.set_ylim(0, 60)
    ax.set_xticks(range(2005, 2026, 5))
    ax.tick_params(labelsize=9, colors="#333333")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    fig.savefig(OUT / "figS1_annual_noise_JOA.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figS1_annual_noise_JOA.png")


def supp_concordance():
    """Supplementary Figure S2: OLS slope vs negative binomial rate ratio."""
    r = pd.read_csv("Revisions/01_analyses/trend_robustness_JOA.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=DPI)

    colors = {"Increasing": "#2E7D46", "Stable": "#4878D0", "Decreasing": "#9E4A4A"}
    for cls, sub in r.groupby("ols_class"):
        ax.scatter(sub.ols_slope, sub.nb_log_irr, s=34, alpha=0.85,
                   color=colors.get(cls, "#777777"), label=cls,
                   edgecolor="white", linewidth=0.8, zorder=3)

    ax.axhline(0, color="#B0B0B0", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#B0B0B0", linewidth=1.0, zorder=1)
    ax.set_xlabel("OLS slope, annual change in publication share (% per year)",
                  fontsize=9.5, color="#333333")
    ax.set_ylabel("Negative binomial log rate ratio per year",
                  fontsize=9.5, color="#333333")
    ax.set_title("Concordance of linear and count-based trend models "
                 "across 83 topics",
                 fontsize=10.5, fontweight="bold", color=INK, pad=8)
    ax.tick_params(labelsize=9, colors="#333333")
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    ax.legend(title="OLS classification", fontsize=8.5, title_fontsize=9,
              frameon=True, edgecolor=SPINE, loc="upper left")
    fig.savefig(OUT / "figS2_model_concordance_JOA.png", dpi=DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figS2_model_concordance_JOA.png")


if __name__ == "__main__":
    print("renumbering figures:")
    renumber()
    print("supplementary figures:")
    supp_noise()
    supp_concordance()
    print("\nfinal contents:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}  ({p.stat().st_size/1e6:.1f} MB)")

"""
Journal of Arthroplasty BERTopic -- Enhanced Publication Figures (manuscript V2)
================================================================================
Regenerates Figures 1-4 with publication polish (300 DPI print resolution,
refined typography, cleaner gridlines / spines / colorbars) and adds a NEW
Figure 5: the evolution of eight clinical research domains over time.

Figures 1-4 keep the same layout and aspect as the prior version so they drop
into the existing manuscript at their current display widths. Only the visual
quality is improved; the underlying data and meaning are unchanged.

Output folder: outputs/figures_enhanced_v2/
No em dashes in any outputs.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from pathlib import Path
import json, warnings, shutil
warnings.filterwarnings("ignore")

BASE    = Path("/Users/rohan/Documents/BERTopic Studies/JOA BERT")
TBL_DIR = BASE / "outputs/tables"
FIG_DIR = BASE / "outputs/figures_enhanced_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

JOURNAL   = "JOA"
FULL_NAME = "Journal of Arthroplasty"
YEARS     = "2005-2026"

sns.set_theme(style="whitegrid", font="Arial")
plt.rcParams.update({
    "svg.fonttype": "none",
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#BBBBBB",
    "figure.facecolor": "white",
})

TREND_COLORS = {
    "Emerging": "#2CA02C",
    "Hot":      "#D62728",
    "Stable":   "#4878D0",
    "Cold":     "#7F7F7F",
}
SPINE_COLOR = "#BBBBBB"

FS_LABEL  = 17
FS_AXIS   = 19
FS_TITLE  = 21
FS_LEGEND = 15
FS_ANNOT  = 14

PAGE_W = 6.5
DPI    = 300

FIGURE_HEIGHTS = {}

def save_fig(fig, name, height_in, dpi=DPI, pad_inches=0.1):
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
                facecolor="white", edgecolor="none")
    FIGURE_HEIGHTS[name] = height_in
    plt.close(fig)
    kb = path.stat().st_size // 1024
    print(f"  {name}  ({kb} KB)  ->  {PAGE_W:.1f} x {height_in:.1f} in")

def short_label(label, max_words=6):
    words = str(label).split()
    return label if len(words) <= max_words else " ".join(words[:max_words]) + "..."

# ---- Load data ----------------------------------------------------------------
trends     = pd.read_csv(TBL_DIR / f"temporal_trends_{JOURNAL}.csv")
topic_info = pd.read_csv(TBL_DIR / f"topic_info_{JOURNAL}.csv")
ot         = pd.read_csv(TBL_DIR / f"topics_over_time_{JOURNAL}.csv")
doc_topics = pd.read_csv(TBL_DIR / f"doc_topics_{JOURNAL}.csv")

ti = topic_info[topic_info["Topic"] != -1].copy()

label_col = "clinical_label" if "clinical_label" in trends.columns else "topic_name"
label_map = dict(zip(trends["topic_id"], trends[label_col]))
ti["label"] = ti["Topic"].map(label_map)
ti = ti.sort_values("Count", ascending=False).reset_index(drop=True)

n_topics = len(trends)
n_total  = int(trends["n_documents"].sum())

print(f"Generating ENHANCED figures for {FULL_NAME} ({YEARS})\n"
      f"  {n_topics} topics  |  {n_total} non-noise docs\n")


# ==============================================================================
# FIG 1 -- Topics Over Time (top 10 largest topics, annual % share)
# ==============================================================================
print("Fig 1: Topics Over Time")
ot_filt = ot[ot["Topic"] != -1].copy()
ot_filt["year"] = ot_filt["Timestamp"].round().astype(int)
ot_filt["label"] = ot_filt["Topic"].map(label_map)
annual = doc_topics[doc_topics["topic_id"] != -1].groupby("year").size().rename("total")
ot_filt = ot_filt.merge(annual.reset_index(), on="year", how="left")
ot_filt["pct"] = ot_filt["Frequency"] / ot_filt["total"] * 100
top10_ids = ot_filt.groupby("Topic")["Frequency"].sum().nlargest(10).index.tolist()
ot_top    = ot_filt[ot_filt["Topic"].isin(top10_ids)].copy()
palette10 = sns.color_palette("tab10", 10)

H1 = 7.5
fig, ax = plt.subplots(figsize=(PAGE_W, H1))
handles, labels_leg = [], []
for i, tid in enumerate(top10_ids):
    sub    = ot_top[ot_top["Topic"] == tid].sort_values("year")
    clabel = short_label(label_map.get(tid, f"Topic {tid}"), 5)
    trend  = (trends.loc[trends["topic_id"] == tid, "trend_class"].values[0]
              if tid in trends["topic_id"].values else "Stable")
    line,  = ax.plot(sub["year"], sub["pct"],
                     color=palette10[i], linewidth=2.2,
                     linestyle="--" if trend == "Cold" else "-",
                     alpha=0.92, marker="o", markersize=3.8,
                     markeredgecolor="white", markeredgewidth=0.5)
    handles.append(line)
    labels_leg.append(clabel)

ax.set_title(
    f"Figure 1. Publication Trends for the 10 Largest Research Topics\n"
    f"{FULL_NAME} ({YEARS}) -- Annual publication share (%) | Dashed = Cold (declining)",
    fontsize=FS_TITLE * 0.75, fontweight="bold", pad=8, color="#1A1A1A")
ax.set_xlabel("Year", fontsize=FS_AXIS * 0.8, color="#333333", labelpad=4)
ax.set_ylabel("Annual Publication Share (%)", fontsize=FS_AXIS * 0.8, color="#333333", labelpad=4)
ax.tick_params(labelsize=FS_LABEL * 0.75, colors="#333333")
years_present = sorted(ot_top["year"].unique())
year_ticks = [y for y in years_present if y % 5 == 0]
ax.set_xticks(year_ticks)
ax.set_xticklabels([str(y) for y in year_ticks])
ax.set_ylim(bottom=0)
ax.grid(True, color="#EBEBEB", linewidth=0.6)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
ax.legend(handles, labels_leg,
          loc="upper center", bbox_to_anchor=(0.5, -0.13),
          fontsize=FS_LEGEND * 0.75, ncol=3,
          framealpha=0.95, edgecolor=SPINE_COLOR,
          handlelength=2.5, columnspacing=1.0)
fig.subplots_adjust(bottom=0.28, top=0.88, left=0.11, right=0.97)
save_fig(fig, f"fig1_topics_over_time_{JOURNAL}.png", H1)


# ==============================================================================
# FIG 2 -- Temporal Heatmap (all topics x 5 time-period bins)
# ==============================================================================
print("Fig 2: Temporal Heatmap")
bins_list  = [2004, 2008, 2013, 2018, 2022, 2026]
labels_bin = ["2005-2008", "2009-2013", "2014-2018", "2019-2022", "2023-2026"]
doc_topics["period"] = pd.cut(doc_topics["year"], bins=bins_list, labels=labels_bin)
dn    = doc_topics[doc_topics["topic_id"] != -1].copy()
pivot = dn.groupby(["topic_id", "period"]).size().unstack(fill_value=0)
pivot.columns = pivot.columns.astype(str)
pivot_norm = pivot.div(pivot.sum(axis=1), axis=0) * 100

order_cats = ["Hot", "Emerging", "Stable", "Cold"]
pivot_norm["trend_class"] = pivot_norm.index.map(
    lambda t: trends.loc[trends["topic_id"] == t, "trend_class"].values[0]
    if t in trends["topic_id"].values else "Stable")
pivot_norm["slope"] = pivot_norm.index.map(
    lambda t: trends.loc[trends["topic_id"] == t, "slope"].values[0]
    if t in trends["topic_id"].values else 0.0)
pivot_norm["trend_order"] = pivot_norm["trend_class"].map(
    {t: i for i, t in enumerate(order_cats)})
pivot_norm = pivot_norm.sort_values(["trend_order", "slope"], ascending=[True, False])
heat_cols  = [c for c in labels_bin if c in pivot_norm.columns]
heat_data  = pivot_norm[heat_cols].copy()
heat_data.index = heat_data.index.map(
    lambda t: short_label(label_map.get(t, f"Topic {t}"), 5))

N_ITEMS = len(heat_data)

W2     = 14.0
ROW_H  = 0.45
MARGIN = 5.0
H2     = N_ITEMS * ROW_H + MARGIN

FS_Y  = 13
FS_X  = 17
FS_CB = 14

fig, ax = plt.subplots(figsize=(W2, H2))

hm = sns.heatmap(
    heat_data, ax=ax,
    cmap="YlOrRd",
    linewidths=0.4,
    linecolor="#EBEBEB",
    vmin=0, vmax=100,
    cbar_kws={
        "shrink": 0.30,
        "pad":    0.015,
        "aspect": 25,
        "ticks":  [0, 25, 50, 75, 100],
    },
    xticklabels=True, yticklabels=True,
    annot=False,
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45, ha="right", rotation_mode="anchor",
    fontsize=FS_X, color="#111111", fontweight="bold",
)
ax.tick_params(axis="x", length=4, width=1, pad=5)
ax.xaxis.set_tick_params(which="both", bottom=True)

ax.set_yticklabels(ax.get_yticklabels(), fontsize=FS_Y, color="#222222")
ax.tick_params(axis="y", length=0, pad=4)

try:
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS_CB)
    cbar.set_label("Within-Topic\nPublication Share (%)",
                   size=FS_CB, labelpad=8, multialignment="center")
    cbar.outline.set_edgecolor("#BBBBBB")
    cbar.outline.set_linewidth(0.6)
except Exception:
    pass

strip_w  = 0.022
bar_ax   = ax.inset_axes([-strip_w - 0.004, 0, strip_w, 1])
bar_ax.set_xlim(0, 1)
bar_ax.set_ylim(0, N_ITEMS)
bar_ax.axis("off")

pivot_trend = pivot_norm["trend_class"].values
for k, tc in enumerate(reversed(pivot_trend)):
    bar_ax.barh(k + 0.5, 1, height=0.92,
                color=TREND_COLORS.get(tc, "#4878D0"),
                edgecolor="white", linewidth=0.4)

group_sizes = pivot_norm["trend_class"].value_counts(sort=False).reindex(
    order_cats, fill_value=0)
cumulative = 0
for grp in order_cats:
    sz = group_sizes.get(grp, 0)
    cumulative += sz
    if 0 < cumulative < N_ITEMS:
        ax.axhline(y=cumulative, color="#555555", linewidth=1.2, linestyle="--", alpha=0.5)

active_trends = [t for t in order_cats if t != "Hot"]
patches = [mpatches.Patch(facecolor=TREND_COLORS[t], label=t, linewidth=0)
           for t in active_trends]
ax.legend(
    handles=patches,
    fontsize=FS_X * 0.85,
    title="Temporal Trend Classification",
    title_fontsize=FS_X * 0.85,
    framealpha=0.95, edgecolor="#CCCCCC",
    bbox_to_anchor=(0.5, -0.072), loc="upper center",
    ncol=3, handlelength=1.4, handletextpad=0.6, columnspacing=1.2,
)

ax.set_xlabel("Publication Period", fontsize=FS_X * 1.05,
              color="#111111", fontweight="bold", labelpad=14)
ax.set_ylabel("", labelpad=0)
ax.set_title(
    f"Figure 2. Temporal Evolution of Research Topics -- {FULL_NAME} ({YEARS})\n"
    "Within-topic publication share (%) by period | Ordered by trend class and regression slope",
    fontsize=FS_TITLE * 0.78, fontweight="bold", pad=14, color="#1A1A1A")

fig.subplots_adjust(bottom=0.06, top=0.97, left=0.25, right=0.93)
save_fig(fig, f"fig2_temporal_heatmap_{JOURNAL}.png", H2)


# ==============================================================================
# FIG 3 -- Topic Sizes (document count per topic, colored by trend)
# ==============================================================================
print("Fig 3: Topic Sizes")
t3_data = trends.copy().sort_values("n_documents", ascending=True)
colors3 = [TREND_COLORS.get(c, "#4878D0") for c in t3_data["trend_class"]]
labels3 = [short_label(l, 6) for l in t3_data[label_col]]

N3 = len(t3_data)
H3 = N3 * 0.38 + 2.0
fig, ax = plt.subplots(figsize=(PAGE_W, H3))
ax.barh(range(N3), t3_data["n_documents"], color=colors3,
        edgecolor="white", height=0.78)
ax.set_yticks(range(N3))
ax.set_yticklabels(labels3, fontsize=FS_LABEL)
for i, v in enumerate(t3_data["n_documents"]):
    ax.text(v + 0.5, i, str(v), va="center", fontsize=FS_ANNOT, color="#333333")
patches = [mpatches.Patch(facecolor=c, label=t) for t, c in TREND_COLORS.items()]
ax.legend(handles=patches, fontsize=FS_LEGEND, title="Research Trend",
          title_fontsize=FS_LEGEND, framealpha=0.95, edgecolor=SPINE_COLOR,
          loc="lower right")
ax.set_title(
    f"Figure 3. Document Count per Research Topic -- {FULL_NAME} ({YEARS})\n"
    f"All {n_topics} BERTopic clusters, colored by temporal trend classification",
    fontsize=FS_TITLE, fontweight="bold", pad=12, color="#1A1A1A")
ax.set_xlabel("Number of Abstracts", fontsize=FS_AXIS, color="#333333")
ax.tick_params(axis="x", labelsize=FS_LABEL, colors="#333333")
ax.grid(axis="x", color="#EBEBEB", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
ax.set_xlim(0, t3_data["n_documents"].max() * 1.16)
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
fig.tight_layout()
save_fig(fig, f"fig3_topic_sizes_{JOURNAL}.png", H3)


# ==============================================================================
# FIG 4 -- Trend Slopes (OLS slope per topic, FDR asterisks)
# ==============================================================================
print("Fig 4: Trend Summary")
t4      = trends.copy().sort_values("slope", ascending=True).reset_index(drop=True)
colors4 = [TREND_COLORS.get(c, "#4878D0") for c in t4["trend_class"]]
labels4 = [short_label(l, 6) for l in t4[label_col]]

N4 = len(t4)
H4 = N4 * 0.38 + 2.0
fig, ax = plt.subplots(figsize=(PAGE_W, H4))
ax.barh(range(N4), t4["slope"], color=colors4, edgecolor="white", height=0.78)
ax.axvline(0, color="#333333", linewidth=1.1)
ax.set_yticks(range(N4))
ax.set_yticklabels(labels4, fontsize=FS_LABEL)
for i, (_, row) in enumerate(t4.iterrows()):
    if row["p_adj"] < 0.05:
        xv  = row["slope"]
        off = 0.0015 if xv >= 0 else -0.0015
        ax.text(xv + off, i, "*", va="center",
                ha="left" if xv >= 0 else "right",
                fontsize=FS_LABEL, color="#333333", fontweight="bold")
patches = [mpatches.Patch(facecolor=c, label=t) for t, c in TREND_COLORS.items()]
ax.legend(handles=patches, fontsize=FS_LEGEND, title="Research Trend",
          title_fontsize=FS_LEGEND, framealpha=0.95, edgecolor=SPINE_COLOR,
          loc="lower right")
ax.set_title(
    f"Figure 4. Temporal Trend Slopes for All Research Topics -- {FULL_NAME} ({YEARS})\n"
    "OLS regression slope (annual change in % publication share) | * = FDR-corrected p < 0.05",
    fontsize=FS_TITLE, fontweight="bold", pad=12, color="#1A1A1A")
ax.set_xlabel("Regression Slope (delta % per year)", fontsize=FS_AXIS, color="#333333")
ax.tick_params(axis="x", labelsize=FS_LABEL, colors="#333333")
ax.grid(axis="x", color="#EBEBEB", linewidth=0.6)
ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)
fig.tight_layout()
save_fig(fig, f"fig4_trend_summary_{JOURNAL}.png", H4)


# ==============================================================================
# FIG 5 -- Evolution of Clinical Research Domains (NEW)
# ==============================================================================
# Each of the 83 topics is mapped to one of eight clinically meaningful
# arthroplasty research domains. The figure shows how the composition of the
# clustered literature (share of clustered abstracts) has shifted across the
# eight domains over the 2005-2026 study period, providing a clinically
# organized view of where the field's research effort has moved.
print("Fig 5: Clinical Research Domains Over Time")

# ---- Topic -> clinical domain map (all 83 topics; see manuscript caption) ----
DOMAIN_MAP = {
    # Periprosthetic Joint Infection
    3: "Periprosthetic Joint Infection", 5: "Periprosthetic Joint Infection",
    9: "Periprosthetic Joint Infection", 38: "Periprosthetic Joint Infection",
    52: "Periprosthetic Joint Infection", 71: "Periprosthetic Joint Infection",
    # Implant Survivorship & Bearing Surfaces
    10: "Implant Survivorship & Bearing Surfaces", 12: "Implant Survivorship & Bearing Surfaces",
    14: "Implant Survivorship & Bearing Surfaces", 16: "Implant Survivorship & Bearing Surfaces",
    17: "Implant Survivorship & Bearing Surfaces", 18: "Implant Survivorship & Bearing Surfaces",
    21: "Implant Survivorship & Bearing Surfaces", 24: "Implant Survivorship & Bearing Surfaces",
    25: "Implant Survivorship & Bearing Surfaces", 26: "Implant Survivorship & Bearing Surfaces",
    33: "Implant Survivorship & Bearing Surfaces", 35: "Implant Survivorship & Bearing Surfaces",
    46: "Implant Survivorship & Bearing Surfaces", 49: "Implant Survivorship & Bearing Surfaces",
    55: "Implant Survivorship & Bearing Surfaces", 57: "Implant Survivorship & Bearing Surfaces",
    70: "Implant Survivorship & Bearing Surfaces",
    # Patient-Reported Outcomes & Satisfaction
    4: "Patient-Reported Outcomes & Satisfaction", 40: "Patient-Reported Outcomes & Satisfaction",
    58: "Patient-Reported Outcomes & Satisfaction", 79: "Patient-Reported Outcomes & Satisfaction",
    81: "Patient-Reported Outcomes & Satisfaction", 20: "Patient-Reported Outcomes & Satisfaction",
    # Health Policy, Value-Based & Care Delivery
    15: "Health Policy, Value & Care Delivery", 23: "Health Policy, Value & Care Delivery",
    27: "Health Policy, Value & Care Delivery", 28: "Health Policy, Value & Care Delivery",
    32: "Health Policy, Value & Care Delivery", 47: "Health Policy, Value & Care Delivery",
    62: "Health Policy, Value & Care Delivery", 64: "Health Policy, Value & Care Delivery",
    65: "Health Policy, Value & Care Delivery", 68: "Health Policy, Value & Care Delivery",
    74: "Health Policy, Value & Care Delivery",
    # Perioperative & Medical Management
    0: "Perioperative & Medical Management", 11: "Perioperative & Medical Management",
    13: "Perioperative & Medical Management", 34: "Perioperative & Medical Management",
    36: "Perioperative & Medical Management", 44: "Perioperative & Medical Management",
    54: "Perioperative & Medical Management", 60: "Perioperative & Medical Management",
    66: "Perioperative & Medical Management", 67: "Perioperative & Medical Management",
    76: "Perioperative & Medical Management", 78: "Perioperative & Medical Management",
    80: "Perioperative & Medical Management",
    # Hip Arthroplasty (Technique & Anatomy)
    1: "Hip Arthroplasty", 2: "Hip Arthroplasty", 31: "Hip Arthroplasty",
    37: "Hip Arthroplasty", 39: "Hip Arthroplasty", 45: "Hip Arthroplasty",
    50: "Hip Arthroplasty", 53: "Hip Arthroplasty", 59: "Hip Arthroplasty",
    69: "Hip Arthroplasty",
    # Knee Arthroplasty (Technique & Alignment)
    8: "Knee Arthroplasty", 19: "Knee Arthroplasty", 30: "Knee Arthroplasty",
    42: "Knee Arthroplasty", 51: "Knee Arthroplasty", 56: "Knee Arthroplasty",
    63: "Knee Arthroplasty", 75: "Knee Arthroplasty", 77: "Knee Arthroplasty",
    # Digital Health, AI & Research Methods
    6: "Digital Health, AI & Research Methods", 7: "Digital Health, AI & Research Methods",
    22: "Digital Health, AI & Research Methods", 29: "Digital Health, AI & Research Methods",
    41: "Digital Health, AI & Research Methods", 43: "Digital Health, AI & Research Methods",
    48: "Digital Health, AI & Research Methods", 61: "Digital Health, AI & Research Methods",
    72: "Digital Health, AI & Research Methods", 73: "Digital Health, AI & Research Methods",
    82: "Digital Health, AI & Research Methods",
}

# Safety check: every clustered topic must be assigned to exactly one domain
all_topic_ids = set(trends["topic_id"].tolist())
mapped_ids    = set(DOMAIN_MAP.keys())
missing = all_topic_ids - mapped_ids
extra   = mapped_ids - all_topic_ids
assert not missing, f"Topics missing a domain: {sorted(missing)}"
assert not extra,   f"Domain map has unknown topics: {sorted(extra)}"

# Display order (bottom -> top of the stack): growth-oriented domains on top
DOMAIN_ORDER = [
    "Hip Arthroplasty",
    "Knee Arthroplasty",
    "Implant Survivorship & Bearing Surfaces",
    "Perioperative & Medical Management",
    "Periprosthetic Joint Infection",
    "Patient-Reported Outcomes & Satisfaction",
    "Health Policy, Value & Care Delivery",
    "Digital Health, AI & Research Methods",
]
# Distinct, color-blind-considerate palette aligned to the order above
DOMAIN_COLORS = {
    "Hip Arthroplasty":                          "#8C6D31",
    "Knee Arthroplasty":                         "#BCBD22",
    "Implant Survivorship & Bearing Surfaces":   "#7F7F7F",
    "Perioperative & Medical Management":        "#17BECF",
    "Periprosthetic Joint Infection":            "#D62728",
    "Patient-Reported Outcomes & Satisfaction":  "#1F77B4",
    "Health Policy, Value & Care Delivery":      "#9467BD",
    "Digital Health, AI & Research Methods":     "#2CA02C",
}

dn5 = doc_topics[doc_topics["topic_id"] != -1].copy()
dn5["domain"] = dn5["topic_id"].map(DOMAIN_MAP)
counts = dn5.groupby(["year", "domain"]).size().unstack(fill_value=0)
counts = counts.reindex(columns=DOMAIN_ORDER, fill_value=0).sort_index()
share  = counts.div(counts.sum(axis=1), axis=0) * 100   # rows sum to 100%

years = share.index.values
stack_vals  = [share[d].values for d in DOMAIN_ORDER]
stack_cols  = [DOMAIN_COLORS[d] for d in DOMAIN_ORDER]

# Legend labels carry each domain's total share of the clustered corpus
domain_totals = counts.sum(axis=0)
grand_total   = domain_totals.sum()
leg_labels = [f"{d}  ({domain_totals[d] / grand_total * 100:.0f}%)"
              for d in DOMAIN_ORDER]

H5 = 5.4
fig, ax = plt.subplots(figsize=(PAGE_W, H5))
ax.stackplot(years, stack_vals, colors=stack_cols,
             edgecolor="white", linewidth=0.4, alpha=0.95)
ax.set_xlim(years.min(), years.max())
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.set_xticks([y for y in years if y % 5 == 0])
ax.set_xticklabels([str(y) for y in years if y % 5 == 0])
ax.tick_params(labelsize=FS_LABEL * 0.75, colors="#333333")
ax.set_xlabel("Year", fontsize=FS_AXIS * 0.8, color="#333333", labelpad=4)
ax.set_ylabel("Share of Clustered Abstracts (%)",
              fontsize=FS_AXIS * 0.8, color="#333333", labelpad=4)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_edgecolor(SPINE_COLOR)

# Legend below, ordered top-of-stack first so it reads with the figure
handles = [mpatches.Patch(facecolor=DOMAIN_COLORS[d], edgecolor="white", label=lab)
           for d, lab in zip(DOMAIN_ORDER, leg_labels)][::-1]
ax.legend(handles=handles,
          loc="upper center", bbox_to_anchor=(0.5, -0.22),
          fontsize=FS_LEGEND * 0.72, ncol=2,
          framealpha=0.95, edgecolor=SPINE_COLOR,
          handlelength=1.4, columnspacing=1.4, labelspacing=0.5,
          title="Clinical Research Domain (share of clustered corpus)",
          title_fontsize=FS_LEGEND * 0.74)
fig.subplots_adjust(bottom=0.40, top=0.88, left=0.10, right=0.98)
save_fig(fig, f"fig5_clinical_domains_{JOURNAL}.png", H5, pad_inches=0.2)

# Print domain composition summary for the manuscript text / caption
print("\n  Clinical domain composition (share of clustered corpus):")
for d in DOMAIN_ORDER:
    early = share.loc[share.index <= 2009, d].mean()
    late  = share.loc[share.index >= 2022, d].mean()
    print(f"    {d:<44} {domain_totals[d]:>4} docs | "
          f"2005-09 {early:4.1f}%  ->  2022-26 {late:4.1f}%")


# ---- Save figure height index ------------------------------------------------
(FIG_DIR / "figure_heights.json").write_text(json.dumps(FIGURE_HEIGHTS, indent=2))

print(f"\nFigure heights (for DOCX embedding):")
for name, h in FIGURE_HEIGHTS.items():
    print(f"  {name}: {PAGE_W} x {h:.1f} in")

print(f"\nEnhanced figures written to: {FIG_DIR}")
print("Done.")

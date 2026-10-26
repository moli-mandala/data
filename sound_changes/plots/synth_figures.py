#!/usr/bin/env python3
"""Family-wide figures from synth.py outputs (results/summary.csv, results/tags.csv).

    uv run --group plots python sound_changes/plots/synth_figures.py

Writes plots/family-*.pdf (+ .png): scores by group, tag chronology, isogloss maps, and the
lect × change heatmap ordered by hierarchical clustering (also results/clusters.txt).
"""
import csv
import os

import numpy as np
import pandas as pd
from plotnine import (
    aes, coord_fixed, coord_flip, element_blank, element_line, element_text, facet_wrap, geom_errorbar,
    geom_point, geom_tile, ggplot, labs, position_dodge, scale_color_brewer, scale_fill_brewer,
    scale_fill_manual, theme, theme_bw, theme_set,
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
RESULTS = os.path.join(SC, "results")
NONPHON = {"@notation", "@morph", "@acc-loss", "@untagged", "@stress"}

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

GROUP_ORDER = ["mia-insular", "dardic-north", "dardic-south", "nuristani", "northwest",
               "west-pahari", "central-east-pahari-romani", "hindi", "rajasthani-gujarati",
               "marathi-bhil", "bihari-tharu", "eastern"]


def save(p, name):
    out = os.path.join(HERE, f"family-{name}.pdf")
    p.save(out, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=250, verbose=False)
    print(out)


def fig_scores(summary):
    rows = []
    for _, r in summary.iterrows():
        for key, label in (("null_test_ned", "No rules"), ("auto_test_ned_at_hand_n", "Greedy (same #rules)"),
                           ("hand_test_ned", "Hand-written")):
            if pd.notna(r.get(key)) and r.get(key) != "":
                rows.append({"Group": r["Group"], "Lang": r["Lang"], "System": label, "NED": float(r[key])})
    df = pd.DataFrame(rows)
    df["Group"] = pd.Categorical(df["Group"], [g for g in GROUP_ORDER if g in set(df["Group"])][::-1])
    df["System"] = pd.Categorical(df["System"], ["No rules", "Greedy (same #rules)", "Hand-written"])
    p = (ggplot(df, aes("Group", "NED", color="System"))
         + geom_point(position=position_dodge(width=0.6), size=0.8, alpha=0.7)
         + coord_flip()
         + scale_color_brewer(type="qual", palette="Set1")
         + labs(x="", y="Test NED (lower is better)", color="")
         + theme(figure_size=(3.85, 3.2), axis_text_x=element_text(rotation=0)))
    save(p, "scores")


def fig_chronology(tags, min_lects=8):
    # one entry per rule identity ("change"), never per tag: differently conditioned rules differ
    t = tags[tags["change"].fillna("") != ""].drop_duplicates(["Lang", "line"])
    g = t.groupby("change").agg(tag=("tag", "first"), lects=("Lang", "nunique"), med=("rel_pos", "median"),
                             q1=("rel_pos", lambda x: np.percentile(x, 25)),
                             q3=("rel_pos", lambda x: np.percentile(x, 75))).reset_index()
    g = g[g["lects"] >= min_lects].sort_values("med", ascending=False)
    g["change"] = pd.Categorical(g["change"], list(g["change"]))
    p = (ggplot(g, aes("change", "med"))
         + geom_errorbar(aes(ymin="q1", ymax="q3"), width=0, size=0.4, color="#377eb8")
         + geom_point(aes(size="lects"), color="#e41a1c")
         + coord_flip()
         + labs(x="", y="Relative position in chain (0 = first, 1 = last)", size="Lects")
         + theme(figure_size=(3.85, 0.12 * len(g) + 0.8), axis_text_x=element_text(rotation=0)))
    save(p, "chronology")
    g.to_csv(os.path.join(RESULTS, "chronology.csv"), index=False)


def presence(tags, summary):
    t = tags[~tags["tag"].isin(NONPHON)]
    m = pd.crosstab(t["Lang"], t["tag"]).clip(upper=1)
    return m


def fig_heatmap_and_clusters(m, summary, min_lects=4):
    m = m.loc[:, m.sum() >= min_lects]
    m = m[m.sum(axis=1) > 0]
    Z = linkage(pdist(m.values.astype(bool), metric="jaccard"), method="average")
    order = dendrogram(Z, no_plot=True, labels=list(m.index))["ivl"]
    col_order = list(m.sum().sort_values(ascending=False).index)
    grp = summary.set_index("Lang")["Group"].to_dict()
    long = m.reset_index().melt(id_vars="Lang", var_name="tag", value_name="has")
    long = long[long["has"] > 0].copy()
    long["Group"] = long["Lang"].map(grp)
    long["Lang"] = pd.Categorical(long["Lang"], order[::-1])
    long["tag"] = pd.Categorical(long["tag"], col_order)
    p = (ggplot(long, aes("tag", "Lang", fill="Group"))
         + geom_tile()
         + scale_fill_brewer(type="qual", palette="Set1")
         + labs(x="", y="", fill="")
         + theme(figure_size=(0.09 * len(col_order) + 1.5, 0.075 * len(order) + 1.2),
                 axis_text_x=element_text(size=5, rotation=90, hjust=0.5),
                 axis_text_y=element_text(size=4.5), legend_position="right",
                 legend_direction="vertical"))
    save(p, "heatmap")
    # flat clusters at a few cut heights, for the report
    from scipy.cluster.hierarchy import fcluster
    with open(os.path.join(RESULTS, "clusters.txt"), "w", encoding="utf-8") as f:
        for k in (6, 10, 16):
            lab = fcluster(Z, k, criterion="maxclust")
            f.write(f"== {k} clusters (UPGMA, Jaccard over shared change tags)\n")
            for c in sorted(set(lab)):
                members = [lang for lang, l in zip(m.index, lab) if l == c]
                f.write(f"  {c:2d}: {' '.join(sorted(members, key=lambda x: grp.get(x, '')))}\n")
        f.write("\nleaf order: " + " ".join(order) + "\n")


def fig_isoglosses(m, tags_of_interest):
    L = pd.read_csv(os.path.join(os.path.dirname(SC), "cldf", "languages.csv"))
    L = L.set_index("ID")[["Latitude", "Longitude"]].dropna()
    rows = []
    for tag in tags_of_interest:
        if tag not in m.columns:
            continue
        for lang, has in m[tag].items():
            if lang in L.index:
                rows.append({"tag": tag.lstrip("@"), "Lang": lang, "lat": L.loc[lang, "Latitude"],
                             "lon": L.loc[lang, "Longitude"], "has": "yes" if has else "no"})
    df = pd.DataFrame(rows)
    # South Asia only: the migratory lects' European/Near-Eastern coordinates would squash the map
    df = df[(df["lon"].between(60, 98)) & (df["lat"].between(5, 38))]
    if df.empty:
        return
    df["has"] = pd.Categorical(df["has"], ["yes", "no"])
    df = df.sort_values("has", ascending=False)  # draw "yes" on top
    p = (ggplot(df, aes("lon", "lat", color="has"))
         + geom_point(size=0.7, alpha=0.85)
         + coord_fixed()
         + facet_wrap("~tag", ncol=4)
         + scale_color_brewer(type="qual", palette="Set1")
         + labs(x="", y="", color="Change present")
         + theme(figure_size=(5.5 * 0.7, 0.9 * ((len(set(df["tag"])) + 3) // 4) + 0.6),
                 axis_text_x=element_text(rotation=0, size=5), axis_text_y=element_text(size=5),
                 strip_text=element_text(size=6)))
    save(p, "isoglosses")


def main():
    summary = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    tags = pd.read_csv(os.path.join(RESULTS, "tags.csv"))
    fig_scores(summary)
    fig_chronology(tags)
    m = presence(tags, summary)
    m.to_csv(os.path.join(RESULTS, "tag_matrix.csv"))
    fig_heatmap_and_clusters(m, summary)
    import sys
    iso = sys.argv[1:] or ["@degem-lengthen", "@degem-plain", "@sibilant-ś", "@sibilant-s",
                           "@int-stop-loss", "@cluster-simplify", "@nasal-voicing", "@s-h",
                           "@a-rounding", "@deaspiration", "@anusvara-vowel", "@final-vowel-loss"]
    fig_isoglosses(m, iso)


if __name__ == "__main__":
    main()

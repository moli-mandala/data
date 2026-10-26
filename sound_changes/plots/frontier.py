#!/usr/bin/env python3
"""Plot #rules vs. test score for each rule set's frontier CSV.

    uv run --group plots python sound_changes/plots/frontier.py [--pair OIA-Pk]

Reads results/<pair>.hand.frontier.csv (greedy backward elimination of the hand rules) and
results/<pair>.auto.frontier.csv (greedy induction prefixes), whichever exist, and writes
plots/<pair>.frontier.pdf. The 0-rule point of both curves is the no-change baseline.
"""
import argparse
import os

import pandas as pd
from plotnine import (
    aes, element_blank, element_line, element_text, facet_wrap, geom_line, geom_point, ggplot,
    labs, scale_color_brewer, theme, theme_bw, theme_set,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

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

SERIES = {"hand": "Hand-written (ablated)", "auto": "Greedy induction"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="OIA-Pk")
    args = ap.parse_args()

    frames = []
    for key, label in SERIES.items():
        path = os.path.join(RESULTS, f"{args.pair}.{key}.frontier.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Rule set"] = label
            frames.append(df)
    df = pd.concat(frames)
    long = pd.concat([
        df.assign(metric="Test NED ↓", value=df["test_ned"]),
        df.assign(metric="Test exact match (%) ↑", value=100 * df["test_exact"]),
    ])
    long["Rule set"] = pd.Categorical(long["Rule set"], [v for v in SERIES.values()
                                                         if v in set(long["Rule set"])])

    p = (
        ggplot(long, aes("n_rules", "value", color="Rule set"))
        + geom_line(size=0.5)
        + geom_point(size=0.6)
        + facet_wrap("~metric", scales="free_y")
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Number of rules", y="", color="")
        + theme(figure_size=(3.85, 1.8), axis_text_x=element_text(rotation=0))
    )
    out = os.path.join(HERE, f"{args.pair}.frontier.pdf")
    p.save(out, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=300, verbose=False)
    print(out)


if __name__ == "__main__":
    main()

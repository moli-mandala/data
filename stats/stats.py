import csv
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    theme,
    element_text,
    ylab,
    xlab,
    element_blank,
    geom_map,
    coord_cartesian,
    scale_x_continuous,
    scale_y_continuous,
    scale_size_continuous,
    xlim,
    ylim,
    geom_point,
    geom_histogram,
    scale_x_log10,
)
from plotnine.scales import scale_y_log10
from matplotlib import pyplot as plt
import geopandas as gp
from tqdm import tqdm
from collections import Counter, defaultdict


def categorise(clade: str):
    if clade == "Other":
        return clade
    if "Dravidian" in clade:
        return "Dravidian"
    if "Munda" in clade:
        return "Munda"
    if "Nuristani" in clade:
        return "Nuristani"
    if "Burushaski" in clade:
        return "Burushaski"
    return "Indo-Aryan"


def old(row):
    return (
        "Old" in row[1]
        or "Proto" in row[1]
        or "Middle" in row[1]
        or row[5] in ["OIA", "MIA"]
    )


def load_data():
    langs = {}
    with open("../cldf/languages.csv") as fin:
        reader = csv.reader(fin)
        next(reader)
        for x in reader:
            langs[x[0]] = x

    data = []
    with open("../cldf/forms.csv") as fin:
        reader = csv.DictReader(fin)
        for x in tqdm(reader):
            if x["Form"]:
                data.append(
                    [
                        langs[x["Language_ID"]][1],
                        x["Form"],
                        x["Parameter_ID"],
                        categorise(langs[x["Language_ID"]][5]),
                        x["Gloss"],
                    ]
                )

    df = pd.DataFrame(data, columns=["lang", "word", "cogset", "Grouping", "gloss"])
    return df, langs


def plot_top_counts(df: pd.DataFrame):
    order = df["lang"].value_counts().index.tolist()
    cat = pd.Categorical(df["lang"], categories=order)
    df = df.assign(lang_order=cat)
    df = df[df["lang"].isin(order[:50])]
    g = (
        ggplot(df)
        + geom_bar(aes(x="lang_order", fill="Grouping"))
        + theme(
            axis_text_x=element_text(rotation=45, size=6, hjust=1),
            axis_title_x=element_blank(),
            text=element_text(family='Times')
        )
        + scale_y_log10()
        + ylab("Lemmata")
    )
    g.draw()
    g.save("figures/bar.pdf", width=7.5, height=1.5)


def plot_lemma_counts(df: pd.DataFrame):
    df = df.groupby(["lang", "Grouping"]).count().reset_index()
    df = df.rename(columns={"word": "Lemmata"})
    print(df)
    g = (
        ggplot(df)
        + geom_histogram(aes(x="Lemmata", fill="Grouping"), bins=20)
        + scale_x_log10()
        + theme(axis_title_y=element_blank(), text=element_text(family="Times"))
    )
    g.draw()
    g.save("figures/lemmas.pdf", width=3, height=2)


def summary_table(df: pd.DataFrame, langs: dict[str, list[str]]):
    total_lemmata = len(df)
    lemmata_counts = df.groupby(["Grouping"]).count()["lang"]
    cols = lemmata_counts.index.tolist()
    lemmata_counts = {cols[i]: x for i, x in enumerate(lemmata_counts)}
    cogset_counts = {
        cols[i]: x
        for i, x in enumerate(
            df.groupby(["cogset", "Grouping"])
            .count()
            .groupby(["Grouping"])
            .count()["lang"]
        )
    }
    total_cogsets = len(df.groupby(["cogset"]).count())
    lang_counts = Counter([categorise(lang[5]) for lang in langs.values()])

    # for lang in lang_counts:
    #     print(f"{lang:<15} & {lang_counts[lang]:>10,d} & {cogset_counts[lang]:>10,d} & {lemmata_counts[lang]:>10,d} \\\\")
    # t = "\\textbf{Total}"
    # print(
    #     f"\\midrule\n{t:<15} & {sum(lang_counts.values()):>10,d} & {total_cogsets:>10,d} & {total_lemmata:>10,d} \\\\"
    # )


def map(df: pd.DataFrame, langs: dict[str, list[str]]):
    continents = gp.read_file("maps/World_Continents.shp")
    asia = continents.query('CONTINENT=="Asia"')
    lang_ct = {x[1]: 0 for x in langs.values()}
    for row in df["lang"]:
        lang_ct[row] += 1

    # convert langs to df of short name, full name, glottocode, lat, long, grouping
    l = [
        [
            x[0],
            x[1],
            x[2],
            float(x[3]),
            float(x[4]),
            categorise(x[5]),
            old(x),
            lang_ct[x[1]],
        ]
        for x in langs.values()
    ]
    l.sort(key=lambda x: x[-1], reverse=True)
    lang_df = pd.DataFrame(
        l,
        columns=[
            "short",
            "name",
            "glottocode",
            "lat",
            "long",
            "Family",
            "Historical",
            "Lemmata",
        ],
    )

    g = (
        ggplot()
        + geom_map(asia, fill="#ddd")
        + xlim(67, 95)
        + ylim(0, 37)
        + xlab("")
        + ylab("")
        + geom_point(
            lang_df,
            aes(x="long", y="lat", fill="Family", shape="Historical", size="Lemmata"),
        )
        + theme(
            text=element_text(family='Times'),
            axis_ticks=element_blank(),
            axis_text=element_blank()
        )
    )

    g.draw()
    g.save("figures/map.pdf", width=6, height=8.25)


def top_glosses(df: pd.DataFrame):
    # get top counts from df['gloss']
    gloss_counts = df["gloss"].value_counts()
    gloss_counts = gloss_counts[gloss_counts > 10]
    print(gloss_counts, sum(gloss_counts))

to_retro = {
    'ṉ': 'ṇ',
    'l': 'ḷ',
    'r': 'r̤',
}

def k_q_counts_ndr(df: pd.DataFrame):
    data = list(df.itertuples(index=False))

    # get Kurux and Malto lemmas
    malto = [x for x in data if x.lang == "Malto"]
    kurux = [x for x in data if x.lang == "Kurux"]
    brahui = [x for x in data if x.lang == "Brahui"]

    # get initial k and q words in Malto, k and x in Kurux
    k_malto = [x for x in malto if x.word[0] == "k"]
    q_malto = [x for x in malto if x.word[0] == "q"]
    k_kurux = [x for x in kurux if x.word[0] == "k"]
    x_kurux = [x for x in kurux if x.word[0] == "x"]
    k_brahui = [x for x in brahui if x.word[0] == "k"]
    x_brahui = [x for x in brahui if x.word[0] == "x"]

    # get counts over second phoneme
    k_malto = Counter([x.word[1] for x in k_malto])
    q_malto = Counter([x.word[1] for x in q_malto])
    k_kurux = Counter([x.word[1] for x in k_kurux])
    x_kurux = Counter([x.word[1] for x in x_kurux])
    k_brahui = Counter([x.word[1] for x in k_brahui])
    x_brahui = Counter([x.word[1] for x in x_brahui])

    # print counts pairwise
    print("Malto")
    print(f"{'':<5} {'k-':>5} {'q-':>5}")
    for key in set(list(k_malto.keys()) + list(q_malto.keys())):
        print(f"-{key + '-':<5} {k_malto[key]:>5} {q_malto[key]:>5}")
    
    print("Kurux")
    print(f"{'':<5} {'k-':>5} {'x-':>5}")
    for key in set(list(k_kurux.keys()) + list(x_kurux.keys())):
        print(f"-{key + '-':<5} {k_kurux[key]:>5} {x_kurux[key]:>5}")

    print("Brahui")
    print(f"{'':<5} {'k-':>5} {'x-':>5}")
    for key in set(list(k_brahui.keys()) + list(x_brahui.keys())):
        print(f"-{key + '-':<5} {k_brahui[key]:>5} {x_brahui[key]:>5}")


def drav_retro_alternants(df: pd.DataFrame):
    # filter all Dravidian lemmas
    data = list(df.itertuples(index=False))
    drav = [x for x in data if "Dravidian" in x.Grouping]

    # get retro alternants
    lemmas = defaultdict(list)
    for lemma in tqdm(drav):
        word = lemma.word[:4]
        change = False
        for char in to_retro:
            if char in word:
                change = True
            word = word.replace(char, to_retro[char])
        if change:
            lemmas[word].append(lemma)

    # print retro alternants
    for lemma in lemmas:
        words = [x.word[:4] for x in lemmas[lemma]]
        cogsets = [x.cogset for x in lemmas[lemma]]
        if len(set(words)) > 1 and len(set(cogsets)) > 1:
            for word in lemmas[lemma]:
                print(f"{word.cogset:<7} {word.word:<20} {word.lang:<10} {word.gloss}")
            print()

def main():
    df, langs = load_data()
    # plot_lemma_counts(df)
    # top_glosses(df)
    # plot_top_counts(df)
    # summary_table(df, langs)
    # map(df, langs)
    # drav_retro_alternants(df)
    k_q_counts_ndr(df)


if __name__ == "__main__":
    main()

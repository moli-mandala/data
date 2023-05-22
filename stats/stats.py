import csv
import pandas as pd
from plotnine import ggplot, aes, geom_bar, theme, element_text, ylab, xlab, element_blank, geom_map, coord_cartesian,\
    scale_x_continuous, scale_y_continuous, scale_size_continuous, xlim, ylim, geom_point
from plotnine.scales import scale_y_log10
from matplotlib import pyplot as plt
import geopandas as gp
from tqdm import tqdm
from collections import Counter

def categorise(clade: str):
    if clade == "Other": return clade
    if "Dravidian" in clade: return "Dravidian"
    if "Munda" in clade: return "Munda"
    if "Nuristani" in clade: return "Nuristani"
    if "Burushaski" in clade: return "Burushaski"
    return "Indo-Aryan"

def old(row):
    return 'Old' in row[1] or 'Proto' in row[1] or 'Middle' in row[1] or row[5] in ['OIA', 'MIA']

def load_data():
    langs = {}
    with open('../cldf/languages.csv') as fin:
        reader = csv.reader(fin)
        next(reader)
        for x in reader:
            langs[x[0]] = x

    data = []
    with open('../cldf/forms.csv') as fin:
        reader = csv.reader(fin)
        next(reader)
        for x in tqdm(reader):
            if x[3]: data.append([langs[x[1]][1], x[3], x[2], categorise(langs[x[1]][5])])

    df = pd.DataFrame(data, columns=['lang', 'word', 'cogset', 'Grouping'])
    return df, langs

def plot_top_counts(df: pd.DataFrame):
    order = df['lang'].value_counts().index.tolist()
    cat = pd.Categorical(df['lang'], categories=order)
    df = df.assign(lang_order = cat)
    df = df[df['lang'].isin(order[:50])]
    g = (ggplot(df) + geom_bar(aes(x='lang_order', fill='Grouping')) +
        theme(axis_text_x=element_text(rotation=45, size=6, hjust=1), axis_title_x=element_blank()) + scale_y_log10() +
        ylab("Lemmata"))
    g.draw()
    g.save('figures/bar.pdf', width=7.5, height=1.5)

def summary_table(df: pd.DataFrame, langs: dict[str, list[str]]):
    total_lemmata = len(df)
    lemmata_counts = df.groupby(['Grouping']).count()['lang']
    cols = lemmata_counts.index.tolist()
    lemmata_counts = {cols[i]: x for i, x in enumerate(lemmata_counts)}
    cogset_counts = {cols[i]: x for i, x in enumerate(df.groupby(['cogset', 'Grouping']).count().groupby(['Grouping']).count()['lang'])}
    total_cogsets = len(df.groupby(['cogset']).count())
    lang_counts = Counter([categorise(lang[5]) for lang in langs.values()])

    for lang in lang_counts:
        print(f"{lang:<15} & {lang_counts[lang]:>10,d} & {cogset_counts[lang]:>10,d} & {lemmata_counts[lang]:>10,d} \\\\")
    t = '\\textbf{Total}'
    print(f"\\midrule\n{t:<15} & {sum(lang_counts.values()):>10,d} & {total_cogsets:>10,d} & {total_lemmata:>10,d} \\\\")

def map(df: pd.DataFrame, langs: dict[str, list[str]]):
    continents = gp.read_file('maps/World_Continents.shp')
    asia = continents.query('CONTINENT=="Asia"')
    lang_ct = {x[1]: 0 for x in langs.values()}
    for row in df['lang']:
        lang_ct[row] += 1

    # convert langs to df of short name, full name, glottocode, lat, long, grouping
    l = [[x[0], x[1], x[2], float(x[3]), float(x[4]), categorise(x[5]), old(x), lang_ct[x[1]]] for x in langs.values()]
    l.sort(key=lambda x: x[-1], reverse=True)
    lang_df = pd.DataFrame(l, columns=['short', 'name', 'glottocode', 'lat', 'long', 'Family', 'Historical', 'Lemmata'])

    g = (ggplot() + geom_map(asia, fill='#ddd') + xlim(67, 95) + ylim(0, 37) + xlab('') + ylab('') +
        geom_point(lang_df, aes(x='long', y='lat', fill='Family', shape='Historical', size='Lemmata')))

    g.draw()
    g.save('figures/map.pdf', width=6, height=8.25)

def main():
    df, langs = load_data()
    plot_top_counts(df)
    summary_table(df, langs)
    map(df, langs)

if __name__ == "__main__":
    main()
import csv
import pandas as pd
from plotnine import ggplot, aes, geom_bar, theme, element_text, ylab, element_blank
from plotnine.scales import scale_y_log10
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import Counter

def categorise(clade: str):
    if clade == "Other": return clade
    if "Dravidian" in clade: return "Dravidian"
    if "Munda" in clade: return "Munda"
    if "Nuristani" in clade: return "Nuristani"
    if "Burushaski" in clade: return "Burushaski"
    return "Indo-Aryan"

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
    print('Total:', len(df))
    print(df.groupby(['Grouping']).count())
    print(df.groupby(['cogset', 'Grouping']).count().groupby(['Grouping']).count())
    print(len(df.groupby(['cogset']).count()))
    l = [categorise(lang[5]) for lang in langs.values()]
    print(Counter(l))
    print(len(l))

def main():
    df, langs = load_data()
    plot_top_counts(df)
    summary_table(df, langs)

if __name__ == "__main__":
    main()
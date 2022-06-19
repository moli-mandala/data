import csv
import pandas as pd
from plotnine import *
import matplotlib.pyplot as plt

data = []
with open('stats.tsv', 'r') as fin:
    reader = csv.reader(fin, delimiter='\t')
    for row in reader:
        row[0] = row[0].split('/')[0]
        row = row[:1] + row[2:]
        row[2] = int(row[2])
        row[3] = int(row[3])
        row[4] = float(row[4])
        row[7] = int(row[7])
        row[8] = float(row[8])
        if len(row) > 9:
            row[9] = float(row[9])
            row[10] = float(row[10])
        data.append(row)

df = pd.DataFrame(data, columns=['Name', 'Bidirectional', 'EmbeddingSize', 'BatchSize', 'LearningRate', 'Langs', 'Sparsemax', 'Epoch', 'Perplexity', 'PER', 'WER'])

g = ggplot(df, aes(x='Epoch', y='Perplexity', group='Name', color='EmbeddingSize')) + \
    geom_line() + scale_y_log10() \
    + facet_grid('LearningRate~BatchSize')
g.draw()
plt.show()

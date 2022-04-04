import csv
import pandas as pd
from plotnine import *
from matplotlib import pyplot as plt
from tqdm import tqdm

data = []
with open('../cldf/forms.csv') as fin:
    reader = csv.reader(fin)
    for x in tqdm(reader):
        if x[3] and x[1] == 'Pa': data.append([x[1], x[3]])

df = pd.DataFrame(data, columns=['Lang', 'Len'])

p = ggplot(df, aes(x='Len')) + geom_histogram()
plt.show()
p.show()
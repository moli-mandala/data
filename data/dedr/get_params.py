import csv
from collections import defaultdict

pdr = defaultdict(list)
gloss = defaultdict(set)
with open('pdr.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pdr[row[1]].append(row[2])
        gloss[row[1]].add(row[3])


done = set()
with open('dedr_new.csv') as fin, open('params.csv', 'w') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        source = row[1]
        if source not in done:
            done.add(source)
            writer.writerow([source,  ', '.join(pdr.get(source, [''])), '', '; '.join(list(gloss.get(source, ['']))), ''])
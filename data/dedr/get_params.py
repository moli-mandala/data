import csv
import os
from collections import defaultdict

pdr = defaultdict(list)
gloss = defaultdict(set)
with open('pdr.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        pdr[row[1]].append(row[2].strip())
        gloss[row[1]].add(row[3].strip())

existing = {}
if os.path.exists('params.csv'):
    with open('params.csv', 'r') as f:
        existing = {row[0]: row for row in csv.reader(f)}
existing_ids = set(existing)

sources = list(existing)
with open('dedr_new.csv') as fin, open('params.csv', 'w') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        source = row[1]
        if source not in existing:
            existing[source] = [source, '', '', '', '']
            sources.append(source)

    for source in sources:
        if source in existing_ids:
            writer.writerow(existing[source][:5])
            continue
        old = existing[source] + [''] * (5 - len(existing[source]))
        writer.writerow([
            source,
            ', '.join(pdr[source]) if source in pdr else old[1],
            '',
            '; '.join(sorted(gloss[source])) if source in gloss else old[3],
            old[4],
        ])

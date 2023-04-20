import csv

pdr = {}
with open('dedr.csv', 'r') as f, open('pdr.csv', 'w') as fout:
    reader = csv.reader(f)
    writer = csv.writer(fout)
    for row in reader:
        if row[3].strip() == '\'PDr.\'':
            row = [x.strip(' \'\n') for x in row]
            row[1] = f'd{int(row[1])}'
            if row[1] not in pdr:
                pdr[row[1]] = row[4]
            else:
                pdr[row[1]] += f', {row[4]}'
            writer.writerow(['PDr', row[1], row[4], row[5], '', '', '', 'krishnamurti'])

done = set()
with open('dedr_new.csv') as fin, open('params.csv', 'w') as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        source = row[1]
        if source not in done:
            done.add(source)
            writer.writerow([source,  pdr.get(source, ''), '', '', ''])
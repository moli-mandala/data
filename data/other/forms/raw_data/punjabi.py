import csv

with open('punjabi', 'r') as fin, open('punjabi.csv', 'w') as fout:
    rows = []
    reader = csv.DictReader(fin)
    for row in reader:
        lang = row['Language'].replace(' ', '_').lower()
        for key, val in row.items():
            if key == 'Language': continue
            for v in val.split(','):
                if v: rows.append([key, v.strip(), lang])
    
    rows.sort()
    writer = csv.writer(fout)
    for row in rows:
        writer.writerow([row[2], '', row[1], row[0], '', '', '', 'gill'])
    
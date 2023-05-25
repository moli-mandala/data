import csv

with open('tharu', 'r') as fin, open('tharu.csv', 'w') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    for row in reader:
        for col in row:
            if col in ['Gloss', 'CDIAL']: continue
            for form in row[col].split('/'):
                form = form.strip()
                if form:
                    writer.writerow([col, '', form, row['Gloss'], '', '', '', 'boehm'])
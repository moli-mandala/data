import csv

with open('toulmin.txt', 'r') as fin, open('toulmin.csv', 'w') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    for row in reader:
        notes = ''
        if not row['CDIAL'].isdigit():
            notes = row['CDIAL']
            row['CDIAL'] = ''
        for key in row:
            if key not in ['CDIAL', 'Gloss']:
                for word in row[key].split(','):
                    word = word.strip()
                    if word:
                        writer.writerow([key, row['CDIAL'], word, row['Gloss'], '', word, notes, 'toulmin'])
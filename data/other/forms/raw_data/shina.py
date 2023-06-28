import csv

with open('shina.csv', 'r') as fin, open('../20230621-shina.csv', 'w') as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    for row in reader:
        for key in row:
            if key not in ['CDIAL', 'Gloss']:
                for word in row[key].split(','):
                    word = word.strip(' \n')
                    notes = ''
                    if '(' in word:
                        word, notes = word.split('(')
                        word = word.strip()
                        notes = notes.strip(')')
                    if word:
                        writer.writerow([key, row['CDIAL'], word, row['Gloss'], '', word, notes, 'schmidt'])
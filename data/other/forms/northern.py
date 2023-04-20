import csv

with open("northern_param", "r") as fin:
    reader = csv.reader(fin)
    next(reader)
    params = {row[0]: (row[1], row[2]) for row in reader}

with open("northern", "r") as fin, open("northern.csv", "w") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    next(reader)
    for row in reader:
        lang, param, form, segments, notes = row[2], row[3], row[5], row[6], row[7]
        writer.writerow([lang, '', form, params[param][0], '', segments.replace(' ', ''), notes, 'backstrom1992'])
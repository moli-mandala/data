import csv

from shackle import tagged_notes

with open("old_punjabi", "r") as fin, open("old_punjabi.csv", "w") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        writer.writerow([
            "OP",
            row[3],
            row[0],
            row[2],
            '',
            '',
            tagged_notes(row[1], row[4]),
            'shackle'
        ])

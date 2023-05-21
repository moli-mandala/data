import csv

with open("old_punjabi", "r") as fin, open("old_punjabi.csv", "w") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)
    for row in reader:
        writer.writerow([
            "OP",
            row[3],
            row[0],
            (f"({row[1]}) " if row[1] else '') + row[2],
            '',
            '',
            row[4],
            'shackle'
        ])
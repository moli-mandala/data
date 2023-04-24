from collections import defaultdict
import csv

notes = defaultdict(int)
with open('dedr_new.csv', 'r') as f:
    read = csv.reader(f)
    for row in read:
        for item in row[6].split(' '):
            if row[6]:
                notes[(row[0], row[6])] += 1

# sort dict and print
for k, v in sorted(notes.items(), key=lambda item: item[1]):
    print(k, v)
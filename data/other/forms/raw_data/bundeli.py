import csv
import re

langs = [
    "CtD",
    "SaB",
    "HaG",
    "TiS",
    "ShL",
    "PaB",
    "CtH",
    "HaK",
    "DmJ",
    "BhK",
    "DaK",
    "JaH",
    "SeB",
    "HoP",
    "CnC",
    "HaT",
    "BaK",
    "BaM",
    "BaA",
    "BlG",
    "BlK",
    "CnB",
    "ReR",
    "MaB",
    "MaG",
    "Hin",
]

match_str = r'^(\*)?\d+\.? ?'
idx = 0

rows = []

with open('bundeli', 'r') as fin, open('bundeli.csv', 'w') as fout:

    gloss = None
    writer = csv.writer(fout)

    for line in fin:
        line = line.strip()
        if line in langs or line == "" or line.isdigit():
            continue
        if line[0].isdigit():
            if idx != len(langs) and idx != 0:
                print("ERROR")
                exit(1)
            gloss = re.sub(match_str, '', line)
            idx = 0
            print(gloss)
        else:
            word = line
            print(langs[idx] if idx < len(langs) else idx, word, gloss)
            if langs[idx] != 'Hin':
                for lemma in word.split(','):
                    lemma = lemma.strip()
                    rows.append([langs[idx], '', lemma, gloss, '', lemma, '', 'bundeli'])
            idx += 1
    
    rows.sort(key=lambda x: (x[3], x[2], x[0]))
    for row in rows:
        writer.writerow(row)
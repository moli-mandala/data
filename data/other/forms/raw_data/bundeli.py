import csv
import re
from pathlib import Path

from _curated_parameters import apply, load


HERE = Path(__file__).parent
OUTPUT = HERE.parent / "20230522-bundeli.csv"

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

LECT_IDS = dict(zip(langs[:-1], [
    "bundeli_malupur", "bundeli_karera", "bundeli_gudah", "bundeli_sakera",
    "chaurasi_lautna", "bundeli_bagaran", "bundeli_hardwar", "bundeli_kurra",
    "bundeli_jamunia", "jatbara_katva", "bundeli_kaliputra", "bundeli_hadrokh",
    "dehati_bijana", "mugalai_panjarakala", "dehati_chand", "bundeli_kuthupur",
    "bundeli_kaptia", "bundeli_atarra", "dehati_asoh", "lodhi_gara",
    "pawar_kashpur", "nagpuri_beradi", "bagheli_lakshman", "braj_bundi",
    "braj_gokul",
]))

match_str = r'^(\*)?\d+\.? ?'
idx = 0

rows = []

with (HERE / 'bundeli').open(encoding='utf-8') as fin:

    gloss = None
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
                    # The PDF text layer inserts the first word of the English prompt here.
                    row_gloss = gloss
                    if gloss == "heart" and lemma == "dʒɪũ̠life":
                        lemma = "dʒɪũ̠"
                        row_gloss = "life, heart"
                    rows.append([LECT_IDS[langs[idx]], '', lemma, row_gloss, '', lemma, '', 'bundeli'])
            idx += 1
    
rows.sort(key=lambda x: (x[3], x[2], x[0]))
apply(rows, load(OUTPUT))
with OUTPUT.open('w', encoding='utf-8', newline='') as fout:
    csv.writer(fout, lineterminator='\n').writerows(rows)

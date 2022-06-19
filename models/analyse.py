import csv
from collections import defaultdict, Counter
from tqdm import tqdm
import math

sets = defaultdict(list)
langs = defaultdict(set)
tot = set()
dialects = {}
name_to_id = {}

ignore = ['PDr.']
with open('../cldf/languages.csv') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in reader:
        name_to_id[row[1]] = row[0]
        if row[-1] in ['MIA', 'OIA']:
            ignore.append(row[0])
        if ':' in row[1]:
            a, b = row[1].split(':')
            dialects[row[0]] = a

name_to_id['Hindko'] = 'L'
name_to_id['Saraiki'] = 'L'

with open('../cldf/forms.csv') as fin:
    reader = csv.reader(fin)
    next(reader)
    for row in tqdm(reader):
        if 'dedr' in row[-1]: continue
        if row[1] in ignore:
            continue
        if row[1] in dialects:
            row[1] = name_to_id[dialects[row[1]]]
        sets[row[2]].append(row[1])
        langs[row[1]].add(row[2])
        tot.add(row[2])

sets2 = Counter()
for entry in sets:
    sets2[tuple(sorted(set(sets[entry])))] += 1
exp = Counter()
exp2 = Counter()
agr = Counter()

for i in tqdm(sets2.most_common()):
    # disagree = 0
    # agree = 0
    # for j in sets2:
    #     inter = set(i[0]).intersection(set(j))
    #     if inter == set(i[0]) or inter == set(j):
    #         agree += sets2[j]
    #     elif inter == set():
    #         ...
    #     else:
    #         disagree += sets2[j]
    # print(i[0], agree, disagree, agree / (agree + disagree))
    # agr[i[0]] = agree / (agree + disagree)

    expected = 1
    if i[1] <= 1:
        exp[i[0]] = 0
        continue
    for lang in langs:
        if lang in i[0]:
            expected *= (len(langs[lang]) / len(tot))
        else:
            expected *= 1 - (len(langs[lang]) / len(tot))
    # print(i, expected)
    exp[i[0]] = i[1] * (math.log(i[1] / len(tot)) - math.log(expected))
    exp2[i[0]] = i[1]

ranks = Counter()
for i, j in enumerate(exp.most_common()):
    ranks[j[0]] = -i
for i, j in enumerate(exp2.most_common()):
    ranks[j[0]] += i
print(ranks.most_common(10))


with open('res.txt', 'w') as fout:
    for i in exp.most_common(1000):
        fout.write(f'{i[0]}: {i[1]}\n')
    fout.write('=========================\n')
    for i in exp2.most_common(100):
        fout.write(f'{i[0]}: {i[1]}\n')
    # fout.write('=========================\n')
    # for i in agr.most_common(100):
    #     if i[1] <= 0:
    #         break
    #     fout.write(f'{i[0]}: {i[1]}\n')
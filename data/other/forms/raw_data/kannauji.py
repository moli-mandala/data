import re
import csv

lines = []
labels = ['Dehati-Sikandarpur', 'Dehati-Badeli', 'Hindi-Rohili', 'Hindi-Sanayak',
            'Dehati-Kirkkichiyapur', 'Hindi-Dhubar', 'Kannauji-Central', 'Hindi-Jamniya',
            'Hindi-Gohaniya', 'Hindi-Gabchariyapur', 'Hindi-Sarhati', 'Hindi-Saraiyya',
            'Dehati-Madnapur', 'Hindi', 'Bundeli', 'Braj-Bhasha']

match_str = r'^(\*)?\d+\. ?'

def starts(line: str):
    return any([line.startswith(label) for label in labels])

with open('kannauji', 'r') as fin:
    for line in fin:
        line = line[:-1]
        if line.isdigit():
            continue
        elif re.match(match_str, line):
            lines.append(re.sub(match_str, '', line))
            print(line)
        elif not starts(line):
            lines[-1] = lines[-1] + line
        else:
            lines.append(line)

rows = []
cur_gloss = None
for line in lines:
    if not starts(line):
        cur_gloss = line.strip()
    else:
        toks = list(line.split())
        lang = toks[0]
        if lang in ['Hindi', 'Bundeli', 'Braj-Bhasha']: continue
        for tok in toks[1:]:
            tok = tok.strip(' ,.')
            if not tok.isdigit() and tok != '——':
                rows.append([lang, '', tok, cur_gloss, '', tok, '', 'kannauji'])

rows.sort(key=lambda x: (x[3], x[2]))

with open('20230526-kannauji.csv', 'w') as fout:
    writer = csv.writer(fout)
    for row in rows:
        writer.writerow(row)


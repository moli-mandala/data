import re
import csv

lines = []
labels = ['HIN', 'RNS', 'RNK', 'BNM', 'SkP', 'TkN', 'RNS', 'RkM', 'RKB', 'BNT', 'BNM', 'BNT', 'DGC', 'RKB', 'DKS', 'KkP', 'DkR', 'DDK', 'CCC']

match_str = r'^(\*)?\d+\. ?'

with open('tharu2', 'r') as fin:
    for line in fin:
        line = line[:-1]
        if line.isdigit():
            continue
        elif re.match(match_str, line):
            lines.append(re.sub(match_str, '', line))
            print(line)
        elif len(line) < 3 or line[:3] not in labels:
            lines[-1] = lines[-1] + line
        else:
            line = line[:3] + ' ' + line[3:]
            lines.append(line)

rows = []
cur_gloss = None
with open('tharu2.csv', 'w') as fout:
    writer = csv.writer(fout)
    for line in lines:
        if len(line) < 3 or line[:3] not in labels:
            cur_gloss = line.strip()
        else:
            toks = list(line.split())
            lang = toks[0]
            if lang in ['HIN']: continue
            for tok in toks[1:]:
                tok = tok.strip(' ,.')
                if not tok.isdigit() and tok != '——':
                    writer.writerow([lang, '', tok, cur_gloss, '', tok, '', 'webster'])



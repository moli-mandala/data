import re
import csv

lines = []
labels = ['NDu', 'DRp', 'BRg', 'IRp', 'KBp', 'PRj', 'MBs', 'JBp', 'TRp', 'KBs', 'SSr', 'HIn', 'ORi']

match_str = r'^(\*)?\d+\. ?'

with open('chattisgarhi', 'r') as fin:
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
            lines.append(line)

rows = []
cur_gloss = None
with open('chattisgarhi.csv', 'w') as fout:
    writer = csv.writer(fout)
    for line in lines:
        if len(line) < 3 or line[:3] not in labels:
            cur_gloss = line
        else:
            toks = list(line.split())
            lang = toks[0]
            if lang in ['HIn', 'ORi']: continue
            for tok in toks[1:]:
                tok = tok.strip(' ,.')
                if not tok.isdigit() and tok != '——':
                    writer.writerow([lang, '', tok, cur_gloss, '', tok, '', 'chattisgarhi'])



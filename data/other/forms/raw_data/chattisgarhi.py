import re
import csv
from pathlib import Path

lines = []
labels = ['NDu', 'DRp', 'BRg', 'IRp', 'KBp', 'PRj', 'MBs', 'JBp', 'TRp', 'KBs', 'SSr', 'HIn', 'ORi']
HERE = Path(__file__).parent
OUTPUT = Path(__file__).parents[1] / "20230517-chattisgarhi.csv"

match_str = r'^(\*)?\d+\. ?'

with (HERE / 'chattisgarhi').open(encoding="utf-8") as fin:
    raw_lines = [line.rstrip("\n") for line in fin]

# OCR occasionally splits a three-letter dialect label across two physical lines (``IR`` +
# ``p ...``). Rejoin only an exact prefix/suffix pair from the declared label inventory.
joined_lines = []
i = 0
while i < len(raw_lines):
    line = raw_lines[i]
    following = raw_lines[i + 1] if i + 1 < len(raw_lines) else ""
    label = next(
        (
            candidate for candidate in labels
            if candidate.startswith(line) and following.startswith(candidate[len(line):] + " ")
        ),
        None,
    )
    if label:
        line = line + following
        i += 1
    joined_lines.append(line)
    i += 1

for line in joined_lines:
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
with OUTPUT.open('w', encoding="utf-8", newline="") as fout:
    writer = csv.writer(fout, lineterminator="\n")
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
                    # The English prompt was inserted between a vowel and its combining tilde in
                    # this one OCR token. Removing that exact intrusion recovers the printed form;
                    # a broad alphabetic cleanup would risk changing genuine transcription.
                    if tok == 'ɐnɐ̆another̃':
                        tok = 'ɐnɐ̆̃'
                    writer.writerow([lang, '', tok, cur_gloss, '', tok, '', 'chattisgarhi'])

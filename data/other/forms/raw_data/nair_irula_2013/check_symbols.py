"""Literal non-IPA route and complete output character accounting."""
import collections
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=list(csv.reader((HERE/'proposed.csv').open()))
characters=collections.Counter(c for r in rows for c in r[2])
assert all(unicodedata.is_normalized('NFC',r[2]) and not r[4] and not r[5] for r in rows)
assert all(unicodedata.category(c) not in {'Co','Cc'} and c!='\ufffd' for c in characters)
report={'status':'passed source-preservation character inventory','rows':len(rows),
        'policy':'convert:false preserves exact source Roman notation, drawn lowlines, en dashes, internal spaces, slash-readings and supplied quantity/diaeresis. No modern IPA tier is inferred.',
        'structural_policy':'Quoted wrappers and commas separating a source list are editorial boundaries, while printed form-internal characters are retained. Slash alternatives are separate direct printed readings, not invented expanded forms. Backslash in one table cell retained with typed unexplained-notation uncertainty.',
        'source_lookalike_policy':'One text-layer Cyrillic ї, visually i with diaeresis, is retained literally and typed source-character-encoding uncertainty. It is not normalized into an asserted phonemic value.',
        'characters':[{'character':c,'codepoint':f'U+{ord(c):04X}','name':unicodedata.name(c,'UNNAMED'),'count':n} for c,n in sorted(characters.items())],
        'csv_sha256':hashlib.sha256((HERE/'proposed.csv').read_bytes()).hexdigest()}
(HERE/'symbol-inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'rows':len(rows),'characters':len(characters),'status':'passed'}))

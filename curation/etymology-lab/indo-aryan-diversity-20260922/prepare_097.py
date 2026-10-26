import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
form_id = 'f_ygiupjenfbrh6'
with open('cldf/forms.csv') as stream:
    record = next(row for row in csv.DictReader(stream) if row['ID'] == form_id)
with open('cldf/edges.csv') as stream:
    compiled_linked = {row['Child_ID'] for row in csv.DictReader(stream) if row['Rank'] == '1'}
overlay_linked = {row['Form_ID'] for row in read_assignments()}
assert form_id not in compiled_linked | overlay_linked
decision = {
    'record': {key: record[key] for key in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
    'parent': '13682',
    'kind': 'reflex',
    'citation': f'{record["Source"]};CDIAL[13682];schmidt',
    'evidence': 'Buddruss prints Gilgit Shina tom tree and compares Burushaski. Turner 13682 stambha includes Wg. štɔ̈̄ma and Kt. štom stem/tree; the independently imported Schmidt Shina database assigns exact tom tree from Gilgit, Kohistani, Guresi and Astori to 13682. This is an inherited stem hypothesis supported by several Shina dialects, while the Burushaski comparison and possible contact influence remain noted.',
}
held = [
    {'form': 'muk', 'gloss': 'gemstones (collective)', 'reason': 'Schmidt Shina muk face is a homonym, not a suitable 10158 target for this gloss.'},
    {'form': 'ṣak', 'gloss': 'full', 'reason': 'Schmidt Shina ṣak neck is a homonym, not a suitable 13576 target for this gloss.'},
    {'form': 'lup', 'gloss': 'burn', 'reason': 'Buddruss proposes a complex tentative derivation that needs separate analysis.'},
]
(here / 'pass097-decisions.json').write_text(json.dumps({'accepted': [decision], 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted 1 held 3')

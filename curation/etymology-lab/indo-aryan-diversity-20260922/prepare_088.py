import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
targets = [
    ('f_4xwkbqv44r52u', '424', 'Batote bacːa child corresponds to Turner 424 Punjabi baccā child and West Pahari Pangwali baccā child. The SIL response also contains a distinct second alternative, jakt, which is held; only the separately installed bacːa child is linked.'),
    ('f_4kmqnkwry2qhu', '4749', 'The original SIL-font response is t͡ʃɔl rice (cooked), and Turner 4749 gives Punjabi caul and West Pahari cau under *cāmala/*cāvala rice. The source specifies cooked rice, a narrower local usage than the dictionary’s husked-rice headword; its gloss is preserved.'),
]
with open('cldf/forms.csv') as stream:
    records = {row['ID']: row for row in csv.DictReader(stream) if row['ID'] in {x[0] for x in targets}}
with open('cldf/edges.csv') as stream:
    compiled_linked = {row['Child_ID'] for row in csv.DictReader(stream) if row['Rank'] == '1'}
overlay_linked = {row['Form_ID'] for row in read_assignments()}
accepted = []
for form_id, parent, evidence in targets:
    assert form_id not in compiled_linked | overlay_linked, form_id
    row = records[form_id]
    accepted.append({
        'record': {key: row[key] for key in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f'{row["Source"]};CDIAL[{parent}]',
        'evidence': evidence,
    })
held = [
    {'form': 'jakt', 'gloss': 'child', 'reason': 'Indic versus Iranian route in the jātak family remains unresolved.'},
    {'form': 'go', 'gloss': 'cow', 'reason': 'Turner places the close northern go forms under both gṓ 4255 and gava 4093, preferring the latter for some regions; the Batote form alone does not distinguish them.'},
    {'form': 'jo', 'gloss': 'millet', 'reason': 'Turner yava 10431 gives nearby jo barley, whereas the survey says millet; the crop identity is not verified.'},
    {'form': 'uḍrava', 'gloss': 'it is flying', 'reason': 'Turner 1697 §3 records a Dogri uḍḍar- verb but the installed 1697-3 node represents §4 causative; competing *uddrāti 2001 requires finer branch analysis.'},
]
(here / 'pass088-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted', len(accepted), 'held', len(held))

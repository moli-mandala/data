import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
targets = [
    ('f_xdlrxv5to3pxu', '3030', 'Buddruss Grangali kandā́ wild almond matches Turner kāntā 3030: Kalasha kānda almond and Khowar kándu wild almond tree are explicit neighboring comparanda. The source contrasts this wild almond with edible Persian bādām.'),
    ('f_j6ibjx7pqkotg', '4368', 'Buddruss lists -lām as a village suffix in Grangali place names Cukon-lām and Wuriglām. Turner grāma 4368 explicitly lists Ningalami and Shumashti lām village, as well as Pashai lām, in the same regional stem. This link records the inherited village element in a bound name component; it does not reclassify the source suffix as a free noun.'),
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
    {'form': 'desāī', 'language': 'Ningalami', 'gloss': 'tomorrow', 'reason': 'Turner 6573 has Lahnda tomorrow forms, while the separate Gawri comparison in the homepage database is not in the primary CDIAL article; the Kunar route needs more evidence.'},
    {'form': 'lik', 'language': 'Grangali', 'gloss': 'write', 'reason': 'Turner 11048 has the widespread likh family, but the Grangali source form lacks the aspirate and a contact versus inheritance route has not been established.'},
]
(here / 'pass092-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted', len(accepted), 'held', len(held))

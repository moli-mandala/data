import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
rows = {r['ID']: r for r in json.loads((here / 'pashai-remaining-current.json').read_text())}
linked = {r['Form_ID'] for r in read_assignments()}

analyses = [
    ('f_z75v74thv5wi6', '4368', 'Turner explicitly records Darrai Pashai lām under grāma village. Lehr gives lām home, village, retaining the broader local gloss.'),
    ('f_4gxz3woolnkoo', '6884', 'Turner section 2 *dhaura² explicitly records Pashai dōr face, mouth, side. Lehr Darrai dōr face corresponds in form and sense; the installed CDIAL graph has one node for the entire 6884 article, so the section is specified here.'),
    ('f_gm4cycnloy6la', '9330', 'Turner section 1 *bhakkha explicitly records Darrai Pashai bakoṭā́ stout, fat. Lehr Darrai bakoṭa fat is the same lexeme; section 2 *bhikkha is excluded.'),
    ('f_hae3bj7wblibu', '8042', 'Turner section 1 *pāḍḍa explicitly records Darrai and Nuristani Pashai pāṛek goat, precisely matching Lehr Darrai pāṛek. Section 2 *pēḍḍa is excluded; Turner notes probable Dravidian origin for the deeper etymon.'),
    ('f_6vtascyju6jz4', '7067', 'Turner explicitly records Pashai Laurowan nām and other regional nām name forms under nāman. Lehr Darrai nām name matches; the addendum questions a Persian loan only for Khowar, not Pashai.'),
]

accepted = []
for form_id, parent, evidence in analyses:
    assert form_id not in linked, form_id
    row = rows[form_id]
    accepted.append({
        'record': {k: row[k] for k in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': parent,
        'kind': 'reflex',
        'citation': f'lehr;CDIAL[{parent}]',
        'evidence': evidence,
    })

held = [
    {'form': 'aṇā', 'gloss': 'egg', 'reason': 'The nearby CDIAL Pashai āṇḍa is marked as borrowed from Indic and does not establish Lehr aṇā or its immediate donor.'},
    {'form': 'kāṛ', 'gloss': 'ear', 'reason': 'CDIAL Pashai kan/kaṇḍ refers to an ear orifice, but correspondence to Lehr kāṛ remains unverified.'},
    {'form': 'ainc', 'gloss': 'eye', 'reason': 'CDIAL Pashai ačhī/aᵉċ lacks the nasal in Lehr ainc; no secure analysis yet.'},
]
(here / 'pass084-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted', len(accepted), 'held', len(held))

import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
ids = ('f_jlpeeizo3c3j6', 'f_pyf2f47cgq2jw', 'f_cumnfhilo76zq')
with open('cldf/forms.csv') as stream:
    records = {row['ID']: row for row in csv.DictReader(stream) if row['ID'] in ids}
with open('cldf/edges.csv') as stream:
    compiled_linked = {row['Child_ID'] for row in csv.DictReader(stream) if row['Rank'] == '1'}
overlay_linked = {row['Form_ID'] for row in read_assignments()}
accepted = []
for form_id in ids:
    assert form_id not in compiled_linked | overlay_linked, form_id
    row = records[form_id]
    dialect = 'Nager' if 'nager' in row['Tags'].lower() else 'Hunza'
    accepted.append({
        'record': {key: row[key] for key in ('ID', 'Language_ID', 'Form', 'Gloss', 'Source', 'Tags')},
        'parent': '4661',
        'kind': 'reflex',
        'citation': f'{row["Source"]};CDIAL[4661]',
        'evidence': f'Weinreich §8.22 separately records this {dialect} Domaaki moon form. Turner 4661 §2 explicitly gives Ḍomaaki ċōnc̣ moon from candrá via a Dardic *ċanJ̣ branch; the installed article has one 4661 node, so the moon subsection is specified here. Weinreich records the other Nager/Hunza spellings as distinct source-defined entries.',
    })
held = [
    {'form': 'tíir', 'gloss': 'arrow', 'reason': 'Weinreich explicitly compares Persian/Urdu tīr but does not resolve the immediate donor.'},
    {'form': 'khakaí', 'gloss': 'walnut', 'reason': 'The Shina and Burushaski comparanda do not establish the direction of contact; Turner 48 applies to a different Domaaki walnut form.'},
    {'form': 'bras', 'gloss': 'rice', 'reason': 'Weinreich gives identical Burushaski bras; Turner 12233 orients a different Nager briyú form, not this Hunza form.'},
]
(here / 'pass096-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted', len(accepted), 'held', len(held))

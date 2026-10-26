import csv
import json
from pathlib import Path

from etymology_assignments import read_assignments


here = Path(__file__).resolve().parent
targets = [
    ('f_outljmm5ngkna', '12646', 'Turner 12646 explicitly records Pashai Darrai susuwā́ and Wegali sosəwá dry under śōṣu. Lehr Darrai sos- has the same s-s dry stem; its shorter lexical-stem notation is retained, without importing Turner’s adjectival ending.'),
    ('f_5js3uqgwjoxbk', '10181', 'Turner 10181 explicitly records Pashai muč- to flee and fall of a star under mucyátē escapes, alongside Ashkun muč- escape/flee. Lehr Darrai muc- flee matches the stem and meaning directly.'),
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
        'citation': f'lehr;CDIAL[{parent}]',
        'evidence': evidence,
    })
held = [
    {'form': 'maṛ-', 'gloss': 'rub', 'reason': 'Turner 9890 notes merger of mardati and *mr̥ndati branches; the relevant Pashai forms differ by dialect/nasalization from Lehr Darrai maṛ-.'},
    {'form': 'cir-', 'gloss': 'tear', 'reason': 'Turner 4844 has northern split/tear stems, but Waigali is explicitly via Pashto and the immediate route into Darrai Pashai is not established.'},
    {'form': 'piṛ-', 'gloss': 'squeeze', 'reason': 'Turner 8226 has exact northern squeeze stems, but no Pashai comparator and a nearby contact route remains possible.'},
]
(here / 'pass101-decisions.json').write_text(json.dumps({'accepted': accepted, 'held': held}, ensure_ascii=False, indent=2) + '\n')
print('accepted', len(accepted), 'held', len(held))

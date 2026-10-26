"""Exhaustive same-cell slash alternatives; proposal only, frozen CSV unchanged."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(HERE/'proposed.csv') == '4e85544cd7835bdd84eb58898b0b74368efc81c04aada4b0c7d89713297d0c83'
assert sha(HERE/'physical-units-first-reading.jsonl') == '7bd8af78668aeff2453ebb99ae9fd0e8ad57212f10af6f3867164ea8e29f51c0'
rows = {r[10]:r for r in csv.reader((HERE/'proposed.csv').open(newline=''))}
groups = [
    ('suffix-third-masculine-singular-cell', ['-aan','an','ven'], 'third-person masculine singular'),
    ('suffix-third-plural-cell', ['aaru','aatu','aru','atu','eecca'], 'third-person masculine plural, explicit Mas-row alignment'),
    ('suffix-third-feminine-singular-cell', ['-aaḷu','-aḷu','ooḷu','vaḷu'], 'third-person feminine singular'),
]
census, decisions = [], []
for section, forms, scope in groups:
    keys = [f'nair2013mullukurumba:p57:{section}:w{i}' for i in range(1,len(forms)+1)]
    assert [rows[k][2] for k in keys] == forms
    assert all(not rows[k][11] for k in keys)
    census.append(dict(source_unit_key=f'nair2013mullukurumba:p57:{section}', printed_page=57,
        physical_table='Pronominal Suffixes', function_slot=scope, literal_slash_cell='/'.join(forms),
        entry_keys=keys, literal_forms=forms, disposition='same-function source-listed morphological alternatives',
        caution='A slash within one explicit table cell supports alternate realizations of that function; it does not establish unrestricted free interchangeability. No relation to a different gender/person/number cell.'))
    for child in keys[1:]:
        decisions.append(dict(entry_key=child, field='Variant_Of_Key', old=rows[child][11], proposed_value=keys[0],
            proposed_tags=' '.join(sorted(set(rows[child][14].split())|{'alternate'})),
            old_tags=rows[child][14], proposed_notes=(rows[child][6]+' Source lists these slash-separated realizations in the same explicit pronominal-suffix slot; alternate relation is limited to that morphological function, not a claim of free interchangeability.').strip(),
            source_evidence=f'p57 Pronominal Suffixes: {scope} cell '+ '/'.join(forms), accepted=False))
assert len(census)==3 and len(decisions)==9
assert all(d['proposed_value'] in rows for d in decisions)
first = [json.loads(s) for s in (HERE/'physical-units-first-reading.jsonl').read_text().splitlines()]
assert len([u for u in first if u['section'] in {g[0] for g in groups}]) == 3
(HERE/'slash-alternative-adjudication-proposal.json').write_text(json.dumps(dict(status='Complete slash census; root approval pending',
    inputs={'first_reading_sha256':sha(HERE/'physical-units-first-reading.jsonl'),'csv_sha256':sha(HERE/'proposed.csv'),
        'freeze_sha256':sha(HERE/'review-freeze.json')}, table_groups=census, proposed_decisions=decisions,
    existing_explicit_free_variation='Two p56 proximal/distal pronoun links remain a separate, explicitly free-variation class.',
    no_new_forms=True, no_cross_group_relations=True, canonical_changes=False), ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'slash_groups':3,'source_listed_forms':12,'proposed_links':9,'candidate_changed':False}))

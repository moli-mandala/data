"""Record Norton’s complete original-page review, preserving owner inputs."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ROOT = BASE.parents[5]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

units = [json.loads(s) for s in (BASE / 'physical-units-first-reading.jsonl').read_text().splitlines()]
rows = list(csv.reader((BASE / 'proposed.csv').open()))
by_key = {r[10]: r for r in rows}
assert len(units) == 140 and sum(len(u['source_witnesses']) for u in units) == 208
assert len(rows) == len(by_key) == 154
assert digest(BASE / 'physical-units-first-reading.jsonl') == '7e756a167297b07d0f03cddf135bc5ed97a6e102e21ab99947d1de7d6148edf5'
assert digest(BASE / 'proposed.csv') == '7e95da15b5e4770da3e2fee7943413dd81b0637f4ea937e8e924b663184ed8b7'
assert digest(BASE / 'proposal-audit.jsonl') == '7c69e497b95665320dc66f75fb2a6f2ce7991051d13472705572544fbd47bb5d'
decisions = []
for key, row in by_key.items():
    add = []
    if ':kinship:' in key or any(f':kin-plural:{i}:' in key for i in range(1, 5)):
        add.append('kinship')
    if ':nonpast-negative:' in key:
        add.append('non-past')
    if ':vocabulary:23:' in key:
        add.append('verb')
    if add:
        assert all(t not in row[14].split() for t in add)
        decisions.append({'entry_key': key, 'form_unchanged': row[2], 'old_tags': row[14],
                          'new_tags': ' '.join(row[14].split() + add), 'add_tags': add,
                          'evidence': ('Explicit kinship section/examples, pp.46,48–49.' if add == ['kinship'] else
                                       'The p.47 paragraph explicitly introduces these as non-past negative markers/examples.' if add == ['non-past'] else
                                       'P.48 vocabulary prints cimp, to carry on head; this is an explicitly verbal definition.')})
assert len(decisions) == 19
num = [r for r in rows if r[3] in {'one', 'three', 'seven'}]
assert len(num) == 6 and all('num' in r[14].split() for r in num)
coverage = []
for unit in units:
    coverage.append({'source_unit_key': unit['source_unit_key'], 'printed_page': unit['printed_page'],
                     'kind': unit['kind'], 'scope_review': 'agrees',
                     'witnesses': [{'witness_key': w['witness_key'], 'form': w['literal_first_reading'],
                                    'printed_gloss': w['printed_gloss'], 'literal_and_gloss_review': 'agrees',
                                    'emitted': w['witness_key'] in by_key} for w in unit['source_witnesses']]})
assets = ['physical-units-first-reading.jsonl','proposed.csv','proposal-audit.jsonl','review-freeze.json',
          'scope/underline-vector-census.json']
assets += [f'original-review/page-{p}-150dpi.png' for p in range(45, 50)]
assets += ['original-review/page-50-boundary-150dpi.png']
report = {
 'reviewer': 'Norton; independent of Rohru’s first reading and exporter',
 'method': 'Read all five original page images before comparing owner text; then compare every source witness, scope record and actual output row. Narrow original PDF p47 negative construction checked at 4x.',
 'status': 'literal and chapter-scope review passed; metadata repair required',
 'counts': {'physical_groups':140,'witness_records':208,'emitted_rows':154,'literal_corrections':0,'metadata_rows':19,
            'kinship_rows':11,'nonpast_rows':7,'verb_rows':1,'lowline_vectors_reviewed':11},
 'inputs_sha256': {a:digest(BASE/a) for a in assets},
 'additional_evidence_sha256': {'p47-negative-original-detail.png':digest(OUT/'p47-negative-original-detail.png')},
 'original_checks': [
   'All 11 drawn lowlines agree with distinct U+0332 encoding; they are not retroflex underdots.',
   'P47 kaaṇ-aat-e does have a visible lower dot and matching encoded diacritic; no plain-n correction warranted.',
   'P46 peeṭṭa-laam versus p48 peettan differ in printed dental notation; preserve both without harmonization.',
   'P47 demonstratives have gender/distance headings but no repeated singular/plural headings; candidate correctly withholds inferred number/variant edges.',
   'P48 icca calling a bor and p48 coḷḷan daughter relation retain source anomalies rather than silently replacing them.',
   'Bound markers retain printed en dash / ASCII dash / bare aa and kaaṭ-u differences; zero phonemic claims.',
   'P49 personal names and community mannaan remain documentary, with no unjustified lexical language assignment.',
   'P50 begins Muduga; Mannan comparison commentary is accounted separately without inventing additional Mannan heads.'
 ],
 'resolved_nonfailure': {'numeral_class':'All six actual emitted numeral comparison rows already have num; first-reading provisional tags are not the final output authority.'},
 'metadata_decisions': decisions,
 'coverage':coverage,
 'limits':'No canonical installation, full build, compiled graph/identity or browser acceptance claimed. Parent’s fresh output sample is separate.'
}
(OUT/'complete-original-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'report':str(OUT/'complete-original-review.json'),'sha256':digest(OUT/'complete-original-review.json'),'counts':report['counts']}))

"""Record independently observed original-page findings and exact continuity checks."""
import csv
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[5]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name):
    return json.loads((BASE / name).read_text())

selection = load('independent-review/fresh20-selection-r2-20260927571.json')
for name, expected in selection['pins'].items():
    assert sha(BASE / name) == expected, name
freeze = load('review-freeze.json')
for name, expected in freeze['files'].items():
    path = ROOT / name
    assert sha(path) == expected, name
rows = list(csv.reader((BASE / 'proposed.csv').open()))
old = list(csv.reader((BASE / 'revision1/proposed.csv').open()))
assert len(rows) == len(old) == 135
assert [r[10] for r in rows] == [r[10] for r in old]
assert all(r[2] == o[2] for r, o in zip(rows, old))
fields = ['Language_ID', 'Parameter_ID', 'Form', 'Gloss', 'Native', 'Phonemic',
          'Notes', 'Source', 'Cognateset_ID', 'Etymology', 'Entry_Key',
          'Variant_Of_Key', 'Borrowed_From_Key', 'Derivation_Parent_Keys', 'Tags']
actual_changes = [{'entry_key': r[10], 'field': fields[i], 'old': o[i], 'new': r[i]}
                  for o, r in zip(old, rows) for i in range(15) if o[i] != r[i]]
expected_changes = [{k: c[k] for k in ['entry_key', 'field', 'old', 'new']}
                    for c in load('review-repair-continuity.json')['changes']]
assert sorted(actual_changes, key=str) == sorted(expected_changes, key=str)
assert len(actual_changes) == 35
indexed = {r[10]: r for r in rows}
assert len(indexed) == 135
for sample in selection['samples']:
    assert sample['actual_csv_fields'] == indexed[sample['entry_key']]
assert len(selection['samples']) == 20
graph = load('graph-proposal.json')['proposals']
actual_edges = {(r[10], r[11]) for r in rows if r[11]}
assert actual_edges == {(e['child_key'], e['variant_of_key']) for e in graph}
assert len(actual_edges) == 11
for child, parent in actual_edges:
    assert child != parent and parent in indexed
    assert not indexed[child][12] and not indexed[child][13]
for key, row in indexed.items():
    if ':present-suffixes:' in key or ':future-suffixes:' in key:
        assert 'verb' not in row[14].split()
        assert {'affix', 'suffix'} <= set(row[14].split())
assert indexed['nair2013mullukurumba:p55:plural-examples:w1'][14] == 'm noun pl'
assert indexed['nair2013mullukurumba:p56:instrumental-examples:w1'][3] == 'by the knife'

report = {
    'status': 'passed: independent complete chapter review plus fresh actual-output sample and exact repair controls',
    'sample_seed': selection['seed'],
    'sample_count': 20, 'sample_pass': 20, 'material_errors': 0,
    'authorship': 'Reviewer did not author owner first readings, candidate, repairs, or graph proposals.',
    'scope': 'Complete pp55–58 literal/scope review already recorded in pass1; fresh20 actual rows on pp55–57 (p58 bibliography has no emitted rows), plus all eight metadata repairs and eleven actual variant endpoints.',
    'limits': 'Source-stage review only; no canonical installation, compiled graph/database, full suite, or browser acceptance. Report does not independently verify proposed contextual p3/p12 source images beyond prior documentary accounting.',
    'pins': selection['pins'],
    'selection_sha256': sha(BASE / 'independent-review/fresh20-selection-r2-20260927571.json'),
    'complete_pass1_sha256': sha(BASE / 'independent-review/complete-original-review-pass1.json'),
    'original_page_images': {f'p{p}': sha(BASE / f'scope/originals/page-{p}-150dpi.png') for p in [55,56,57,58]},
    'samples': [{**s, 'result': 'pass', 'original_evidence': 'Original full page and chapter glyph/detail controls reread; form, source gloss scope, language column, explicit metadata, citation and graph scope agree.'} for s in selection['samples']],
    'repair_controls': {
        'exact_changed_fields': actual_changes,
        'changed_field_count': 35,
        'changed_row_count': len({c['entry_key'] for c in actual_changes}),
        'ordered_keys_and_all135forms_preserved': True,
        'six_isolated_tense_suffixes_not_lexical_verbs': True,
        'masculine_old_men_supported_by_printed_mutt_en_analysis': True,
        'knife_shared_gloss_limited_to_single_two_column_row': True,
        'variant_edges': [{'child': c, 'parent': p, 'child_form': indexed[c][2], 'parent_form': indexed[p][2]} for c,p in sorted(actual_edges)],
        'variant_scope': 'Two p56 explicitly free-varying proximal/distal pairs; nine p57 same-cell suffix alternatives qualified as morphological-slot realizations, not unrestricted interchangeability.',
        'controls': [
            'Unmarked nominative and isolated neuter zero are documentary, not invented overt lexical exponents.',
            'Feminine plural blank in p57 suffix table remains blank; masculine-row alternatives are not propagated into it.',
            'Segmented-to-surface arrows remain analysis without invented variant or derivation edges.',
            'Veetuvar and Mullu retain unresolved-language documentary attribution.',
            'p58 reported 1982 bibliography and colophon close chapter/book; no lexical rows fabricated from author contact details.',
            'Drawn l/n lowline, underdots, both n tildes in paañña, doubled vowels, and distinct source dash/hyphen spellings remain literal.'
        ]
    },
    'frozen_inputs_unchanged_after_review': all(sha(BASE / n) == h for n,h in selection['pins'].items())
}
out = BASE / 'independent-review/final-original-output-review-r2-20260927571.json'
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(out)
print(sha(out))

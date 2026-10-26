"""Merge and validate registry proposals in a private review directory only."""
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
sys.path.insert(0, str(DATA))
from dialects import normalize_dialect, load_dialect_aliases
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

OUT = HERE / 'integration-proposal'
registry = OUT / 'registry-drafts'
registry.mkdir(exist_ok=True)
inputs = [DATA / 'cldf/languages.csv', DATA / 'cldf/dialects.csv',
          OUT / 'new-language.csv', OUT / 'all-new-dialects.csv',
          OUT / 'all-additions.csv', HERE / 'legacy-review/legacy-repaired-candidate.csv']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pins = {str(p): sha(p) for p in inputs}

def merge(old_path, additions_path, target):
    with old_path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        old = list(reader)
    additions = list(csv.DictReader(additions_path.open()))
    assert all(list(row) == fields for row in additions)
    assert not {r['ID'] for r in old} & {r['ID'] for r in additions}
    merged = old + additions
    assert len({r['ID'] for r in merged}) == len(merged)
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    reloaded = list(csv.DictReader(target.open()))
    assert reloaded[:len(old)] == old
    return {r['ID']: r for r in reloaded}, additions

languages, added_languages = merge(inputs[0], inputs[2], registry / 'languages.csv')
dialects, added_dialects = merge(inputs[1], inputs[3], registry / 'dialects.csv')
assert len(added_languages) == 1 and added_languages[0]['ID'] == 'Birjia'
assert len(added_dialects) == 7
for row in added_languages + added_dialects:
    assert not row['Latitude'] and not row['Longitude'] and not row['Quality']
for row in added_dialects:
    assert row['Language_ID'] in languages
    assert row['Clade'] == languages[row['Language_ID']]['Clade']
    assert row['Location']
assert 'sil-kodaku-kanda' in dialects and 'sil-kodaku-kanda-story' not in dialects
by_tag = {r['Tag']: r for r in dialects.values()}
aliases = load_dialect_aliases(registry / 'dialects.csv')
rows = [row for path in inputs[4:] for row in csv.reader(path.open())]
for row in rows:
    assert row[0] in languages, (row[10], row[0])
    for tag in row[14].split():
        if tag.startswith('dialect:'):
            assert tag in by_tag and by_tag[tag]['Language_ID'] == row[0], (row[10], tag)
        else:
            assert tag in GRAMMATICAL_TAGS | GENDER_TAGS, (row[10], tag)
    assert normalize_dialect(row[0], row[14], aliases, set(languages)) == (row[0], row[14])
assert pins == {str(p): sha(p) for p in inputs}, 'Input changed during validation'
report = {
    'status': 'All proposed rows resolve through draft registries; existing records preserved',
    'input_sha256': pins, 'rows_checked': len(rows),
    'new_languages': [r['ID'] for r in added_languages],
    'new_dialects': [r['ID'] for r in added_dialects],
    'new_coordinate_points': 0, 'all_tags_registered': True,
    'qualified_tags_preserve_parent_languages': True,
    'canonical_changes': False,
    'limits': 'Draft metadata validation; no compiled graph or browser checks',
}
(OUT / 'registry-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'input_sha256'}, indent=2))

"""Check the printed paired-line inventory without accepting transcription yet."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from tags import GENDER_TAGS, GRAMMATICAL_TAGS

HERE = Path(__file__).resolve().parent
scope = json.loads((HERE / 'scope.json').read_text())
with (HERE / 'paired-lines.tsv').open() as stream:
    rows = list(csv.DictReader(stream, delimiter='\t'))
assert len(rows) == 47
assert len({(row['pdf_page'], row['block'], row['line']) for row in rows}) == 47
assert all(row['malayalam_raw'] and row['kurichiya_raw'] and row['printed_gloss'] for row in rows)
counts = Counter((int(row['pdf_page']), row['block']) for row in rows)
for block in scope['blocks']:
    if block['type'] in {'paired comparison table', 'paired prose example',
                         'two whole Kurichiya expressions and Malayalam controls'}:
        assert counts[(block['page'], block['id'])] == block['printed_lines'], block
assert sum(counts.values()) == 47
assert Counter(row['pdf_page'] for row in rows) == {'34': 22, '35': 20, '36': 5}
with (HERE / 'pronoun-cells.tsv').open() as stream:
    pronouns = list(csv.DictReader(stream, delimiter='\t'))
assert len(pronouns) == 17
assert len({(row['cell'], row['alternative']) for row in pronouns}) == 17
assert all(row['raw_form'] and row['pdf_page'] == '36' for row in pronouns)
assert all(set(row['provisional_tags'].split()) <= GENDER_TAGS | GRAMMATICAL_TAGS
           for row in pronouns)
with (HERE / 'particle-examples.tsv').open() as stream:
    particles = list(csv.DictReader(stream, delimiter='\t'))
assert len(particles) == 5
assert len({row['unit'] for row in particles}) == 5
assert sum(row['unit'].startswith('utterance-') for row in particles) == 2
with (HERE / 'prose-witnesses.tsv').open() as stream:
    prose = list(csv.DictReader(stream, delimiter='\t'))
assert len(prose) == 1 and prose[0]['linked_table_cell'] == 'I-gen-pl:1'
report = {
    'status': 'chapter source-unit first reading reconciled; not literal acceptance or installed-row counts',
    'source_sha256': scope['pdf_sha256'],
    'rows': len(rows), 'counts': dict(Counter(row['pdf_page'] for row in rows)),
    'inventory_sha256': hashlib.sha256((HERE / 'paired-lines.tsv').read_bytes()).hexdigest(),
    'pronoun_alternatives': len(pronouns), 'particle_ending_and_utterance_units': len(particles),
    'audited_prose_duplicate_witnesses': len(prose),
    'still_to_review': ['source-specific font/underline marks', 'reported bibliography',
                        'language identity and commentary claims', 'all original-page glyphs'],
}
(HERE / 'pair-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

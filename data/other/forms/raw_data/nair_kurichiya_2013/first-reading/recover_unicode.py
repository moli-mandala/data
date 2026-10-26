"""Propose literal Unicode recovery for all inventoried Kurichiya form witnesses.

This is a source-layer preview, not an installed CSV or phonemic conversion.
"""
import csv
import hashlib
import json
import unicodedata as ud
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOW = '\u0332'
PUA = {'\uf019': '\u0303', '\uf02c': '\u0323'}
UNDERLINED_PAIRS = {
    ('34', 'a-to-e', '6', 'Malayalam'): 'l',
    ('34', 'a-to-e', '6', 'Kurichiya'): 'l',
    ('34', 'i-to-y-or-v', '1', 'Malayalam'): 'l',
    ('34', 'i-to-y-or-v', '2', 'Malayalam'): 'l',
    ('34', 'i-to-y-or-v', '4', 'Malayalam'): 'l',
    ('34', 'i-to-y-or-v', '5', 'Malayalam'): 'l',
    ('35', 's-to-y-or-c', '2', 'Malayalam'): 'n',
    ('35', 's-to-y-or-c', '2', 'Kurichiya'): 'n',
    ('35', 'medial-elision', '1', 'Malayalam'): 'n',
}
UNDERLINED_PRONOUNS = {('I-sg', '1'): 'n',
                       ('II-sg', '1'): 'n', ('II-pl', '1'): 'n'}
assert len(UNDERLINED_PAIRS) + len(UNDERLINED_PRONOUNS) == 12


def read_tsv(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def recover(raw, underline=None):
    value = ''.join(PUA.get(char, char) for char in raw)
    if underline:
        assert value.count(underline) >= 1, (raw, underline)
        pos = value.index(underline)
        value = value[:pos + 1] + LOW + value[pos + 1:]
    value = ud.normalize('NFC', value)
    assert not any(ud.category(char) == 'Co' or char == '\ufffd' for char in value)
    return value


out = []
seen_underlines = set()
for row in read_tsv('paired-lines.tsv'):
    for side in ('Malayalam', 'Kurichiya'):
        key = (row['pdf_page'], row['block'], row['line'], side)
        if key in UNDERLINED_PAIRS:
            seen_underlines.add(key)
        raw = row[side.lower() + '_raw']
        out.append({'witness_key': f'nair2013kurichiya:p{row["pdf_page"]}:{row["block"]}:{row["line"]}:{side}',
                    'pdf_page': int(row['pdf_page']), 'block': row['block'], 'line': int(row['line']),
                    'source_side': side, 'source_raw_text_layer': raw,
                    'literal_unicode_proposal': recover(raw, UNDERLINED_PAIRS.get(key)),
                    'printed_gloss': row['printed_gloss'],
                    'status': 'literal proposal; source comparison or target not yet accepted'})
for row in read_tsv('pronoun-cells.tsv'):
    key = (row['cell'], row['alternative'])
    if key in UNDERLINED_PRONOUNS:
        seen_underlines.add(key)
    out.append({'witness_key': f'nair2013kurichiya:p36:pronouns:{row["cell"]}:{row["alternative"]}',
                'pdf_page': 36, 'block': 'pronoun-table', 'source_side': 'Kurichiya',
                'source_raw_text_layer': row['raw_form'],
                'literal_unicode_proposal': recover(row['raw_form'], UNDERLINED_PRONOUNS.get(key)),
                'printed_scope': row['printed_scope'], 'provisional_tags': row['provisional_tags'],
                'typed_uncertainty': row['uncertainty'], 'status': 'literal proposal; grammar needs review'})
for row in read_tsv('particle-examples.tsv'):
    out.append({'witness_key': f'nair2013kurichiya:p{row["pdf_page"]}:{row["unit"]}',
                'pdf_page': int(row['pdf_page']), 'block': row['unit'], 'source_side': 'Kurichiya',
                'source_raw_text_layer': row['raw_form'],
                'literal_unicode_proposal': recover(row['raw_form']),
                'printed_gloss': row['printed_gloss'], 'source_analysis': row['analysis'],
                'status': 'literal proposal; morphology and full utterance need review'})
for row in read_tsv('prose-witnesses.tsv'):
    out.append({'witness_key': f'nair2013kurichiya:p{row["pdf_page"]}:{row["unit"]}',
                'pdf_page': int(row['pdf_page']), 'block': row['unit'], 'source_side': 'Kurichiya',
                'source_raw_text_layer': row['raw_form'],
                'literal_unicode_proposal': recover(row['raw_form']),
                'linked_table_cell': row['linked_table_cell'], 'reason': row['reason'],
                'status': row['status']})
assert seen_underlines == set(UNDERLINED_PAIRS) | set(UNDERLINED_PRONOUNS)
assert len(out) == 117 == len({row['witness_key'] for row in out})
target = HERE / 'unicode-preview.jsonl'
target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in out))
report = {'status': 'all inventoried form witnesses have provisional Unicode; no installed rows',
          'witnesses': len(out), 'source_drawn_underlines_represented': len(seen_underlines),
          'unicode_preview_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'remaining': ['independent original-glyph audit', 'source claim and language routing',
                        'slash alternative expansion', 'audit and stable CSV fields',
                        'bibliography and profile validation', 'full chapter installation gates']}
(HERE / 'unicode-preview-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))

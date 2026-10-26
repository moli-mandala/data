"""Reproduce the complete source-only homophone transcription from reviewed TSV.

No OCR acceptance, lexical installation, inferred translation, normalization,
variant linkage, or homograph deduplication occurs here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[5]
PDF = WORKSPACE / 'tmp/pdfs/chakma-discovery/niranjan-2010.pdf'
TSV = HERE / 'homophones-manual-first-reading.tsv'
CENSUS = HERE / 'independent-structure-pages60-74-20260927.json'
OUT = HERE / 'homophones-first-reading.jsonl'
REPORT = HERE / 'homophones-first-reading-validation.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def build():
    census = json.loads(CENSUS.read_text())
    assert sha(PDF) == census['source']['sha256'], 'Original PDF changed'
    raw = list(csv.DictReader(TSV.open(newline=''), delimiter='\t'))
    assert len(raw) == 427
    page_counts = Counter()
    group_page_counts = Counter()
    cells = defaultdict(list)
    rows = []
    for source in raw:
        p, g, c = (int(source[k]) for k in ('page', 'group', 'cell'))
        col = source['column']
        assert col in ('left', 'right') and source['form'] and source['definition_bn']
        assert g != 146 and '\ufffd' not in ''.join(source.values())
        page_counts[p] += 1
        group_page_counts[p, g] += 1
        cells[g].append(c)
        parts = [{'printed_page': p, 'pdf_page': p + 2, 'column': col,
                  'group': g, 'cell': c, 'raw_definition': source['definition_bn']}]
        definition = source['definition_bn']
        note = source['uncertainty']
        if (g, c) == (133, 2):
            parts.append({'printed_page': 66, 'pdf_page': 68, 'column': 'left',
                          'position': 'top; preceding group134',
                          'raw_definition': 'কথা বলে।', 'continuation_only': True})
            definition += ' কথা বলে।'
            note = ''
        row = {
            'source_unit_key': f'niranjanchakma2010:p{p}:homophones:{col}:g{g}:c{c}',
            'printed_page': p, 'pdf_page': p + 2, 'column': col,
            'printed_group': g, 'group_cell': c,
            'target_form': source['form'], 'definition_bn': definition,
            'definition_language': 'Bengali',
            'source_parts': parts,
            'language_scope': 'source Chakma homophone section; explicit qualifiers retained',
            'source_qualifiers': [],
            'printed_alternative_separator': '/' if '/' in source['form'] else None,
            'relation_policy': 'no inferred variant or synonym edge; no homograph merge',
            'transcription_uncertainty': note or None,
            'review_status': 'owner original-page first reading; independent literal review pending',
        }
        if (g, c) == (62, 1):
            row['source_qualifiers'] = ['তঞ্চঙ্গ্যা উপভাষা']
            row['language_scope'] = 'explicit source Tanchangya variety qualifier; canonical routing unresolved'
        if (g, c) == (126, 1):
            row['source_qualifiers'] = ['অপ্র.']
        if (g, c) == (149, 3):
            row['source_qualifiers'] = ['চাকমাদের এক গোষ্ঠী নাম']
        rows.append(row)
    assert set(cells) == set(range(1, 199)) - {146}
    for g, cell_ids in cells.items():
        assert sorted(cell_ids) == list(range(1, len(cell_ids) + 1)), (g, cell_ids)
    for page in census['homophones']['pages']:
        p = page['printed_page']
        assert page_counts[p] == page['physical_cells'], p
        assert {str(g): n for (pp, g), n in group_page_counts.items() if pp == p} == page['cells_by_group'], p
    assert len({r['source_unit_key'] for r in rows}) == 427
    assert sum(bool(r['printed_alternative_separator']) for r in rows) == 3
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    report = {
        'status': 'complete source-only first-reading coverage; not installation approval',
        'physical_cells': 427, 'actual_groups': 197, 'terminal_group': 198,
        'missing_printed_groups': [146],
        'per_page_cells': {str(p): n for p, n in sorted(page_counts.items())},
        'uncertain_cells': [r['source_unit_key'] for r in rows if r['transcription_uncertainty']],
        'slash_alternative_cells': [r['source_unit_key'] for r in rows if r['printed_alternative_separator']],
        'definition_continuation': 'g133:c2 printed65–66; one cell, both source parts retained',
        'input_sha256': {'pdf': sha(PDF), TSV.name: sha(TSV), CENSUS.name: sha(CENSUS)},
        'output_sha256': hashlib.sha256(payload.encode()).hexdigest(),
        'lexical_policy': 'Literal target forms and Bengali definitions only. No English translations, phonemic values, variants, or deduplication inferred.',
        'structural_refinement': 'Group179 first head is ফিরি/পিরি; following dash separates its definition. Second ফিরি is a separate sense, not a wrapped compound.',
        'remaining_gates': ['Fine-glyph resolution of individually flagged readings', 'Independent full literal review', 'Whole-book integration and source-stage metadata/profile/parser/audit gates'],
    }
    return payload, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    payload, report = build()
    if args.check:
        assert OUT.read_text() == payload, 'Generated JSONL stale'
        assert json.loads(REPORT.read_text()) == report, 'Validation report stale'
    else:
        OUT.write_text(payload)
        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cells': report['physical_cells'], 'groups': report['actual_groups'],
                      'uncertain_cells': len(report['uncertain_cells']), 'sha256': report['output_sha256']}))

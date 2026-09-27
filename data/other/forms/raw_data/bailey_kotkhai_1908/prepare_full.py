"""Stage Bailey's complete Kotkhai chapter (pp. 23–24), without a data build.

The first-reading input is explicitly provisional. Canonical installation must use
the independently reconciled full-transcription.jsonl and a passing frozen audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import unicodedata
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE.parents[4]
INPUT = PACKAGE / "full-transcription.jsonl"
DRAFT = PACKAGE / "full-transcription-first-reading-20260926.jsonl"
SOURCE = "bailey1908kotkhai"
DIALECT = "dialect:Kotkhai:bailey1908-kotkhai:Kotkhai"
OUTPUT = DATA / "data/other/forms/20260925-bailey-kotkhai.csv"


def generate(input_path: Path = INPUT):
    units = [json.loads(line) for line in input_path.read_text().splitlines()]
    assert len(units) == 77 and {u['printed_page'] for u in units} == {23, 24}
    assert len({u['source_unit_key'] for u in units}) == len(units)
    rows, audit = [], []
    for unit in units:
        u = dict(unit)
        u['entry_keys'] = []
        u['gloss_basis'] = 'Explicit lexical gloss or meaning of source-labelled grammatical paradigm.'
        if not u['forms']:
            assert u['status'] in {'source_blank', 'excluded_control'}
            audit.append(u)
            continue
        u['status'] = 'ingested'
        for n, form in enumerate(u['forms']):
            key = u['source_unit_key'] + (f':answer{n+1}' if n else '')
            tags = [{'postposition': 'postp', 'infinitive': 'inf'}.get(t, t) for t in u['tags']]
            gloss = u['gloss']
            notes = []
            if u['section'] == 'pronoun' and 'f.' in u['raw_source'] and len(u['forms']) == 2:
                tags.append('f' if n else 'm')
                if '; ' in gloss:
                    gloss = gloss.split('; ')[n]
            if u['section'] == 'auxiliary' and u['cell'].startswith('pret'):
                tags.append('f' if n else 'm')
            if u['section'] == 'noun':
                notes.append('Source uses the horse paradigm referenced to Kiunthali p. 11; plural forms are explicitly the same as singular.')
            if u['section'] == 'noun-comment':
                notes.append('Source contrasts Kotkhai kē and āgō with Kiunthali khē and hāgō respectively.')
            if u['section'] == 'adverb' and u['cell'] == '5':
                tags.append('uncertain')
                notes.append('Source prints “these” among place adverbs; this apparent gloss anomaly is retained.')
                u['uncertainty'] = {'type': 'source_gloss', 'reason': 'Printed these in Place adverb column; not silently changed to there.'}
            if u['section'] == 'imperfect':
                notes.append('Source describes this as the usually preferred imperfect construction.')
            if u['section'] == 'lexical-difference' and u['cell'] == '2':
                notes.append('Zoller 2023 p. 686, entry 1223 quotes this same Bailey attestation; it is not an independent elicitation.')
                u['same_print_overlap'] = {'entry_key': 'zoller2023:18.1:p686:1223:span5:lect1:form1', 'relation': 'later quotation of this print cell; no linguistic graph edge inferred'}
            if ' ' in form:
                tags.append('multiword-expression')
            tags.append(DIALECT)
            row = ['Kotkhai', '', unicodedata.normalize('NFC', form), gloss, '', '',
                   ' '.join(notes), f"{SOURCE}[p. {u['printed_page']}, {u['section']}, {u['cell']}]",
                   '', '', key, '', '', '', ' '.join(dict.fromkeys(tags))]
            rows.append(row)
            u['entry_keys'].append(key)
        audit.append(u)
    assert len(rows) == 73 and len({r[10] for r in rows}) == 73
    assert {r[10] for r in csv.reader(OUTPUT.open())} <= {r[10] for r in rows}
    return rows, audit


def encoded(rows, audit):
    out = io.StringIO(newline='')
    csv.writer(out).writerows(rows)
    csv_text = out.getvalue()
    audit_text = ''.join(json.dumps(u, ensure_ascii=False, sort_keys=True) + '\n' for u in audit)
    chars = sorted({c for r in rows for c in r[2]})
    profile = 'Grapheme\tIPA\n' + ''.join(f"{c}\t{'#' if c == ' ' else c.lower()}\n" for c in chars)
    return {'proposal.csv': csv_text, 'proposal-audit.jsonl': audit_text, 'proposal-profile.txt': profile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--check-pdf', action='store_true')
    args = parser.parse_args()
    if args.check_pdf:
        pdf = DATA.parent / 'tmp/pdfs/bailey-sainji/bailey1908.pdf'
        with pdf.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == '953f5da5ee9bc341cdf7eb558920e824dc6f15f7473a8c0d5c8134c401ad60b5'
    assert not (args.draft and args.install), 'A provisional reading cannot be installed'
    rows, audit = generate(DRAFT if args.draft else INPUT)
    artifacts = encoded(rows, audit)
    if args.install:
        verdict = json.loads((PACKAGE / 'independent-full-audit-final.json').read_text())
        assert verdict['status'] in {'pass', 'passed'} and verdict['material_errors'] == 0
        assert verdict['sample_size'] >= 20
        for name, contents in artifacts.items():
            assert hashlib.sha256(contents.encode()).hexdigest() == verdict['hashes'][name]
        OUTPUT.write_bytes(artifacts['proposal.csv'].encode())
        (PACKAGE / 'audit.jsonl').write_bytes(artifacts['proposal-audit.jsonl'].encode())
        (DATA / 'conversion/bailey-kotkhai-1908.txt').write_bytes(artifacts['proposal-profile.txt'].encode())
    for name, contents in artifacts.items():
        (PACKAGE / name).write_bytes(contents.encode())
    print(json.dumps({'rows': len(rows), 'audit_units': len(audit), 'provisional': args.draft,
                      'installed': args.install, 'hashes': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in artifacts.items()}}, indent=2))


if __name__ == '__main__':
    main()

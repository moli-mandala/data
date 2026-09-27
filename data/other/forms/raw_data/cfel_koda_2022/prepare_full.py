"""Stage all reviewed Koda print attestations; never install or build a database.

English-edition IPA belongs to English control headings and is never imported.
The paired native edition supplies Koda IPA only where its complete native head
is aligned. Publisher readings are prepared separately, without print emendation.
"""
import csv
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def nfc(value):
    return unicodedata.normalize('NFC', value or '').strip()


def load_json(path):
    return json.loads(path.read_text())


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify_audited(package):
    report = load_json(ROOT / 'independent-full-audit-20260926-pass1.json')
    assert report['sample_size'] == 20 and all(x['status'] == 'pass' for x in report['sample'])
    expected = {Path(path).name: digest for path, digest in report['sha256'].items()
                if Path(path).parent.name == package.name}
    for name in ('full-proposal.csv', 'full-proposal-audit.jsonl'):
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected[name], 'Staging changed since independent audit'


def grammar_fields(grammar, domain):
    grammar = grammar.replace('‘', '')
    notes = []
    if '+' in grammar:
        tags = ['multiword-expression']
        notes.append('Source component labels: ' + grammar + '.')
    else:
        tags = [{'noun': 'noun', 'verb': 'verb', 'adjective': 'adj',
                 'adverb': 'adv'}[grammar.lower()]]
    if domain == 'Compound Verb':
        tags.append('compound')
        if 'verb' not in tags:
            notes.append('Source domain classifies this as a compound verb; printed part-of-speech label retained.')
    if domain == 'Causative Verb':
        tags.append('caus')
    return tags, notes


def build(work=ROOT):
    inventory = load_lines(work / 'paired-review-inventory.jsonl')
    semantic = load_json(work / 'semantic-review-decisions.json')
    native_review = load_json(work / 'native-transcription-visual-review.json')
    ipa_review = load_json(work / 'native-ipa-visual-review.json')
    ipa_inventory = {x['entry_key']: x for x in load_json(work / 'native-ipa-review-inventory.json')}
    english_review = load_json(work / 'english-native-visual-review.json')
    english_blocks = {x['entry_key']: x for x in load_lines(work / 'english-native-blocks.jsonl')}
    rows, audit = [], []
    for unit in inventory:
        key = unit['entry_key']
        nk = f"{unit['native_xps_page']}:{unit['native_item']}"
        reviewed_native = native_review.get(nk, {})
        publisher = unit['publisher_candidate'] or {}
        native = nfc(reviewed_native.get('printed_native') or reviewed_native.get('reviewed_native') or publisher.get('word'))
        if nk == '20:2':
            native = native.strip('/')
        assert native, (key, 'native head not reviewed')
        if key in ipa_review:
            reviewed_ipa = ipa_review[key]
        else:
            exact = ipa_inventory[key]
            assert exact['ocr_ipa'] == exact['api_ipa'], key
            reviewed_ipa = {'printed_ipa': exact['api_ipa'], 'decision': 'Independent OCR and publisher IPA agree exactly.'}
        ipa = nfc(reviewed_ipa['printed_ipa'])
        block = english_blocks[key]
        if key in english_review:
            english_native = [nfc(s) for s in english_review[key]['native_readings']]
        else:
            assert block['native_agrees'], key
            english_native = [nfc(s) for s in block['api_expected'] if nfc(s)]
        assert english_native and all(english_native), key
        if nk == '20:2':
            english_native = [s.strip('/') for s in english_native]
        tags, notes = grammar_fields(unit['english_source_grammar'], unit['source_domain'])
        if semantic[nk]['note']:
            notes.append(semantic[nk]['note'])
        if nk == '216:6':
            notes.append('Source grammatical claim: intensifier.')
        native_source = f"pradhan-tripathi2022koda-native[p. {unit['native_xps_page'] - 3}, entry {unit.get('native_physical_item', unit['native_item'])}]"
        english_source = f"pradhan-tripathi2022koda[p. {unit['english_pdf_page'] - 3}, entry {unit['english_item']}]"
        scope = reviewed_ipa.get('transcription_scope', '')
        if nk == '132:5':
            scope = reviewed_native['transcription_issue']
        alternatives = reviewed_ipa.get('printed_ipa_alternatives', [])
        aligned = bool(ipa) and not scope
        if key == 'cfelkoda2022:p12:entry:3':
            ipa, aligned = alternatives[0], True
        emitted, readings = [], []

        def emit(row_key, spelling, transcription, sources, parent='', extra_notes=(), issues=()):
            row_notes = list(notes) + list(extra_notes)
            row_tags = list(tags)
            row_issues = list(issues)
            if 'source-transcription-scope' in row_issues:
                row_tags.append('uncertain')
            if not transcription:
                row_notes.append('Source-native Form retained; no independently aligned full-form IPA.')
                row_issues.append('missing-aligned-full-form-IPA')
            if transcription and any(ch in transcription for ch in '?յɁᴐͻᴂͪ'):
                row_tags.append('uncertain')
                row_issues.append('source-transcription-symbol-interpretation-unresolved')
            form = nfc(transcription or spelling)
            row = ['Koda', '', form, unit['english_head'], spelling, '',
                   ' '.join(row_notes), ';'.join(sources), '', '', row_key,
                   parent, '', '', ' '.join(dict.fromkeys(row_tags))]
            rows.append(row)
            emitted.append(row_key)
            readings.append({'entry_key': row_key, 'row': row, 'issues': row_issues,
                             'form_layer': 'native-print-IPA' if transcription else 'source-native'})

        primary_sources = [native_source]
        if native in english_native:
            primary_sources.append(english_source)
        extra = []
        if scope:
            extra.append('Source transcription scope: ' + scope)
            if ipa:
                extra.append('Printed transcription: /' + ipa + '/.')
        emit(key, native, ipa if aligned else '', primary_sources, extra_notes=extra,
             issues=['source-transcription-scope'] if scope else [])
        for index, spelling in enumerate(english_native, 1):
            if spelling == native:
                continue
            alt_ipa = ''
            sources = [english_source]
            if key == 'cfelkoda2022:p12:entry:3' and spelling == 'মাকুরি':
                alt_ipa = alternatives[1]
                sources.append(native_source)
            # Semantic notes are explicitly from the native edition, which is
            # cited separately even when only the English edition prints this spelling.
            if semantic[nk]['note'] and native_source not in sources:
                sources.append(native_source)
            emit(key + f':variant:{index}', spelling, alt_ipa, sources, key)
        if key == 'cfelkoda2022:p153:entry:2':
            for index, (spelling, transcription) in enumerate(zip(
                    ['ভাজি আলু', 'ভাজি মান্ডি', 'ভাজি হাকু'], alternatives), 1):
                emit(key + f':expansion:{index}', spelling, transcription,
                     [native_source, english_source], key,
                     extra_notes=['Expanded only the explicitly printed parenthesized alternatives.'])
        audit.append({'entry_key': key, 'status': 'staged-complete-print-unit',
                      'emitted_keys': emitted, 'readings': readings,
                      'source_unit': unit, 'native_review': reviewed_native,
                      'ipa_review': reviewed_ipa, 'semantic_review': semantic[nk],
                      'english_native_readings': english_native,
                      'english_native_review': english_review.get(key, {'decision': block['review_status']}),
                      'publisher_attestation_policy': 'Prepared independently under publisher record keys; print differences never silently replace API evidence.'})
    assert len(inventory) == len(audit) == 2450
    assert len(rows) == len({r[10] for r in rows})
    keys = {r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    return rows, audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    rows, audit = build()
    with (ROOT / 'full-proposal.csv').open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    (ROOT / 'full-proposal-audit.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in audit))
    if args.install:
        verify_audited(ROOT)
        with (ROOT.parents[1] / '20260925-cfel-koda-print-pilot.csv').open('w', newline='') as handle:
            csv.writer(handle).writerows(rows)
    print(len(rows), 'print rows;', len(audit), 'fully accounted units;', 'source-stage installed' if args.install else 'not installed')

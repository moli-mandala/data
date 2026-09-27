"""Stage every print-bounded publisher attestation without print emendation."""
import csv
import argparse
import importlib.util
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent / 'cfel_koda_2022'
spec = importlib.util.spec_from_file_location('koda_print_full', WORK / 'prepare_full.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
nfc = shared.nfc


def build(work=WORK):
    inventory = shared.load_lines(work / 'paired-review-inventory.jsonl')
    rows, audit = [], []
    for unit in inventory:
        record = unit['publisher_candidate']
        if not record:
            audit.append({'entry_key': unit['entry_key'], 'status': 'print-only-no-matching-publisher-record', 'emitted_keys': [], 'query_evidence': unit['publisher_evidence']})
            continue
        key = f"cfel-koda-api:record:{record['id']}"
        native = nfc(record['word'])
        if (unit['native_xps_page'], unit['native_item']) == (20, 2):
            native = native.strip('/')
        raw = nfc(record['ipa'])
        ipa = raw.strip('/')
        tags, notes = shared.grammar_fields(nfc(record['grammatical_category']), unit['source_domain'])
        source = f"cfel2026koda[online dictionary, Koda record {record['id']}, concept {record['abs_ref']}]"
        emitted, readings = [], []
        scope = {
            'cfelkoda2022:p13:entry:6': 'IPA covers only the initial phrase; final native পেণ্টুল has no aligned transcription.',
            'cfelkoda2022:p61:entry:5': 'IPA covers only the final native component ঝুটি.',
            'cfelkoda2022:p69:entry:8': 'Native head আরে গ্যাল মরেয়া differs from IPA upɔn kuɽi mɔrea.',
            'cfelkoda2022:p176:entry:5': 'Native head তিহি তাপরি differs from IPA hat̪t̪ali.',
            'cfelkoda2022:p130:entry:5': 'Native head ends পরব while IPA ends ut̪sɔb.',
            'cfelkoda2022:p153:entry:2': 'Parenthesized alternatives are expanded into three separately aligned forms.'
        }.get(unit['entry_key'], '')
        def emit(row_key, spelling, transcription, parent='', extra_notes=(), issues=()):
            row_notes = list(notes) + list(extra_notes)
            row_tags = list(tags)
            row_issues = list(issues)
            if 'source-transcription-scope' in row_issues:
                row_tags.append('uncertain')
            if not transcription:
                row_notes.append('Source-native Form retained; publisher supplies no independently aligned full-form IPA.')
                row_issues.append('missing-aligned-full-form-IPA')
            elif any(ch in transcription for ch in '?յͪͻᴂɁᴐ') or '/' in transcription:
                row_tags.append('uncertain')
                row_issues.append('publisher-transcription-symbol-interpretation-unresolved')
            row = ['Koda', '', nfc(transcription or spelling), unit['english_head'], spelling, '', ' '.join(row_notes), source, '', '', row_key, parent, '', '', ' '.join(dict.fromkeys(row_tags))]
            rows.append(row); emitted.append(row_key)
            readings.append({'entry_key': row_key, 'row': row, 'issues': row_issues, 'form_layer': 'publisher-IPA' if transcription else 'source-native'})
        extra = []
        if scope:
            extra = ['Publisher transcription scope: ' + scope, 'Publisher transcription: /' + ipa + '/.']
        primary_ipa = ipa if not scope else ''
        if unit['entry_key'] == 'cfelkoda2022:p12:entry:3':
            assert ipa == 'lutur na/makuri'
            primary_ipa = 'lutur na'
        emit(key, native, primary_ipa, extra_notes=extra, issues=['source-transcription-scope'] if scope else [])
        for field in ('word2', 'word3'):
            alternate = nfc(record.get(field))
            if not alternate:
                continue
            alt_ipa = 'makuri' if unit['entry_key'] == 'cfelkoda2022:p12:entry:3' and alternate == 'মাকুরি' else ''
            emit(key + ':' + field, alternate, alt_ipa, key)
        if unit['entry_key'] == 'cfelkoda2022:p153:entry:2':
            assert ipa == 'bʰaʤi (alu/ manɖi/ haku)'
            for index, (spelling, transcription) in enumerate(zip(['ভাজি আলু', 'ভাজি মান্ডি', 'ভাজি হাকু'], ['bʰaʤi alu', 'bʰaʤi manɖi', 'bʰaʤi haku']), 1):
                emit(key + f':expansion:{index}', spelling, transcription, key)
        issues = []
        if raw and (not raw.startswith('/') or not raw.endswith('/') or raw.endswith('//')):
            issues.append('source-IPA-delimiter-anomaly')
        audit.append({'entry_key': key, 'status': 'staged-complete-publisher-record', 'emitted_keys': emitted, 'readings': readings, 'issues': issues, 'publisher_raw': record, 'publisher_evidence': unit['publisher_evidence'], 'paired_print_key': unit['entry_key'], 'print_policy': 'Print transcriptions and semantic notes are separate print attestations, not silent corrections of publisher evidence.'})
    assert len(audit) == 2450
    assert sum(bool(x['emitted_keys']) for x in audit) == 2448
    assert len(rows) == len({r[10] for r in rows})
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
        shared.verify_audited(ROOT)
        with (ROOT.parents[1] / '20260925-cfel-koda-adornments.csv').open('w', newline='') as handle:
            csv.writer(handle).writerows(rows)
    print(len(rows), 'publisher rows;', len(audit), 'print-bounded queries;', 'source-stage installed' if args.install else 'not installed')

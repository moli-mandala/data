"""Bounded independent-sidecar checks; no canonical parse or installation."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import runpy
import unicodedata as ud

HERE = Path(__file__).resolve().parent


def main():
    proposal = HERE / 'notation-keyed-proposal-20260927.jsonl'
    rows = [json.loads(line) for line in proposal.read_text().splitlines()]
    by_key = {r['source_unit_key']: r for r in rows}
    assert len(by_key) == len(rows) == 1420
    ledger = runpy.run_path(str(HERE.parent / 'manual_review_data.py'))['ROWS']
    expected = {}
    for page in ledger:
        if page['Site'] in {'TEL', 'ORI'}:
            continue
        for i, literal in enumerate(page['Forms'], page['First_Item']):
            expected[f"wordlist:{page['Site']}:{i}"] = (literal, page['PDF_Page'])
    assert set(by_key) == set(expected)
    for key, (old, page) in expected.items():
        row = by_key[key]
        assert row['previous_ledger_form'] == old, key
        assert row['pdf_page'] == page, key
        assert row['normalized_ipa_proposal'] is None, key
        assert row['installation_approved'] is False, key
        literal = row['literal_source_proposal']
        assert '\ufffd' not in literal and not any(ud.category(c) in {'Co', 'Cs'} for c in literal), key
        assert ud.normalize('NFC', ud.normalize('NFD', literal)) == ud.normalize('NFC', literal), key
        if '?' in literal or 'ʔ' in literal:
            assert row['question_shaped_sign_review']['status'] == 'interpretation_uncertain', key
            assert row['status'] == 'uncertain_question_shaped_sign_interpretation', key
    assert not by_key['wordlist:MAL:200']['literal_source_proposal']
    assert by_key['wordlist:UTN:16']['literal_source_proposal'] == 'bot:ʌ'
    assert by_key['wordlist:CHI:144']['literal_source_proposal'] == 'bʰaru·'
    assert by_key['wordlist:MAL:33']['literal_source_proposal'] == 'moʈ·al'
    assert by_key['wordlist:POD:186']['literal_source_proposal'] == 'eɾund ᵘaɣt̪ɛ'
    assert by_key['wordlist:JAG:105']['narrow_adjudication']['alternatives'] == ['i:la']
    assert all('ɪ' not in r['literal_source_proposal'] for r in rows)
    symbols = Counter(c for r in rows for c in r['literal_source_proposal'])
    profile = HERE.parents[5] / 'conversion' / 'sil-koya.txt'
    profile_symbols = set(''.join(line.split('\t')[0] for line in profile.read_text().splitlines()[1:]))
    report = {
        'status': 'passed_sidecar_integrity_only',
        'source_cells': len(rows),
        'physically_printed_cells': 1419,
        'absent_slot': 'wordlist:MAL:200',
        'attested_nonblank_cells': sum(bool(r['literal_source_proposal']) for r in rows),
        'literal_changes_from_previous_ledger': sum(r['changed'] for r in rows),
        'keyed_narrow_adjudications': sum(bool(r['narrow_adjudication']) for r in rows),
        'typed_uncertain_cells': sum(r['status'].startswith('uncertain_') for r in rows),
        'remaining_first_reading_notes': sum(r['status'] == 'review_note_or_pending_question' for r in rows),
        'question_shaped_sign_cells': sum(bool(r['question_shaped_sign_review']) for r in rows),
        'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
        'symbol_inventory': [{'symbol': c, 'codepoint': f'U+{ord(c):04X}', 'name': ud.name(c, 'UNNAMED'), 'count': n} for c, n in sorted(symbols.items())],
        'symbols_absent_from_existing_profile_graphemes': sorted(set(symbols) - profile_symbols - {' ', '/'}),
        'profile_assessment': 'Existing canonical profile is only inventoried, not changed or approved for the new literal notation. Normalized IPA remains null. A separate source-literal route or reviewed mapping is required before installation.',
        'deferred': ['root independent whole-output audit', 'integration with full-source proposal', 'approved normalization/profile routing', 'canonical installation'],
    }
    (HERE / 'notation-validation-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'symbol_inventory'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

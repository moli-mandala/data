"""Run actual source parser in a small isolated sandbox; never build CLDF."""
from pathlib import Path
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata as ud

HERE = Path(__file__).resolve().parent
OUT = HERE / 'literal-integration'
DATA = HERE.parents[5]
SANDBOX = DATA.parent / 'tmp/koya-notation/parser-sandbox'
STEMS = {'wordlist': '20260828-sil-koya', 'narrative': '20260927-sil-koya-narrative', 'comparative': '20260927-sil-koya-comparative'}


def main():
    target = SANDBOX / 'data/other/forms'
    target.mkdir(parents=True, exist_ok=True)
    (SANDBOX / 'conversion').mkdir(exist_ok=True)
    code_hashes = {}
    for name in ['make_cldf.py', 'source_meta.py', 'form_note_policy.py']:
        shutil.copyfile(DATA / name, SANDBOX / name)
        code_hashes[name] = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    for kind, stem in STEMS.items():
        for extension in ['csv', 'yaml']:
            shutil.copyfile(OUT / f'proposed-{kind}.{extension}', target / f'{stem}.{extension}')
        profile = 'sil-koya-literal' if kind == 'wordlist' else f'sil-koya-{kind}'
        shutil.copyfile(OUT / f'proposed-{profile}.txt', SANDBOX / f'conversion/{profile}.txt')
    os.chdir(SANDBOX)
    sys.path[:0] = [str(SANDBOX), str(DATA)]
    import make_cldf
    from form_note_policy import apply_form_note_policy
    meta = make_cldf.source_meta.load()
    results = {}
    all_keys = set()
    for kind, stem in STEMS.items():
        path = target / f'{stem}.csv'
        with path.open(newline='') as handle:
            rows = list(csv.reader(handle))
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(path), errors, name=stem)
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == len(rows) == stats['converted'], (kind, len(parsed), stats)
        for actual, expected in zip(parsed, rows):
            assert (actual.entry_key, actual.old_form, actual.notes, actual.source, actual.tags) == (expected[10], expected[2], expected[6], expected[7], expected[14]), (kind, expected[10])
            assert actual.ipa == '', (kind, actual.entry_key)
            assert actual.variant_of_key == expected[11], actual.entry_key
            assert make_cldf.source_entry_dedupe_key(actual, meta) == actual.entry_key, actual.entry_key
            assert actual.entry_key not in all_keys
            all_keys.add(actual.entry_key)
            public_notes, _, _, _ = apply_form_note_policy(actual.notes, actual.source, '')
            assert public_notes == expected[6], (actual.entry_key, public_notes)
            if kind == 'wordlist':
                assert ud.normalize('NFC', actual.form) == ud.normalize('NFC', expected[2]), (actual.entry_key, actual.form, expected[2])
            else:
                display = expected[2].strip('1234⁴5⁵67⁷,;.')
                display = display.replace('G', 'ɣ') if kind == 'comparative' else display.replace('[stutter]', '')
                assert ' '.join(actual.form.split()) == ' '.join(display.split()), (actual.entry_key, actual.form, display)
        results[kind] = {'rows': len(parsed), 'conversion_errors': 0, 'keys_originals_notes_citations_tags_preserved': True, 'phonemic_blank': True, 'display_contract_verified': True, 'source_key_identity_policy_verified': True}
    aliases = json.loads((OUT / 'proposed-entry-key-aliases.json').read_text())
    assert all(a['canonical_entry_key'] in all_keys and a['alias_entry_key'] not in all_keys for a in aliases)
    report = {'status': 'passed_focused_actual_parser_checks', 'results': results, 'total_active_keys': len(all_keys), 'proposed_aliases': len(aliases), 'code_sha256': code_hashes, 'scope': 'Actual parser, metadata loader and Notes policy in a source-only sandbox; no canonical, CLDF or database writes.', 'limits': 'Whole-output independent review and alias approval still pending. Literal preservation does not assert phonemic interpretation.'}
    (OUT / 'parser-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

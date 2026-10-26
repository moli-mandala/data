"""Exercise unmodified parser/metadata/notes code against all three draft files."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
D = P.parents[4]
OUT = P / 'whole-source-recovery'
SANDBOX = D.parent / 'tmp/koya-whole/parser-sandbox'
STEMS = {'wordlist': '20260828-sil-koya',
         'narrative': '20260927-sil-koya-narrative',
         'comparative': '20260927-sil-koya-comparative'}


def main():
    target = SANDBOX / 'data/other/forms'
    target.mkdir(parents=True, exist_ok=True)
    (SANDBOX / 'conversion').mkdir(exist_ok=True)
    hashes = {}
    for name in ['make_cldf.py', 'source_meta.py', 'form_note_policy.py']:
        shutil.copyfile(D / name, SANDBOX / name)
        hashes[name] = hashlib.sha256((D / name).read_bytes()).hexdigest()
    for kind, stem in STEMS.items():
        for extension in ['csv', 'yaml']:
            shutil.copyfile(OUT / f'proposed-{kind}.{extension}', target / f'{stem}.{extension}')
        source = D / 'conversion/sil-koya.txt' if kind == 'wordlist' else OUT / f'proposed-sil-koya-{kind}.txt'
        profile = 'sil-koya' if kind == 'wordlist' else f'sil-koya-{kind}'
        shutil.copyfile(source, SANDBOX / f'conversion/{profile}.txt')
    os.chdir(SANDBOX)
    sys.path[:0] = [str(SANDBOX), str(D)]
    import make_cldf
    from form_note_policy import apply_form_note_policy
    results = {}
    for kind, stem in STEMS.items():
        path = target / f'{stem}.csv'
        rows = list(csv.reader(path.open()))
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(path), errors, name=stem)
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == len(rows) == stats['converted'], (kind, len(parsed), len(rows), stats)
        for actual, expected in zip(parsed, rows):
            assert (actual.entry_key, actual.old_form, actual.notes, actual.source, actual.tags) == (
                expected[10], expected[2], expected[6], expected[7], expected[14]), actual.entry_key
            public_notes, _, _, _ = apply_form_note_policy(actual.notes, actual.source, '')
            assert public_notes == expected[6], (actual.entry_key, public_notes)
            if kind != 'wordlist':
                display = expected[2].strip('1234⁴5⁵67⁷,;.')
                display = display.replace('G', 'ɣ') if kind == 'comparative' else display.replace('[stutter]', '')
                assert ' '.join(actual.form.split()) == ' '.join(display.split()), (actual.entry_key, actual.form, display)
                assert actual.ipa == '', actual.entry_key
        results[kind] = {'rows': len(parsed), 'conversion_errors': 0,
                         'keys_originals_notes_citations_tags_preserved': True,
                         'public_notes_preserved': True}
    report = {'status': 'draft parser checks pass; independent acceptance pending',
              'results': results, 'code_sha256': hashes,
              'scope': 'Unmodified current parser, metadata loader and notes policy in a local source sandbox. No CLDF/database/canonical writes.',
              'spacing': 'One editorial stutter removal leaves adjacent spaces; compared after whitespace normalization. Original remains exact.'}
    (OUT / 'whole-draft-parser-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

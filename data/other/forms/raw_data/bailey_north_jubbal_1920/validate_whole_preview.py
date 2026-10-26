"""Exercise the actual source parser in an isolated source tree, without a build."""
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import unicodedata
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[4]
SANDBOX = DATA.parent / 'tmp/koya-whole/north-parser-sandbox'
STEM = '20260925-bailey-north-jubbal'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(preview='whole-unfrozen-preview.csv', profile='whole-preview-profile.txt', report_name='whole-preview-parser-validation.json'):
    target = SANDBOX / 'data/other/forms'
    target.mkdir(parents=True, exist_ok=True)
    (SANDBOX / 'conversion').mkdir(exist_ok=True)
    protected = [DATA / f'data/other/forms/{STEM}.csv',
                 DATA / f'data/other/forms/{STEM}.yaml',
                 DATA / 'conversion/bailey-north-jubbal-1920.txt']
    before = {str(p): sha(p) for p in protected}
    code = {}
    for name in ['make_cldf.py', 'source_meta.py']:
        shutil.copyfile(DATA/name, SANDBOX/name)
        code[name] = sha(DATA/name)
    meta = yaml.safe_load(protected[1].read_text())
    meta['defaults']['transcription'].update(input='form', preserve_hyphens=True)
    (target/f'{STEM}.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
    shutil.copyfile(HERE/profile, SANDBOX/'conversion/bailey-north-jubbal-1920.txt')
    with (HERE/preview).open(newline='') as stream:
        rows = list(csv.reader(stream))
    os.chdir(SANDBOX)
    sys.path[:0] = [str(SANDBOX), str(DATA)]
    import make_cldf
    checks = {}
    for norm in ['NFC', 'NFD']:
        inputs = [[unicodedata.normalize(norm, v) if i == 2 else v for i,v in enumerate(row)] for row in rows]
        path = target/f'{STEM}.csv'
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(inputs)
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(path), errors, name=STEM)
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == len(rows) == stats['converted']
        assert [r.entry_key for r in parsed] == [r[10] for r in inputs]
        for actual, expected in zip(parsed, inputs):
            assert (actual.old_form,actual.notes,actual.source,actual.tags) == (expected[2],expected[6],expected[7],expected[14]), actual.entry_key
            display = unicodedata.normalize('NFC', expected[2]).lower().replace('w','v').replace('ṅ','ŋ')
            assert unicodedata.normalize('NFC', actual.form) == display, (actual.entry_key, actual.form, display)
        checks[norm] = dict(rows=len(parsed), errors=0, keys_originals_notes_sources_tags_preserved=True,
                            display_matches_literal_policy=True)
    assert before == {str(p): sha(p) for p in protected}
    report = dict(status='preview actual-parser checks passed; editorial approval and full build pending',
                  checks=checks, code_sha256=code, canonical_unchanged=before,
                  preview_sha256=sha(HERE/preview),
                  profile_sha256=sha(HERE/profile),
                  transcription_settings=dict(input='form', preserve_hyphens=True),
                  method='Unmodified make_cldf/source_meta copied into a bounded isolated source tree; no compiled output or database writes.')
    (HERE/report_name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', default='whole-unfrozen-preview.csv')
    parser.add_argument('--profile', default='whole-preview-profile.txt')
    parser.add_argument('--report', default='whole-preview-parser-validation.json')
    args = parser.parse_args()
    main(args.preview, args.profile, args.report)

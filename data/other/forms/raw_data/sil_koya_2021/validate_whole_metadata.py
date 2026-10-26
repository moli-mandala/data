"""Validate draft registry/reference integration without writing canonical data."""
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
D = P.parents[4]
OUT = P / 'whole-source-recovery'
TMP = D.parent / 'tmp/koya-whole/metadata-validation'
KEY = 'devagnanavaram-et-al2021koya'
STEMS = {'wordlist': '20260828-sil-koya',
         'narrative': '20260927-sil-koya-narrative',
         'comparative': '20260927-sil-koya-comparative'}


def main():
    sys.path[:0] = [str(D), str(D / 'tests')]
    import pybtex
    import pybtex.database
    from source_meta import SourceMeta, yaml_files
    from coordinate_policy import assert_reviewed_point

    TMP.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, stem in STEMS.items():
        target = TMP / f'{stem}.yaml'
        shutil.copyfile(OUT / f'proposed-{kind}.yaml', target)
        paths.append(target)
    # Replace only this source's settings in the complete current registry.
    meta = SourceMeta([p for p in yaml_files() if p.stem not in STEMS.values()] + paths)
    dialects = {r['Tag']: r for r in csv.DictReader((D / 'cldf/dialects.csv').open())}
    languages = {r['ID'] for r in csv.DictReader((D / 'cldf/languages.csv').open())}
    seen = set()
    counts = {}
    hashes = {}
    for kind, stem in STEMS.items():
        path = OUT / f'proposed-{kind}.csv'
        rows = list(csv.reader(path.open()))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_profile = 'sil-koya' + (f'-{kind}' if kind != 'wordlist' else '')
        for row in rows:
            assert row[0] in languages
            assert row[7].startswith(KEY + '[') and row[7].endswith(']')
            assert meta.transcription(KEY, stem, row[0]) == (expected_profile, True)
            tags = [t for t in row[14].split() if t.startswith('dialect:')]
            assert len(tags) == 1, row[10]
            for tag in tags:
                dialect = dialects[tag]
                assert dialect['Language_ID'] == row[0]
                assert_reviewed_point(dialect)
                seen.add(tag)
        counts[kind] = len(rows)
    assert len(seen) == 7
    assert meta.flag(KEY, 'reference', 'ocr') is True
    assert meta.flag(KEY, 'reference', 'editor') == 'OpenAI Codex'
    entry = pybtex.database.parse_file(str(D / 'cldf/sources.bib')).entries[KEY]
    proposal = json.loads((OUT / 'bibliography-update-proposal.json').read_text())
    entry.fields.update(proposal['fields'])
    formatted = pybtex.PybtexEngine().format_from_string(
        entry.to_string('bibtex'), 'plain', output_backend='markdown')[3:].strip()
    assert formatted and '2021' in formatted
    report = {'status': 'draft registry and reference checks pass; independent audit pending',
              'rows': counts, 'dialects': sorted(seen), 'formatted_reference': formatted,
              'editor': meta.flag(KEY, 'reference', 'editor'), 'ocr': True,
              'input_sha256': hashes,
              'scope': 'Current complete source-settings registry with three proposed replacements; every draft row checked. No canonical writes or compiled-reference build.'}
    (OUT / 'whole-draft-metadata-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

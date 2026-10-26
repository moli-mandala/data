"""Read-only registered-route source validation; no compiled output writes."""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'whole-snapshot-recovery'
DATA = ROOT.parents[4]
sys.path.insert(0, str(DATA))

import make_cldf
import pybtex
import pybtex.database


def main():
    path = DATA / 'data/other/forms/20260927-birhor-living.csv'
    raw = {r[10]: r for r in csv.reader(path.open())}
    errors = io.StringIO()
    parsed, stats = make_cldf.parse_file(str(path), errors, name=path.stem)
    assert not errors.getvalue(), errors.getvalue()
    assert len(parsed) == len(raw) == 3480
    assert {r.entry_key for r in parsed} == set(raw)
    for r in parsed:
        row = raw[r.entry_key]
        assert (r.old_form, r.native, r.ipa, r.notes, r.source, r.gloss) == tuple(
            row[i] for i in (2, 4, 5, 6, 7, 3))
        assert set(row[14].split()) <= set(r.tags.split())
        assert r.lang == 'Birhor' and '�' not in r.form
    assert all(not r[11] or r[11] in raw for r in raw.values())
    languages = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/languages.csv').open())}
    assert languages['Birhor']['Glottocode'] == 'birh1242'
    bib = pybtex.database.parse_file(str(DATA / 'cldf/sources.bib'))
    used = {t.split('[', 1)[0] for r in raw.values() for t in r[7].split(';')}
    assert used <= set(bib.entries)
    engine = pybtex.PybtexEngine()
    formatted = {key: engine.format_from_string(bib.entries[key].to_string('bibtex'),
                 'plain', output_backend='markdown').strip() for key in sorted(used)}
    assert all(formatted.values())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((OUT / 'frozen-manifest-20260927.json').read_text())
    assert sha(path) == manifest['files']['proposal.csv']
    assert sha(DATA / 'conversion/birhor-living.txt') == manifest['files']['proposal-profile.txt']
    assert sha(path.with_suffix('.yaml')) == manifest['files']['20260927-birhor-living.yaml']
    result = {'status': 'passed source-stage validation; no full build/browser claim',
              'rows': len(raw), 'keys': len(raw), 'parser_stats': stats,
              'conversion_errors': errors.getvalue(), 'variant_links': sum(bool(r[11]) for r in raw.values()),
              'language': languages['Birhor'], 'formatted_references': formatted,
              'canonical_csv_sha256': sha(path),
              'frozen_manifest_sha256': sha(OUT / 'frozen-manifest-20260927.json'),
              'deferred': ['full CLDF/database build', 'compiled graph/reference checks', 'browser QA']}
    (OUT / 'post-install-validation-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

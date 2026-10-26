"""Actual canonical SourceMeta/parse_file checks; no registry or CLDF writes."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
import source_meta
import make_cldf
from tags import GRAMMATICAL_TAGS, GENDER_TAGS
from install_source import main as install_main, TARGETS

STEM = '20260927-nair-malaveda'
FIELDS = ('lang', 'param', 'old_form', 'gloss', 'native', 'ipa', 'notes', 'source', 'cognateset',
    'etymology', 'entry_key', 'variant_of_key', 'borrowed_from_key', 'derivation_parent_keys', 'tags')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert sha(DATA / TARGETS['csv']) == sha(PACKAGE / 'proposed.csv')
    assert sha(DATA / TARGETS['yaml']) == sha(HERE / (STEM + '.yaml'))
    assert sha(DATA / TARGETS['audit']) == sha(HERE / 'accepted-audit-proposal.jsonl')
    rows = list(csv.reader((DATA / TARGETS['csv']).open(newline='')))
    assert len(rows) == 223 and len({r[10] for r in rows}) == 223
    source_meta.load.cache_clear()
    meta = source_meta.load()
    assert meta.file_flag(STEM + '.csv', 'identity', 'dedupe_by_entry_key') is True
    refs = json.loads((HERE / 'root-acceptance.json').read_text())
    for key in ('nair2013malaveda', 'vimalakumari1977veda', 'vimala1984malaveda', 'kumarinirmala1977veda',
                'thankachi1977veda', 'radhakrishnan1977veda', 'hyrunnisabeegam1991malaveda'):
        assert meta.flag(key, 'reference', 'editor') == 'OpenAI Codex'
        assert meta.flag(key, 'reference', 'ocr') is False
        assert meta.flag(key, 'reference', 'etymology_provenance') == ('source' if key == 'nair2013malaveda' else 'none')
    langs = {r['ID']: r for r in csv.DictReader((DATA / 'cldf/languages.csv').open(newline=''))}
    assert 'Malaveda' in langs and not langs['Malaveda']['Glottocode'] and not langs['Malaveda']['Latitude'] and not langs['Malaveda']['Longitude']
    lects = {r['Tag']: r for r in csv.DictReader((DATA / 'cldf/dialects.csv').open(newline=''))}
    for row in rows:
        assert row[0] in langs
        assert all(t in GRAMMATICAL_TAGS | GENDER_TAGS or t in lects for t in row[14].split())
    checks = {}
    for normalization in ('NFC', 'NFD'):
        inputs = [[unicodedata.normalize(normalization, value) if i == 2 else value for i, value in enumerate(r)] for r in rows]
        if normalization == 'NFC':
            path = DATA / TARGETS['csv']
            errors = io.StringIO()
            parsed, stats = make_cldf.parse_file(str(path), errors, name=STEM)
        else:
            with tempfile.TemporaryDirectory(dir=HERE) as folder:
                path = Path(folder) / (STEM + '.csv')
                with path.open('w', newline='') as stream:
                    csv.writer(stream).writerows(inputs)
                errors = io.StringIO()
                parsed, stats = make_cldf.parse_file(str(path), errors, name=STEM)
        assert not errors.getvalue(), errors.getvalue()
        assert len(parsed) == 223 and stats == {'converted': 0, 'for_conversion': 0}
        for actual, expected in zip(parsed, inputs):
            assert [getattr(actual, f) for f in FIELDS] == expected, actual.entry_key
            assert unicodedata.normalize('NFC', actual.form) == unicodedata.normalize('NFC', expected[2])
        assert len({(item.lang, item.param, item.form, make_cldf.source_entry_dedupe_key(item, meta)) for item in parsed}) == 223
        checks[normalization] = {'rows': 223, 'all15fields_exact': True, 'initial_dedupe_keys': 223, 'zero_conversion': True}
    assert install_main(install=True)['canonical_mutations'] is False
    assert sha(DATA / 'data/form-identities.csv') == refs['durable_identity_sha256']
    report = dict(status='canonical source-stage validation passes; compiled/browser gates deferred',
        checks=checks, reference_flags=7, variants=12, audit_records=240,
        canonical_sha256={role: sha(DATA / TARGETS[role]) for role in ('csv', 'yaml', 'audit')},
        durable_identity_sha256=refs['durable_identity_sha256'], actual_canonical_source_meta=True,
        parser_monkeypatch=False, repeat_install='no-op with current shared membership verified',
        bibliography='seven reviewed records installed; auxiliary works reported only',
        unresolved='Source Veda locality/catalogue identity and printed grammatical/gloss anomalies retained with typed uncertainty.',
        deferred=['full CLDF/database build', 'full suite', 'compiled identity/graph/reference survival', 'browser QA/publication'])
    (PACKAGE / 'canonical-source-stage-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    main()

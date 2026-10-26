"""Bounded source-local checks through the actual Jambu parser and reference formatter."""
import csv
import hashlib
import io
import json
import sys
import tempfile
import unicodedata
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
import make_cldf
import source_meta
import pybtex
from pybtex.database import parse_file
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows = list(csv.reader((HERE / 'proposed.csv').open()))
audit = [json.loads(line) for line in (HERE / 'proposal-audit.jsonl').read_text().splitlines()]
assert len(rows) == 223 and all(len(r) == 15 for r in rows)
by_key = {r[10]: r for r in rows}
assert len(by_key) == 223
assert len(audit) == 240
assert sum(a['status'] == 'proposed' for a in audit) == 223
assert sum(a['status'] == 'excluded' for a in audit) == 15
assert sum(a['status'] == 'documentary_only' for a in audit) == 2
assert sum(r[2].count('\u0332') for r in rows) == 14
assert all(r[0] == 'Malaveda' and r[1] == r[4] == r[5] == '' for r in rows)
for r in rows:
    assert r[2] and unicodedata.is_normalized('NFC', r[2])
    assert all(unicodedata.category(c) != 'Co' and c != '\ufffd' for c in r[2])
    assert all(t in (GRAMMATICAL_TAGS | GENDER_TAGS) or t.startswith('dialect:Malaveda:') for t in r[14].split())
    assert r[7].startswith('nair2013malaveda[p. ') and r[10].startswith('nair2013malaveda:p')
    assert not any(r[12:14])
    if r[11]:
        assert r[11] in by_key and r[11] != r[10]
        assert by_key[r[11]][0] == r[0]
assert sum(bool(r[11]) for r in rows) == 12
assert by_key['nair2013malaveda:p41:gender-markers:w1'][2] == '– aan'
assert by_key['nair2013malaveda:p41:dative-markers:w1'][2] == '– aaru'
assert by_key['nair2013malaveda:p43:copula:w3'][2] == 'avaṛite peeru kooran enṛaata'
assert by_key['nair2013malaveda:p41:dative-free-variants:w2'][3] == 'I'
assert by_key['nair2013malaveda:p41:accusative-examples:w1'][3] == 'dog'
assert by_key['nair2013malaveda:p40:first-person-plural:w1'][14].split()[:2] == ['pron', '1pl']
assert 'uncertain' in by_key['nair2013malaveda:p41:numerals:w9'][14].split()
assert by_key['nair2013malaveda:p41:numerals:w9'][3] == 'fifty'
assert 'uncertain' in by_key['nair2013malaveda:p42:personal-terminations:w6'][14].split()
for key, expected in {
    'p41:genitive-examples:w1': {'pron', 'gen', '3pl', 'poss'},
    'p41:genitive-examples:w5': {'pron', 'gen', '3sg', 'f', 'poss'},
    'p41:genitive-examples:w4': {'pron', 'gen', '2sg', 'poss'},
    'p41:accusative-examples:w2': {'pron', 'acc', '2pl'},
    'p41:locative-examples:w7': {'pron', 'loc', '3sg', 'f'},
    'p41:genitive-examples:w2': {'noun', 'gen', 'kinship'},
    'p41:locative-examples:w2': {'noun', 'loc', 'kinship'},
}.items():
    assert expected <= set(by_key[f'nair2013malaveda:{key}'][14].split())
assert 'poss' not in by_key['nair2013malaveda:p41:locative-examples:w7'][14].split()
assert 'm' not in by_key['nair2013malaveda:p41:genitive-examples:w1'][14].split()
stem = '20260927-nair-malaveda'
fields = ('lang', 'param', 'old_form', 'gloss', 'native', 'ipa', 'notes', 'source', 'cognateset', 'etymology', 'entry_key', 'variant_of_key', 'borrowed_from_key', 'derivation_parent_keys', 'tags')
with tempfile.TemporaryDirectory(dir=HERE / 'metadata') as temp:
    settings = Path(temp) / f'{stem}.yaml'
    settings.write_bytes((HERE / 'metadata/proposed-settings.yaml').read_bytes())
    meta = source_meta.SourceMeta(paths=[settings])
    assert meta.transcription('nair2013malaveda', f'{stem}.csv', 'Malaveda') == (None, False)
    assert meta.transcription_rule('nair2013malaveda', f'{stem}.csv', 'Malaveda')['preserve_hyphens']
    assert meta.file_flag(f'{stem}.csv', 'identity', 'dedupe_by_entry_key') is True
    with patch.object(make_cldf.source_meta, 'load', return_value=meta):
        errors = io.StringIO()
        parsed, stats = make_cldf.parse_file(str(HERE / 'proposed.csv'), errors, name=stem)
    assert not errors.getvalue(), errors.getvalue()
    assert stats == {'converted': 0, 'for_conversion': 0}
    assert len(parsed) == 223
    for item, row in zip(parsed, rows):
        assert [getattr(item, field) for field in fields] == row, row[10]
        assert item.form == row[2]
bib = parse_file(HERE / 'metadata/proposed-references.bib')
assert len(bib.entries) == 7
assert 'Recounstruction' in bib.entries['kumarinirmala1977veda'].fields['note']
assert 'Morophology' in bib.entries['thankachi1977veda'].fields['note']
formatter = pybtex.PybtexEngine()
references = []
for key, entry in bib.entries.items():
    rendered = formatter.format_from_string(entry.to_string('bibtex'), 'plain', output_backend='markdown')[3:].strip()
    assert rendered and entry.fields['year'] in rendered
    references.append({'key': key, 'formatted': rendered, 'reported_only': key != 'nair2013malaveda'})
(HERE / 'metadata/formatter-review.json').write_text(json.dumps({'status': 'seven references format successfully; independent original-title review pending', 'bib_sha256': sha(HERE / 'metadata/proposed-references.bib'), 'references': references}, ensure_ascii=False, indent=2) + '\n')
report = {'status': 'source-local parser/metadata/count checks pass; independent review pending',
          'rows': 223, 'audit_records': 240, 'comparison_exclusions': 10, 'documentary_name_exclusions': 5,
          'parsed_rows': len(parsed), 'conversion_stats': stats, 'reference_count': 7,
          'csv_sha256': sha(HERE / 'proposed.csv'), 'audit_sha256': sha(HERE / 'proposal-audit.jsonl'),
          'canonical_changes': False, 'full_build': 'not run', 'browser_refresh': 'not requested'}
(HERE / 'candidate-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report))

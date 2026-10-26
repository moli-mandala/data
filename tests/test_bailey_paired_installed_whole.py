"""Checks for the accepted paired whole source, including real canonical parsing."""
import csv
import hashlib
import io
import json
import unicodedata
from pathlib import Path
import pytest
import yaml
from pybtex.database import parse_file

D = Path(__file__).resolve().parents[1]
P = D/'data/other/forms/raw_data/bailey_rampur_1920'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.reader(stream))


@pytest.mark.parametrize('name,count,legacy_count', [('rampur',647,288),('baghi',698,296)])
def test_accepted_output_identity_and_closed_variants(name,count,legacy_count):
    package = P.parent/f'bailey_{name}_1920'
    canonical = D/f'data/other/forms/20260925-bailey-{name}.csv'
    data = rows(canonical)
    assert len(data) == count and len({row[10] for row in data}) == count
    assert canonical.read_bytes() == (package/'pass4-preview.csv').read_bytes()
    old = rows(package/'legacy-before-whole-recovery.csv')
    assert len(old) == legacy_count
    assert [row[10] for row in data[:legacy_count]] == [row[10] for row in old]
    by_key = {row[10]:row for row in data}
    for row in data:
        if row[11]:
            assert row[11] in by_key and row[11] != row[10]
            assert by_key[row[11]][0] == row[0]
    assert (package/'audit.jsonl').read_bytes() == (package/'pass4-preview-audit.jsonl').read_bytes()
    assert len((package/'audit.jsonl').read_text().splitlines()) == 640


@pytest.mark.parametrize('name,count', [('rampur',647),('baghi',698)])
def test_actual_canonical_parser_and_metadata(name,count):
    import make_cldf
    from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    path = D/f'data/other/forms/20260925-bailey-{name}.csv'
    raw = rows(path)
    errors = io.StringIO()
    parsed,stats = make_cldf.parse_file(str(path),errors,name=f'20260925-bailey-{name}')
    assert not errors.getvalue()
    assert len(parsed) == stats['converted'] == count
    for item,row in zip(parsed,raw):
        assert (item.entry_key,item.old_form,item.notes,item.source,item.tags) == (row[10],row[2],row[6],row[7],row[14])
        expected = unicodedata.normalize('NFC',row[2]).lower().replace('w','v').replace('ṅ','ŋ')
        assert unicodedata.normalize('NFC',item.form) == expected
        assert set(row[14].split()) <= GRAMMATICAL_TAGS|GENDER_TAGS
    meta = yaml.safe_load(path.with_suffix('.yaml').read_text())
    assert meta['defaults']['transcription']['input'] == 'form'
    assert meta['defaults']['transcription']['preserve_hyphens'] is True
    assert meta['sources'][f'bailey1920{name}']['identity']['dedupe_by_entry_key'] is True
    assert meta['sources'][f'bailey1920{name}']['forms']['split_alternates'] is False
    assert meta['defaults']['importer']['commands'][0][0].endswith('install_reviewed_whole.py')
    reference = parse_file(str(D/'cldf/sources.bib'),'bibtex').entries[f'bailey1920{name}']
    assert f'{count} rows' in reference.fields['included']
    assert 'whole-paired-pass4-freeze' in reference.fields['provenance']


def test_source_corrections_and_complete_expressions():
    data = {r[10]:r for name in ['rampur','baghi'] for r in rows(D/f'data/other/forms/20260925-bailey-{name}.csv')}
    assert data['paired:sentence:142:16:baghi:1'][2] == 'Ĭs ănda rupaue mŏṅga ōruh'
    assert data['bailey1920rampur:p147:item:31'][2] == 'băṛo'
    assert data['bailey1920rampur:p147:item:25:answer3'][2] == 'dŭjjau'
    body = data['bailey1920rampur:p144:item:19']
    assert body[2] == 'jĕā́' and 'uncertain' in body[14].split()
    sentences = [r for key,r in data.items() if key.startswith('paired:sentence:')]
    assert len(sentences) == 45
    assert all({'sentential','multiword-expression'} <= set(r[14].split()) for r in sentences)
    assert sum(row[0] == 'kc' for row in data.values()) == 4
    assert all('uncertain' in row[14].split() for row in data.values() if row[0] == 'kc')


def test_current_audit_and_regeneration_evidence():
    frozen = P/'whole-paired-pass4-freeze-20260927.json'
    audit = json.loads((P/'root-independent-whole-output-audit-20260927-pass4.json').read_text())
    assert audit['sample_size'] == 20 and audit['material_errors'] == 0
    assert audit['freeze_sha256'] == sha(frozen)
    regeneration = json.loads((P/'pass4-regeneration-validation.json').read_text())
    assert regeneration['freeze_sha256'] == sha(frozen)
    for name, files in json.loads(frozen.read_text())['packages'].items():
        for filename,digest in files.items():
            assert sha(P.parent/f'bailey_{name}_1920'/filename) == digest

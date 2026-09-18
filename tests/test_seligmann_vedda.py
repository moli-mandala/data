import csv
import importlib.util
import io
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from make_cldf import parse_file
from segments import Tokenizer
from assign_form_ids import assign_ids
from coordinate_policy import assert_reviewed_or_blank, assert_reviewed_point

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/other/forms/raw_data'
spec = importlib.util.spec_from_file_location('vedda', RAW / 'seligmann_vedda.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

def test_complete_reproducible_source():
    rows, audit = v.emit()
    with (RAW.parent / f'{v.STEM}.csv').open() as f:
        assert list(csv.reader(f)) == rows
    assert len(rows) == len(audit) == 502
    assert len(v.load()) == 196
    assert {a['article'] for a in audit} == set(range(1,186))
    assert len({r[10] for r in rows}) == 502
    assert all(len(r) == 15 and r[0] == 'Vedda' and r[2] for r in rows)
    assert all(not r[1] and not any(r[8:10]) and not any(r[11:14]) for r in rows)
    assert all('�' not in x and unicodedata.is_normalized('NFC', x) for r in rows for x in r)
    assert all(a['pdf_page'] == a['printed_page']+168 and a['raw_ocr'] for a in audit)
    assert sum(bool(a['issues']) for a in audit) == 46

def test_transcription_corpus_and_rare_symbols():
    errors = io.StringIO()
    rows, stats = parse_file(str(RAW.parent / f'{v.STEM}.csv'), errors)
    assert stats == {'for_conversion':502, 'converted':502}
    assert not errors.getvalue()
    assert all(r.form == r.old_form and not r.ipa and not r.native for r in rows)
    tokenizer = Tokenizer(str(ROOT/'conversion/seligmann-vedda.txt'))
    for row in v.emit()[0]:
        for norm in ('NFC','NFD'):
            result = tokenizer(unicodedata.normalize(norm,row[2]),column='IPA')
            assert '�' not in result
            assert unicodedata.normalize('NFC',result.replace(' ','').replace('#',' ')) == row[2]
    by_key = {r.entry_key:r.form for r in rows}
    assert by_key[f'{v.SOURCE}:128:g1:f2'] == 'naidaṇḍa'
    assert by_key[f'{v.SOURCE}:172:g3:f1'] == 'dëula'
    assert by_key[f'{v.SOURCE}:36:g3:f1'] == 'paiga damapumu'

def test_scopes_grammar_provenance_and_no_inferred_variants():
    rows,audit=v.emit()
    by_key={r[10]:r for r in rows}
    assert by_key[f'{v.SOURCE}:17.ii:g1:f1'][3] == 'Stingless bee (Trigona sp.)'
    assert by_key[f'{v.SOURCE}:57.iii:g2:f1'][3] == 'Mouse deer (Tragulus minima)'
    assert 'verb' in by_key[f'{v.SOURCE}:65:g1:f1'][14].split()
    assert 'impv' in by_key[f'{v.SOURCE}:150:g2:f1'][14].split()
    assert all('(v.)' not in r[3] for r in rows)
    assert 'm' in by_key[f'{v.SOURCE}:116.m:g1:f1'][14].split()
    assert 'f' in by_key[f'{v.SOURCE}:116.f:g1:f1'][14].split()
    assert sum('Wannaku' in r[6] for r in rows) == sum('O' in a['source_labels'] for a in audit)
    assert all('uncertain' in a['tags'] for a in audit if 'T' in a['source_labels'])
    assert all(not any('Tamil' in t or ':T:' in t for t in a['tags']) for a in audit)

def test_registry_and_seeded_audit():
    languages=list(csv.DictReader((ROOT/'cldf/languages.csv').open()))
    assert [r['Glottocode'] for r in languages if r['ID']=='Vedda'] == ['vedd1240']
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    assert len(v.LECTS)==11
    for label in v.LECTS:
        r=dialects[v.dialect_tag(label)]
        assert r['Language_ID']=='Vedda'
        assert_reviewed_or_blank(r)
    manifest=json.loads((v.PACKAGE/'manifest.json').read_text())
    for name,expected in manifest['input_hashes'].items():
        assert hashlib.sha256((v.PACKAGE/name).read_bytes()).hexdigest()==expected
    sample=manifest['sample']
    assert sorted(random.Random(sample['seed']).sample(range(1,186),20))==sample['articles']
    assert sample['material_errors']==0

def test_stable_ids_survive_reorder_and_corrections():
    rows=v.emit()[0]
    initial=[dict(ID=f'tmp-{i}', Language_ID=r[0], Original=r[2], Form=r[2],
                  Gloss=r[3], Native=r[4], Source=r[7], Status='unlinked') for i,r in enumerate(rows)]
    keys={row['ID']:r[10] for row,r in zip(initial,rows)}
    first,registry=assign_ids(initial,[],keys)
    corrected=[dict(row,ID='new-'+row['ID'],Original='corrected',Form='corrected',Gloss='corrected') for row in reversed(initial)]
    second,_=assign_ids(corrected,registry,{r['ID']:keys[r['ID'][4:]] for r in corrected})
    assert len(set(first.values()))==502
    assert all(second['new-'+old]==opaque for old,opaque in first.items())

def test_compiled_source_survives():
    forms={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if v.SOURCE in r['Source']}
    assert len(forms)==502
    assert all(r['Language_ID']=='Vedda' and r['Status']=='unlinked' for r in forms.values())
    assert all(r['Form']==r['Original'] for r in forms.values())
    assert {r['Gloss'] for r in forms.values() if r['Form']=='dia'} == {'Tears','Water'}
    keys=[r for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith(v.SOURCE+':')]
    assert len(keys)==502
    assert {r['Source_Key'] for r in keys}=={r[10] for r in v.emit()[0]}
    assert not any(r['Child_ID'] in forms for r in csv.DictReader((ROOT/'cldf/edges.csv').open()))
    refs=[r for r in csv.DictReader((ROOT/'cldf/references.csv').open()) if r['ID']==v.SOURCE]
    assert len(refs)==1

import csv
import importlib.util
import io
import json
import unicodedata
from pathlib import Path
from make_cldf import parse_file
from segments import Tokenizer
from coordinate_policy import assert_reviewed_or_blank, assert_reviewed_point

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('muduga_import',ROOT/'data/other/forms/raw_data/muduga_2022.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def source_rows():
    lines={int(k):v for k,v in json.loads((m.RAW/'source-lines.json').read_text()).items()}
    return m.build(m.records(lines))

def test_source_counts_and_reproducibility():
    rows,audit=source_rows()
    assert len(audit)==91 and len(rows)==109
    assert list(csv.reader(m.OUT.open()))==rows
    assert len({r[10] for r in rows})==109
    assert sum(bool(r[11]) for r in rows)==18
    assert all(len(r)==15 and r[0]=='Muduga' and r[2] and m.SOURCE in r[7] for r in rows)
    assert all('�' not in v and unicodedata.is_normalized('NFC',v) for r in rows for v in r)
    keys={r[10] for r in rows}
    assert all(not r[i] or r[i] in keys for r in rows for i in (11,13))

def test_source_caveats_and_relations():
    rows,audit=source_rows();byform={r[2]:r for r in rows}
    assert byform['mɯɡa'][1]=='' and 'uncertain' in byform['mɯɡa'][14]
    assert byform['bɯːɳe'][1]=='' and 'loanword' in byform['bɯːɳe'][14]
    assert byform['ʧoːre'][1]=='' and 'dedr[2885]' in byform['ʧoːre'][7]
    assert byform['kɤːkkæ'][13]=='muduga2022:table6:b:1'
    assert byform['kɤːkkæ'][1]==''
    assert byform['kɤːɭɯ'][1]=='d2017'
    assert all(r[11] and not r[1] for r in rows if r[10].endswith(':phonetic'))
    assert {r['table'] for r in audit if r['table']}==set(range(3,20))

def test_profile_and_registry():
    e=io.StringIO();rows,stats=parse_file(str(m.OUT),e)
    assert not e.getvalue() and stats['converted']==109
    t=Tokenizer(str(ROOT/'conversion/muduga.txt'))
    def convert(s):return unicodedata.normalize('NFC',t(s,column='IPA').replace(' ','').replace('#',' '))
    assert convert('t̪otti')=='toṯṯi'
    assert convert('are')=='aṟe'
    assert convert('ɾaː')=='rā'
    assert convert('y')=='ü'
    for r in source_rows()[0]:assert '�' not in convert(r[2])
    dialects={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    assert dialects[m.DIALECT]['Language_ID']=='Muduga'
    assert_reviewed_point(dialects[m.DIALECT])

def test_compiled_source():
    keys={r['Source_Key']:r for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith('muduga2022:')}
    assert set(keys)=={r[10] for r in source_rows()[0]}
    refs={r['ID'] for r in csv.DictReader((ROOT/'cldf/references.csv').open())}
    assert m.SOURCE in refs


def test_compiled_typed_derivations_and_variants():
    aliases={r['Legacy_ID']:r['Form_ID'] for r in csv.DictReader((ROOT/'cldf/form-id-aliases.csv').open())}
    keys={r['Source_Key']:aliases[r['Legacy_ID']] for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open()) if r['Source_Key'].startswith('muduga2022:')}
    edges={(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank']) for r in csv.DictReader((ROOT/'cldf/edges.csv').open())}
    for r in source_rows()[0]:
        for col,kind in [(11,'variant'),(13,'derived')]:
            if r[col]:assert (keys[r[10]],keys[r[col]],kind,'1') in edges

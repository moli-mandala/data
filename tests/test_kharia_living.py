import csv
import importlib.util
import io
import json
from pathlib import Path
import unicodedata

from make_cldf import parse_file
from segments.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/other/forms/raw_data'
STEM = '20260911-kharia-living'
SPEC = importlib.util.spec_from_file_location('kharia_living',RAW/'kharia_living.py')
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)


def records():
    return [json.loads(s) for s in (RAW/f'{STEM}-audit.jsonl').read_text().splitlines()]


def forms():
    return list(csv.reader((ROOT/f'data/other/forms/{STEM}.csv').open()))


def test_full_count_and_offline_reproduction():
    audit, rows = records(), forms()
    assert len(audit) == 452
    assert rows == [row for record in audit for row in K.emit_record(record)]
    assert len(rows) == 521
    assert len({r[10] for r in rows}) == 521
    assert sum(bool(r[11]) for r in rows) == 68
    assert sum(not r[5] for r in rows) == 6
    assert all(len(r)==15 and r[2] and r[3] and r[4] for r in rows)
    assert all(unicodedata.is_normalized('NFC',f) and '�' not in f for r in rows for f in r)


def test_variants_optional_segments_and_prose_are_separate():
    assert K.expand_phonetic('gʰurserel ~ gʰuŋs(e)rel') == (['gʰurserel','gʰuŋserel','gʰuŋsrel'],[])
    assert K.expand_phonetic('kuluʔŭneiɲ (same as child)') == (['kuluʔŭneiɲ'],['(same as child)'])
    rows=forms();keys={r[10] for r in rows}
    assert all(not r[11] or r[11] in keys for r in rows)
    assert all(not r[1] and not r[12] and not r[13] for r in rows)
    assert all('same as' not in r[2] and '~' not in r[2] and '(' not in r[2] for r in rows)


def test_numbered_senses_and_homographs_survive():
    records_by_id={r['upstream_entry_id']:r for r in records()}
    coming=K.emit_record(records_by_id['ij7crwDASiJI4giJTKMD'])
    wing=K.emit_record(records_by_id['G0NphmdbTtsUekhPqaDy'])
    assert [r[3] for r in coming] == ['to come','he will come']
    assert wing[0][3]=='wing'
    assert len({r[10] for r in coming+wing})==3


def test_source_dialects_registered_without_invented_coordinates():
    registry={r['Tag']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
    for row in forms():
        assert row[0]=='kh'
        for tag in row[-1].split():
            if tag.startswith('dialect:'):
                assert registry[tag]['Language_ID']=='kh'
                assert not registry[tag]['Latitude'] and not registry[tag]['Longitude']


def test_profile_all_symbols_and_rich_source_layers():
    tokenizer=Tokenizer(str(ROOT/'conversion/kharia-living.txt'))
    for row in forms():
        for norm in ('NFC','NFD'):
            value=unicodedata.normalize(norm,row[2])
            out=tokenizer(value,column='IPA').replace(' ','').replace('#',' ')
            assert unicodedata.normalize('NFC',out)==row[2]
    error=io.StringIO()
    parsed,stats=parse_file(str(ROOT/f'data/other/forms/{STEM}.csv'),errors=error)
    assert not error.getvalue()
    assert len(parsed)==521
    assert stats=={'converted':521,'for_conversion':521}
    raw={r[10]:r for r in forms()}
    for r in parsed:
        assert r.form==raw[r.entry_key][2]
        assert r.old_form==raw[r.entry_key][2]
        assert r.ipa==raw[r.entry_key][5]


def test_compiled_source_identity_and_variant_graph():
    raw = {r[10]: r for r in forms()}
    aliases = {r['Legacy_ID']: r['Form_ID'] for r in csv.DictReader(
        (ROOT/'cldf/form-id-aliases.csv').open())}
    identities = {r['Source_Key']: aliases.get(r['Legacy_ID'], r['Legacy_ID'])
                  for r in csv.DictReader((ROOT/'cldf/form-source-keys.csv').open())
                  if r['Source_Key'] in raw}
    assert identities.keys() == raw.keys()
    assert len(set(identities.values())) == 521
    compiled = {r['ID']: r for r in csv.DictReader((ROOT/'cldf/forms.csv').open())
                if 'living-kharia2026' in r['Source']}
    assert set(compiled) == set(identities.values())
    for key, row in raw.items():
        built = compiled[identities[key]]
        assert built['Language_ID'] == 'kh'
        assert built['Form'] == row[2] and built['Gloss'] == row[3]
        assert built['Native'] == row[4] and built['Phonemic'] == row[5]
    actual = {(r['Child_ID'], r['Parent_ID'], r['Kind'], r['Rank'])
              for r in csv.DictReader((ROOT/'cldf/edges.csv').open())
              if r['Child_ID'] in compiled}
    expected = {(identities[key], identities[row[11]], 'variant', '1')
                for key, row in raw.items() if row[11]}
    assert len(expected) == 68
    assert actual == expected
    references = list(csv.DictReader((ROOT/'cldf/references.csv').open()))
    assert any(r['ID'] == 'living-kharia2026' for r in references)


def test_corporate_author_short_citation():
    from make_refs import create_short_ref
    from pybtex.database import parse_string
    bib = parse_string('@misc{test,author={{Living Tongues Institute}},year={2026}}', 'bibtex')
    assert create_short_ref(bib.entries['test']) == 'L2026'

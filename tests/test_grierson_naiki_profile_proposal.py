"""Exercise actual source-parser routing without building CLDF or a database."""
import csv
import io
import unicodedata as ud
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_naiki_1906'


def test_corpus_profile_and_canonical_equivalence():
    from segments import Tokenizer
    tokenizer=Tokenizer(str(P/'proposal-profile.txt'))
    for row in csv.reader((P/'whole-review-proposal.csv').open()):
        displays=[]
        for norm in ['NFC','NFD']:
            display=tokenizer(ud.normalize(norm,row[2]),column='IPA').replace(' ','').replace('#',' ')
            assert '\ufffd' not in display
            displays.append(ud.normalize('NFC',display))
        assert displays[0]==displays[1]==ud.normalize('NFC',row[2].lower().replace('ṅ','ŋ'))
    for form in ['nāhī̃','kā̃-kī̃','kaktēn anlēn','Aunnē-lōpul']:
        display=tokenizer(form,column='IPA').replace(' ','').replace('#',' ')
        assert ud.normalize('NFC',display)==ud.normalize('NFC',form.lower())


def test_actual_parser_keeps_all_source_keys_and_fields(monkeypatch):
    import make_cldf,source_meta
    from segments import Tokenizer
    metadata=source_meta.SourceMeta([P/'whole-review-proposal.yaml'])
    monkeypatch.setattr(source_meta,'load',lambda:metadata)
    monkeypatch.setitem(make_cldf.convertors,'grierson-naiki-1906',Tokenizer(str(P/'proposal-profile.txt')))
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(P/'whole-review-proposal.csv'),errors,name='whole-review-proposal')
    rows=list(csv.reader((P/'whole-review-proposal.csv').open()))
    assert len(parsed)==stats['converted']==len(rows)==694
    assert not errors.getvalue()
    assert {r.entry_key:r.old_form for r in parsed}=={r[10]:r[2] for r in rows}
    assert all(not r.ipa for r in parsed)
    assert {r.entry_key:ud.normalize('NFC',r.form) for r in parsed}=={
        r[10]:ud.normalize('NFC',r[2].strip('1234⁴5⁵67⁷,;.').lower().replace('ṅ','ŋ')) for r in rows}

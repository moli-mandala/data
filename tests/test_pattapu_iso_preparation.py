"""Reproduce primary-PDF recovery; no lexical installation or build."""
import importlib.util
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/rebbavarapu_pattapu_2013'
PDF=Path(__file__).resolve().parents[2]/'tmp/pdfs/pattapu-iso-2013/2013-020.pdf'

def test_numbered_source_cells_are_all_accounted_for():
    rows=[json.loads(x) for x in (ROOT/'font-recovered-scaffold.jsonl').read_text().splitlines()]
    assert len(rows)==210 and [r['item'] for r in rows]==list(range(1,211))
    assert {n:sum(r['pdf_page']==n for r in rows) for n in (6,7,8)}=={6:87,7:112,8:11}
    assert [r['item'] for r in rows if '\ue000' in r['raw_text']]==[81,89,121,129,142,155,178,179,189]
    assert all(r['status']=='font-recovered-pending-visual-review' for r in rows)
    assert rows[3]['raw_text']=='Face- mũᶮdʒ͡ĩ'
    assert rows[10]['raw_text']=='breast (woman’s)- ed̪uɹumaːɹu, mola'
    assert rows[86]['raw_text']=='chicken- koːli'
    assert rows[198]['raw_text']=='speak!- peːsu'
    assert all('Language Code' not in r['raw_text'] for r in rows)

def test_visual_review_accounts_for_unanswered_and_unresolved_cells():
    rows=[json.loads(x) for x in (ROOT/'visual-review.jsonl').read_text().splitlines()]
    assert [r['item'] for r in rows]==list(range(1,211))
    assert [r['item'] for r in rows if r['status']=='source-unanswered']==[73]
    assert [r['item'] for r in rows if r['status']=='unresolved-glyph']==[81,89,121,129,142,155,178,179,189]
    assert rows[68]['status']=='transcription-review'

def test_embedded_font_recovery_is_reproducible():
    if not PDF.exists():pytest.skip('Pinned external PDF not present')
    sys.path.insert(0,str(ROOT))
    spec=importlib.util.spec_from_file_location('pattapu_font',ROOT/'recover_font.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows,report=m.recover(PDF)
    assert rows==[json.loads(x) for x in (ROOT/'font-recovered-scaffold.jsonl').read_text().splitlines()]
    assert report==json.loads((ROOT/'font-recovery.json').read_text())
    assert report['repairs']['0086']=='d'
    assert report['repairs']['04C1']=='̪'
    assert report['repairs']['04F2']=='͡'
    assert report['unmapped_cids']==['1345']

def test_overlap_review_preserves_semantic_disagreements():
    rows=[json.loads(x) for x in (ROOT/'lindgren-overlap-review.jsonl').read_text().splitlines()]
    assert len(rows)==101 and len({r['existing_entry_key'] for r in rows})==101
    by_key={r['existing_entry_key']:r for r in rows}
    assert by_key['lindgren:2057']['status']=='conflicting-gloss'
    assert by_key['lindgren:2057']['primary_items']==[75]
    assert by_key['lindgren:2039']['primary_items']==[203,204]
    assert by_key['lindgren:2132']['status']=='conflicting-grammatical-analysis'
    assert all(r['decision'].startswith('retain derived transcription and cognate analysis') for r in rows)
    assert sum('annotated uncertain' in r['decision'] for r in rows)==8

def test_draft_retains_prompt_grammar_and_excludes_unresolved_readings():
    spec=importlib.util.spec_from_file_location('pattapu_importer',ROOT/'import_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit=module.build()
    assert len(rows)==201 and len(audit)==210
    assert [r['item'] for r in audit if r['status']=='withheld-transcription']==[69,81,89,121,129,142,155,178,179,189]
    assert [r['item'] for r in audit if r['status']=='excluded-unanswered']==[73]
    by_item={r['item']:r['rows'] for r in audit}
    assert [r[2] for r in by_item[11]]==['ed̪uɹumaːɹu','mola']
    assert [r[2] for r in by_item[96]]==['paːmu','paᵐbu']
    assert by_item[203][0][3]=='you' and by_item[203][0][14].split()[:-1]==['pron','2sg','informal']
    assert by_item[204][0][14].split()[:-1]==['pron','2sg','formal']
    assert by_item[208][0][14].split()[:-1]==['pron','first-person','du']
    assert by_item[207][0][14].split()[:-1]==['pron','1pl']
    assert by_item[75][0][3]=='chili' and by_item[137][0][3]=='cold things'
    assert by_item[184][0][2]=='an̪d̪aˈji pasaːjaːra'
    assert by_item[184][0][3]=='he is hungry'
    assert all(len(r)==15 and not r[11] and '\ue000' not in r[2] for r in rows)

def test_profile_preserves_unexplained_marks_and_covers_both_normalizations():
    import unicodedata
    from segments.tokenizer import Tokenizer
    converter=Tokenizer(str(ROOT/'pattapu-iso.txt'))
    spec=importlib.util.spec_from_file_location('pattapu_importer_profile',ROOT/'import_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit=module.build()
    def convert(text):
        return unicodedata.normalize('NFC',converter(text,column='IPA').replace(' ','').replace('#',' '))
    assert convert('kaˈnu')=='kaˈnu'
    assert convert('maᶮdʒ͡')=='maᶮdź͡'
    assert convert('an̪d̪aˈji pasaːjaːra')=='an̪d̪aˈyi pasāyāra'
    for row in rows:
        assert '�' not in convert(row[2])
        assert convert(unicodedata.normalize('NFD',row[2]))==convert(row[2])
    assert [r['item'] for r in audit if r.get('issues')]==[4,76,138,139]
    assert all('uncertain' in row[14] for r in audit if r.get('issues') for row in r['rows'])

def test_installed_source_files_and_registered_locality():
    import csv
    data_root=ROOT.parents[4]
    spec=importlib.util.spec_from_file_location('pattapu_installed',ROOT/'import_source.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows,audit=module.build()
    installed=list(csv.reader((ROOT.parent.parent/'20260922-pattapu-iso.csv').open()))
    assert installed==rows
    assert [json.loads(x) for x in (ROOT/'audit.jsonl').read_text().splitlines()]==audit
    assert (data_root/'conversion/pattapu-iso.txt').read_bytes()==(ROOT/'pattapu-iso.txt').read_bytes()
    dialects={r['Tag']:r for r in csv.DictReader((data_root/'cldf/dialects.csv').open())}
    assert dialects[module.DIALECT]['Language_ID']=='Pattapu'
    assert not dialects[module.DIALECT]['Latitude'] and not dialects[module.DIALECT]['Longitude']
    assert all(module.DIALECT in r[14].split() for r in rows)

def test_installed_rows_use_global_metadata_and_preserve_source_fields():
    import csv,io
    import make_cldf
    path=ROOT.parent.parent/'20260922-pattapu-iso.csv'
    errors=io.StringIO()
    parsed,stats=make_cldf.parse_file(str(path),errors,name='20260922-pattapu-iso')
    assert not errors.getvalue()
    assert len(parsed)==stats['converted']==201
    raw={r[10]:r for r in csv.reader(path.open())}
    assert {r.entry_key for r in parsed}==set(raw)
    for row in parsed:
        assert row.old_form==raw[row.entry_key][2]
        assert row.gloss==raw[row.entry_key][3]
        assert 'dialect:Pattapu:pattapu_ethamukkala:Ethamukkala' in row.tags.split()

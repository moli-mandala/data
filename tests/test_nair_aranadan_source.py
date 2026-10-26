"""Installed whole Aranadan chapter: source-stage contracts, no full build."""
import csv,io,json,unicodedata as ud
from pathlib import Path
import make_cldf,source_meta,pybtex
from pybtex.database import parse_file
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'data/other/forms/raw_data/nair_aranadan_2013'
STEM='20260927-nair-aranadan'
CSV=ROOT/f'data/other/forms/{STEM}.csv'
SOURCE='nair2013aranadan'
def rows():return list(csv.reader(CSV.open()))
def audit():return [json.loads(l) for l in (PACKAGE/'installed-audit.jsonl').read_text().splitlines()]
def test_exact_installed_assets_and_all_parser_fields():
    assert CSV.read_bytes()==(PACKAGE/'integration/review-candidate.csv').read_bytes()
    assert (PACKAGE/'installed-audit.jsonl').read_bytes()==(PACKAGE/'integration/accepted-audit.jsonl').read_bytes()
    raw=rows();assert len(raw)==len({r[10] for r in raw})==102
    errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(CSV),errors,name=STEM)
    assert not errors.getvalue();assert len(parsed)==102;assert stats=={'converted':0,'for_conversion':0}
    fields=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags')
    for p,r in zip(parsed,raw):
        assert [getattr(p,f) for f in fields]==r
        assert p.form==r[2]
        assert not any(ud.category(c)=='Co' or c=='\ufffd' for c in p.form)
        assert not p.ipa and not p.native
        assert p.source.split('[',1)[0]==SOURCE

def test_complete_physical_accounting_documentary_ethnonyms():
    a=audit();assert len(a)==len({r['source_unit_key'] for r in a})==97
    first=[json.loads(l) for l in (PACKAGE/'first-reading/complete-first-reading.jsonl').read_text().splitlines()]
    assert sum(len(r['witnesses']) for r in first)==105
    assert {r['source_unit_key'] for r in a}=={r['source_unit_key'] for r in first}
    assert all(r['accepted_for_installation'] for r in a)
    assert {k for r in a for k in r['entry_keys']}=={r[10] for r in rows()}
    names=[r for r in a if ':ethnonyms:' in r['source_unit_key']]
    assert len(names)==4 and all(not r['entry_keys'] for r in names)

def test_explicit_variants_and_shared_malayalam_suffix():
    raw=rows();keyed={r[10]:r for r in raw};links=[r for r in raw if r[11]]
    assert {(r[2],keyed[r[11]][2]) for r in links}=={('naayi','naaji'),('puye','puje')}
    assert all(not r[12] and not r[13] for r in raw)
    shared=[r for r in raw if r[0]=='Malayalam' and r[2]=='-um']
    assert len(shared)==1 and {'suffix','fut'}<=set(shared[0][14].split())
    assert sum(r[0]=='Aranadan' for r in raw)==81
    assert sum(r[0]=='Malayalam' for r in raw)==20
    assert sum(r[0]=='Kannada' for r in raw)==1

def test_literal_notation_and_source_specific_grammar():
    raw=rows();forms={r[2] for r in raw}
    assert {'ñãã','kool̲i','naal̲i','pul̲a','n̲in','n̲iinu','n̲iin naaḷe baa','niina beeṇoo','ulaakke','–kaḷ','-um'}<=forms
    assert 'niina beeṇoo' in forms and 'n̲iina beeṇoo' not in forms
    for r in raw:
        if r[2] in ('naam','yem'):assert not {'inclusive','exclusive'}&set(r[14].split())
    kure=[r for r in raw if r[2]=='kuṛe'];assert len(kure)==1 and 'adj' in kure[0][14].split()

def test_actual_metadata_and_formatted_bibliography():
    meta=source_meta.load()
    for lang in ('Aranadan','Malayalam','Kannada'):
        assert meta.transcription(SOURCE,CSV,lang)==(None,False)
        assert meta.transcription_rule(SOURCE,CSV,lang)['preserve_hyphens']
    for key,provenance in ((SOURCE,'source'),('usha1992aranadanmar','none')):
        assert meta.flag(key,'reference','editor')=='OpenAI Codex'
        assert meta.flag(key,'reference','ocr') is False
        assert meta.flag(key,'reference','etymology_provenance')==provenance
    assert meta.file_flag(CSV,'identity','dedupe_by_entry_key') is True
    assert meta.file_flag(CSV,'forms','split_alternates') is False
    assert source_meta.importer_commands(meta,STEM)
    expected=parse_file(PACKAGE/'root-review/proposed-references.bib');actual=parse_file(ROOT/'cldf/sources.bib')
    for key,entry in expected.entries.items():
        assert actual.entries[key]==entry
        assert pybtex.PybtexEngine().format_from_string(entry.to_string('bibtex'),'plain',output_backend='markdown')
    languages={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
    assert {r[0] for r in rows()}<=languages

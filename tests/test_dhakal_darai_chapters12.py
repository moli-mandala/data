"""Whole chapters1–2 source inventory; canonical source stays untouched."""
import csv,io,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/dhakal_darai_2011'
def rows():
 with (P/'chapters12-proposal.csv').open() as s:return list(csv.reader(s))
def audit():
 return [json.loads(x) for x in (P/'chapters12-proposal-audit.jsonl').read_text().splitlines()]
def test_all_pages_and_attestations_accounted():
 c=json.loads((P/'chapters12-page-census.json').read_text());a=audit();r=rows()
 assert [x['printed_page'] for x in c]==list(range(1,42))
 assert len(a)==171 and len(r)==136
 assert {x['key'] for x in a}=={k for page in c for k in page['source_units']}
 assert {x[10] for x in r}=={x['key'] for x in a if x['status']=='target'}
 assert all(x['native_evidence'] and x['pdf_page']==x['printed_page']+22 for x in a)
 assert not any(x['unresolved_glyphs'] for x in a)
def test_source_conflicts_and_lect_boundaries():
 a=audit();s=[x for x in a if x['form']=='səmdʰinin']
 assert len(s)==2 and any('father' in x['gloss'] for x in s) and any('mother' in x['gloss'] for x in s)
 n=[x for x in a if x['form']=='nai']
 assert {x['lect'] for x in n}=={'Chitwan','Tanahun'}
 assert len([x for x in a if x['printed_page']==24 and 'sentential' in x['tags']])==3
 assert all('pl' not in x['tags'] for x in a if x['form'] in ['kẽwari','pʰokra','ɡuləi'])
 assert next(x for x in a if x['form']=='puja')['phonemic_status'].startswith('unresolved')
def test_existing_profile_covers_unambiguous_ipa_rows():
 import make_cldf
 from tags import GRAMMATICAL_TAGS,GENDER_TAGS
 e=io.StringIO();p,stats=make_cldf.parse_file(str(P/'chapters12-proposal.csv'),e,name='20260921-dhakal-darai')
 assert len(p)==stats['converted']==136 and not e.getvalue()
 # Successful codepoint conversion is not approval of puja's conventional Roman interpretation.
 for x in rows():
  assert {t for t in x[14].split() if not t.startswith('dialect:')}<=GRAMMATICAL_TAGS|GENDER_TAGS
  if x[2]!='puja':assert x[5]==x[2]

def test_separate_conventional_roman_route_uses_existing_metadata_schema(monkeypatch):
    import make_cldf,source_meta
    meta=source_meta.SourceMeta([P/'20260921-dhakal-darai.yaml',P/'chapters12-ipa-proposal.yaml',P/'chapters12-roman-preservation.yaml'])
    monkeypatch.setattr(source_meta,'load',lambda:meta)
    assert list(meta.sources)==['dhakal2011darai']
    e=io.StringIO()
    roman,stats=make_cldf.parse_file(str(P/'chapters12-roman-preservation.csv'),e)
    ipa,istats=make_cldf.parse_file(str(P/'chapters12-ipa-proposal.csv'),e)
    assert not e.getvalue() and len(roman)==1 and len(ipa)==135
    assert roman[0].form==roman[0].old_form=='puja' and not roman[0].ipa
    assert stats['converted']==0 and istats['converted']==135
    assert 'uncertain' in roman[0].tags.split()

"""Full Padari staging gates; no database build or canonical installation."""
import csv,importlib.util,io,json
from pathlib import Path
from segments import Tokenizer
DATA=Path(__file__).resolve().parents[1]
P=DATA/'data/other/forms/raw_data/bailey_padari_1908'
spec=importlib.util.spec_from_file_location('padari_full_stage',P/'prepare_full.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def rows():return list(csv.reader((P/'proposal.csv').open()))
def test_whole_scope_and_keys():
 actual,audit=module.generate()
 assert actual==rows()
 assert len(audit)==753 and len(actual)==768
 assert sum(a['status']=='source-blank' for a in audit)==5
 assert {k for a in audit for k in a['entry_keys']}=={r[10] for r in actual}
 assert len({r[10] for r in actual})==768
 assert all(not r[11] or r[11] in {x[10] for x in actual} for r in actual)
 assert len({r[10] for r in csv.reader((DATA/'data/other/forms/20260925-bailey-padari.csv').open())}&{r[10] for r in actual})>=13

def test_typography_and_source_variation():
 r={r[10]:r for r in rows()}
 assert r['bailey1908padari:p83:left:item:22'][2]=='dīsū'
 assert r['bailey1908padari:part4:p34:item:49'][2]=='gīh'
 assert r['bailey1908padari:part4:p34:item:48'][2]=='paaiṇʸⁱ̆'
 assert r['bailey1908padari:p77:pronoun:1:sg:gen'][2]=='mĕe͞uṇ'
 assert r['bailey1908padari:p77:pronoun:1:sg:dat:variant:2'][2]=='maī̃'
 assert 'uncertain' not in r['bailey1908padari:p77:pronoun:1:sg:dat:variant:2'][14].split()
 assert 'uncertain' in r['bailey1908padari:p77:pronoun:1:sg:erg'][14].split()
 assert r['bailey1908padari:part4:p35:item:86'][2]=='ghōrī'
 assert r['bailey1908padari:part4:p35:item:84'][2]=='ghōṛī'

def test_profile_and_registered_tags():
 import tags
 t=Tokenizer(str(P/'proposal-profile.txt'))
 for r in rows():
  assert len(r)==15
  assert t(r[2],column='IPA').replace(' ','').replace('#',' ')==r[2]
  assert set(r[14].split())<=tags.GRAMMATICAL_TAGS|tags.GENDER_TAGS

def test_scoped_parser_with_staged_profile(monkeypatch):
 import make_cldf
 monkeypatch.setitem(make_cldf.convertors,'bailey-padari-1908',Tokenizer(str(P/'proposal-profile.txt')))
 error=io.StringIO();parsed,stats=make_cldf.parse_file(str(P/'proposal.csv'),error,name='20260925-bailey-padari')
 assert not error.getvalue(),error.getvalue()
 assert len(parsed)==stats['converted']==768
 assert {r.entry_key for r in parsed}=={r[10] for r in rows()}

def test_explicit_header_metadata():
 r={r[10]:r for r in rows()}
 for i,tag in enumerate(('demonstrative','correlative','interr','relative')*2,1):
  assert tag in r[f'bailey1908padari:p77:correlative:{i}'][14].split()
 audit=[json.loads(s) for s in (P/'proposal-audit.jsonl').read_text().splitlines()]
 for x in audit:
  if x['section'] in ('time-adverbs','place-adverbs'):
   expected='temporal' if x['section']=='time-adverbs' else 'spatial'
   assert all(expected in r[k][14].split() for k in x['entry_keys'])
 assert 'ind' in r['bailey1908padari:p80:verb:give:present'][14].split()
 assert 'ind' not in r['bailey1908padari:p80:verb:beat:present:sg:m'][14].split()
 assert not set(r['bailey1908padari:p80:verb:beat:fut:f:3'][14].split())&{'1sg','2sg','3sg','1pl','2pl','3pl'}


def test_correlative_schema_frontend_parity():
 import tags
 assert 'correlative' in tags.GRAMMATICAL_TAGS
 frontend=(DATA.parent/'jambu-static/src/lib/tags.ts').read_text()
 grammatical=frontend.split('const GRAMMATICAL',1)[-1] if 'const GRAMMATICAL' in frontend else frontend
 assert "'demonstrative', 'correlative', 'personal'" in grammatical
 assert "correlative: 'correlative'" in frontend

def test_n_u_variation_and_comparison_class():
 r={r[10]:r for r in rows()}
 assert r['bailey1908padari:part4:p33:aux:pres:1'][2]=='hauᵃ̆'
 assert r['bailey1908padari:part4:p33:aux:pres:2'][2]=='hanᵃ̆'
 assert r['bailey1908padari:p84:sentence:14'][2].endswith('hauᵃ')
 assert r['bailey1908padari:p84:sentence:12'][2].endswith('hanᵃ')
 for key in ('bailey1908padari:p77:adjective:comparison','bailey1908padari:part4:p33:comparison:better','bailey1908padari:part4:p33:comparison:best'):
  assert {'adj','degree'}<=set(r[key][14].split())

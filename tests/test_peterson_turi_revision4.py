"""Inline metadata must describe the attested unit, not its conditioning context."""
import json
import pytest
import test_peterson_turi_whole_proposal as contract
import test_peterson_turi_revision3 as revision3

@pytest.fixture
def revised(monkeypatch):
 h=contract.HERE/'revision4'
 monkeypatch.setattr(contract,'HERE',h);monkeypatch.setattr(contract,'OUT',h/'proposal')
 return {r[10]:r for r in contract.rows()}

def test_previous_classes_and_actual_parser(revised,monkeypatch):
 revision3.test_preserve_all_legacy_fields_except_additive_tags(revised)
 revision3.test_excerpt_and_documented_ipl_analysis(revised)
 revision3.test_all_revision2_regressions(revised)
 revision3.test_actual_profiles_and_legacy_parser(revised,monkeypatch)

def test_all_inline_units_have_scoped_decisions(revised):
 emitted=[a for a in contract.audit() if a['section']=='inline' and a['entry_keys']]
 assert len(emitted)==101
 for a in emitted:
  u=a['evidence'];assert u['structured_metadata_basis']
  for key in a['entry_keys']:
   actual={t for t in revised[key][14].split() if not t.startswith('dialect:')}
   assert set(u['structured_tags'])<=actual
   assert actual-set(u['structured_tags']) <= {'uncertain','loanword','multiword-expression'}

def test_context_does_not_become_marker_features(revised):
 for tail in ['p271:inline:finite-raised','p271:inline:finite-surface','p273:inline:negative-surface','p273:inline:negative-underlying']:
  ts=set(revised['peterson2024turi:'+tail][14].split())
  assert not ts & {'first-person','second-person','third-person','sg','pl','animate','subj','obj','pron'}
 assert 'negator' in revised['peterson2024turi:p273:inline:negative-surface'][14].split()
 assert 'affix' not in revised['peterson2024turi:p273:inline:negative-surface'][14].split()

def test_explicit_analyses_and_index_scope(revised):
 checks={'p270:inline:to-monkeys':{'pl','obl'},'p270:inline:of-children':{'pl','gen'},'p272:inline:jashpur-fruitful':{'verb','middle','pret'},'p271:inline:third-index-1':{'3sg','obj'},'p271:inline:third-index-2':{'3sg','subj'},'p271:inline:third-index-jh':{'3sg','obj','subj'},'p271:inline:jh-object':{'3sg','obj','animate'},'p274:inline:recipient-index':{'obj','animate'},'p276:inline:inanimate-index':{'3sg','obj','inanimate'}}
 for tail,tags in checks.items():assert tags<=set(revised['peterson2024turi:'+tail][14].split()),tail
 assert 'monkey=pl=obl' in revised['peterson2024turi:p270:inline:to-monkeys'][6]
 for a in contract.audit():
  u=a['evidence']
  if a['section']=='tables' and u.get('table')==5 and a['entry_keys']:
   tail=u['source_unit_key'].rsplit(':',1)[1]
   want={'obj'} if tail=='3sg-obj' else {'subj'} if tail=='3sg-subj' else {'subj','obj'}
   assert want<=set(revised[a['entry_keys'][0]][14].split())

def test_copula_stems_full_forms_and_enclitics(revised):
 for tail in ['p275:inline:loc-copula','p275:inline:ranchi-past-cop','p275:inline:ranchi-present-cop','p276:inline:negative-copula']:
  ts=set(revised['peterson2024turi:'+tail][14].split())
  assert {'verb','copula'}<=ts and 'affix' not in ts
 assert {'affix','copula'}<=set(revised['peterson2024turi:p275:inline:ident-copula'][14].split())

def test_prior_three_freezes_unchanged(revised):
 import hashlib
 for h in [contract.HERE.parent,contract.HERE.parent/'revision2',contract.HERE.parent/'revision3']:
  for n,digest in json.loads((h/'frozen-manifest.json').read_text())['files'].items():
   assert hashlib.sha256((h/n).read_bytes()).hexdigest()==digest,n

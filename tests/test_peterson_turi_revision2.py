"""Metadata recovery regressions plus the actual source-route contract."""
import csv,json
import pytest
import test_peterson_turi_whole_proposal as contract

@pytest.fixture
def revised(monkeypatch):
 root=contract.HERE/'revision2'
 monkeypatch.setattr(contract,'HERE',root)
 monkeypatch.setattr(contract,'OUT',root/'proposal')
 return {r[10]:r for r in contract.rows()}

def test_revision2_scope_and_identity(revised):
 contract.test_existing_keys_and_complete_numbered_scope()
 contract.test_all_alignment_and_reuse_targets_preserve_literal_forms()
 contract.test_control_and_lect_boundaries_and_real_variant_links()
 contract.test_literal_analysis_conflicts_and_tags_are_visible()

def test_revision2_actual_parser(revised,monkeypatch):
 contract.test_actual_parser_with_declared_profiles_preserves_original_and_identity(monkeypatch)

def test_contextual_numerals_and_pronouns(revised):
 for a in contract.audit():
  for w in a.get('word_alignment',[]):
   r=revised[w['entry_key']]
   if w['printed_gloss'].split('=')[0] in {'one','two','three','four','five'}:
    assert 'num' in r[14].split(),w
   if w['printed_gloss'] in {'what','which'}:assert {'interr','pron'} <= set(r[14].split())

def test_readable_glosses_and_exact_printed_analysis(revised):
 assert not any(r[3] in {'cop','rdp'} or 'appl.pst' in r[3] or '=du' in r[3] for r in revised.values())
 for a in contract.audit():
  for w in a.get('word_alignment',[]):
   if w['printed_gloss'].startswith('rdp-go'):
    r=revised[w['entry_key']]
    assert r[3]=='go' and 'reduplicated' in r[14].split()
 r=revised['peterson2024turi:p276:example:29:word5']
 assert r[3]=='copula' and 'cop=pst.mid=fin=du' in r[6]

def test_localized_uncertainty_and_epenthesis(revised):
 prefix='peterson2024turi:p276:example:29:'
 for suffix in ['whole','word2']:assert 'uncertain' in revised[prefix+suffix][14].split()
 for suffix in ['word1','word3','word4','word5']:assert 'uncertain' not in revised[prefix+suffix][14].split()
 r=revised['peterson2024turi:p290:inline:hiatus']
 assert 'epenthetic' in r[14].split() and 'affix' not in r[14].split()
 assert not any(t.startswith('dialect:') for t in r[14].split())
 assert 'Jashpur' in r[6] and 'Sarangarh' in r[6]

def test_original_failed_freeze_unchanged(revised):
 import hashlib
 h=contract.HERE.parent
 manifest=json.loads((h/'frozen-manifest.json').read_text())
 for name,digest in manifest['files'].items():
  assert hashlib.sha256((h/name).read_bytes()).hexdigest()==digest,name

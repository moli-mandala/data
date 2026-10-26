"""Whole-source metadata includes the previously installed Appendix1 rows."""
import csv,json,io
import pytest
import test_peterson_turi_whole_proposal as contract
import test_peterson_turi_revision2 as revision2

@pytest.fixture
def revised(monkeypatch):
 h=contract.HERE/'revision3'
 monkeypatch.setattr(contract,'HERE',h);monkeypatch.setattr(contract,'OUT',h/'proposal')
 return {r[10]:r for r in contract.rows()}

def test_preserve_all_legacy_fields_except_additive_tags(revised):
 old=list(csv.reader((contract.ROOT/'data/other/forms/20260921-peterson-turi.csv').open()))
 assert len(old)==246
 for r in old:
  assert revised[r[10]][:14]==r[:14]
  assert set(r[14].split())<=set(revised[r[10]][14].split())
 for n in range(179,184):
  rs=[r for k,r in revised.items() if f':p295:item{n}:' in k]
  assert rs and all('num' in r[14].split() for r in rs)
 for n in range(227,235):
  rs=[r for k,r in revised.items() if f':p296:item{n}:' in k]
  assert rs and all('pron' in r[14].split() for r in rs)
 assert all(not {'m','f'} & set(r[14].split()) for k,r in revised.items() if ':p296:item' in k)

def test_excerpt_and_documented_ipl_analysis(revised):
 for a in contract.audit():
  u=a['evidence']
  if a['section']=='numbered' and '…' in u['raw_surface']:
   r=revised[u['source_unit_key']+':whole']
   assert 'sentential' not in r[14].split() and 'multiword-expression' in r[14].split()
 r=revised['peterson2024turi:p298:example:45:word4']
 assert {'first-person','pl','1pl','inclusive','uncertain'}<=set(r[14].split())
 assert 'ipl.incl' in r[6]

def test_all_revision2_regressions(revised):
 revision2.test_contextual_numerals_and_pronouns(revised)
 revision2.test_readable_glosses_and_exact_printed_analysis(revised)
 revision2.test_localized_uncertainty_and_epenthesis(revised)
 contract.test_all_alignment_and_reuse_targets_preserve_literal_forms()
 contract.test_control_and_lect_boundaries_and_real_variant_links()
 contract.test_literal_analysis_conflicts_and_tags_are_visible()

def test_actual_profiles_and_legacy_parser(revised,monkeypatch):
 contract.test_actual_parser_with_declared_profiles_preserves_original_and_identity(monkeypatch)
 p=contract.OUT/'20260921-peterson-turi.csv'
 errors=io.StringIO()
 parsed,stats=contract.make_cldf.parse_file(str(p),errors=errors,name=p.stem)
 assert not errors.getvalue()
 assert len(parsed)==246
 for r in parsed:
  raw=revised[r.entry_key]
  assert r.old_form==raw[2] and r.ipa==raw[5]
  assert set(raw[14].split())<=set(r.tags.split())

def test_prior_frozen_versions_unchanged(revised):
 import hashlib
 for h in [contract.HERE.parent,contract.HERE.parent/'revision2']:
  manifest=json.loads((h/'frozen-manifest.json').read_text())
  for n,digest in manifest['files'].items():assert hashlib.sha256((h/n).read_bytes()).hexdigest()==digest,n

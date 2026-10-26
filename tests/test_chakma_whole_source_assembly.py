"""Whole-book preview coverage and strict proposal/acceptance separation."""
import csv
import importlib.util
import json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/niranjan_chakma_2010'
spec=importlib.util.spec_from_file_location('chakma_whole_assembly',P/'assemble_whole_source_preview.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
@pytest.fixture(scope='module')
def assembly():return m.build(canonical=P/'whole-installation/legacy-742.csv')

def test_complete_section_keys_and_legacy_fields(assembly):
 rows,status=assembly;by={r['key']:r for r in rows}
 assert len(by)==len(rows)
 assert len(rows)==5232
 for section,(name,count) in m.SECTIONS.items():
  source=[json.loads(x) for x in (P/name).read_text().splitlines()]
  assert len(source)==count
  assert {r['source_unit_key'] for r in source} <= set(by)
  assert status['sections'][section]['raw_units']==count
 canonical=list(csv.reader((P/'whole-installation/legacy-742.csv').open()))
 assert len(canonical)==742
 assert {r[10] for r in canonical}=={r['Entry_Key'] for r in rows if 'Entry_Key' in r}
 for old in canonical:assert by[old[10]]['canonical_fields_unchanged']==old
 assert all(r['accepted_for_installation'] is False for r in rows)
 assert status['canonical_changes'] is False

def test_only_explicit_root_deltas_change_preview(assembly):
 rows,status=assembly
 assert status['root_glyph_corrections_applied']==157
 assert status['root_comparison_fields_accepted']==36
 for r in rows:
  if r['preview_reading']!=r['source_form']:
   a=r['accepted_root_glyph_delta'];assert a['old']==r['source_form'] and a['new']==r['preview_reading']
 by={r['key']:r for r in rows}
 # Independently proposed grammar correction remains pending, not adopted.
 chicken=by['niranjan2010:p10:grammar:b1:feminine:5']
 assert chicken['preview_reading']=='কুর্ঘী কুরাহ্'
 assert chicken['proposed_fields_pending']['form']=='কুহরী কুরাহ্'
 assert chicken['typed_independent_uncertainties']
 # Withdrawn root h approval must not become a changed reading.
 h=by['niranjan2010chakma:p92:brokskat-comparison:row11:chakma']
 assert h['preview_reading']=='গোহর'
 assert h['proposed_fields_pending']['form']=='গোহ্‌র'


def test_controls_and_unresolved_attribution_are_not_dropped(assembly):
 rows,status=assembly;by={r['key']:r for r in rows}
 groups=[json.loads(x) for x in (P/'numbered-structured-review.jsonl').read_text().splitlines()]
 comps=[c for g in groups for c in g['comparisons']]
 assert len(comps)==395 and all(c['key'] in by for c in comps)
 assert sum(by[c['key']]['disposition']=='blank' for c in comps)==2
 assert by['niranjan2010chakma:numbered-comparison:group48']['disposition']=='headless commentary group'
 assert len([r for r in rows if r['section']=='abbreviations'])==88
 assert sum(r.get('bengali_control_disposition')=='explicit blank' for r in rows)==2
 t=by['niranjanchakma2010:p62:homophones:right:g62:c1']
 assert t['language_proposal']=='Tanchangya' and 'registered' in t['disposition']
 assert status['sections']['documentary']['source_metadata_rows']==94
 assert len([r for r in rows if r['section']=='frontmatter' and r['original_review_proof']])==7
 assert status['root_metadata_fields_accepted']==347
 assert all(r['disposition']=='documentary context; no target lexical attestation' for r in rows if r['section']=='frontmatter')


def test_marked_ha_keeps_mandatory_uncertainty_and_language_scope(assembly):
 rows,status=assembly;by={r['key']:r for r in rows}
 approval=json.loads((P/'font-control-root-adjudication-20260927.json').read_text())
 for d in approval['decisions']:
  r=by[d['source_unit_key']]
  assert r['preview_reading']==d['accepted_value']
  assert 'uncertain' in r['required_tags']
  assert any(u['type']=='marked-ha-encoding' for u in r['typed_root_uncertainties'])
 for c in approval['candidate_uncertainty_evidence']:
  assert c in by[c['source_unit_key']]['font_control_evidence']
 numbered=[r for r in rows if r.get('root_identity_decision')]
 assert len(numbered)==395
 assert sum(r['disposition']=='control comparator outside requested families' for r in numbered)==37
 assert sum(r.get('accepted_language_id') is not None and r['root_identity_decision']['disposition']=='accepted existing target-family base' for r in numbered)==209
 one=by['niranjan2010chakma:p106:basic:left:row11']
 assert one['disposition']=='audit-only unresolved glyph' and not one['lexical_emission_permitted']
 dog=by['niranjan2010chakma:p106:basic:left:row21']
 assert dog['disposition']=='eligible target with typed uncertainty'
 assert 'uncertain' in dog['required_tags']

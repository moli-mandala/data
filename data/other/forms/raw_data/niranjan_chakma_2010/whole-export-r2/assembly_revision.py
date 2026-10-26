"""Exact two-field identity addendum; preserve original assembly and its freeze."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import assemble_whole_source_preview as original
sha=original.sha
CANONICAL=ROOT/'whole-installation/legacy-742.csv'
def build():
 rows,status=original.build(canonical=CANONICAL)
 path=ROOT/'further-identity-root-adjudication-20260927.json';overlay=json.loads(path.read_text())
 for name,h in overlay['input_sha256'].items():assert sha(ROOT/name)==h,name
 by={r['key']:r for r in rows}
 for d in overlay['decisions']:
  r=by[d['source_unit_key']]
  assert r['language_proposal']==d['old_language_id']
  assert r['source_form']==d['source_form'] and r['source_language_label']==d['source_label']
  r['language_proposal']=r['accepted_language_id']=d['accepted_language_id']
  r['required_identity_source_note']=d['required_source_note']
  r['disposition']='eligible comparator; root-approved generic source language/stage'
  r['root_further_identity_decision']=d
 assert len(overlay['decisions'])==2
 status['inputs'][path.name]=sha(path)
 status['identity_addendum']={'accepted_decisions':2,'no_other_source_field_changes':True,'script_sha256':sha(Path(__file__))}
 return rows,status

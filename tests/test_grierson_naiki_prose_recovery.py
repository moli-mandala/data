"""Source-local Naiki correction-layer contract, without a compiled build."""
import csv,hashlib,importlib.util,json,shutil
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_naiki_1906'
def jl(name):return [json.loads(s) for s in (HERE/name).read_text().splitlines()]
def test_reviewed_literals_and_immutable_inputs():
 review=json.loads((HERE/'independent-prose-review.json').read_text())
 for name,digest in review['input_sha256'].items():
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
 original={r['source_unit_key']:r for name in review['input_sha256'] for r in jl(name)}
 updated={r['source_unit_key']:r for name in ['prose-reviewed.jsonl','pronoun-table-reviewed.jsonl'] for r in jl(name)}
 assert len(original)==94 and len(updated)==95
 changes={(r['source_unit_key'],r['field']):r for r in review['corrections']}
 for key,before in original.items():
  for field in ['raw_form','raw_form_group','raw_gloss','raw_gloss_or_analysis']:
   if field in before:
    expected=changes.get((key,field),{}).get('after',before[field])
    assert updated[key][field]==expected
 assert updated['grierson1906naiki:p570:pronoun-table:col2:row2']['raw_form']=='inun'
 assert updated['grierson1906naiki:p571:prose:done-phrase']['raw_form_group']=='kaktēn anlēn'
def test_bound_plural_witness_and_reconciled_counts():
 rows=list(csv.reader((HERE/'whole-review-proposal.csv').open()))
 bykey={r[10]:r for r in rows};audit=jl('whole-review-proposal-audit.jsonl')
 assert len(rows)==len(bykey)==694 and len(audit)==699
 assert sum(r['disposition']=='exclude-control' for r in audit)==5
 key='grierson1906naiki:p571:prose:plural-personal-ending'
 row=bykey[key]
 assert row[2]=='r' and row[3]=='' and row[5]==''
 assert set(row[14].split())=={'suffix','pl','dialect:Naiki:naiki_chanda_lsi1906:Chanda'}
 assert 'plural personal ending' in row[6] and 'singular translations' in row[6]
 assert row[7]=='grierson1906naiki[p. 571]'
 assert bykey['grierson1906naiki:p571:prose:done-phrase'][2]=='kaktēn anlēn'
 assert 'uncertain' not in bykey['grierson1906naiki:p571:prose:done-phrase'][14].split()
 # The independent correction applies only to this occurrence, never globally.
 assert bykey['grierson1906naiki:p570:pronoun-table:col2:row1'][2]=='īn'
 assert bykey['grierson1906naiki:p570:pronoun-table:col2:row4'][2]=='īm'
def test_stale_review_cannot_reconcile(tmp_path):
 spec=importlib.util.spec_from_file_location('naiki_reconcile',HERE/'reconcile_prose.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for name in ['independent-prose-review.json','prose-first-reading.jsonl','pronoun-table-first-reading.jsonl']:
  shutil.copyfile(HERE/name,tmp_path/name)
 with (tmp_path/'prose-first-reading.jsonl').open('a') as f:f.write('\n')
 with pytest.raises(AssertionError,match='Stale independent review'):
  module.reconcile(tmp_path)

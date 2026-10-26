"""Notes-only source reconciliation: actual parser, immutable r2 and repeatability."""
import csv,hashlib,importlib.util,io,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'data/other/forms/raw_data/grierson_naiki_1906'
O=P/'metadata-addendum'
def test_exact_notes_only_change_and_partial_components():
 old=list(csv.reader((P/'whole-review-proposal.csv').open()));new=list(csv.reader((O/'whole-review-proposal.csv').open()))
 assert len(old)==len(new)==694
 assert sum(a[6]!=b[6] for a,b in zip(old,new))==30
 assert all(all(a[i]==b[i] for i in range(15) if i!=6) for a,b in zip(old,new))
 audit=list(map(json.loads,(O/'whole-review-proposal-audit.jsonl').read_text().splitlines()))
 assert len(audit)==699
 assert sum(len(a.get('partial_affix_component_claims',[])) for a in audit)==7
 assert all(not r[11] and not r[12] and not r[13] for r in new)
 changes=json.loads((O/'receipt.json').read_text())['changed_notes']
 assert len(changes)==30 and all(c['source_note'] in c['after'] for c in changes)
def test_actual_parser_all_694_notes_and_fields(monkeypatch):
 import make_cldf,source_meta
 from segments import Tokenizer
 meta=source_meta.SourceMeta([O/'whole-review-proposal.yaml']);monkeypatch.setattr(source_meta,'load',lambda:meta)
 monkeypatch.setitem(make_cldf.convertors,'grierson-naiki-1906',Tokenizer(str(P/'proposal-profile.txt')))
 err=io.StringIO();parsed,_=make_cldf.parse_file(str(O/'whole-review-proposal.csv'),err,name='whole-review-proposal')
 raw={r[10]:r for r in csv.reader((O/'whole-review-proposal.csv').open())}
 assert len(parsed)==len(raw)==694 and not err.getvalue()
 for row in parsed:
  x=raw[row.entry_key];assert row.old_form==x[2] and row.native==x[4] and row.ipa==x[5]
  assert set(x[14].split())<=set(row.tags.split())
  assert '\ufffd' not in row.form
  assert x[6] in row.notes

def test_byte_identical_reproduction_and_frozen_r2():
 names=['whole-review-proposal.csv','whole-review-proposal-audit.jsonl','whole-review-proposal.yaml','receipt.json']
 before={n:(O/n).read_bytes() for n in names}
 spec=importlib.util.spec_from_file_location('naiki_notes',P/'prepare_notes_addendum.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
 assert all((O/n).read_bytes()==b for n,b in before.items())
 frozen=json.loads((P/'review-manifest-r2.json').read_text())
 assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in frozen['sha256'].items())

def test_install_guard_rejects_unexpected_existing_canonical(monkeypatch):
 import pytest
 spec=importlib.util.spec_from_file_location('naiki_install_guard',P/'import_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class Target:
  def __init__(self,exists,hash_value):self.present=exists;self.hash_value=hash_value
  def exists(self):return self.present
 monkeypatch.setattr(m,'sha',lambda target:target.hash_value)
 m.guard_canonical_target(Target(False,None),'approved')
 m.guard_canonical_target(Target(True,'approved'),'approved')
 m.guard_canonical_target(Target(True,'baseline'),'approved','baseline')
 with pytest.raises(AssertionError,match='unexpected canonical'):
  m.guard_canonical_target(Target(True,'user-edited'),'approved','baseline')

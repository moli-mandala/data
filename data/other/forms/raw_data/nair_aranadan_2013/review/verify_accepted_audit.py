"""Independent comparison of accepted derivative against immutable review audit."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json
H=Path(__file__).resolve().parent;P=H.parent;I=P/'integration'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 old=jl(I/'review-audit.jsonl');new=jl(I/'accepted-audit.jsonl');status=json.loads((I/'accepted-audit-status.json').read_text());approval=P/'root-review/source-acceptance.json';a=json.loads(approval.read_text())
 assert a['approved'] is True and a['result']=='pass'
 for name,pin in a['files'].items():assert sha(P/name)==pin
 assert len(old)==len(new)==97
 preserve_excluding={'accepted_for_installation','pending','disposition','review_candidate_keys'}
 for before,after in zip(old,new):
  assert {k:v for k,v in before.items() if k not in preserve_excluding}=={k:v for k,v in after.items() if k not in {'entry_keys','accepted_for_installation','disposition','acceptance','historical_review'}}
  assert after['entry_keys']==before['review_candidate_keys']
  assert after['accepted_for_installation'] is True
  assert after['disposition']==('emitted' if after['entry_keys'] else 'documentary-excluded')
  assert after['acceptance']['sha256']==sha(approval) and after['acceptance']['result']=='pass'
  assert after['historical_review']['sha256']==sha(I/'review-audit.jsonl')
  assert after['historical_review']['prior_disposition']==before['disposition']
  assert 'pending' not in after and 'review_candidate_keys' not in after
 rows=list(csv.reader((I/'review-candidate.csv').open()));mapped=[k for r in new for k in r['entry_keys']]
 assert len(rows)==len(mapped)==len(set(mapped))==102 and set(mapped)=={r[10] for r in rows}
 assert sum(r['physical_witness_slots'] for r in new)==105
 empty=[r for r in new if not r['entry_keys']];assert len(empty)==16
 assert sum(r['source_record']['section']=='ethnonyms' for r in empty)==4
 assert sum(not r['source_record']['witnesses'] for r in empty)==12
 assert status['accepted_audit_sha256']==sha(I/'accepted-audit.jsonl') and status['accepted_csv_sha256']==sha(I/'review-candidate.csv')
 assert status['documentary_units']==16 and status['all_emitted_keys_closed'] is True
 assert Counter(r[0] for r in rows)==Counter(Aranadan=81,Malayalam=20,Kannada=1)
 bib=(P/'root-review/proposed-references.bib').read_text()
 importer=(P/'import_source.py').read_text()
 assert 'installed-audit.jsonl' in bib and "'audit':'integration/accepted-audit.jsonl'" in importer and "HERE/'installed-audit.jsonl':(HERE/INPUTS['audit']).read_bytes()" in importer
 provenance={'proposed_reference_names_canonical_installed_audit':True,'installer_copies_exact_accepted_derivative':True,'canonical_not_installed_by_reviewer':True}
 report=dict(status='passed',reviewer='asur_finish',counts=dict(rows=102,units=97,physical_witnesses=105,documentary_ethnonyms=4,documentary_context=12),raw_record_and_context_exact=True,historical_flags_preserved_inside_source_record=True,top_level_acceptance_separate=True,all_entry_keys_exact=True,source_audit_provenance=provenance,hashes={p.name:sha(p) for p in [I/'prepare_accepted_audit.py',I/'review-audit.jsonl',I/'accepted-audit.jsonl',I/'accepted-audit-status.json',approval]},scope='Source-stage audit derivative only; canonical installation and compiled/browser gates not asserted.')
 (H/'accepted-audit-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Apply approved, source-keyed scholarly Notes without changing frozen r2."""
import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'metadata-addendum'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 manifest=json.loads((HERE/'review-manifest-r2.json').read_text())
 for name,h in manifest['sha256'].items():assert sha(HERE/name)==h,name
 review=json.loads((HERE/'independent-r2-original-review.json').read_text())
 assert review['result']=='pass' and review['material_errors_in_sample']==0
 proposal=json.loads((HERE/'independent-morphology-notes-proposal-20260927.json').read_text())
 baseline=HERE/'whole-review-proposal.csv'
 assert sha(baseline)==proposal['input_csv_sha256']
 before=list(csv.reader(baseline.open()));rows=[r[:] for r in before]
 bykey={r[10]:r for r in rows};changes=[]
 for item in proposal['notes_additions']:
  row=bykey[item['entry_key']]
  assert row[2]==item['form'] and row[6]==item['existing_notes']
  old=row[6];row[6]=' '.join(s for s in [old,item['proposed_addition']] if s)
  changes.append(dict(entry_key=row[10],before=old,after=row[6],source_note=item['proposed_addition'],citation=item['citation']))
 assert len(changes)==30
 for a,b in zip(before,rows):assert all(a[i]==b[i] for i in range(15) if i!=6)
 audit=[json.loads(x) for x in (HERE/'whole-review-proposal-audit.jsonl').read_text().splitlines()]
 corrections={r['entry_key']:r for r in changes}
 for unit in audit:
  additions=[corrections[k] for k in unit['emitted_keys'] if k in corrections]
  if additions:unit['source_morphology_notes_addendum']=additions
  components=[c for c in proposal['affix_only_component_candidates_for_separate_review'] if c['child_key'] in unit['emitted_keys']]
  if components:unit['partial_affix_component_claims']=[dict(c,graph_disposition='No edge: current Derivation_Parent_Keys represents generic derivation, not this incomplete affix-only component analysis.') for c in components]
 OUT.mkdir(exist_ok=True)
 with (OUT/'whole-review-proposal.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
 (OUT/'whole-review-proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
 (OUT/'whole-review-proposal.yaml').write_bytes((HERE/'whole-review-proposal.yaml').read_bytes())
 receipt={'status':'Notes-only addendum; independent owner acceptance and installation approval pending','base_revision':'r2','rows':len(rows),'audit_units':len(audit),'changed_notes':changes,'non_notes_csv_fields_identical':True,'graph_fields_unchanged_blank':all(not r[11] and not r[12] and not r[13] for r in rows),'partial_affix_components_retained':7,'base_manifest_sha256':sha(HERE/'review-manifest-r2.json'),'base_csv_sha256':sha(baseline),'notes_proposal_sha256':sha(HERE/'independent-morphology-notes-proposal-20260927.json'),'outputs':{n:sha(OUT/n) for n in ['whole-review-proposal.csv','whole-review-proposal-audit.jsonl','whole-review-proposal.yaml']},'deferred':['Independent owner review of 30 Notes changes','Canonical installation and metadata registration','Full compiled/build/browser gates']}
 (OUT/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps(receipt['outputs']))
if __name__=='__main__':main()

"""Reproduce a fresh independent audit selection from a frozen whole-source proposal."""
import argparse,csv,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent
FILES=['proposal.csv','proposal-audit.jsonl','proposal-profile.txt','grammar-reviewed.jsonl','specimens-reviewed.jsonl','table-reviewed.jsonl','source-wide-attestations.jsonl']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sample(seed,size=20):
 freeze=json.loads((P/'full-stage-freeze-20260926.json').read_text())
 for f,h in freeze['hashes'].items():assert sha(P/f)==h,(f,'frozen input changed')
 audit=[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()];rows={r[10]:r for r in csv.reader((P/'proposal.csv').open())}
 exclusions=set()
 historical=P/'sample-audit-20260925-full.json'
 if historical.exists():
  for r in json.loads(historical.read_text()).get('rows',[]):
   if 'prompt' in r:exclusions.add(f"grierson1906lsi4:korvi_belgaum:{r['prompt']}")
 for f in sorted(P.glob('independent-full-audit-*.json')):
  d=json.loads(f.read_text())
  for r in d.get('rows',d.get('sample',[])):
   if isinstance(r,dict) and r.get('source_unit_key'):exclusions.add(r['source_unit_key'])
 candidates=[u for u in audit if u['entry_keys'] and u['source_unit_key'] not in exclusions]
 selected=random.Random(seed).sample(sorted(candidates,key=lambda u:u['source_unit_key']),size)
 records=[]
 for u in selected:
  path='tmp/korvi-full-scope/alternate-leaf708.jpg' if u['section']=='later-erratum' else f"tmp/korvi-full-scope/source-page{u['djvu_page']}.jpg"
  records.append({'source_unit_key':u['source_unit_key'],'unit':u,'proposed_rows':[rows[k] for k in u['entry_keys']],'original_path':path})
 return {'status':'selected_not_audited','seed':seed,'sample_size':size,'excluded_source_unit_keys':sorted(exclusions),'frozen_hashes':freeze['hashes'],'rows':records}
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',required=True,type=int);parser.add_argument('--output',required=True);a=parser.parse_args();target=Path(a.output).resolve();assert P.resolve() in target.parents,'Write sampler output inside source package'
 target.write_text(json.dumps(sample(a.seed),ensure_ascii=False,indent=2)+'\n');print(target)

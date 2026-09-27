"""Seeded independent whole-source audit with explicit historical exclusions."""
import argparse,csv,hashlib,json,random
from pathlib import Path
P=Path(__file__).resolve().parent
FILES=['proposal.yaml','proposal.csv','proposal-audit.jsonl','proposal-profile.txt','full_inventory.tsv','reviewed_inventory.tsv','expression-recovery-p154-161-reviewed.jsonl','expression-recovery-p162-169-first-reading.jsonl','expression-recovery-song-first-reading.jsonl','held-recovery-first-reading-20260926.jsonl','root-held-song-reuse-decisions-20260926.json']
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def excluded():
 keys=set()
 for path in P.glob('independent-audit-20260926-pass*.json'):
  data=json.loads(path.read_text())
  for u in data.get('entries',[]):
   if {'printed_page','section','item'}<=u.keys():keys.add(f"hahn1900asur:p{u['printed_page']}:s{u['section']}:{int(u['item']):02}")
 for path in P.glob('independent-whole-audit*.json'):
  data=json.loads(path.read_text())
  for u in data.get('rows',[]):
   if u.get('source_unit_key'):keys.add(u['source_unit_key'])
 return keys

def sample(seed):
 freeze=json.loads((P/'whole-source-freeze-20260926.json').read_text())
 for f,h in freeze['hashes'].items():assert sha(P/f)==h,f
 a=[json.loads(x) for x in (P/'proposal-audit.jsonl').read_text().splitlines()];r={x[10]:x for x in csv.reader((P/'proposal.csv').open())};ex=excluded()
 pool=sorted([u for u in a if u['entry_keys'] and u['source_unit_key'] not in ex],key=lambda u:u['source_unit_key']);selected=random.Random(seed).sample(pool,20)
 return dict(status='selected_not_reviewed',seed=seed,excluded_source_unit_keys=sorted(ex),frozen_hashes=freeze['hashes'],rows=[dict(source_unit_key=u['source_unit_key'],source_unit=u,proposed_rows=[r[k] for k in u['entry_keys']],original_pdf='/private/tmp/hahn-asur-1900-jasb.pdf',printed_page=int(u['printed_page']),pdf_page=int(u['printed_page'])+12) for u in selected])
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output',required=True);args=ap.parse_args();path=Path(args.output).resolve();assert P.resolve() in path.parents;path.write_text(json.dumps(sample(args.seed),ensure_ascii=False,indent=2)+'\n')

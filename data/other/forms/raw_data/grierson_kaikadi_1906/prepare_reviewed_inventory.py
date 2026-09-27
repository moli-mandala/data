"""Apply preserved original-page review sidecars without changing first readings."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def read(n):return [json.loads(l) for l in (P/n).read_text().splitlines() if l]
def main():
 rows=read('specimens-first-reading.jsonl');by={r['source_unit_key']:r for r in rows};reviewed=set();changes=[]
 for pg in range(336,343):
  f=P/f'owner-specimen{pg}-review-20260926.json';rep=json.load(f.open());units=[r for r in rows if r['printed_page']==pg];assert len(units)==rep['units_reviewed']
  reviewed.update(r['source_unit_key'] for r in units)
  for c in rep['corrections']:
   r=by[c['source_unit_key']];assert r['forms']==c['old'];r['forms']=c['new'];r.setdefault('review_corrections',[]).append(c);changes.append(c)
 assert reviewed==set(by)
 for r in rows:r['review']='complete owner original-grayscale second reading; final independent sample pending'
 (P/'specimens-reviewed.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 (P/'specimen-review-reconciliation.json').write_text(json.dumps({'units':len(rows),'owner_corrections':len(changes),'first_reading_sha256':hashlib.sha256((P/'specimens-first-reading.jsonl').read_bytes()).hexdigest(),'reviewed_sha256':hashlib.sha256((P/'specimens-reviewed.jsonl').read_bytes()).hexdigest(),'all_units_second_read':True,'independent_output_audit':'pending'},indent=2)+'\n')
 print(len(rows),len(changes))
if __name__=='__main__':main()

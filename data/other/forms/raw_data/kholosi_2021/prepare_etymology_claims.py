"""Preserve attributed etymological statements separately from Jambu assignments."""
import csv
import hashlib
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
repo = P.parents[4]
target_file = repo/'data/cdial/params.csv'
targets = {r[0]:dict(form=r[1], text=r[3]) for r in csv.reader(target_file.open())}
legacy = list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
conflicts = {
 'kholosi:pdf:p1:c1:u24':(9,'11714'),
 'kholosi:pdf:p8:c1:u7':(147,'12415'),
 'kholosi:pdf:p8:c1:u19':(154,'12372'),
}
claims=[]
for filename in ('pdf-first-reading.jsonl','structured-first-reading.jsonl'):
 for line in (P/filename).open():
  r=json.loads(line)
  prose=r.get('etymology','')
  if not prose:
   continue
  key=r['source_unit_key']
  pdf=filename.startswith('pdf')
  ids=re.findall(r'\((\d{1,5})\)',prose) if pdf else re.findall(r'CDIAL\s+(\d{1,5})',prose)
  ids=list(dict.fromkeys(ids))
  c=dict(claim_key=key+':etymology',source_unit_key=key,source_form=r['form'],
         source_gloss=r['gloss'],literal_claim=prose,
         provenance='printed author dictionary' if pdf else 'pinned author repository lexicon',
         locator=dict(printed_page=r['printed_page'],column=r['column'],unit=key) if pdf else dict(file=r['source_file'],physical_csv_row=r['physical_csv_row']),
         qualifier_words=re.findall(r'\b(?:probably|perhaps|possibly|uncertain|unknown)\b',prose,re.I),
         cited_target_ids=[dict(cdial_id=i,exists=i in targets,target_head=targets.get(i,{}).get('form','')) for i in ids],
         graph_disposition='Attributed source claim only; no automatic Jambu graph assignment.')
  if key in conflicts:
   n,printed=conflicts[key];old=legacy[n-1]
   assert printed in ids and old[10].endswith(':'+str(n))
   c['legacy_conflict']=dict(entry_key=old[10],historical_form=old[2],historical_phonemic=old[5],
      historical_jambu_parameter_id=old[1],printed_target_id=printed,status='unresolved competing attributions; historical assignment preserved',
      resolution='No winner selected; printed claim and historical editorial assignment remain distinct.')
  claims.append(c)
out=P/'source-etymology-claims.jsonl'
out.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in claims))
missing=[dict(claim_key=c['claim_key'],**t) for c in claims for t in c['cited_target_ids'] if not t['exists']]
(P/'source-etymology-claim-validation.json').write_text(json.dumps(dict(
 status='source claims preserved; no graph mutations',claim_count=len(claims),
 numeric_citations=sum(len(c['cited_target_ids']) for c in claims),missing_targets=missing,
 known_legacy_conflicts=[c for c in claims if c.get('legacy_conflict')],
 target_inventory=str(target_file.relative_to(repo)),target_inventory_sha256=hashlib.sha256(target_file.read_bytes()).hexdigest(),
 claims_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
 method='Check cited numeric CDIAL identifiers against tracked params.csv; existence does not establish that a printed number is semantically correct.'),ensure_ascii=False,indent=2)+'\n')
print('claims',len(claims),'numeric citations',sum(len(c['cited_target_ids']) for c in claims),'missing targets',missing)

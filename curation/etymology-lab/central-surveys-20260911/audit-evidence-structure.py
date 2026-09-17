import json,re,datetime
from pathlib import Path
p=Path(__file__).resolve().parent;arts=json.load(open(p/'cdial-articles.json'));issues=[];n=0
for lid in ['mewari_basad','Nimadi','bagheli_lakshman']:
 nums=[]
 for f in sorted((p.parent/lid).glob('batch-*.json')):
  d=json.load(open(f))
  if d.get('researchDirectory')!=str(p):continue
  for x in d['proposals']:
   n+=1;nums.append(x['number'])
   if not x.get('evidence') or not x.get('citation'):issues.append([lid,x['number'],'missing evidence or citation'])
   for entry in re.findall(r'CDIAL\[(\d+[a-z]?)',x['citation']):
    if entry not in arts:issues.append([lid,x['number'],'uncached primary article',entry])
   if x['kind']=='borrowed' and not x.get('parentLanguage'):issues.append([lid,x['number'],'missing donor language'])
 if nums!=list(range(1,len(nums)+1)):issues.append([lid,'numbering not contiguous'])
r={'checkedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proposals':n,'checks':['contiguous per-survey proposal numbers','nonempty evidence and citation','cited CDIAL articles cached, preserving alphabetic suffixes','borrowed proposals name donor language'],'issues':issues,'limitation':'Structural evidence audit only; not a fresh scholarly re-review of every claim.','auditCorrection':'Initial digit-only citation regex misread 11813a as 11813; fixed. The actual 11813a primary text was already cached and supports the two bhinsar proposals.'}
(p/'evidence-structure-audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

import json
from pathlib import Path
p=Path(__file__).resolve().parent
parents=json.loads((p/'parents.json').read_text())
blocked={}
lines=['# Acceptance dependencies','','All analyses remain pending. Approving a derived or compound proposal also requires review of its listed base proposals; approval of the whole survey includes those bases unless you exclude them. A base record anchors the lexical analysis and does not by itself establish a donor village.','','| Survey and proposal | Required base or donor | Affected records |','|---|---|---:|']
for lid in ['mewari_basad','Nimadi','bagheli_lakshman']:
 for f in sorted((p.parent/lid).glob('batch-*.json')):
  d=json.load(open(f))
  if d.get('researchDirectory')!=str(p):continue
  for x in d['proposals']:
   deps=x.get('acceptanceDependencies',[])+([x['acceptanceDependency']] if x.get('acceptanceDependency') else [])
   if not deps:continue
   labels=[]
   for a in deps:
    if a['type']=='unlinked-donor':blocked[a['parentId']]=blocked.get(a['parentId'],0)+len(x['formIds'])
    labels.append(f"{a['survey']} proposal {a['proposal']}" if a['type']=='pending-proposal' else f"Donor ancestry unresolved: `{a['parentId']}`")
   lines.append(f"| [{d['survey']} {x['number']}](../{lid}/{f.stem}-review.md) | {', '.join(labels)} | {len(x['formIds'])} |")
lines+=['',f"{sum(blocked.values())} links require unresolved donor ancestry: "+'; '.join(f"{parents.get(k,{}).get('word',k)} ({n} links, `{k}`)" for k,n in blocked.items())+'. These are excluded from temporary-graph eligibility until the dependencies are handled; this is distinct from dependent bases already included among the pending proposals.','','The latest [validation report](validation.json) is authoritative for current IDs and conflicts. A successful temporary graph check verifies structure, not the scholarly correctness of an etymology.']
(p/'DEPENDENCIES.md').write_text('\n'.join(lines)+'\n')

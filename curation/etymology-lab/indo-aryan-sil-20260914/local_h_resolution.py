import json,collections
from pathlib import Path
P=Path(__file__).resolve().parent;inv=json.loads((P/'inventory.json').read_text());audit=json.loads((P/'audit-records.json').read_text());xs=[x for x in audit if x['reason'].startswith('Initial s/ś')];tags={x['record']['Tags'] for x in xs}
gs={'dry','gold','snake','one hundred','hundred','seven'};matrix=collections.defaultdict(list)
for r in inv:
 if r['Tags'] in tags and r['Gloss'].lower() in gs and (r['Form'].startswith('h') or r['Gloss']=='one hundred' and r['Form'].startswith('ek h')):matrix[r['Tags']].append(r)
acc=[]
for x in xs:
 r=x['record'];cmps=[z for z in matrix[r['Tags']] if z['ID']!=r['ID']];assert len({z['Gloss'] for z in cmps})>=2
 q=x['candidates'][0];ev=q['evidence'];ev=ev[0] if isinstance(ev,list) else ev
 ev+=' The same survey locality independently records '+', '.join(dict.fromkeys(z['Form']+' “'+z['Gloss']+'”' for z in cmps))+', corroborating initial s/ś > h locally. These are source-local comparisons, not transferred locality evidence. Exact comparative records are preserved in local-h-matrix.json. Intra-IA transmission, if any, remains open.'
 cite=q['citation'];cite=';'.join(cite) if isinstance(cite,list) else cite
 acc.append(dict(record=r,parent=q['parent'],citation=cite,evidence=ev,family=q['parent'],priorHold=x['reason']))
(P/'local-h-matrix.json').write_text(json.dumps(matrix,ensure_ascii=False,indent=1))
(P/'local-h-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1))
s=(P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','local-h-decisions.json').replace('sixth','local-h');(P/'local_h_save.py').write_text(s)
print(len(acc))

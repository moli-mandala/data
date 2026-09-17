import json,csv,collections
from pathlib import Path
P=Path(__file__).resolve().parent
inv=json.loads((P/'inventory.json').read_text());eligible={r['ID'] for r in csv.DictReader(open(P/'unresearched-records.csv'))};eligible.update(x['record']['ID'] for x in json.loads((P/'audit-records.json').read_text()))
cm=collections.defaultdict(list)
for r in inv:
 if r['Gloss'].lower() in {'dry','snake','seven','hundred','one hundred'} and (r['Form'].startswith('h') or r['Form'].startswith('ek h')):cm[r['Language_ID'],r['Tags']].append(r)
ev='CDIAL 13519 groups the regional sonā/sonu gold family under suvarṇa or sauvarṇa. The user selected sauvarṇa when this branch is indistinguishable. The h-initial gold forms are supported by independent s/ś > h comparisons from the same language and exact survey locality; possible intra-IA transfer remains open.'
q=dict(parent='13519-2',citation='CDIAL[13519.2]',evidence=ev);acc=[];held=[];matrix={}
for r in inv:
 if r['ID'] not in eligible or r['Gloss']!='gold' or not r['Form'].startswith('h'):continue
 comparanda=[x for x in cm[r['Language_ID'],r['Tags']] if ',' not in x['Form'] and '[' not in x['Form']];matrix[r['ID']]=comparanda
 reason=None
 if 'ŋ' in r['Form']:reason='Local s/ś > h is supported, but the velar nasal in this gold form needs an independent correspondence check.'
 elif ' ' in r['Form']:reason='The internal spacing in the gold response needs checking against the original survey before treating it as a single word.'
 elif len({x['Gloss'] for x in comparanda})<2:reason='Only one independent h-initial comparison is available at this exact locality; further local sound evidence is needed.'
 if reason:held.append(dict(record=r,families=[0],reason=reason,passNumber=33));continue
 e=ev+' Exact-locality comparanda: '+', '.join(dict.fromkeys(x['Form']+' “'+x['Gloss']+'”' for x in comparanda))+'. Their full source records are preserved in gold-h-matrix.json.'
 acc.append(dict(record=r,family=0,parent=q['parent'],citation=q['citation'],evidence=e))
for name,obj in [('gold-h-rules.json',[q]),('gold-h-decisions.json',dict(accepted=acc,held=held)),('gold-h-matrix.json',matrix)]: (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=1))
(P/'gold_h_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','gold-h'))
print('accepted',len(acc),'held',len(held))

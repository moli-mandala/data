import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass125';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4124',citation='CDIAL[4124]',evidence='CDIAL 4124 gātra gives Prakrit gāa body, Assamese gā/gāw trunk of body and Bengali gā/gāo body. Hajong gau directly fits this eastern contracted family with the source diphthong retained; local Indo-Aryan transmission remains unresolved.'),dict(parent='12335',citation='CDIAL[12335]',evidence='CDIAL 12335 śarīra gives Pali/Prakrit sarīra and western Pahari sarīr body. Survey çarir fits the regional sibilant notation; Dang saril retains r/l variation. These family links leave learned or local Indo-Aryan transmission unresolved and do not assert that every regional pronunciation is quoted in the article.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Gloss']=='body' and r['Language_ID']=='Hajong' and r['Form']=='gau' else 1 if r['Gloss']=='body' and r['Form'] in {'çarir','saril'} else None
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Gloss']=='head' and r['Form'] in {'munḍ','munḍi','mūnḍ'}:held.append(dict(record=r,families=[],passNumber=125,reason='Full CDIAL 10247 mūrdhan discusses unaspirated muṇḍ head forms as perhaps from or crossed with 10191 muṇḍa shaven. The matching survey form does not distinguish those roots. Existing homepage 10191 links are not sufficient to resolve this explicitly primary-attested ambiguity; retain for comparative review.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass125-body-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))

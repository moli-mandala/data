import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass105';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='1135',citation='CDIAL[1135, sense 1];CDIAL[6648]',evidence='The complete response is the self-derived inclusive pronoun plus two/both. CDIAL 1135 gives Old Marwari āpa and Gujarati āpaṇ inclusive we; CDIAL 6648 gives Marwari do, Hindi donõ both, Apabhramsha doṇṇi and regional dunni both. Two ordered components preserve the whole expression and source vowel/nasal/ending variation. The source we or we(two) label is retained; no inherited Sanskrit compound or settled local transmission route is asserted.'),dict(parent='986',citation='CDIAL[986];CDIAL[6648]',evidence='The complete response is ham we plus dono-type both. CDIAL 986 explicitly gives regional ham we; 6648 gives Hindi donõ both and related don/donni forms. Two ordered components preserve the response, including nasal/retroflex notation and final vowels. The source exclusive/two qualification is retained without attributing it to the old pronoun stem; local Indo-Aryan transmission remains open.'),dict(parent='986',citation='CDIAL[986];CDIAL[6648];CDIAL[11119]',evidence='The complete ami/ame duilok expression contains we plus two plus people. CDIAL 986 gives Oriya ami/āme, 6648 Oriya dui and 11119 Prakrit lōga people. Three ordered component-family edges preserve this full expression, including the final voiceless k in lok as a regional qualification and the exclusive elicitation label. No single inherited Sanskrit compound or settled local loan route is asserted.')]
sets=[{'āpe dɔī','āpā̃ donīyɔ','āpā̃ donyɔ','āpā̃ dɔnū','āpādɔno','apa dɔnyɔ','apā donyɔ','āpaṇa donī','āpaṇa dɔy','āpaṇa dɔnī'},{'ham dɔnɔ','ham doṇɔ','ham donōo'},{'ame duilok','ami duilok'}]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or not r['Gloss'].lower().startswith('we'):continue
 i=next((i for i,ss in enumerate(sets) if r['Form'] in ss),None)
 if i is None:continue
 q=rules[i];cs=[q['parent'],'6648']+(['11119'] if i==2 else [])
 acc.append(dict(record=r,parent=q['parent'],components=cs,family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['1135']=json.loads((P/'pass102-primary-articles.json').read_text())['1135']
for k in ['986','11119']:a[k]=json.loads((P/'pass101-primary-articles.json').read_text())[k]
f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass105_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print(len(acc),sum(len(x['components']) for x in acc))

import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='expansion-pass78';assert not (P/(stem+'-decisions.json')).exists()
spec={
'11175':'CDIAL 11175 vaṃśa section 1 explicitly gives Nepali, Bengali and Hindi bā̃s bamboo. The survey bamboo tree gloss denotes that plant; it does not require an extra lexical tree component. Nasalization and vowel-length notation remain preserved.',
'3735':'CDIAL 3735 kṣetra explicitly gives Nepali and neighboring khet field. These simple khet responses to irrigated field identify the field noun; irrigation is the elicited subtype, not an unrepresented compound member.',
'6778':'CDIAL 6778 dhānya explicitly gives Nepali and neighboring dhān growing or unhusked rice. The survey unhusked rice gloss directly matches this primary sense, with source vowel length and aspiration retained.',
'11443':'CDIAL 11443 vasā explicitly gives Nepali boso fat and Kumaoni baso. The regional boso/buso responses denote animal fat or the fat part of flesh, directly matching this noun; buso retains its raised initial vowel and possible intra-Indo-Aryan transmission is open.'}
rules=[dict(parent=p,citation='CDIAL['+p+']',evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=r['Form'];g=r['Gloss'].lower();p=None
 if w in ['bãs','bā̃s'] and g in ['bamboo tree','bamboo']:p='11175'
 if w in ['kʰet','khet','khēt','kʰēt'] and g=='irrigated field':p='3735'
 if w in ['dʰan','dʰān','dhan','dhān'] and g=='unhusked rice':p='6778'
 if w in ['boso','buso'] and g in ['fat part of flesh','fat (of meat)']:p='11443'
 if p:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+w+' is preserved.'))
 elif w=='mati' and g=='soil/ground':held.append(dict(record=r,families=[],reason='CDIAL 10286 mṛttikā gives eastern māṭi with retroflex ṭ, whereas these survey records spell mati with dental t. Verify the Bengali/Hajong/Bishnupriya source transcription before treating the distinction as irrelevant; the Marathi dental form is not independent eastern evidence.',passNumber=78))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'expansion_pass78_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})

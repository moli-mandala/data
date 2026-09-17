import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass122';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='6726',citation='CDIAL[6726];platts1884[548]',evidence='Platts p. 548 explicitly gives Hindi dhanush both bow and rainbow; CDIAL 6726 dhanus gives Prakrit dhaṇu and regional bow forms. Survey dʰanus/dʰanūs/danu rainbow fits this family with final-sibilant retention/loss and aspiration/length notation preserved. The rainbow sense is primary-supported, not inferred merely from the English bow metaphor. Local Indo-Aryan transmission and learned influence remain unresolved.'),dict(parent='1577',citation='CDIAL[1577]',evidence='CDIAL 1577 indradhanuṣ explicitly means rainbow and gives Pali indradhanu and Prakrit iṁdadhaṇu/iṁdahaṇu. Selected indra/indar/indor/id ar-type plus dhanu(s) forms preserve the recognizable whole compound with regional vowel, rhotic and aspiration variation. These are links to the attested rainbow compound, with learned or regional Indo-Aryan transmission unresolved.')]
sets=[{'dʰanus','dʰanūs','danu'},{'indra danūś','indradhaṇu','īndrɨdanuś','indra danuś','indra danu','indor dʰonu','indordʰonu','indro dʰonu','indar dʰanuś','indər dʰanuś','idardhanuś','indardhanus','īndrdānūś'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='rainbow':continue
 for i,ss in enumerate(sets):
  if r['Form'] in ss:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['platts-dhanush']=[x for x in json.loads((P/'pass122-platts-research.json').read_text()) if x['word']=='dhanush'];f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

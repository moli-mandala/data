import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass114';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='f_a3zgar2k27pzm',citation='centralbank-saral-kannada[PDF p. 11, vegetables, item 4]',evidence='The Hindi column of the primary vegetable list gives फूलगोबी cauliflower, verified again in the preserved page image. Hindi survey fulgobi/phulgobi are spelling and pronunciation variants of this same-language pʰūlgobī entry; f/ph and unmarked vowel length are retained. The whole compound is linked to the Hindi lexical entry.'),dict(parent='f_ody2xkuwe2bqg',citation='cstt-agriculture-eng-hin-dogri[s.v. cabbage]',evidence='The preserved primary-source transcription of the CSTT glossary gives Hindi बंदगोभी cabbage. Hindi survey bandgobi/bandigobi fit this same-language compound entry, retaining medial epenthetic i in bandigobi and aspiration notation. These are variant links, not loans from another language; the source does not itself quote the epenthetic survey spelling.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Language_ID']!='H':continue
 i=0 if r['Form'] in {'fulgobi','phulgobi'} and r['Gloss']=='cauliflower' else 1 if r['Form'] in {'bandgobi','bandigobi'} and r['Gloss']=='cabbage' else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='variant',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for name,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+name+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
a=json.loads((P/'loan-third-primary-articles.json').read_text());(P/(stem+'-primary-articles.json')).write_text(json.dumps({q['parent']:a[q['parent']] for q in rules},ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

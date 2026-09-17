import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass141';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='700',citation='CDIAL[700]',evidence='CDIAL 700 alagna unconnected explicitly gives Punjabi alagg, Nepali alag/algga, Assamese ālag/ālgā, Bengali ālag/ālgā, Oriya alaga/algā and Hindi alag/algā separate. Selected simple and repeated alag/olag/alga forms denote different/separate, with repetition interpreted distributively. Regional initial vowels, unstressed-vowel reduction and shortened repeated elements are preserved; exact local borrowing routes are unresolved. Extra suffixes and mixed alag-plus-another-lexeme responses are excluded from this rule.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='different' and r['Form'] in {'ʌlʌgæʌlʌgæ','ʌlʌgælʌg','ʌlʌgeʌlʌge','ʌlʌglʌg','olag','lagalag','olga','olga olga','olagalag'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

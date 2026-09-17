import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass120';assert not (P/(stem+'-decisions.json')).exists()
spec=[({'dī̃','dẽ','deõ','diũ','dẽ̤','de̤','dʰeõ','deo'},'CDIAL 6333 divasa explicitly gives Awan dèõ sun, Lahnda dēhũ sun and Punjabi dĩh/dẽh/dihũ day, sun. The selected northwestern nasal, breathy or h-less dī̃/dẽ/deõ/diũ/deo forms fit this family; loss or weakening of h, vowel differences and source nasal notation remain qualifications. The nasal forms are not linked to dina merely because both name a day.'),({'dahaḍu','ḍahaḍu','dahəḍo','dāhaḍo'},'CDIAL 6333 gives Old Marwari dihāḍo/dhyāḍo and Old Gujarati dihāḍaü, Gujarati dahāṛɔ day, alongside explicit Punjabi and western Pahari dihāṛā sun. The selected Bhil dahaḍu/dahəḍo/dāhaḍo and initial-retroflex ḍahaḍu fit this extended day/sun family. Initial retroflex notation, vowel syncope, final vowel and stop/flap differences are retained, without claiming a source-glyph correction. Turner considers MIA h possibly due to crossing with ahar day.')]
rules=[dict(parent='6333',citation='CDIAL[6333, addendum]',evidence=e+' Local Indo-Aryan transmission remains unresolved.') for ss,e in spec]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='sun':continue
 for i,(ss,e) in enumerate(spec):
  if r['Form'] in ss:
   q=rules[i];acc.append(dict(record=r,parent='6333',family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

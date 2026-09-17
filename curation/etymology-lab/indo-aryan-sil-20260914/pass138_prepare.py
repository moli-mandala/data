import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass138';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='2830',citation='CDIAL[2830]',evidence='CDIAL 2830 karṇa ear explicitly gives Oriya kāna and Gujarati/Marathi kān as well as Prakrit kaṇṇa and extended kaṇṇaḍaya ear; it also records Pashai kaṇ(ḍ)- orifice of ear. Simple kano/kaṇo/kane and Marathi karna support the family directly, with learned retention in karna possible. The western kānḍo/kānḍā/kānəḍio responses are qualified comparisons to the old ḍ-extension, not claims of an exact uninterrupted suffix history. Regional vowel quantity, inflection and inter-Indo-Aryan transmission remain unresolved; all spellings are retained.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='ear':continue
 if r['Form'] in {'karna','kaṇo','kano','kane','kānḍo','kānḍā','kānəḍio'}:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
 elif r['Form'] in {'kānṭu','kānṭo','kānṭhā','kaṇṭu','kānṭā'}:
  held.append(dict(record=r,families=[],reason='The ear root karṇa 2830 is a plausible base, but its explicit kaṇṇaḍaya / kaṇ(ḍ)- extensions do not establish the ṭ-/ṭh- extension in these western kānṭ-/kaṇṭ- responses. Karṇaka 2831 gives a different suffix and mostly rim/handle senses. Retain for morphological or source-transcription evidence rather than treating all retroflex stops as interchangeable.',passNumber=138))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc),len(held))

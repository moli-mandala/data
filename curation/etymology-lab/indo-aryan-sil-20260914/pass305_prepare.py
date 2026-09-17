import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass305';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='5329',citation='CDIAL[5329]',evidence='Full jhadi explicitly records Lahnda jhar large cloud, Punjabi jhar clouds covering sky/heavy rain and Sindhi jhuru heavy clouds. These support the selected Saraiki, Awan, Gojri and Pothwari retroflex cloud forms. Source c/ch versus jh and y/aspirated-y realizations, vowels and endings are preserved; exact local phonetics and cross-IA transmission remain qualified. This links the regional IA family and does not assert a direct Dravidian loan into these survey languages. Plain dental-r car and mixed badal/car responses are excluded because separate evidence or alternative analysis is needed.'),dict(parent='11567',citation='CDIAL[11567]',evidence='Full vardala gives regional badal/badar, and its addendum explicitly gives Jaunsari badli clouds and Kotgarhi badal. Dotyali baldo cloud is a qualified metathesized regional match, preserving the transposed l/d sequence and final vowel. Precise local development and cross-IA transmission remain qualified.')]
sets=[('cloud',{'cāṛā','cāṛ','caṛ','jʰaṛ','yāṛ','yāṛa','yāṛa / yʰāṛ','yʰāṛ','cʰaṛʰ','cʰa̤ṛ'}),('cloud',{'bāldo'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare305.py').read_text());print('accepted',len(acc))

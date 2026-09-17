import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass181';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='5501',citation='CDIAL[5501.1]',evidence='Full CDIAL *ṭhiṅga subsection 1 explicitly supplies Punjabi ṭhiṅgṇā, Hindi ṭhĩgnā and Gujarati ṭhĩgṇũ dwarfish. Selected western survey ṭhiŋgəṇo/ṭhīŋgaṇɔ/ṭīgɨṇɔ/ṭigṇɔ short match that i-vowel velar-plus-nasal family; aspiration, medial nasal realization and vowel notation remain qualified. The e-vowel *ṭhēṅga subsection 2 is not selected. The source short gloss is retained, with short stature as the comparison; local IA transmission remains unresolved.'),
 dict(parent='11538',citation='CDIAL[11538.1]',evidence='Full CDIAL vāmana subsection 1 explicitly gives Punjabi vauṇā/bauṇā/bāunā, Hindi baunā/bāunā and West Pahari bauṇɔ/baoṇā dwarf. Punjabi bɔṇā and Kaithal boṇːa short match this regional au-to-o contracted family, with nasal gemination and vowel notation preserved. Short stature is the semantic comparison; local IA transmission remains unresolved. The L. vā̃varā subsection 2 is excluded.')]
sets=[{'ṭʰīŋgaṇɔ','ṭīgɨṇɔ','ṭigṇɔ','ṭhiŋgəṇo'},{'bɔṇā','boṇːa'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='short':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))

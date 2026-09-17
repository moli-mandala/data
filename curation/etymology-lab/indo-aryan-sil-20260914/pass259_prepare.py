import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass259';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10560',citation='CDIAL[10560]',evidence='Full raṅga dye/colour explicitly documents Prakrit raṃga red colour, Bengali rāṅā red and Oriya raṅgā red/red colour. These support the eastern raŋ/raŋŋa/roŋ/roṇ survey responses. Bengali supplies a comparison for loss of the stop after the nasal; local retroflexion, gemination and final vowels remain explicitly qualified rather than treated as established sound laws. Source red gloss and all phonetics remain intact. Cross-IA transmission is uncertain. The unrelated raṅga defective/wretch subsection 10538.5 was inspected and excluded.')]
fs={'raŋŋa','roṇ','roṇo','ṛəṇ','ṛaṇ','ṛaṇg','roŋg','roŋ','raŋ'};remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='red' and r['Form'] in fs:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare259.py').read_text());print('accepted',len(acc))

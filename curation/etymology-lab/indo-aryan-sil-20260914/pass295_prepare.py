import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass295';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4676-2',citation='CDIAL[4676,2]',evidence='Full cammakka section 2 gives Bengali camak flash, regional camaknā/camkanu/camakṇā to shine, and Punjabi camkār/camkārā flash. These support simple camak/cemek, verbal camkni/cemekna/cemkev and camkṛi-type lightning responses. Preserve source vowels and ordinary endings with local formation and cross-IA transmission qualified. Use section 2, not the separate startle branch 1.'),dict(parent='10993-2',citation='CDIAL[10993]',evidence='Full lasati explicitly gives its kk-extension Hindi lasaknā to shine and Lahnda lask flashing, tentatively borrowed from Hindi, alongside lahakṇu/lahaknā flare or shine. These support lasak/laskoṇ/laske/lasske/laśk and both laske/laskaṇ alternatives. Use the existing lasakkati extension node; the extension is unnumbered in the prose, so cite the base article. Ordinary local verbal endings and precise transmission remain qualified.')]
sets=[('lightning',{'camkā','cemekna','cemkev','cemek','camak','camkni','camki','camkṛi'}),('lightning',{'lasak','laskoṇ','laske / laskaṇ','laske','lasske','laśk'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare295.py').read_text());print('accepted',len(acc))

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass279';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9982',citation='CDIAL[9982]',evidence='Full māṃsa gives Pali/Prakrit maṃsa and Prakrit māsa, eastern mā̃s/mās and Oriya māũsa, Jaunsari mās, Gujarati/Marathi mā̃s and Sindhi māhu. These support the eastern maŋso/maŋśo/maŋʃo and māus/mə̃us-type meat forms and western maso/moso/maṣ. Western mah/māh/maha/mahã and māx are provisional family matches with weakened sibilant; the local conditioning is not established merely by the Sindhi comparison. Preserve source vowels, nasal/affricate realizations and endings. Cross-IA transmission is qualified; no Sindhi or Sinhalese loan is asserted.')]
sets=[{'maŋso','maŋśo','maŋʃo','māūs','maūs','maus','maṇʦo','mɔu̯s','mə̃u̯s','mãus','mʌ̃us','mə̃us','māx','māh','mah','maha','mahã','maũs','mãṽs','mous','meus','moso','maso','maṣ'}];remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='meat':continue
 for i,fs in enumerate(sets):
  if r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare279.py').read_text());print('accepted',len(acc))

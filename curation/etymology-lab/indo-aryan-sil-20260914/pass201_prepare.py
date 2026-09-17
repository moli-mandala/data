import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass201';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9828',citation='CDIAL[9828]',evidence='Full manuṣya article gives Lahnda muṇas husband, Awankari muṇus, Oriya/Sambalpuri munus labourer and the munisa branch influenced by Middle IA purisa. Selected Bhatri/Adivasi Oriya munus/munos/munəs man/husband and Kochila Tharu munsa man match this regional human/person family, preserving vowel and nasal notation. Husband is a contextual specialization of man/person and is independently documented in the article, not imposed on the original gloss. Turner explicitly notes shortening and collision with puruṣa; local IA transmission remains unresolved. Extra -kh, -k and mixed or compound responses are excluded.'),dict(parent='10049',citation='CDIAL[10049]',evidence='Full mānuṣa article explicitly gives Gujarati māṇas man and Marathi māṇūs man, alongside Old Gujarati māṇisa and Old Marwari mā̃ṇasa. Bhilali/Rathawi mānas and Khandesi mānos man match this western family, retaining nasal and vowel notation as qualifications. The parallel manuṣya article was inspected separately; local IA transmission remains unresolved.')]
sets=[{'munus','munos','munəs','munsa'},{'mānas','mānos'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in {'man','husband'}:continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))

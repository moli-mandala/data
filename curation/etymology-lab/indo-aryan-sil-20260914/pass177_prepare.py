import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass177';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='2963',citation='CDIAL[2963]',evidence='Full CDIAL kavāṭa includes kapāṭa leaf of a door and Prakrit kavāḍa/kavāla/kiviḍī, then Bihari kewāṛī, Maithili kebāṛ/kebāṛī, Hindi kiwāṛ/kiwār/kewār and Gujarati kamāṛ. These support the selected velar-initial labial-medial door family, including conservative kapat/kabat forms and rhotic kiwara/kebari forms. Labial voicing, dental/retroflex notation and vowels remain as transcribed. Learned or cross-IA transmission is unresolved; the entry explicitly reports some IA borrowing and proposes a deeper Dravidian origin without establishing it.'),
 dict(parent='6663',citation='CDIAL[6663]',evidence='Full CDIAL dvāra gives Prakrit bāra and Gujarati bār/bārũ door; its addendum explicitly supplies Gujarati bārṇũ and Sindhi Kachchhi bāyṇo door. These regional nasal-extension comparanda support survey barana/bayiṇo/bayiṇu/bayno and glottal baʔaṇo/baʔaŋo, with rhotic weakening or loss and source nasal notation qualified. The similar vāraṇa 11553 means obstruction and supplies an Old Marwari doorstep, not this door family. Local IA transmission is unresolved.')]
sets=[set('kʸevar kebārī kowaɾ kibaɾa keːwaɾi kebari kebār kiwara kamar kəpat kʌpat kabat kopaṭ kobato kopʌt kapat kopat kobat kobaṭ'.split()),set('barana bāyiṇo bāyiṇu bayno baʔaṇo baʔaŋo'.split())]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='door':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))

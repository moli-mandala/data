import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass205';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='5071',components=['5071','9661'],citation='CDIAL[5071];CDIAL[9661]',evidence='The complete younger-brother expression is small/young plus brother. Full CDIAL 5071 gives Punjabi choṭṭā, Hindi choṭā, Bengali/Oriya choṭa and Gujarati choṭũ small; 9661 gives bhāi/bhāī, Lahnda bhrā and Punjabi bharā brother. Preserve source aspiration, vowel and retroflex notation, including Pothwari pra̤ as the regional breathy/p-initial counterpart of bhrā. Save two ordered lexical components, without claiming a single ancient compound or a settled cross-IA transmission route.'),
 dict(parent='5071',components=['5071','9349'],citation='CDIAL[5071];CDIAL[9349]',evidence='The complete younger-sister expression is small/young plus sister. Full CDIAL 5071 supplies choṭa/choṭā/choṭṭā small, here with feminine agreement; 9349 explicitly gives bahin/bahan, Gujarati bahɛn/bɛn, and Lahnda bhēṇ/Punjabi bhaiṇ sister. Source bahīn and bayeṇ retain vowel and metathesis qualifications; Pothwari pe̤n/pe̤ṇ retains regional breathy/p-initial notation. Save both lexical components in source order. Local Indo-Aryan transmission remains unresolved; no single ancient compound is asserted.')]
sets=[{'coṭɔbʰāī','caṭɔ bʰāī','coṭābʰāi','coṭa bʰai','choṭo bɦāi','coṭā bāi','coṭːu bʰai','coṭːa bʰai','coṭa bʰayi','cʰoṭa bʰay','choṭobhāy','choṭobhāi','coṭobhay','coṭobhāi','cʰoṭa pra̤'},
 {'coṭī bahīn','coṭīben','coṭībahan','coṭībahaṇ','coṭībahīn','coṭī bahin','coṭibahen','choṭi bəhin','choṭi bayeṇ','coṭi ben','choṭi bhen','cʰoṭi bahaṇ','coṭi behen','cʰoṭi pe̤n','cʰoṭi pe̤ṇ'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss'] not in ('younger brother','younger sister'):continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms and r['Gloss']==('younger brother' if i==0 else 'younger sister'):
   q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if 'components' in q else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
   if 'components' in q:x['components']=q['components']
   acc.append(x);break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print('accepted',len(acc))

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass288';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9425',citation='CDIAL[9425,1]',evidence='Full bhasman gives Kashmiri bas fine dust, Lahnda/Punjabi bhas/bhass ashes and West Pahari bhass dust. These support Kullu baːs/bʰaːs ash, preserving source vowel quantity and aspiration with cross-IA transmission qualified. Do not choose the separate bhāsma branch 2 solely from long vowel spelling.'),dict(parent='5020',citation='CDIAL[5020]',evidence='Full chādi explicitly gives Bshk čī ashes, regional chāī and Assamese sāi. These support Bshk cē and both cī/ce alternatives, plus provisionally Goj sayī with a sibilant realization. Preserve source vowels; local sound history and cross-IA transmission remain qualified. The separate kṣāma addendum gives nasal Assamese chā̃i, which does not require moving these nonnasal responses to that family.')]
sets=[('ash',{'baːs','bʰaːs'}),('ash',{'cē','cī / ce','sayī'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
held=[dict(record=r,families=[],passNumber=288,reason='Full kṣāra 3674 documents khār/chār alkali and ash, but the reduced kh-forms lack evidence for the lost liquid and may also invite comparison with regional khāk ash/dust. No supported immediate donor or diagnostic local development establishes the choice yet. Hold the competing analysis and preserve glottal onset/hiatus spellings; do not treat this as only uncertainty about IA borrowing.') for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in remaining and r['Gloss']=='ash' and r['Form'] in {'khaʔa','kha','khaa','khe'}]
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare288.py').read_text());print('accepted',len(acc))

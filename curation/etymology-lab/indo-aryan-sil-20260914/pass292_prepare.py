import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass292';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11497-2',citation='CDIAL[11497,2]',evidence='Full vātala section 2 explicitly gives Bshk bālā, Torwali balai and Hindi bāl wind. These support belāy/balla/bāle/baḷ across the selected northern records, preserving vowels and lateral quality. Turner allows an alternative vātālī origin for Hindi bāl; retain that published alternative for the Kaithal forms rather than claim exclusive descent. Regional transmission remains qualified.'),dict(parent='11497',citation='CDIAL[11497,1]',evidence='Full vātara section 1 gives Gujarati vāyrɔ, Marathi vārā/vārẽ and Konkani vāro/vārẽ wind. These support Noiri varu and Bhili vāiri with source vowel shape and final inflection preserved and local history/cross-IA transmission qualified.'),dict(parent='11491',citation='CDIAL[11491]',evidence='Full vāta wind and its Old Marwari vāi addendum support Jaunsari bat/bāt with retained stop and Bhilali vāi with loss of the stop. Turner notes that vāta and vāyu are not always distinguishable in NIA; the vāi link follows the explicit Old Marwari comparison provisionally, not a resolved exclusive ancestry.'),dict(parent='11544',citation='CDIAL[11544]',evidence='Full vāyu gives Lahnda vā and Awankari vā (oblique vāū) wind, directly supporting Pothwari va/vǎ̤. Preserve source tone/breathy marking. The dictionary cautions that vāyu and vāta can converge; use the explicit regional comparison without asserting that deeper ambiguity is settled.')]
sets=[('wind',{'belāy','balla','bāle','baḷ'}),('wind',{'varu','vāiri'}),('wind',{'bat','bāt','vāi'}),('wind',{'va','vǎ̤'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare292.py').read_text());print('accepted',len(acc))

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass282';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='4287',citation='CDIAL[4287]',evidence='Full godhūma gives the regional gohum/gahum/gahũ/gehũ and reduced gyũ/gíũ/gī̃ū̃ forms, Gujarati ghaũ, Konkani gaṁv and eastern retained m. These support the selected wheat responses with source nasalization, hiatus, glottal stop and h/gh spellings preserved. The article itself qualifies some eastern m forms; these are family links with local history and cross-IA transmission unresolved, not newly proved sound laws.'),dict(parent='10431',citation='CDIAL[10431]',evidence='Full yava gives Torwali yōu, Maiya yåw and Shina yō/yū barley, directly supporting Maiya/Chilisso/Gowro yaoū/yaov/yāū. Preserve the source vowel sequences and qualify cross-IA transmission. Forms with unexplained final r or a wheat gloss are not included.'),dict(parent='2665',citation='CDIAL[2665]',evidence='Full kaṇika gives Lahnda/Punjabi kaṇak wheat and Kashmiri kaṇakh. These support Pothwari kankʰ wheat with syncope and the recorded aspiration, while the exact local history and possible regional transmission remain qualified.')]
sets=[('wheat',{'ɡʊhõ','gohomõ','gə̃u̯','gɦahũ','gɦauhũ','gohõ','gā̃o','gõve','goṽ','gõː','gyu','gheu','gyũ','ge(h)ũ','gʰĩẽ','gʰĩjũ','gẽõn','ghoʔẽ','gome','gov'}),('barley',{'yaoū','yaov','yāū'}),('wheat',{'kankʰ'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare282.py').read_text());print('accepted',len(acc))

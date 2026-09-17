import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass285';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='11443',citation='CDIAL[11443]',evidence='Full vasā gives Kashmiri was marrow/brain, Kumaoni baso, Nepali boso and Oriya basā fat, plus West Pahari bɔ̄ fat in the addendum. These support eastern bos/bu.so/bɔ̃s/bə̃s/bõs/bõːs and Jaunsari bo. Chilisso vāz is a provisional voiced-sibilant family match. Preserve the source vowels, nasalization and internal spacing and qualify local changes/cross-IA transmission.'),dict(parent='10323',citation='CDIAL[10323]',evidence='Full medas gives Bshk mä̃ fat, Shina mī̃ fat and Shumasti mīə̃ animal fat, alongside Kalasha meʰ/mɛ̃. These support Bshk mā/mã/mae/mā̃/maʔ and Maiya/Bhateri mīū̃/mīõ/mī̃õ. Keep source glottalization and vowel sequences; the precise local phonetics and transmission remain qualified. Do not collapse the distinct majjan and medya families into this parent.')]
sets=[('fat',{'vāz','bos','bu.so','bɔ̃s','bə̃s','bõs','bõːs','bo'}),('fat',{'mā','mã','mae','mā̃','maʔ','mīū̃','mīõ','mī̃õ'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
held=[dict(record=r,families=[],passNumber=285,reason='Full majjan 9712 documents irregular miñj/mijjh and fat comparanda, while medya 10326 explicitly links Lahnda mẽjh and Bhalessi mènj fat to crossing with the marrow family. These mīny/mī̃y/mī̃nź responses need evidence distinguishing those histories and explaining any palatal-consonant reduction. Hold competing etyma, not merely IA transmission.') for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in remaining and r['Gloss']=='fat' and r['Form'] in {'mī̃nź','mī̃y','mī̃ny','mīny'}]
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare285.py').read_text());print('accepted',len(acc))

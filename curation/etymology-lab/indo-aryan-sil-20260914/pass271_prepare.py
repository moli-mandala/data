import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass271';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9250',citation='CDIAL[9250]',evidence='Full bīja explicitly gives Nepali biu seed, Kumaoni bĩyo/biyõ and regional bīa/bīyā seed. These directly support Majhi/Bote bʸu seed as a glide-bearing reduced regional form. Source superscript glide and vowel notation remain intact; local phonetic history and cross-IA transmission remain qualified. The separately compounded bījadhāna/bījadhānya seed-corn entries were inspected and are not substituted.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='seed':continue
 if r['Form']=='bʸu':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']))
 elif r['Form'] in {'bicahan','bicahān','biha'}:held.append(dict(record=r,families=[],passNumber=271,reason='Full bījadhānya 9254 gives Bihari bīhan and Oriya bihana seed-corn, and bījadhāna 9252 separately gives Sindhi bīhaṇu and Marathi biyāṇe. Danuwar bicahan/bicahān has a retained affricate and extra vowel sequence without a direct local comparison, while biha could be shortened from one of these or compared with bīja regional bīh. The exact stem/compound and local reductions are unresolved, beyond uncertainty solely about IA transmission.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare271.py').read_text());print('accepted',len(acc),'held',len(held))

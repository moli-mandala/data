import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass232';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='3277',citation='CDIAL[3277]',evidence='Full CDIAL kuttira explicitly gives West Pahari kutar/kuttar and feminine kutrī, Old Gujarati kūtiraü, Gujarati kutrɔ/kutrī/kutrũ and Marathi kutrā/kutrī/kutrẽ. The selected Bhil-area kutəro/kutəru/kutru/kuṭaro/kuṭara/kuṭri dog responses fit this rhotic family; source medial vowels and dental/retroflex t/ṭ remain intact, as do ordinary final-vowel forms. This uses the specifically attested rhotic kuttira family rather than assigning all dog words to kutta. Local IA transmission remains unresolved; mixed huṇo-plus-kuṭro responses are excluded.')]
forms={'kutəro','kutəru','kutru','kuṭaro','kuṭara','kuṭri'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in remaining and r['Gloss']=='dog' and r['Form'] in forms:
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare232.py').read_text());print('accepted',len(acc))

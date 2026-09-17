import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass149';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='10893',citation='CDIAL[10893]',evidence='CDIAL 10893 lagna explicitly gives Prakrit lagga touching/connected, Oriya lāga contiguous, Gujarati lāgu near, Marathi lāgī̃ near, and oblique forms used as postpositions. The near responses in lag/lagge/loge match this family rather than a present-tense verb under lagyati 10895. Regional vowel height, gemination, aspiration and final vowels are qualified and preserved; exact local Indo-Aryan transmission is unresolved.'),dict(parent='10893',citation='CDIAL[10893]',evidence='The addendum to CDIAL 10893 lagna explicitly gives Old Punjabi lavai near/equal to, and Punjabi/Hindi lave. Selected lave/love/lava near responses fit that documented postpositional family; vowels and the local transmission route remain qualified, and source notation is retained.')]
sets=[{'līgah','legʰā̃','lagɛ','lɛgā','lɛgɛ','ligʰe','lʌgːe','ləgə','ləg','lag','loge','ləge','lʌge','lege','leghe','lage','logge','logke','ləgghə','ləggɛ','ləggye','lɪghyɛ','ləgge'},{'lave','love','lava'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='near':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

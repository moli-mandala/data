import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass190';assert not (P/(stem+'-decisions.json')).exists()
rules=[
 dict(parent='13291',citation='CDIAL[13291]',evidence='Full *savēla article gives Awankari savēl/savēlā/savēlē in the morning, Punjabi saver morning/dawn, and Gujarati saveḷā/saverā. Both alternatives of savela / savele belong to this family and are preserved as one source response. Bundeli suvere preserves the initial vowel variation; Bhili haver preserves the regional s-to-h correspondence. Local IA transmission remains unresolved.'),
 dict(parent='8707',citation='CDIAL[8707]',evidence='Full prabhāta means daybreak and includes Pali pabhāta dawn. Selected perbāt/pārbʰāt morning retain the pr-bh-t frame with metathesis/epenthesis and aspiration differences stated as qualifications; the retained medial consonant suggests a learned or regionally transmitted form rather than the fully reduced Prakrit pahāya branch. Local IA mediation versus learned transmission is unresolved. Nasal/spirant pārəvā̃t is excluded pending closer evidence.'),
 dict(parent='8747',citation='CDIAL[8747];CDIAL[8707]',evidence='Full prarōha ray/sprout article explicitly derives Gujarati paroḍh/paroṛ dawn through *parōha-ḍa, explaining o better than prabhāta. Full prabhāta article likewise prefers prarōha for these forms. Malvi paroḍo morning matches the regional dawn family with final vowel and absent aspiration retained as qualifications. The link follows Turner’s explicit preference, retaining the historical alternative rather than treating the search duplicate under 8707-2 as independent evidence. Local IA transmission is unresolved.')]
sets=[{'savela / savele','suvere','haver'},{'perbāt','pārbʰāt'},{'paroḍo'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='morning':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))

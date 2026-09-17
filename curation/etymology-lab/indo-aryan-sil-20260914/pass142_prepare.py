import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass142';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9498',citation='CDIAL[9498]',evidence='CDIAL 9498 bhinna explicitly gives different, Bengali/Assamese bhin separate and Punjabi bhinn-bhinn separate/apart. These simple and repeated bhin-/bin- forms support the family, retaining deaspiration, vowel endings and Bengali learned/geminate nn. Repetition is distributive. Responses containing additional lexical items or unexplained d are excluded; local Indo-Aryan transmission remains unresolved.'),dict(parent='404',citation='CDIAL[404]',evidence='CDIAL 404 *anyākāra explicitly gives Punjabi neārā, Hindi niyārā/nyārā different/separate and Gujarati nyārũ. Selected western nyara/nyare/nyaru forms and repetitions fit this regional family. CDIAL 7277 *nirālaya was compared: its contracted nyār is specifically Kumaoni lonely/aloof, whereas the present forms are western different responses. Vowel quantity and inflection remain as elicited; local transmission is unresolved.'),dict(parent='11826-2',citation='CDIAL[11826]',evidence='CDIAL 11826 explicitly marks the l-extension Prakrit veggala separate/distinct and Nepali beglo separate/different. Danuwar/Majhi begle and repeated begle are assigned to that specific extended node, retaining the final vowel and repetition. The competing 11827 *viyagra concerns beglo foolish, not the attested separate/different sense used here. Local Indo-Aryan transmission remains unresolved.')]
sets=[{'bʰini','bʰino','bine bine','bhinno','bhinːo','bin','bʰin','bʰin bʰin','bʰine bʰine','bin bin'}, {'nyare nyare','nyara nyara','nyarānyarā','nyaru'}, {'begle','begle begle'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='different':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

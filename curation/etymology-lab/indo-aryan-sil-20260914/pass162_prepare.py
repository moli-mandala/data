import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass162';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9757',citation='CDIAL[9757]',evidence='CDIAL 9757 matsara explicitly gives Sindhi macharu, Lahnda macchur, northern macchar and Gujarati machrũ mosquito/gnat. The selected macchar/machar/machriya series preserves that r-bearing family, with aspiration, vowel variation, rhotic notation and regional iya endings qualified. The entry discusses but does not establish a deeper connection with makṣa; this link does not settle it or local transmission.'),dict(parent='9917',citation='CDIAL[9917]',evidence='CDIAL 9917 maśaka explicitly gives Bengali maśā, Oriya masā, Assamese mah and Kashmiri moh(u) mosquito. The selected moṣa/moha/mesa series fits this mosquito family, preserving sibilant and h variation and vowels. Local IA transmission and any learned reinforcement remain unresolved.'),dict(parent='6110',citation='CDIAL[6110]',evidence='CDIAL 6110 daṃśa explicitly includes Bhojpuri ḍās mosquito, Hindi ḍā̃s large mosquito, eastern dās/ḍās and Assamese ḍā̃h gadfly. The simple daś/dah mosquito forms fit the documented biting-insect family, with nasal absence, dental/retroflex notation and s/h development qualified. The survey mosquito sense is preserved; exact local transmission remains unresolved.')]
sets=[{'macʰor','macʰal','mõsər','miciriyā','micəriyũ','misiryā','miciryā','macəriyā','macad','məśər','maśar','machaṛ','macheṛ','motʃəɽ','macriya','macariya'}, {'moha','moṣa','mesa'}, {'daśa','ḍāsyā','daha','ḍaha'}]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='mosquito':continue
 for i,forms in enumerate(sets):
  if r['Form'] in forms:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'));break
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print('accepted',len(acc))

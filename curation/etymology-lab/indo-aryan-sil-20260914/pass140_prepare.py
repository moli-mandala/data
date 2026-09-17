import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass140';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='14024',citation='CDIAL[14024]',evidence='CDIAL 14024 hasta explicitly gives Maiya hā and Ky. hā̃ with oblique hātha, alongside nearby hath/hatth forms. Both alternatives in Maiya hā / hatʰ hand belong to the documented family. The slash and aspiration notation remain unchanged; this does not assert that the survey pair is specifically a direct/oblique paradigm. Local transmission is unresolved.'),dict(parent='5082',citation='CDIAL[5082]',evidence='CDIAL 5082 jaṅghā explicitly gives Assamese zāṅ leg, Bengali jāṅ/jāṅi thigh/leg and northwestern jaṅ/jaṅg leg forms. Bishnupriya jaŋ and Kullui dʒəŋ elicited foot are linked to this lower-limb family with the foot-versus-leg semantic range explicitly qualified; no source gloss is changed or exact anatomical boundary inferred. Consonant simplification and vowels remain as transcribed; local Indo-Aryan transmission is unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Language_ID']=='Mai' and r['Form']=='hā / hatʰ' and r['Gloss']=='hand' else 1 if r['Gloss']=='foot' and r['Form'] in {'jaŋ','dʒəŋ'} else None
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

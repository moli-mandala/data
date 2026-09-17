import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass146';assert not (P/(stem+'-decisions.json')).exists()
base='Platts page 621 comparative sā like/resembling is explicitly derived through a Prakrit form from sadṛśa plus ka. It is distinct from the adjacent intensifying sā entry. CDIAL 13120 supplies the sadṛśa like family. These same responses are interpreted as comparative ek + sā/so/sī, preserving gender/vowel forms, spacing and regional pronunciation. The sadṛśa link records the base of the documented suffixed derivation, not an unsuffixed direct sound change. Local transmission is unresolved.'
rules=[dict(parent='13120',citation='CDIAL[13120];platts1884[621]',evidence=base),dict(parent='2462-2',citation='CDIAL[2462.2];CDIAL[13120];platts1884[621]',evidence='Ordered components one + like: CDIAL 2462 subsection 2 explicitly gives modern ek/āk one. '+base)]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
simple=set()
compound={'āk sɔ','ek sā','ek sī','ek sɔ','ekso','eksā','ɛkso','ek si','ek sa'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='same':continue
 i=0 if r['Form'] in simple else 1 if r['Form'] in compound else None
 if i is not None:
  q=rules[i];a=dict(record=r,parent=q['parent'],family=i,kind='component' if i else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
  if i:a['components']=['2462-2','13120']
  acc.append(a)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print(len(acc),sum(len(x.get('components',[x['parent']])) for x in acc))

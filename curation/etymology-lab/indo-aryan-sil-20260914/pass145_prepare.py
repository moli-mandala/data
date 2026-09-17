import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass145';assert not (P/(stem+'-decisions.json')).exists()
base='CDIAL 10458 yādṛśa explicitly gives Prakrit jaïsa, Nepali jaso, Hindi jaisā, Old Marwari jaïsaü and Marathi jasā. Selected jaisa/jesa/jasa forms match this sort/likeness family; the survey same sense expresses equivalence. Regional vowels, affrication and the śy notation in jɛśyɔ remain qualified, not normalized. Local Indo-Aryan transmission is unresolved.'
rules=[dict(parent='10458',citation='CDIAL[10458]',evidence=base),dict(parent='2462-2',citation='CDIAL[2462.2];CDIAL[10458]',evidence='Ordered components ek + jaisa: CDIAL 2462 subsection 2 *ekka explicitly gives Hindi and Gujarati ek one. '+base+' Both components are represented while preserving the original spacing and inflection; no ancient compound is reconstructed.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
simple={'jasā','jeso'}
compound={'ek jāīsɔ','ek jesā','ek jesa','ek jɛśyɔ','ekjaiso','ekdʒæse','ek jaso','ekjeso','ekjaso','ekjesa','ekjesā','ekjasā'}
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='same':continue
 i=0 if r['Form'] in simple else 1 if r['Form'] in compound else None
 if i is not None:
  q=rules[i];a=dict(record=r,parent=q['parent'],family=i,kind='component' if i else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
  if i:a['components']=['2462-2','10458']
  acc.append(a)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));print(len(acc),sum(len(x.get('components',[x['parent']])) for x in acc))
